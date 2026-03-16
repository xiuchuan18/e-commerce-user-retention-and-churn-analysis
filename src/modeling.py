import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import build_window_features

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


def build_churn_dataset(
    df: pd.DataFrame,
    feature_window_days: int = 30,
    churn_window_days: int = 14,
    window_list=(7, 14, 30),
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    定义：
    - anchor_date = max_date - churn_window_days
    - feature 使用 anchor_date 及其之前的行为
    - label: anchor_date 之后 churn_window_days 内无行为 => churn=1
    """

    max_date = df["date"].max()
    anchor_date = max_date - pd.Timedelta(days=churn_window_days)
    feature_start = anchor_date - pd.Timedelta(days=feature_window_days - 1)

    # 1) feature 窗口内用户
    feature_df = df[(df["date"] >= feature_start) & (df["date"] <= anchor_date)].copy()
    future_df = df[(df["date"] > anchor_date) & (df["date"] <= max_date)].copy()

    if feature_df.empty:
        raise ValueError("Feature window is empty. Please check the date range and window settings.")

    # 2) 基础聚合特征
    base = (
        feature_df.groupby("visitorid")
        .agg(
            feat_n_events=("event", "count"),
            feat_n_active_days=("date", "nunique"),
            feat_n_items=("itemid", "nunique"),
            feat_first_active=("date", "min"),
            feat_last_active=("date", "max"),
            feat_avg_hour=("hour", "mean"),
            feat_weekend_ratio=("is_weekend", "mean"),
            feat_night_ratio=("hour", lambda x: np.mean((x >= 20) | (x <= 6))),
        )
    )

    event_pivot = (
        feature_df.pivot_table(index="visitorid", columns="event", values="itemid", aggfunc="count", fill_value=0)
        .rename(columns=lambda c: f"feat_cnt_{c}")
    )

    ds = base.join(event_pivot, how="left").fillna(0).reset_index()

    for c in ["feat_cnt_view", "feat_cnt_addtocart", "feat_cnt_transaction"]:
        if c not in ds.columns:
            ds[c] = 0

    ds["feat_days_since_last_active"] = (anchor_date - ds["feat_last_active"]).dt.days
    ds["feat_lifetime_days"] = (ds["feat_last_active"] - ds["feat_first_active"]).dt.days + 1
    ds["feat_views_per_day"] = ds["feat_cnt_view"] / ds["feat_n_active_days"].replace(0, np.nan)
    ds["feat_carts_per_day"] = ds["feat_cnt_addtocart"] / ds["feat_n_active_days"].replace(0, np.nan)
    ds["feat_buys_per_day"] = ds["feat_cnt_transaction"] / ds["feat_n_active_days"].replace(0, np.nan)
    ds["feat_view_to_cart_ratio"] = ds["feat_cnt_addtocart"] / ds["feat_cnt_view"].replace(0, np.nan)
    ds["feat_cart_to_buy_ratio"] = ds["feat_cnt_transaction"] / ds["feat_cnt_addtocart"].replace(0, np.nan)
    ds["feat_view_to_buy_ratio"] = ds["feat_cnt_transaction"] / ds["feat_cnt_view"].replace(0, np.nan)

    # 3) 多时间窗口特征
    window_feat = build_window_features(feature_df, anchor_date=anchor_date, windows=window_list)
    ds = ds.merge(window_feat, on="visitorid", how="left")

    # 4) future label
    future_active = future_df.groupby("visitorid").size().rename("future_events")
    ds = ds.merge(future_active, on="visitorid", how="left")
    ds["future_events"] = ds["future_events"].fillna(0)

    ds["churn"] = (ds["future_events"] == 0).astype(int)

    # 5) 丢掉原始 datetime 列，保留派生数值列
    drop_cols = ["feat_first_active", "feat_last_active"]
    for c in drop_cols:
        if c in ds.columns:
            ds = ds.drop(columns=c)

    numeric_cols = ds.select_dtypes(include=[np.number]).columns
    ds[numeric_cols] = ds[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return ds, anchor_date, feature_start, max_date


def _find_best_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    rows = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1})

    df = pd.DataFrame(rows).sort_values(["f1", "recall", "precision"], ascending=False).reset_index(drop=True)
    best_t = float(df.iloc[0]["threshold"])
    return best_t, df


def extract_feature_importance(pipe, model_name: str, feature_cols):
    model = pipe.named_steps["model"]

    if model_name == "logistic_regression":
        coef = model.coef_[0]
        out = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": np.abs(coef),
                "signed_coef": coef,
            }
        )
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        out = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": fi,
            }
        )
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    return None


def train_models(ds: pd.DataFrame):
    feature_cols = [c for c in ds.columns if c.startswith("feat_") or c.startswith("w")]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(ds[c])]

    X = ds[feature_cols].copy()
    y = ds["churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    numeric_features = feature_cols

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "gbdt": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    if HAS_LGBM:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    results = []
    fitted_models = {}
    threshold_tables = {}

    for name, model in models.items():
        if name == "logistic_regression":
            pipe = Pipeline(
                steps=[
                    ("prep", preprocessor),
                    ("model", model),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    (
                        "prep",
                        ColumnTransformer(
                            transformers=[
                                ("num", SimpleImputer(strategy="median"), numeric_features)
                            ]
                        ),
                    ),
                    ("model", model),
                ]
            )

        pipe.fit(X_train, y_train)

        prob = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
        ap = average_precision_score(y_test, prob)

        best_threshold, threshold_df = _find_best_threshold(y_test, prob)
        pred = (prob >= best_threshold).astype(int)

        p, r, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )

        results.append(
            {
                "model": name,
                "auc": auc,
                "average_precision": ap,
                "precision": p,
                "recall": r,
                "f1": f1,
                "best_threshold": best_threshold,
            }
        )

        fitted_models[name] = pipe
        threshold_tables[name] = threshold_df

    result_df = pd.DataFrame(results).sort_values(
        ["auc", "f1", "average_precision"], ascending=False
    ).reset_index(drop=True)

    best_model_name = result_df.iloc[0]["model"]
    best_threshold = float(result_df.iloc[0]["best_threshold"])
    best_model = fitted_models[best_model_name]

    best_prob = best_model.predict_proba(X_test)[:, 1]
    best_pred = (best_prob >= best_threshold).astype(int)

    test_pred = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_prob": best_prob,
            "y_pred": best_pred,
        }
    )

    report = classification_report(y_test, best_pred, zero_division=0)

    importance_df = extract_feature_importance(best_model, best_model_name, feature_cols)

    return result_df, best_model_name, best_model, test_pred, report, importance_df