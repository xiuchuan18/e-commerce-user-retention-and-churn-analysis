import numpy as np
import pandas as pd


def _safe_ratio(a, b):
    return a / b.replace(0, np.nan)


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["date"].max() + pd.Timedelta(days=1)

    purchase_df = df[df["event"] == "transaction"].copy()
    if purchase_df.empty:
        purchase_df = df.copy()
        monetary_proxy_name = "all_events"
    else:
        monetary_proxy_name = "transactions"

    rfm = (
        purchase_df.groupby("visitorid")
        .agg(
            recency=("date", lambda x: (snapshot_date - x.max()).days),
            frequency=("date", "count"),
            monetary_proxy=("itemid", "nunique"),
        )
        .reset_index()
    )

    rfm["source"] = monetary_proxy_name

    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary_proxy"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def label_user(score):
        if score >= 10:
            return "core_users"
        elif score >= 7:
            return "active_users"
        elif score >= 5:
            return "at_risk_users"
        return "low_value_users"

    rfm["segment"] = rfm["rfm_score"].apply(label_user)
    return rfm


def build_user_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    max_date = df["date"].max()

    base = (
        df.groupby("visitorid")
        .agg(
            n_events=("event", "count"),
            n_active_days=("date", "nunique"),
            n_items=("itemid", "nunique"),
            first_active=("date", "min"),
            last_active=("date", "max"),
            avg_hour=("hour", "mean"),
            weekend_ratio=("is_weekend", "mean"),
            night_ratio=("hour", lambda x: np.mean((x >= 20) | (x <= 6))),
        )
    )

    event_pivot = (
        df.pivot_table(index="visitorid", columns="event", values="itemid", aggfunc="count", fill_value=0)
        .rename(columns=lambda c: f"cnt_{c}")
    )

    feat = base.join(event_pivot, how="left").fillna(0).reset_index()

    for c in ["cnt_view", "cnt_addtocart", "cnt_transaction"]:
        if c not in feat.columns:
            feat[c] = 0

    feat["days_since_last_active"] = (max_date - feat["last_active"]).dt.days
    feat["lifetime_days"] = (feat["last_active"] - feat["first_active"]).dt.days + 1

    feat["views_per_day"] = feat["cnt_view"] / feat["n_active_days"].replace(0, np.nan)
    feat["carts_per_day"] = feat["cnt_addtocart"] / feat["n_active_days"].replace(0, np.nan)
    feat["buys_per_day"] = feat["cnt_transaction"] / feat["n_active_days"].replace(0, np.nan)

    feat["view_to_cart_ratio"] = _safe_ratio(feat["cnt_addtocart"], feat["cnt_view"])
    feat["cart_to_buy_ratio"] = _safe_ratio(feat["cnt_transaction"], feat["cnt_addtocart"])
    feat["view_to_buy_ratio"] = _safe_ratio(feat["cnt_transaction"], feat["cnt_view"])

    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    feat[numeric_cols] = feat[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return feat


def build_window_features(df: pd.DataFrame, anchor_date: pd.Timestamp, windows=(7, 14, 30)) -> pd.DataFrame:
    """
    以 anchor_date 为截点，构建多个时间窗口行为特征。
    """
    all_users = pd.DataFrame({"visitorid": df["visitorid"].astype(str).unique()})

    out = all_users.copy()

    for w in windows:
        start_date = anchor_date - pd.Timedelta(days=w - 1)
        sub = df[(df["date"] >= start_date) & (df["date"] <= anchor_date)].copy()

        if sub.empty:
            tmp = all_users.copy()
            tmp[f"w{w}_n_events"] = 0
            tmp[f"w{w}_n_active_days"] = 0
            tmp[f"w{w}_n_items"] = 0
            tmp[f"w{w}_avg_hour"] = np.nan
            tmp[f"w{w}_weekend_ratio"] = np.nan
            tmp[f"w{w}_night_ratio"] = np.nan
        else:
            base = (
                sub.groupby("visitorid")
                .agg(
                    **{
                        f"w{w}_n_events": ("event", "count"),
                        f"w{w}_n_active_days": ("date", "nunique"),
                        f"w{w}_n_items": ("itemid", "nunique"),
                        f"w{w}_avg_hour": ("hour", "mean"),
                        f"w{w}_weekend_ratio": ("is_weekend", "mean"),
                        f"w{w}_night_ratio": ("hour", lambda x: np.mean((x >= 20) | (x <= 6))),
                    }
                )
                .reset_index()
            )

            event_pivot = (
                sub.pivot_table(index="visitorid", columns="event", values="itemid", aggfunc="count", fill_value=0)
                .rename(columns=lambda c: f"w{w}_cnt_{c}")
                .reset_index()
            )

            tmp = base.merge(event_pivot, on="visitorid", how="left")

        for c in [f"w{w}_cnt_view", f"w{w}_cnt_addtocart", f"w{w}_cnt_transaction"]:
            if c not in tmp.columns:
                tmp[c] = 0

        tmp[f"w{w}_views_per_day"] = tmp[f"w{w}_cnt_view"] / tmp[f"w{w}_n_active_days"].replace(0, np.nan)
        tmp[f"w{w}_carts_per_day"] = tmp[f"w{w}_cnt_addtocart"] / tmp[f"w{w}_n_active_days"].replace(0, np.nan)
        tmp[f"w{w}_buys_per_day"] = tmp[f"w{w}_cnt_transaction"] / tmp[f"w{w}_n_active_days"].replace(0, np.nan)

        tmp[f"w{w}_view_to_cart_ratio"] = _safe_ratio(tmp[f"w{w}_cnt_addtocart"], tmp[f"w{w}_cnt_view"])
        tmp[f"w{w}_cart_to_buy_ratio"] = _safe_ratio(tmp[f"w{w}_cnt_transaction"], tmp[f"w{w}_cnt_addtocart"])
        tmp[f"w{w}_view_to_buy_ratio"] = _safe_ratio(tmp[f"w{w}_cnt_transaction"], tmp[f"w{w}_cnt_view"])

        out = out.merge(tmp, on="visitorid", how="left")

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return out