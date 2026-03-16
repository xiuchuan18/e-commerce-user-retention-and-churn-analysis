from pathlib import Path
import numpy as np
import pandas as pd


def safe_div(a, b):
    return a / b if b not in (0, 0.0, None) else np.nan


def load_events(events_path: str) -> pd.DataFrame:
    df = pd.read_csv(events_path)
    cols = {c.lower(): c for c in df.columns}

    required = ["timestamp", "visitorid", "event", "itemid"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Existing columns: {list(df.columns)}")

    df = df.rename(
        columns={
            cols["timestamp"]: "timestamp",
            cols["visitorid"]: "visitorid",
            cols["event"]: "event",
            cols["itemid"]: "itemid",
        }
    )

    if "transactionid" in cols:
        df = df.rename(columns={cols["transactionid"]: "transactionid"})
    else:
        df["transactionid"] = np.nan

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df = df.dropna(subset=["timestamp", "visitorid", "event", "itemid"]).copy()

    df["visitorid"] = df["visitorid"].astype(str)
    df["itemid"] = df["itemid"].astype(str)
    df["event"] = df["event"].astype(str).str.lower().str.strip()

    valid_events = {"view", "addtocart", "transaction"}
    df = df[df["event"].isin(valid_events)].copy()

    df["date"] = pd.to_datetime(df["timestamp"].dt.date)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    df = df.drop_duplicates(subset=["visitorid", "itemid", "event", "timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def basic_summary(df: pd.DataFrame) -> dict:
    return {
        "n_rows": len(df),
        "n_users": df["visitorid"].nunique(),
        "n_items": df["itemid"].nunique(),
        "start_date": str(df["date"].min().date()),
        "end_date": str(df["date"].max().date()),
        "n_days": df["date"].nunique(),
    }


def build_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    daily_users = df.groupby("date")["visitorid"].nunique().rename("dau")
    daily_events = df.groupby(["date", "event"]).size().unstack(fill_value=0)

    daily = pd.concat([daily_users, daily_events], axis=1).fillna(0).reset_index()

    for col in ["view", "addtocart", "transaction"]:
        if col not in daily.columns:
            daily[col] = 0

    daily["view_to_cart_rate"] = daily.apply(lambda x: safe_div(x["addtocart"], x["view"]), axis=1)
    daily["cart_to_buy_rate"] = daily.apply(lambda x: safe_div(x["transaction"], x["addtocart"]), axis=1)
    daily["view_to_buy_rate"] = daily.apply(lambda x: safe_div(x["transaction"], x["view"]), axis=1)

    return daily.sort_values("date").reset_index(drop=True)


def funnel_summary(df: pd.DataFrame) -> pd.DataFrame:
    users_by_step = (
        df.groupby("event")["visitorid"]
        .nunique()
        .reindex(["view", "addtocart", "transaction"])
        .fillna(0)
        .astype(int)
    )

    funnel = pd.DataFrame(
        {
            "step": ["view", "addtocart", "transaction"],
            "users": users_by_step.values,
        }
    )

    funnel["step_rate_vs_prev"] = [
        1.0,
        safe_div(funnel.loc[1, "users"], funnel.loc[0, "users"]),
        safe_div(funnel.loc[2, "users"], funnel.loc[1, "users"]),
    ]
    funnel["step_rate_vs_view"] = funnel["users"] / max(funnel.loc[0, "users"], 1)
    return funnel


def compute_retention(df: pd.DataFrame, horizons=(1, 7, 14, 30)) -> pd.DataFrame:
    user_dates = df.groupby("visitorid")["date"].apply(lambda x: sorted(set(x))).to_dict()

    records = []
    for user, dates in user_dates.items():
        first_date = dates[0]
        date_set = set(dates)

        for h in horizons:
            retained = pd.Timestamp(first_date + pd.Timedelta(days=h)) in date_set
            records.append(
                {
                    "visitorid": user,
                    "cohort_date": first_date,
                    "horizon": h,
                    "retained": int(retained),
                }
            )

    ret = pd.DataFrame(records)
    out = ret.groupby(["cohort_date", "horizon"])["retained"].mean().reset_index()
    return out


def build_cohort_table(df: pd.DataFrame) -> pd.DataFrame:
    user_first = df.groupby("visitorid")["date"].min().rename("cohort_date")
    tmp = df[["visitorid", "date"]].drop_duplicates().merge(user_first, on="visitorid", how="left")

    tmp["cohort_month"] = tmp["cohort_date"].dt.to_period("M").astype(str)
    tmp["activity_month"] = tmp["date"].dt.to_period("M").astype(str)

    cohort_period = pd.PeriodIndex(tmp["cohort_month"], freq="M")
    activity_period = pd.PeriodIndex(tmp["activity_month"], freq="M")

    tmp["cohort_index"] = (
        (activity_period.year - cohort_period.year) * 12
        + (activity_period.month - cohort_period.month)
    )

    cohort = (
        tmp.groupby(["cohort_month", "cohort_index"])["visitorid"]
        .nunique()
        .reset_index()
    )

    pivot = cohort.pivot(index="cohort_month", columns="cohort_index", values="visitorid").fillna(0)
    base = pivot[0].replace(0, np.nan)
    retention = pivot.div(base, axis=0)

    return retention