from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_dau(daily: pd.DataFrame, outdir: str) -> str:
    out = ensure_output_dir(outdir) / "dau_trend.png"
    plt.figure(figsize=(10, 5))
    plt.plot(daily["date"], daily["dau"])
    plt.xticks(rotation=45)
    plt.title("DAU Trend")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return str(out)


def plot_event_volume(daily: pd.DataFrame, outdir: str) -> str:
    out = ensure_output_dir(outdir) / "event_volume.png"
    totals = daily[["view", "addtocart", "transaction"]].sum()
    plt.figure(figsize=(6, 4))
    plt.bar(totals.index, totals.values)
    plt.title("Event Volume by Type")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return str(out)


def plot_conversion_rates(daily: pd.DataFrame, outdir: str) -> str:
    out = ensure_output_dir(outdir) / "conversion_rates.png"
    plt.figure(figsize=(8, 4))
    plt.plot(daily["date"], daily["view_to_cart_rate"], label="view_to_cart_rate")
    plt.plot(daily["date"], daily["cart_to_buy_rate"], label="cart_to_buy_rate")
    plt.plot(daily["date"], daily["view_to_buy_rate"], label="view_to_buy_rate")
    plt.legend()
    plt.xticks(rotation=45)
    plt.title("Conversion Rates Over Time")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return str(out)


def plot_funnel(funnel: pd.DataFrame, outdir: str) -> str:
    out = ensure_output_dir(outdir) / "funnel.png"
    plt.figure(figsize=(6, 4))
    plt.bar(funnel["step"], funnel["users"])
    plt.title("User Funnel")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return str(out)


def plot_retention_curve(retention_user_level: pd.DataFrame, outdir: str) -> str:
    out = ensure_output_dir(outdir) / "retention_curve.png"
    curve = retention_user_level.groupby("horizon")["retained"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    plt.plot(curve["horizon"], curve["retained"], marker="o")
    plt.title("Retention Curve")
    plt.xlabel("Days Since First Visit")
    plt.ylabel("Retention")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return str(out)


def plot_cohort_heatmap(cohort_retention: pd.DataFrame, outdir: str) -> str:
    out = ensure_output_dir(outdir) / "cohort_retention.png"
    arr = cohort_retention.to_numpy(dtype=float)
    plt.figure(figsize=(10, 5))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="Retention")
    plt.yticks(range(len(cohort_retention.index)), cohort_retention.index)
    plt.xticks(range(len(cohort_retention.columns)), cohort_retention.columns)
    plt.title("Monthly Cohort Retention")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return str(out)


def plot_feature_importance(importance_df: pd.DataFrame, outdir: str, topn: int = 15) -> str:
    out = ensure_output_dir(outdir) / "feature_importance.png"
    top = importance_df.head(topn).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return str(out)