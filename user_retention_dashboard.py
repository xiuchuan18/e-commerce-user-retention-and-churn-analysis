from pathlib import Path
import streamlit as st
import pandas as pd

from src.data_processing import (
    basic_summary,
    build_cohort_table,
    build_daily_metrics,
    compute_retention,
    funnel_summary,
    load_events,
)
from src.feature_engineering import build_rfm
from src.modeling import build_churn_dataset, train_models


st.set_page_config(page_title="E-commerce Retention Dashboard", layout="wide")

st.title("E-commerce User Retention & Churn Dashboard")
st.caption("Interactive portfolio dashboard based on the RetailRocket clickstream dataset.")

st.markdown(
    """
This dashboard summarizes:
- user behavior overview
- DAU and conversion metrics
- funnel analysis
- retention and cohort analysis
- RFM segmentation
- optional churn modeling
"""
)

with st.sidebar:
    st.header("Settings")
    sample_size = st.number_input(
        "Sample rows for faster analysis (0 = full data)",
        min_value=0,
        value=300000,
        step=50000,
    )
    run_model = st.checkbox("Run churn model", value=False)

uploaded_file = st.file_uploader("Upload RetailRocket events.csv", type=["csv"])


@st.cache_data
def load_data(file, sample_size: int) -> pd.DataFrame:
    temp_path = Path("temp_events.csv")
    temp_path.write_bytes(file.getvalue())

    if sample_size > 0:
        raw = pd.read_csv(temp_path, nrows=sample_size)
        sampled_path = Path("temp_events_sampled.csv")
        raw.to_csv(sampled_path, index=False)
        df = load_events(str(sampled_path))
    else:
        df = load_events(str(temp_path))

    return df


if uploaded_file is None:
    st.info("Please upload a RetailRocket-style events.csv file to start.")
    st.stop()

with st.spinner("Loading and cleaning data..."):
    df = load_data(uploaded_file, sample_size)

with st.spinner("Computing daily metrics, funnel, retention, cohort and RFM..."):
    summary = basic_summary(df)
    daily = build_daily_metrics(df)
    funnel = funnel_summary(df)
    retention_user = compute_retention(df, horizons=(1, 7, 14, 30))
    retention_curve = retention_user.groupby("horizon")["retained"].mean().reset_index()
    cohort = build_cohort_table(df)
    rfm = build_rfm(df)

st.subheader("Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Events", f"{summary['n_rows']:,}")
c2.metric("Users", f"{summary['n_users']:,}")
c3.metric("Items", f"{summary['n_items']:,}")
c4.metric("Days", f"{summary['n_days']:,}")

c5, c6, c7 = st.columns(3)
c5.metric("Date Range", f"{summary['start_date']} → {summary['end_date']}")
c6.metric("Avg DAU", f"{daily['dau'].mean():,.0f}")
c7.metric("View→Buy", f"{daily['view_to_buy_rate'].mean():.2%}")

st.markdown("---")

st.subheader("Daily Metrics")
tab1, tab2 = st.tabs(["DAU Trend", "Conversion Trend"])

with tab1:
    st.line_chart(daily.set_index("date")[["dau"]])

with tab2:
    st.line_chart(
        daily.set_index("date")[["view_to_cart_rate", "cart_to_buy_rate", "view_to_buy_rate"]]
    )

st.markdown("---")

st.subheader("Funnel Analysis")
f1, f2 = st.columns([1, 1])

with f1:
    st.dataframe(funnel, use_container_width=True)

with f2:
    st.bar_chart(funnel.set_index("step")["users"])

if len(funnel) >= 3:
    cart_rate = funnel.loc[funnel["step"] == "addtocart", "step_rate_vs_view"].iloc[0]
    buy_rate = funnel.loc[funnel["step"] == "transaction", "step_rate_vs_view"].iloc[0]
    st.info(
        f"Current sample shows view→cart = {cart_rate:.2%}, overall view→buy = {buy_rate:.2%}."
    )

st.markdown("---")

st.subheader("Retention")
r1, r2 = st.columns([1, 1])

with r1:
    st.dataframe(retention_curve, use_container_width=True)

with r2:
    st.line_chart(retention_curve.set_index("horizon")["retained"])

ret_dict = dict(zip(retention_curve["horizon"], retention_curve["retained"]))
st.success(
    f"D1={ret_dict.get(1, float('nan')):.2%} | "
    f"D7={ret_dict.get(7, float('nan')):.2%} | "
    f"D14={ret_dict.get(14, float('nan')):.2%} | "
    f"D30={ret_dict.get(30, float('nan')):.2%}"
)

st.markdown(
    "Note: retention is naturally low in this dataset because visitors are anonymous browser IDs rather than logged-in users."
)

st.markdown("---")

st.subheader("Monthly Cohort Retention")
st.dataframe(cohort.style.format("{:.2%}"), use_container_width=True)

st.markdown("---")

st.subheader("RFM Segmentation")
segment_dist = rfm["segment"].value_counts().rename_axis("segment").reset_index(name="users")

rc1, rc2 = st.columns([1, 1])
with rc1:
    st.dataframe(segment_dist, use_container_width=True)
with rc2:
    st.bar_chart(segment_dist.set_index("segment")["users"])

st.markdown("---")

st.subheader("Churn Modeling")

if run_model:
    with st.spinner("Building churn dataset and training models..."):
        churn_ds, anchor_date, feature_start, max_date = build_churn_dataset(df)
        model_result, best_model_name, best_model, test_pred, report, importance_df = train_models(churn_ds)

    st.caption(
        f"Feature window: {feature_start.date()} to {anchor_date.date()} | "
        f"Label window: {(anchor_date + pd.Timedelta(days=1)).date()} to {max_date.date()}"
    )

    st.dataframe(model_result, use_container_width=True)

    if not model_result.empty:
        best_row = model_result.iloc[0]
        st.success(
            f"Best model = {best_row['model']} | "
            f"AUC = {best_row['auc']:.3f} | "
            f"AP = {best_row['average_precision']:.3f} | "
            f"F1 = {best_row['f1']:.3f}"
        )

    if importance_df is not None:
        st.subheader("Top Feature Importance")
        show_cols = [c for c in ["feature", "importance", "signed_coef"] if c in importance_df.columns]
        st.dataframe(importance_df[show_cols].head(20), use_container_width=True)

    st.subheader("Prediction Sample")
    st.dataframe(test_pred.head(50), use_container_width=True)

    st.subheader("Classification Report")
    st.code(report)
else:
    st.warning("Turn on 'Run churn model' in the sidebar to train and show model results.")

st.markdown("---")

st.subheader("Business Takeaways")
st.markdown(
    f"""
- The dataset contains **{summary['n_rows']:,} events** and **{summary['n_users']:,} users** in the current run.
- Average DAU is **{daily['dau'].mean():,.0f}**.
- Average view→buy conversion is **{daily['view_to_buy_rate'].mean():.2%}**.
- Funnel analysis suggests that the biggest drop-off happens before add-to-cart.
- Retention decays quickly, which is expected for anonymous browsing logs.
- RFM segmentation helps identify core, active, at-risk, and low-value users.
"""
)

st.caption("Use notebooks for full analysis details and this dashboard for portfolio-style presentation.")