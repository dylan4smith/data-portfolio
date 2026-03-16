"""
SaaS Metrics Dashboard — Interactive KPI Explorer

A Streamlit application that provides executive-level visibility into
key SaaS business metrics: MRR growth, churn analysis, unit economics,
and customer health scoring.

Designed for founders, VPs of Finance, and RevOps teams who need
a single pane of glass for subscription business performance.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
BRAND_COLORS = {
    "primary": "#4F46E5",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "neutral": "#6B7280",
}

st.set_page_config(
    page_title="SaaS Metrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_monthly_metrics() -> pd.DataFrame:
    """Load and enrich the monthly metrics CSV."""
    df = pd.read_csv(DATA_DIR / "monthly_metrics.csv", parse_dates=["month"])
    df["arr"] = df["mrr"] * 12
    df["mrr_growth_pct"] = df["mrr"].pct_change() * 100
    df["customers_net_change"] = (
        df["new_customers"] - df["churned_customers"] + df["reactivated_customers"]
    )
    return df


@st.cache_data
def load_segments() -> list[dict[str, Any]]:
    """Load customer segment data."""
    with open(DATA_DIR / "customer_segments.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helper formatters
# ---------------------------------------------------------------------------


def fmt_currency(value: float) -> str:
    """Format a number as USD currency."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a number as a percentage string."""
    return f"{value:.{decimals}f}%"


def delta_color(value: float, higher_is_better: bool = True) -> str:
    """Return 'normal' or 'inverse' for st.metric delta_color."""
    return "normal" if higher_is_better else "inverse"


# ---------------------------------------------------------------------------
# Sidebar — filters & context
# ---------------------------------------------------------------------------


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered dataframe."""
    st.sidebar.header("Filters")

    min_date = df["month"].min().to_pydatetime()
    max_date = df["month"].max().to_pydatetime()

    date_range = st.sidebar.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="MMM YYYY",
    )

    filtered = df[(df["month"] >= date_range[0]) & (df["month"] <= date_range[1])].copy()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**About this dashboard**\n\n"
        "Interactive explorer for subscription business KPIs. "
        "Data is synthetic — generated for portfolio demonstration purposes."
    )
    return filtered


# ---------------------------------------------------------------------------
# KPI cards row
# ---------------------------------------------------------------------------


def render_kpi_cards(df: pd.DataFrame) -> None:
    """Display top-level KPI metric cards."""
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        mrr_delta = latest["mrr"] - previous["mrr"]
        st.metric(
            "Monthly Recurring Revenue",
            fmt_currency(latest["mrr"]),
            delta=fmt_currency(mrr_delta),
            delta_color=delta_color(mrr_delta),
        )

    with col2:
        arr = latest["arr"]
        arr_delta = arr - previous["arr"]
        st.metric(
            "Annual Run Rate",
            fmt_currency(arr),
            delta=fmt_currency(arr_delta),
            delta_color=delta_color(arr_delta),
        )

    with col3:
        st.metric(
            "Active Customers",
            f"{latest['active_customers']:,}",
            delta=f"{int(latest['customers_net_change']):+d}",
            delta_color=delta_color(latest["customers_net_change"]),
        )

    with col4:
        churn_delta = latest["gross_churn_rate"] - previous["gross_churn_rate"]
        st.metric(
            "Gross Churn Rate",
            fmt_pct(latest["gross_churn_rate"] * 100),
            delta=fmt_pct(churn_delta * 100),
            delta_color=delta_color(churn_delta, higher_is_better=False),
        )

    with col5:
        ltv_cac = latest["ltv_cac_ratio"]
        ltv_cac_delta = ltv_cac - previous["ltv_cac_ratio"]
        st.metric(
            "LTV / CAC",
            f"{ltv_cac:.1f}x",
            delta=f"{ltv_cac_delta:+.1f}x",
            delta_color=delta_color(ltv_cac_delta),
        )


# ---------------------------------------------------------------------------
# Revenue section
# ---------------------------------------------------------------------------


def render_revenue_section(df: pd.DataFrame) -> None:
    """Charts related to MRR and revenue composition."""
    st.subheader("Revenue Growth")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_mrr = go.Figure()
        fig_mrr.add_trace(
            go.Scatter(
                x=df["month"],
                y=df["mrr"],
                mode="lines+markers",
                name="MRR",
                line=dict(color=BRAND_COLORS["primary"], width=3),
                marker=dict(size=5),
            )
        )
        fig_mrr.update_layout(
            title="Monthly Recurring Revenue (MRR)",
            yaxis_title="USD",
            yaxis_tickformat="$,.0f",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_mrr, use_container_width=True)

    with col_right:
        fig_rev = go.Figure()
        fig_rev.add_trace(
            go.Bar(x=df["month"], y=df["expansion_revenue"], name="Expansion", marker_color=BRAND_COLORS["success"])
        )
        fig_rev.add_trace(
            go.Bar(x=df["month"], y=-df["contraction_revenue"], name="Contraction", marker_color=BRAND_COLORS["danger"])
        )
        fig_rev.update_layout(
            title="Expansion vs. Contraction Revenue",
            barmode="relative",
            yaxis_title="USD",
            yaxis_tickformat="$,.0f",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_rev, use_container_width=True)


# ---------------------------------------------------------------------------
# Customer section
# ---------------------------------------------------------------------------


def render_customer_section(df: pd.DataFrame) -> None:
    """Charts related to customer acquisition and retention."""
    st.subheader("Customer Health")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_cust = go.Figure()
        fig_cust.add_trace(
            go.Bar(x=df["month"], y=df["new_customers"], name="New", marker_color=BRAND_COLORS["success"])
        )
        fig_cust.add_trace(
            go.Bar(x=df["month"], y=-df["churned_customers"], name="Churned", marker_color=BRAND_COLORS["danger"])
        )
        fig_cust.add_trace(
            go.Bar(x=df["month"], y=df["reactivated_customers"], name="Reactivated", marker_color=BRAND_COLORS["warning"])
        )
        fig_cust.update_layout(
            title="Customer Movements",
            barmode="relative",
            yaxis_title="Customers",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_cust, use_container_width=True)

    with col_right:
        fig_churn = go.Figure()
        fig_churn.add_trace(
            go.Scatter(
                x=df["month"],
                y=df["gross_churn_rate"] * 100,
                mode="lines+markers",
                name="Gross Churn %",
                line=dict(color=BRAND_COLORS["danger"], width=2),
            )
        )
        fig_churn.add_trace(
            go.Scatter(
                x=df["month"],
                y=df["net_revenue_churn"] * 100,
                mode="lines+markers",
                name="Net Revenue Churn %",
                line=dict(color=BRAND_COLORS["warning"], width=2, dash="dash"),
            )
        )
        fig_churn.update_layout(
            title="Churn Trends",
            yaxis_title="Percentage (%)",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_churn, use_container_width=True)


# ---------------------------------------------------------------------------
# Unit economics section
# ---------------------------------------------------------------------------


def render_unit_economics(df: pd.DataFrame) -> None:
    """Charts for LTV, CAC, and payback metrics."""
    st.subheader("Unit Economics")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_ltv = go.Figure()
        fig_ltv.add_trace(
            go.Scatter(x=df["month"], y=df["ltv"], name="LTV", line=dict(color=BRAND_COLORS["primary"], width=2))
        )
        fig_ltv.add_trace(
            go.Scatter(x=df["month"], y=df["cac"], name="CAC", line=dict(color=BRAND_COLORS["danger"], width=2))
        )
        fig_ltv.update_layout(
            title="LTV vs. CAC Over Time",
            yaxis_title="USD",
            yaxis_tickformat="$,.0f",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_ltv, use_container_width=True)

    with col_right:
        fig_ratio = go.Figure()
        fig_ratio.add_trace(
            go.Scatter(
                x=df["month"],
                y=df["ltv_cac_ratio"],
                mode="lines+markers",
                name="LTV/CAC",
                line=dict(color=BRAND_COLORS["success"], width=3),
                fill="tozeroy",
                fillcolor="rgba(16,185,129,0.1)",
            )
        )
        fig_ratio.add_hline(y=3.0, line_dash="dot", line_color=BRAND_COLORS["neutral"], annotation_text="3x Target")
        fig_ratio.update_layout(
            title="LTV / CAC Ratio",
            yaxis_title="Ratio (x)",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_ratio, use_container_width=True)


# ---------------------------------------------------------------------------
# Segment breakdown
# ---------------------------------------------------------------------------


def render_segment_breakdown(segments: list[dict[str, Any]]) -> None:
    """Pie chart and table for customer segments."""
    st.subheader("Customer Segments")

    seg_df = pd.DataFrame(segments)

    col_left, col_right = st.columns(2)

    with col_left:
        fig_pie = px.pie(
            seg_df,
            values="mrr_share",
            names="segment",
            title="MRR Share by Segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig_pie.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        display_df = seg_df.copy()
        display_df["mrr_share"] = display_df["mrr_share"].apply(lambda x: f"{x:.0%}")
        display_df["churn_rate"] = display_df["churn_rate"].apply(lambda x: f"{x:.1%}")
        display_df.columns = ["Segment", "Customers", "MRR Share", "Monthly Churn", "Avg Contract (mo)"]
        st.markdown("**Segment Breakdown**")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Support & NPS section
# ---------------------------------------------------------------------------


def render_support_nps(df: pd.DataFrame) -> None:
    """Charts for support ticket volume and NPS trend."""
    st.subheader("Support & Customer Satisfaction")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_tickets = go.Figure()
        fig_tickets.add_trace(
            go.Bar(x=df["month"], y=df["support_tickets_opened"], name="Opened", marker_color=BRAND_COLORS["warning"])
        )
        fig_tickets.add_trace(
            go.Bar(x=df["month"], y=df["support_tickets_resolved"], name="Resolved", marker_color=BRAND_COLORS["success"])
        )
        fig_tickets.update_layout(
            title="Support Ticket Volume",
            barmode="group",
            yaxis_title="Tickets",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_tickets, use_container_width=True)

    with col_right:
        fig_nps = go.Figure()
        fig_nps.add_trace(
            go.Scatter(
                x=df["month"],
                y=df["nps_score"],
                mode="lines+markers",
                name="NPS",
                line=dict(color=BRAND_COLORS["primary"], width=3),
                fill="tozeroy",
                fillcolor="rgba(79,70,229,0.08)",
            )
        )
        fig_nps.add_hline(y=50, line_dash="dot", line_color=BRAND_COLORS["success"], annotation_text="Excellent (50+)")
        fig_nps.update_layout(
            title="Net Promoter Score (NPS) Trend",
            yaxis_title="NPS",
            yaxis_range=[0, 100],
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_nps, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — compose the dashboard layout."""
    st.title("📊 SaaS Metrics Dashboard")
    st.caption("Executive-level KPI tracking for subscription businesses")

    df = load_monthly_metrics()
    segments = load_segments()

    filtered_df = render_sidebar(df)

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    render_kpi_cards(filtered_df)
    st.markdown("---")
    render_revenue_section(filtered_df)
    st.markdown("---")
    render_customer_section(filtered_df)
    st.markdown("---")
    render_unit_economics(filtered_df)
    st.markdown("---")
    render_segment_breakdown(segments)
    st.markdown("---")
    render_support_nps(filtered_df)

    # Footer
    st.markdown("---")
    st.caption(
        "Built with Streamlit & Plotly · Data is synthetic · "
        "[View source on GitHub](https://github.com/dylansno17/github-portfolio)"
    )


if __name__ == "__main__":
    main()
