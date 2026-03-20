"""
Consulting Utilization Dashboard
================================
Interactive Streamlit dashboard for monitoring consultant utilization rates,
billable revenue, practice area performance, and workforce allocation across
a mid-size consulting firm.

Usage:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Consultant Utilization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR: Path = Path(__file__).parent / "data"

LEVEL_TARGET_UTILIZATION: dict[str, float] = {
    "Analyst": 0.85,
    "Senior Analyst": 0.82,
    "Consultant": 0.78,
    "Senior Consultant": 0.72,
    "Manager": 0.60,
    "Director": 0.45,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and prepare all datasets with proper typing."""
    consultants = pd.read_csv(DATA_DIR / "consultants.csv")
    projects = pd.read_csv(DATA_DIR / "projects.csv")
    timesheets = pd.read_csv(DATA_DIR / "timesheets.csv")

    # Parse dates
    consultants["hire_date"] = pd.to_datetime(consultants["hire_date"])
    projects["start_date"] = pd.to_datetime(projects["start_date"])
    projects["end_date"] = pd.to_datetime(projects["end_date"])
    timesheets["week_start"] = pd.to_datetime(timesheets["week_start"])
    timesheets["week_end"] = pd.to_datetime(timesheets["week_end"])

    return consultants, projects, timesheets


consultants, projects, timesheets = load_data()

# Merge consultant metadata into timesheets for analysis
ts = timesheets.merge(consultants, on="consultant_id", how="left")


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

# Date range
min_date = ts["week_start"].min().date()
max_date = ts["week_start"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Practice area filter
all_practices = sorted(ts["practice_area"].unique())
selected_practices = st.sidebar.multiselect(
    "Practice Area",
    options=all_practices,
    default=all_practices,
)

# Level filter
all_levels = ["Analyst", "Senior Analyst", "Consultant", "Senior Consultant", "Manager", "Director"]
selected_levels = st.sidebar.multiselect(
    "Consultant Level",
    options=all_levels,
    default=all_levels,
)

# Apply filters
if len(date_range) == 2:
    start_filter, end_filter = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_filter, end_filter = pd.Timestamp(min_date), pd.Timestamp(max_date)

mask = (
    (ts["week_start"] >= start_filter)
    & (ts["week_start"] <= end_filter)
    & (ts["practice_area"].isin(selected_practices))
    & (ts["level"].isin(selected_levels))
)
filtered = ts[mask].copy()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def compute_utilization(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """Compute utilization rate as billable hours / 40h available per week."""
    if group_col:
        weekly = df.groupby([group_col, "week_start"]).agg(
            billable=("billable_hours", "sum"),
            n_consultants=("consultant_id", "nunique"),
        ).reset_index()
        weekly["available"] = weekly["n_consultants"] * 40.0
        weekly["utilization"] = (weekly["billable"] / weekly["available"]).clip(0, 1)
        return weekly
    else:
        weekly = df.groupby("week_start").agg(
            billable=("billable_hours", "sum"),
            n_consultants=("consultant_id", "nunique"),
        ).reset_index()
        weekly["available"] = weekly["n_consultants"] * 40.0
        weekly["utilization"] = (weekly["billable"] / weekly["available"]).clip(0, 1)
        return weekly


def format_pct(val: float) -> str:
    """Format a float as a percentage string."""
    return f"{val:.1%}"


def format_currency(val: float) -> str:
    """Format a number as USD currency."""
    return f"${val:,.0f}"


# ---------------------------------------------------------------------------
# Dashboard header
# ---------------------------------------------------------------------------
st.title("Consulting Utilization Dashboard")
st.caption("Workforce performance metrics for firm leadership and practice leads")

# ---------------------------------------------------------------------------
# KPI cards (top row)
# ---------------------------------------------------------------------------
total_billable = filtered["billable_hours"].sum()
total_available = filtered["consultant_id"].nunique() * 40.0 * filtered["week_start"].nunique()
overall_utilization = total_billable / total_available if total_available > 0 else 0

# Estimated revenue
revenue_df = filtered.merge(
    consultants[["consultant_id", "bill_rate_usd"]], on="consultant_id", how="left", suffixes=("", "_lookup")
)
bill_rate_col = "bill_rate_usd_lookup" if "bill_rate_usd_lookup" in revenue_df.columns else "bill_rate_usd"
total_revenue = (revenue_df["billable_hours"] * revenue_df[bill_rate_col]).sum()

n_active_consultants = filtered["consultant_id"].nunique()
avg_billable_per_week = total_billable / filtered["week_start"].nunique() if filtered["week_start"].nunique() > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Utilization", format_pct(overall_utilization))
col2.metric("Total Billable Hours", f"{total_billable:,.0f}")
col3.metric("Estimated Revenue", format_currency(total_revenue))
col4.metric("Active Consultants", n_active_consultants)

st.divider()

# ---------------------------------------------------------------------------
# Row 1: Utilization trend + Practice area breakdown
# ---------------------------------------------------------------------------
row1_left, row1_right = st.columns([3, 2])

with row1_left:
    st.subheader("Firm-Wide Utilization Trend")
    weekly_util = compute_utilization(filtered)
    weekly_util["utilization_pct"] = weekly_util["utilization"] * 100

    fig_trend = px.area(
        weekly_util,
        x="week_start",
        y="utilization_pct",
        labels={"week_start": "Week", "utilization_pct": "Utilization (%)"},
        color_discrete_sequence=["#4F46E5"],
    )
    fig_trend.add_hline(
        y=70,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="70% Target",
        annotation_position="top left",
    )
    fig_trend.update_layout(
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=10, b=20),
        height=350,
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with row1_right:
    st.subheader("Utilization by Practice Area")
    practice_util = (
        filtered.groupby("practice_area")
        .agg(
            billable=("billable_hours", "sum"),
            n_weeks=("week_start", "nunique"),
            n_consultants=("consultant_id", "nunique"),
        )
        .reset_index()
    )
    practice_util["utilization"] = (
        practice_util["billable"]
        / (practice_util["n_consultants"] * 40 * practice_util["n_weeks"])
    ).clip(0, 1) * 100

    fig_practice = px.bar(
        practice_util.sort_values("utilization", ascending=True),
        x="utilization",
        y="practice_area",
        orientation="h",
        labels={"utilization": "Utilization (%)", "practice_area": ""},
        color="utilization",
        color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
        range_color=[0, 100],
    )
    fig_practice.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        height=350,
        showlegend=False,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_practice, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 2: Utilization by level + Revenue by practice
# ---------------------------------------------------------------------------
row2_left, row2_right = st.columns(2)

with row2_left:
    st.subheader("Utilization by Consultant Level")
    level_data = (
        filtered.groupby("level")
        .agg(
            billable=("billable_hours", "sum"),
            n_weeks=("week_start", "nunique"),
            n_consultants=("consultant_id", "nunique"),
        )
        .reset_index()
    )
    level_data["actual_util"] = (
        level_data["billable"] / (level_data["n_consultants"] * 40 * level_data["n_weeks"])
    ).clip(0, 1) * 100

    # Add target utilization
    target_map = {k: v * 100 for k, v in LEVEL_TARGET_UTILIZATION.items()}
    level_data["target_util"] = level_data["level"].map(target_map)

    # Order by seniority
    level_order = ["Analyst", "Senior Analyst", "Consultant", "Senior Consultant", "Manager", "Director"]
    level_data["level"] = pd.Categorical(level_data["level"], categories=level_order, ordered=True)
    level_data = level_data.sort_values("level")

    fig_level = go.Figure()
    fig_level.add_trace(go.Bar(
        x=level_data["level"],
        y=level_data["actual_util"],
        name="Actual",
        marker_color="#4F46E5",
    ))
    fig_level.add_trace(go.Scatter(
        x=level_data["level"],
        y=level_data["target_util"],
        name="Target",
        mode="markers+lines",
        marker=dict(size=10, color="#EF4444"),
        line=dict(dash="dash", color="#EF4444"),
    ))
    fig_level.update_layout(
        yaxis_title="Utilization (%)",
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=10, b=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_level, use_container_width=True)

with row2_right:
    st.subheader("Revenue by Practice Area")
    rev_by_practice = revenue_df.copy()
    rev_by_practice["revenue"] = rev_by_practice["billable_hours"] * rev_by_practice[bill_rate_col]
    practice_revenue = (
        rev_by_practice.groupby("practice_area")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    fig_rev = px.pie(
        practice_revenue,
        values="revenue",
        names="practice_area",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4,
    )
    fig_rev.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        height=350,
    )
    fig_rev.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_rev, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 3: Individual consultant heatmap
# ---------------------------------------------------------------------------
st.subheader("Individual Consultant Utilization (Weekly Heatmap)")

# Compute per-consultant weekly utilization
individual = filtered.groupby(["name", "week_start"]).agg(
    billable=("billable_hours", "sum"),
).reset_index()
individual["utilization"] = (individual["billable"] / 40.0).clip(0, 1) * 100

# Pivot for heatmap
pivot = individual.pivot_table(
    index="name",
    columns="week_start",
    values="utilization",
    aggfunc="mean",
).fillna(0)

# Sort by average utilization
pivot["avg"] = pivot.mean(axis=1)
pivot = pivot.sort_values("avg", ascending=False).drop(columns="avg")

# Limit to top 20 for readability
if len(pivot) > 20:
    pivot = pivot.head(20)
    st.caption("Showing top 20 consultants by average utilization")

fig_heatmap = px.imshow(
    pivot,
    labels=dict(x="Week", y="Consultant", color="Utilization %"),
    color_continuous_scale=["#FEE2E2", "#FCA5A5", "#F59E0B", "#10B981", "#065F46"],
    aspect="auto",
    zmin=0,
    zmax=100,
)
fig_heatmap.update_layout(
    margin=dict(l=20, r=20, t=10, b=20),
    height=500,
    xaxis=dict(tickformat="%b %d", dtick="M1"),
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 4: Hours breakdown + At-risk consultants
# ---------------------------------------------------------------------------
row4_left, row4_right = st.columns(2)

with row4_left:
    st.subheader("Hours Breakdown Over Time")
    hours_weekly = (
        filtered.groupby("week_start")
        .agg(
            Billable=("billable_hours", "sum"),
            Internal=("internal_hours", "sum"),
            Admin=("admin_hours", "sum"),
            PTO=("pto_hours", "sum"),
        )
        .reset_index()
    )
    hours_melted = hours_weekly.melt(
        id_vars="week_start",
        value_vars=["Billable", "Internal", "Admin", "PTO"],
        var_name="Category",
        value_name="Hours",
    )

    fig_stack = px.area(
        hours_melted,
        x="week_start",
        y="Hours",
        color="Category",
        labels={"week_start": "Week"},
        color_discrete_map={
            "Billable": "#4F46E5",
            "Internal": "#8B5CF6",
            "Admin": "#A78BFA",
            "PTO": "#D1D5DB",
        },
    )
    fig_stack.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

with row4_right:
    st.subheader("At-Risk Consultants (Below Target)")
    # Compute individual utilization vs target
    individual_summary = (
        filtered.groupby(["consultant_id", "name", "level", "practice_area", "target_utilization"])
        .agg(
            total_billable=("billable_hours", "sum"),
            weeks_active=("week_start", "nunique"),
        )
        .reset_index()
    )
    individual_summary["actual_util"] = (
        individual_summary["total_billable"]
        / (individual_summary["weeks_active"] * 40)
    )
    individual_summary["gap"] = (
        individual_summary["actual_util"] - individual_summary["target_utilization"]
    )
    at_risk = (
        individual_summary[individual_summary["gap"] < -0.05]
        .sort_values("gap")
        .head(10)
    )

    if not at_risk.empty:
        display_df = at_risk[["name", "level", "practice_area", "actual_util", "target_utilization", "gap"]].copy()
        display_df.columns = ["Name", "Level", "Practice", "Actual", "Target", "Gap"]
        display_df["Actual"] = display_df["Actual"].apply(format_pct)
        display_df["Target"] = display_df["Target"].apply(format_pct)
        display_df["Gap"] = display_df["Gap"].apply(lambda x: f"{x:+.1%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=350)
    else:
        st.success("All consultants are meeting or exceeding their utilization targets.")

# ---------------------------------------------------------------------------
# Row 5: Project allocation table
# ---------------------------------------------------------------------------
st.subheader("Active Project Allocation")
project_hours = (
    filtered[filtered["project_id"].notna()]
    .groupby("project_id")
    .agg(
        total_billable=("billable_hours", "sum"),
        n_consultants=("consultant_id", "nunique"),
        weeks_active=("week_start", "nunique"),
    )
    .reset_index()
    .merge(projects[["project_id", "client", "project_type", "status"]], on="project_id", how="left")
    .sort_values("total_billable", ascending=False)
)
project_hours["avg_weekly_hours"] = (project_hours["total_billable"] / project_hours["weeks_active"]).round(1)

display_projects = project_hours[[
    "project_id", "client", "project_type", "status",
    "n_consultants", "total_billable", "avg_weekly_hours",
]].copy()
display_projects.columns = [
    "Project ID", "Client", "Type", "Status",
    "Team Size", "Total Hours", "Avg Weekly Hours",
]
display_projects["Total Hours"] = display_projects["Total Hours"].apply(lambda x: f"{x:,.0f}")

st.dataframe(display_projects, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Data is synthetically generated for demonstration purposes. "
    "Dashboard built with Streamlit and Plotly. "
    f"Showing {len(filtered):,} timesheet records across {n_active_consultants} consultants."
)


