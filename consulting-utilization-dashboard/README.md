# Consulting Utilization Dashboard

An interactive Streamlit dashboard for monitoring consultant workforce utilization, billable revenue, and project allocation at a mid-size consulting firm.

## Business Problem

Consulting firms live and die by utilization — the percentage of available hours that consultants spend on billable client work. Underutilization means lost revenue, while chronic overutilization signals burnout risk. Practice leads and firm leadership need real-time visibility into utilization trends across the organization to make informed staffing, hiring, and business development decisions.

This dashboard provides that visibility through a single, filterable interface that surfaces firm-wide utilization trends, practice area performance, individual consultant heatmaps, revenue breakdowns, and at-risk consultants falling below target.

## Features

- **KPI Summary Cards**: Overall utilization rate, total billable hours, estimated revenue, and active headcount
- **Utilization Trend**: Week-over-week firm utilization with target threshold overlay
- **Practice Area Breakdown**: Horizontal bar comparison of utilization across practice areas
- **Level Analysis**: Actual vs. target utilization by consultant seniority level
- **Revenue by Practice**: Donut chart showing revenue contribution by practice area
- **Individual Heatmap**: Weekly utilization heatmap for top consultants, revealing patterns and gaps
- **Hours Breakdown**: Stacked area chart showing billable, internal, admin, and PTO hours over time
- **At-Risk Consultants**: Table of consultants significantly below their utilization target
- **Project Allocation**: Active project summary with team size, total hours, and weekly averages

All views respond to sidebar filters for date range, practice area, and consultant level.

## Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web application framework
- **Plotly** — interactive charts and visualizations
- **pandas** — data manipulation and aggregation

## How to Run

```bash
# Clone the repository and navigate to this project
cd consulting-utilization-dashboard

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (already included, but regenerate if needed)
python generate_data.py

# Launch the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

## Project Structure

```
consulting-utilization-dashboard/
├── app.py                  # Main Streamlit dashboard application
├── generate_data.py        # Synthetic data generation script
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
└── data/
    ├── consultants.csv     # Consultant roster (35 consultants)
    ├── projects.csv        # Client engagements (20 projects)
    └── timesheets.csv      # Weekly timesheet records (~2,000 rows)
```

## Sample Output

The dashboard renders a multi-panel layout with:

- A top KPI bar showing ~65-75% overall utilization, 39K+ billable hours, and $10M+ estimated revenue
- An area chart revealing seasonal patterns (December dip, Q1 recovery)
- Practice-level comparison highlighting which teams are over/under-utilized
- A consultant-level heatmap where green cells indicate high utilization weeks and red cells indicate bench time
- An at-risk table flagging consultants more than 5 percentage points below their level's target

## Data Notes

All data is synthetically generated using realistic distributions. The data generator simulates seasonal patterns (holiday slowdowns, summer dips), PTO behavior, multi-project assignments, and seniority-based utilization targets. No real client or employee data is used.
