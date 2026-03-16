# SaaS Metrics Dashboard

An interactive Streamlit dashboard that provides executive-level visibility into key subscription business KPIs — MRR growth, churn analysis, unit economics, and customer health scoring.

## Business Problem

SaaS leadership teams often lack a single, real-time view of the metrics that matter most: recurring revenue trajectory, customer retention health, and the efficiency of their go-to-market spend. Spreadsheet-based reporting is slow, error-prone, and rarely interactive enough to support ad-hoc exploration during board meetings or planning sessions.

This dashboard solves that by consolidating 24 months of subscription business data into a filterable, drill-down-ready interface that any stakeholder can use without SQL knowledge.

## Key Metrics Tracked

- **MRR & ARR** — Monthly and annualized recurring revenue with month-over-month deltas
- **Expansion & Contraction Revenue** — Upsell/cross-sell vs. downgrades
- **Gross & Net Revenue Churn** — Logo churn and dollar-weighted retention
- **LTV / CAC Ratio** — Customer lifetime value relative to acquisition cost, benchmarked against the 3x industry target
- **NPS Trend** — Net Promoter Score over time with threshold indicators
- **Support Health** — Ticket volume, resolution rates, and average resolution time
- **Customer Segments** — Enterprise / Mid-Market / SMB / Startup breakdown by MRR share and churn rate

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend & Server | Streamlit 1.x |
| Visualization | Plotly / Plotly Express |
| Data Processing | pandas |
| Data Generation | Python stdlib (csv, json, random) |
| Language | Python 3.10+ |

## Project Structure

```
saas-metrics-dashboard/
├── app.py                     # Main Streamlit application
├── data/
│   ├── generate_data.py       # Synthetic data generator (reproducible via seed)
│   ├── monthly_metrics.csv    # 24 months of SaaS KPIs
│   └── customer_segments.json # Segment-level breakdown
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

```bash
# Clone the repository
git clone https://github.com/dylansno17/github-portfolio.git
cd github-portfolio/saas-metrics-dashboard

# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Regenerate synthetic data
python data/generate_data.py

# Launch the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

## Sample Output

The dashboard renders five sections:

1. **KPI Cards** — Top-line metrics with period-over-period deltas (MRR, ARR, active customers, churn rate, LTV/CAC)
2. **Revenue Growth** — MRR trend line and expansion vs. contraction waterfall
3. **Customer Health** — Acquisition/churn bar chart and dual-line churn trend
4. **Unit Economics** — LTV vs. CAC time series and ratio tracking against the 3x benchmark
5. **Segment Breakdown** — Donut chart of MRR share and tabular churn by segment
6. **Support & NPS** — Ticket volume comparison and NPS score trajectory

All charts are interactive (zoom, hover tooltips, pan) and respond to the sidebar date range filter.

## Data Notes

All data is **synthetic**, generated with a fixed random seed for reproducibility. The generator (`data/generate_data.py`) simulates realistic growth patterns including seasonal acquisition spikes, improving churn as the "product matures," and correlated expansion/contraction dynamics.
