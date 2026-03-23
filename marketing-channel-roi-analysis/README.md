# Marketing Channel ROI Analysis

## Business Problem

A mid-size B2B SaaS company allocates marketing budget across six channels — Google Ads, LinkedIn Ads, Email, Content/SEO, Webinars, and Trade Shows — but lacks a data-driven framework for evaluating which channels deliver the best return on investment. Leadership needs evidence-based guidance on where to increase or decrease spend, and whether the optimal channel mix varies by customer segment.

## Approach

This project performs a rigorous statistical analysis of 500 marketing campaigns spanning Q3 2024 through Q2 2025:

1. **Exploratory Analysis**: Distribution profiling, summary statistics, and visualization of ROI, CTR, cost-per-lead, and conversion rates across channels.
2. **Hypothesis Testing**: Kruskal-Wallis H-test (non-parametric) with Dunn's post-hoc pairwise comparisons and Bonferroni correction to identify statistically significant ROI differences between channels.
3. **Interaction Effects**: Two-way ANOVA on log-transformed ROI to test whether customer segment (Enterprise / Mid-Market / SMB) moderates channel effectiveness.
4. **Regression Modeling**: OLS regression to quantify the independent effect of channel, segment, region, spend level, CTR, and campaign duration on ROI.
5. **Funnel Efficiency**: Comparative analysis of click-through rates, lead conversion rates, cost-per-lead, and revenue-per-conversion across channels.

## Key Findings

- Statistically significant differences in ROI exist across channels (Kruskal-Wallis p < 0.05)
- Channel effectiveness varies by customer segment — the best channel for Enterprise prospects is not necessarily the best for SMB
- Click-through rate and campaign duration are significant predictors of ROI in multivariate regression
- High-cost channels (Trade Shows) deliver large deal sizes but lower ROI per dollar compared to lower-cost channels (Email, Webinars)

## Tech Stack

- **Python 3.10+**
- **pandas** — data manipulation and aggregation
- **matplotlib / seaborn** — statistical visualizations
- **scipy** — hypothesis testing (Kruskal-Wallis, Mann-Whitney U, Shapiro-Wilk)
- **statsmodels** — two-way ANOVA and OLS regression
- **NumPy** — numerical computation

## Project Structure

```
marketing-channel-roi-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── analysis.ipynb          # Full analysis notebook with narrative
└── data/
    ├── campaign_performance.csv   # Synthetic dataset (500 campaigns)
    └── generate_data.py           # Data generation script
```

## How to Run

```bash
# Clone and set up
git clone <repo-url>
cd marketing-channel-roi-analysis
pip install -r requirements.txt

# (Optional) Regenerate synthetic data
python data/generate_data.py

# Run the analysis
jupyter notebook analysis.ipynb
```

## Sample Output

The analysis produces:
- Channel ROI comparison (box plots, bar charts)
- Spend vs. revenue scatter plots by channel
- Heatmap of median ROI by channel × customer segment
- Regression coefficient plot showing significant ROI drivers
- Funnel efficiency dashboard (CTR, conversion rates, cost-per-lead)
