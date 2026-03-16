"""
Standalone script to run the full analysis without Jupyter.
Executes all analysis steps and saves figures to figures/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import math
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

# Configuration
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 100,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 10,
})
sns.set_theme(style="whitegrid", palette="muted")

print("=" * 60)
print("REGIONAL SALES PERFORMANCE ANALYSIS")
print("=" * 60)

# --- Load Data ---
df = pd.read_csv("data/regional_sales_data.csv", parse_dates=["date"])
print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

region_order = df.groupby("region")["revenue"].median().sort_values(ascending=False).index

# --- Figure 1: Revenue by Region ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(data=df, x="region", y="revenue", order=region_order, ax=axes[0])
axes[0].set_title("Revenue Distribution by Region")
axes[0].set_ylabel("Monthly Revenue ($)")
axes[0].set_xlabel("")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

sns.violinplot(data=df, x="region", y="revenue", order=region_order, ax=axes[1], inner="quartile")
axes[1].set_title("Revenue Distribution Shape by Region")
axes[1].set_ylabel("Monthly Revenue ($)")
axes[1].set_xlabel("")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig("figures/01_revenue_by_region.png", bbox_inches="tight")
plt.close()
print("✓ Figure 1: Revenue by region")

# --- Figure 2: Monthly Trends ---
monthly = df.groupby(["date", "region"])["revenue"].sum().reset_index()
fig, ax = plt.subplots(figsize=(14, 6))
for region in region_order:
    subset = monthly[monthly["region"] == region]
    ax.plot(subset["date"], subset["revenue"], marker="o", markersize=4, label=region, linewidth=2)
ax.set_title("Total Monthly Revenue by Region (2024–2025)")
ax.set_ylabel("Total Revenue ($)")
ax.set_xlabel("")
ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
plt.tight_layout()
plt.savefig("figures/02_monthly_trends.png", bbox_inches="tight")
plt.close()
print("✓ Figure 2: Monthly trends")

# --- Figure 3: Product Mix ---
product_region = df.groupby(["region", "product_line"])["revenue"].sum().reset_index()
product_region_pct = product_region.pivot(index="region", columns="product_line", values="revenue")
product_region_pct = product_region_pct.div(product_region_pct.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
product_region_pct.loc[region_order].plot(kind="barh", stacked=True, ax=axes[0], colormap="Set2")
axes[0].set_title("Revenue Mix by Region (% of Total)")
axes[0].set_xlabel("Share of Revenue (%)")
axes[0].legend(title="Product Line", bbox_to_anchor=(1.0, -0.15), ncol=3)

product_abs = product_region.pivot(index="region", columns="product_line", values="revenue")
product_abs.loc[region_order].plot(kind="barh", stacked=True, ax=axes[1], colormap="Set2")
axes[1].set_title("Absolute Revenue by Region & Product")
axes[1].set_xlabel("Revenue ($)")
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
axes[1].legend().remove()
plt.tight_layout()
plt.savefig("figures/03_product_mix.png", bbox_inches="tight")
plt.close()
print("✓ Figure 3: Product mix")

# --- ANOVA ---
def one_way_anova(*groups) -> Tuple[float, float, bool]:
    k = len(groups)
    N = sum(len(g) for g in groups)
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_between = k - 1
    df_within = N - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within
    df1, df2 = df_between, df_within
    z = (f_stat ** (1/3) * (1 - 2/(9*df2)) - (1 - 2/(9*df1))) / \
        np.sqrt(2/(9*df1) + f_stat ** (2/3) * 2/(9*df2))
    p_value = 0.5 * (1 + math.erf(-z / np.sqrt(2)))
    return f_stat, p_value, p_value < 0.05

region_groups = [df[df["region"] == r]["revenue"].values for r in sorted(df["region"].unique())]
f_stat, p_value, significant = one_way_anova(*region_groups)

print(f"\n{'='*60}")
print("ONE-WAY ANOVA: Revenue by Region")
print(f"{'='*60}")
print(f"F-statistic:  {f_stat:.4f}")
print(f"p-value:      {p_value:.2e}")
print(f"Significant:  {'Yes' if significant else 'No'} (α = 0.05)")

# --- Pairwise t-tests ---
def welch_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    n1, n2 = len(a), len(b)
    mean1, mean2 = a.mean(), b.mean()
    var1, var2 = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(var1/n1 + var2/n2)
    t_stat = (mean1 - mean2) / se
    z = abs(t_stat)
    p_value = 2 * 0.5 * (1 + math.erf(-z / np.sqrt(2)))
    return t_stat, p_value

regions = sorted(df["region"].unique())
n_comparisons = len(regions) * (len(regions) - 1) // 2
alpha_bonferroni = 0.05 / n_comparisons
sig_count = 0
for i in range(len(regions)):
    for j in range(i + 1, len(regions)):
        a = df[df["region"] == regions[i]]["revenue"].values
        b = df[df["region"] == regions[j]]["revenue"].values
        _, p_val = welch_ttest(a, b)
        if p_val < alpha_bonferroni:
            sig_count += 1
print(f"\nPairwise tests: {sig_count}/{n_comparisons} pairs significantly different (Bonferroni α={alpha_bonferroni:.4f})")

# --- Figure 4: Correlation Matrix ---
numeric_cols = ["revenue", "units_sold", "discount_pct", "customer_satisfaction", "deal_cycle_days"]
corr_matrix = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, ax=ax, linewidths=0.5)
ax.set_title("Correlation Matrix: Sales Performance Metrics")
plt.tight_layout()
plt.savefig("figures/04_correlation_matrix.png", bbox_inches="tight")
plt.close()
print("✓ Figure 4: Correlation matrix")

# --- Figure 5: Discount Analysis ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(df["discount_pct"] * 100, df["revenue"], alpha=0.1, s=10, c="steelblue")
z = np.polyfit(df["discount_pct"], df["revenue"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["discount_pct"].min(), df["discount_pct"].max(), 100)
axes[0].plot(x_line * 100, p(x_line), color="red", linewidth=2)
axes[0].set_xlabel("Discount (%)")
axes[0].set_ylabel("Revenue ($)")
axes[0].set_title("Revenue vs Discount Rate")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

axes[1].scatter(df["discount_pct"] * 100, df["customer_satisfaction"], alpha=0.1, s=10, c="darkorange")
z2 = np.polyfit(df["discount_pct"], df["customer_satisfaction"], 1)
p2 = np.poly1d(z2)
axes[1].plot(x_line * 100, p2(x_line), color="red", linewidth=2)
axes[1].set_xlabel("Discount (%)")
axes[1].set_ylabel("Customer Satisfaction (1-5)")
axes[1].set_title("Satisfaction vs Discount Rate")

df["discount_quartile"] = pd.qcut(df["discount_pct"], 4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
sns.boxplot(data=df, x="discount_quartile", y="revenue", ax=axes[2])
axes[2].set_title("Revenue by Discount Quartile")
axes[2].set_ylabel("Revenue ($)")
axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig("figures/05_discount_analysis.png", bbox_inches="tight")
plt.close()
print("✓ Figure 5: Discount analysis")

# --- Figure 6: Seasonal Analysis ---
monthly_total = df.groupby("date").agg(
    total_revenue=("revenue", "sum"),
    total_units=("units_sold", "sum"),
).reset_index()
monthly_total["month"] = monthly_total["date"].dt.month
monthly_total["year"] = monthly_total["date"].dt.year

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
pivot = monthly_total.pivot(index="year", columns="month", values="total_revenue")
pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="YlOrRd", ax=axes[0, 0])
axes[0, 0].set_title("Monthly Revenue Heatmap")

seasonal_idx = df.groupby(df["date"].dt.month)["revenue"].mean()
overall_mean = df["revenue"].mean()
seasonal_factor = (seasonal_idx / overall_mean * 100).values
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
colors = ["#e74c3c" if s < 100 else "#2ecc71" for s in seasonal_factor]
axes[0, 1].bar(months, seasonal_factor - 100, color=colors, edgecolor="black", linewidth=0.5)
axes[0, 1].axhline(y=0, color="black", linewidth=1)
axes[0, 1].set_title("Seasonal Revenue Index")
axes[0, 1].set_ylabel("% Deviation from Mean")

for year in monthly_total["year"].unique():
    subset = monthly_total[monthly_total["year"] == year]
    axes[1, 0].plot(subset["month"], subset["total_units"], marker="o", label=str(year))
axes[1, 0].set_title("Monthly Units Sold by Year")
axes[1, 0].legend()
axes[1, 0].set_xticks(range(1, 13))
axes[1, 0].set_xticklabels(months, rotation=45)

yoy = monthly_total.pivot(index="month", columns="year", values="total_revenue")
years = sorted(monthly_total["year"].unique())
growth = ((yoy[years[-1]] - yoy[years[0]]) / yoy[years[0]] * 100)
axes[1, 1].bar(months, growth.values, color="steelblue", edgecolor="black", linewidth=0.5)
axes[1, 1].set_title(f"YoY Revenue Growth ({years[0]} → {years[-1]})")
axes[1, 1].set_ylabel("Growth (%)")
axes[1, 1].axhline(y=0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig("figures/06_seasonal_analysis.png", bbox_inches="tight")
plt.close()
print("✓ Figure 6: Seasonal analysis")

# --- Figure 7: Rep Performance ---
rep_performance = df.groupby(["rep_id", "region"]).agg(
    total_revenue=("revenue", "sum"),
    avg_satisfaction=("customer_satisfaction", "mean"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(data=rep_performance, x="region", y="total_revenue", order=region_order, ax=axes[0])
axes[0].set_title("Total Rep Revenue by Region (24 months)")
axes[0].set_ylabel("Total Revenue ($)")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

for region in region_order:
    subset = rep_performance[rep_performance["region"] == region]
    axes[1].scatter(subset["avg_satisfaction"], subset["total_revenue"],
                   label=region, alpha=0.7, s=60, edgecolors="white", linewidth=0.5)
axes[1].set_xlabel("Avg Customer Satisfaction")
axes[1].set_ylabel("Total Revenue ($)")
axes[1].set_title("Revenue vs Customer Satisfaction by Rep")
axes[1].legend(title="Region")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
plt.tight_layout()
plt.savefig("figures/07_rep_performance.png", bbox_inches="tight")
plt.close()
print("✓ Figure 7: Rep performance")

print(f"\n{'='*60}")
print("Analysis complete. 7 figures saved to figures/")
print(f"{'='*60}")
