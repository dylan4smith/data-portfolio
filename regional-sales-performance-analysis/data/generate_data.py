"""
Generate synthetic regional sales data for statistical analysis.

Simulates 2 years of monthly sales data across 5 U.S. regions with realistic
seasonal patterns, regional baselines, and controlled noise.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_sales_data(
    seed: int = 42,
    n_reps: int = 12,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Generate synthetic monthly sales data across regions and product lines.

    Args:
        seed: Random seed for reproducibility.
        n_reps: Number of sales reps per region.
        start_date: Start date for the time series.

    Returns:
        DataFrame with columns: date, region, product_line, rep_id,
        revenue, units_sold, discount_pct, customer_satisfaction, deal_cycle_days.
    """
    rng = np.random.default_rng(seed)

    regions = {
        "Northeast": {"base_revenue": 52000, "growth_rate": 0.03},
        "Southeast": {"base_revenue": 45000, "growth_rate": 0.05},
        "Midwest": {"base_revenue": 41000, "growth_rate": 0.02},
        "West": {"base_revenue": 58000, "growth_rate": 0.04},
        "Southwest": {"base_revenue": 38000, "growth_rate": 0.06},
    }

    product_lines = ["Enterprise SaaS", "SMB Platform", "Professional Services"]
    product_weights = {"Enterprise SaaS": 1.4, "SMB Platform": 1.0, "Professional Services": 0.7}

    dates = pd.date_range(start=start_date, periods=24, freq="MS")

    rows = []
    for date in dates:
        month_idx = (date.month - 1)
        # Seasonal multiplier: peak in Q4, trough in Q1
        seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (month_idx - 3) / 12)

        for region_name, params in regions.items():
            months_elapsed = (date.year - 2024) * 12 + date.month - 1
            trend = 1.0 + params["growth_rate"] * (months_elapsed / 12)

            for product in product_lines:
                pw = product_weights[product]
                for rep_idx in range(n_reps):
                    rep_skill = 0.8 + rng.random() * 0.4  # 0.8–1.2 skill factor

                    base = params["base_revenue"] * pw * rep_skill
                    revenue = base * seasonal * trend * (1 + rng.normal(0, 0.08))
                    revenue = max(revenue, 5000)

                    avg_price = 850 if product == "Enterprise SaaS" else 320 if product == "SMB Platform" else 1200
                    units = max(1, int(revenue / avg_price * (1 + rng.normal(0, 0.1))))

                    discount = np.clip(rng.beta(2, 8) * 0.35, 0.0, 0.30)

                    # Satisfaction inversely related to discount, with noise
                    satisfaction = np.clip(
                        4.2 - discount * 3.0 + rng.normal(0, 0.3), 1.0, 5.0
                    )

                    cycle_days = int(np.clip(
                        {"Enterprise SaaS": 45, "SMB Platform": 18, "Professional Services": 30}[product]
                        * (1 + rng.normal(0, 0.2)),
                        5, 120
                    ))

                    rows.append({
                        "date": date,
                        "region": region_name,
                        "product_line": product,
                        "rep_id": f"{region_name[:2].upper()}-{rep_idx+1:03d}",
                        "revenue": round(revenue, 2),
                        "units_sold": units,
                        "discount_pct": round(discount, 4),
                        "customer_satisfaction": round(satisfaction, 2),
                        "deal_cycle_days": cycle_days,
                    })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    df = generate_sales_data()
    output_path = "regional_sales_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} records → {output_path}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Regions: {df['region'].nunique()}, Products: {df['product_line'].nunique()}")
    print(f"Revenue range: ${df['revenue'].min():,.0f} – ${df['revenue'].max():,.0f}")
