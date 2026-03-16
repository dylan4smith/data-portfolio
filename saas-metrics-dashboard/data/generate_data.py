"""
Generate synthetic SaaS metrics data for dashboard demonstration.

Simulates 24 months of subscription business data including MRR,
churn, new customers, expansion revenue, and support tickets.
"""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def generate_monthly_metrics(num_months: int = 24) -> list[dict[str, Any]]:
    """Generate monthly SaaS KPI data with realistic growth patterns."""
    random.seed(42)
    records: list[dict[str, Any]] = []

    # Starting conditions for a mid-stage B2B SaaS company
    active_customers = 185
    mrr = 74_000.0  # Monthly Recurring Revenue in USD
    arpu = mrr / active_customers  # Average Revenue Per User

    start_date = datetime(2024, 1, 1)

    for month_offset in range(num_months):
        current_date = start_date + timedelta(days=30 * month_offset)
        month_label = current_date.strftime("%Y-%m")

        # Simulate organic growth with seasonal variation
        seasonal_factor = 1.0 + 0.08 * (1 if current_date.month in [1, 9, 10] else 0)
        base_new = int(random.gauss(18, 4) * seasonal_factor)
        new_customers = max(5, base_new)

        # Churn rate trends downward over time (product improves)
        base_churn_rate = max(0.025, 0.055 - (month_offset * 0.001))
        churned_customers = max(1, int(active_customers * random.gauss(base_churn_rate, 0.008)))

        # Reactivations (small fraction of churned come back)
        reactivated = random.randint(0, max(1, churned_customers // 4))

        net_new = new_customers - churned_customers + reactivated
        active_customers = max(50, active_customers + net_new)

        # Revenue modeling
        expansion_rate = random.uniform(0.02, 0.06)
        expansion_revenue = mrr * expansion_rate
        contraction_revenue = mrr * random.uniform(0.005, 0.02)
        new_revenue = new_customers * arpu * random.uniform(0.85, 1.15)
        churned_revenue = churned_customers * arpu * random.uniform(0.9, 1.1)

        mrr = mrr + new_revenue + expansion_revenue - churned_revenue - contraction_revenue
        mrr = max(30_000, mrr)
        arpu = mrr / active_customers

        # Gross churn and net revenue churn
        gross_churn_rate = churned_customers / (active_customers + churned_customers - new_customers)
        net_revenue_churn = (churned_revenue + contraction_revenue - expansion_revenue) / mrr

        # Support metrics
        tickets_opened = int(active_customers * random.uniform(0.15, 0.35))
        tickets_resolved = int(tickets_opened * random.uniform(0.85, 1.0))
        avg_resolution_hours = round(random.gauss(18, 5), 1)

        # NPS score (improves over time as product matures)
        nps_score = int(min(80, max(10, random.gauss(42 + month_offset * 0.5, 8))))

        # CAC and LTV
        marketing_spend = random.uniform(12_000, 22_000)
        cac = marketing_spend / max(1, new_customers)
        avg_lifetime_months = 1 / max(0.01, gross_churn_rate)
        ltv = arpu * avg_lifetime_months * 0.7  # 70% gross margin

        records.append({
            "month": month_label,
            "active_customers": active_customers,
            "new_customers": new_customers,
            "churned_customers": churned_customers,
            "reactivated_customers": reactivated,
            "mrr": round(mrr, 2),
            "arpu": round(arpu, 2),
            "expansion_revenue": round(expansion_revenue, 2),
            "contraction_revenue": round(contraction_revenue, 2),
            "gross_churn_rate": round(gross_churn_rate, 4),
            "net_revenue_churn": round(net_revenue_churn, 4),
            "ltv": round(ltv, 2),
            "cac": round(cac, 2),
            "ltv_cac_ratio": round(ltv / max(1, cac), 2),
            "nps_score": nps_score,
            "support_tickets_opened": tickets_opened,
            "support_tickets_resolved": tickets_resolved,
            "avg_resolution_hours": max(4.0, avg_resolution_hours),
            "marketing_spend": round(marketing_spend, 2),
        })

    return records


def generate_customer_segments() -> list[dict[str, Any]]:
    """Generate customer segment breakdown for the latest period."""
    random.seed(42)
    segments = [
        {"segment": "Enterprise", "customers": 28, "mrr_share": 0.42, "churn_rate": 0.015, "avg_contract_months": 18},
        {"segment": "Mid-Market", "customers": 67, "mrr_share": 0.33, "churn_rate": 0.035, "avg_contract_months": 12},
        {"segment": "SMB", "customers": 112, "mrr_share": 0.20, "churn_rate": 0.065, "avg_contract_months": 6},
        {"segment": "Startup", "customers": 43, "mrr_share": 0.05, "churn_rate": 0.09, "avg_contract_months": 3},
    ]
    return segments


def main() -> None:
    """Generate all datasets and write to CSV/JSON."""
    output_dir = Path(__file__).parent

    # Monthly metrics
    monthly = generate_monthly_metrics()
    monthly_path = output_dir / "monthly_metrics.csv"
    with open(monthly_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=monthly[0].keys())
        writer.writeheader()
        writer.writerows(monthly)
    print(f"Wrote {len(monthly)} rows to {monthly_path}")

    # Customer segments
    segments = generate_customer_segments()
    segments_path = output_dir / "customer_segments.json"
    with open(segments_path, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"Wrote {len(segments)} segments to {segments_path}")


if __name__ == "__main__":
    main()
