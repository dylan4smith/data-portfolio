"""
Generate synthetic invoice payment data for the payment delay predictor.

Simulates realistic consulting firm invoice data with features that
correlate with late payment behavior.
"""

import random
import csv
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Reproducibility
random.seed(42)

NUM_RECORDS = 2000

CLIENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "Acme Corp": {"size": "enterprise", "industry": "manufacturing", "base_delay_prob": 0.15},
    "Birch Holdings": {"size": "mid-market", "industry": "finance", "base_delay_prob": 0.10},
    "Cedar Solutions": {"size": "small", "industry": "technology", "base_delay_prob": 0.30},
    "Delta Group": {"size": "enterprise", "industry": "healthcare", "base_delay_prob": 0.20},
    "Echo Ventures": {"size": "small", "industry": "retail", "base_delay_prob": 0.40},
    "Falcon Industries": {"size": "mid-market", "industry": "manufacturing", "base_delay_prob": 0.18},
    "Granite Partners": {"size": "enterprise", "industry": "finance", "base_delay_prob": 0.08},
    "Horizon LLC": {"size": "small", "industry": "technology", "base_delay_prob": 0.35},
    "Ironclad Inc": {"size": "mid-market", "industry": "healthcare", "base_delay_prob": 0.22},
    "Jade Consulting": {"size": "small", "industry": "professional_services", "base_delay_prob": 0.28},
    "Keystone Retail": {"size": "enterprise", "industry": "retail", "base_delay_prob": 0.12},
    "Lumen Analytics": {"size": "mid-market", "industry": "technology", "base_delay_prob": 0.16},
}

SERVICE_TYPES = ["data_migration", "analytics_consulting", "process_automation", "reporting_setup", "training"]

PAYMENT_TERMS_OPTIONS = [15, 30, 45, 60]


def generate_invoice_record(invoice_id: int) -> Dict[str, Any]:
    """Generate a single synthetic invoice record."""
    client_name = random.choice(list(CLIENT_PROFILES.keys()))
    profile = CLIENT_PROFILES[client_name]

    # Invoice metadata
    issue_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 729))
    service_type = random.choice(SERVICE_TYPES)
    payment_terms_days = random.choice(PAYMENT_TERMS_OPTIONS)

    # Amount varies by client size
    amount_ranges = {"small": (1_500, 15_000), "mid-market": (10_000, 75_000), "enterprise": (25_000, 200_000)}
    low, high = amount_ranges[profile["size"]]
    invoice_amount = round(random.uniform(low, high), 2)

    # Project duration in weeks
    duration_weeks = random.randint(1, 16)

    # Historical relationship features
    invoices_ytd = random.randint(1, 24)
    prior_late_payments = random.randint(0, min(invoices_ytd, 8))
    prior_late_rate = round(prior_late_payments / max(invoices_ytd, 1), 2)

    # Whether a purchase order was provided (enterprises more likely)
    has_purchase_order = random.random() < (0.85 if profile["size"] == "enterprise" else 0.40)

    # Contact responsiveness score (1-5, simulated)
    contact_responsiveness = random.randint(1, 5)

    # Determine late payment (target variable)
    delay_prob = profile["base_delay_prob"]

    # Higher amounts increase delay probability
    if invoice_amount > 50_000:
        delay_prob += 0.10
    elif invoice_amount > 100_000:
        delay_prob += 0.18

    # Prior late behavior is predictive
    delay_prob += prior_late_rate * 0.25

    # Shorter payment terms reduce delays
    if payment_terms_days <= 15:
        delay_prob -= 0.05
    elif payment_terms_days >= 60:
        delay_prob += 0.12

    # Purchase orders reduce delays
    if has_purchase_order:
        delay_prob -= 0.08

    # Low responsiveness increases delays
    if contact_responsiveness <= 2:
        delay_prob += 0.15

    # Q4 tends to have more delays (budget cycles)
    if issue_date.month in [10, 11, 12]:
        delay_prob += 0.08

    # Clamp probability
    delay_prob = max(0.02, min(0.85, delay_prob))

    is_late = 1 if random.random() < delay_prob else 0

    # If late, generate days late
    days_late = 0
    if is_late:
        days_late = random.randint(1, 90)

    return {
        "invoice_id": f"INV-{invoice_id:05d}",
        "client_name": client_name,
        "client_size": profile["size"],
        "client_industry": profile["industry"],
        "invoice_amount": invoice_amount,
        "service_type": service_type,
        "issue_date": issue_date.strftime("%Y-%m-%d"),
        "issue_month": issue_date.month,
        "issue_quarter": (issue_date.month - 1) // 3 + 1,
        "issue_day_of_week": issue_date.strftime("%A"),
        "payment_terms_days": payment_terms_days,
        "project_duration_weeks": duration_weeks,
        "invoices_ytd": invoices_ytd,
        "prior_late_payments": prior_late_payments,
        "prior_late_rate": prior_late_rate,
        "has_purchase_order": int(has_purchase_order),
        "contact_responsiveness": contact_responsiveness,
        "is_late": is_late,
        "days_late": days_late,
    }


def main() -> None:
    """Generate dataset and write to CSV."""
    records: List[Dict[str, Any]] = [generate_invoice_record(i) for i in range(1, NUM_RECORDS + 1)]

    output_path = os.path.join(os.path.dirname(__file__), "data", "invoices.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    late_count = sum(1 for r in records if r["is_late"] == 1)
    print(f"Generated {len(records)} invoice records -> {output_path}")
    print(f"Late payments: {late_count} ({late_count / len(records) * 100:.1f}%)")


if __name__ == "__main__":
    main()
