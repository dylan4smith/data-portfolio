"""
Generate synthetic multi-source revenue data for testing the ETL pipeline.

Simulates three business units submitting monthly revenue reports with
slightly different schemas, data quality issues, and formatting inconsistencies
— a realistic scenario for consulting firms managing SMB clients.
"""

import csv
import os
import random
from datetime import date, timedelta

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Shared client pool
CLIENTS = [
    ("C-1001", "Greenfield Organic Foods"),
    ("C-1002", "Summit Peak Athletics"),
    ("C-1003", "Coastal Realty Group"),
    ("C-1004", "Pinnacle Software Solutions"),
    ("C-1005", "Metro Health Partners"),
    ("C-1006", "Riverside Manufacturing Co"),
    ("C-1007", "Beacon Financial Advisors"),
    ("C-1008", "Crestview Hospitality"),
    ("C-1009", "Bridgeport Logistics"),
    ("C-1010", "Evergreen Consulting Group"),
]

SERVICE_TYPES = ["Advisory", "Implementation", "Support", "Training", "Analytics"]
REGIONS = ["West", "East", "Midwest", "South"]


def generate_consulting_unit(filepath: str, num_records: int = 120) -> None:
    """Business Unit A: Consulting Services — clean schema, minor nulls."""
    headers = ["client_id", "client_name", "service_type", "revenue", "invoice_date", "region", "status"]
    rows = []
    for _ in range(num_records):
        client_id, client_name = random.choice(CLIENTS)
        inv_date = date(2025, 1, 1) + timedelta(days=random.randint(0, 364))
        revenue = round(random.uniform(2500, 85000), 2)
        status = random.choice(["paid", "pending", "overdue"])
        # Introduce ~5% missing regions
        region = random.choice(REGIONS) if random.random() > 0.05 else ""
        rows.append([client_id, client_name, random.choice(SERVICE_TYPES), revenue, inv_date.isoformat(), region, status])

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def generate_analytics_unit(filepath: str, num_records: int = 95) -> None:
    """Business Unit B: Analytics Division — different column names, some dirty data."""
    headers = ["ClientCode", "Name", "Category", "Amount", "Date", "Location", "PaymentStatus"]
    rows = []
    for _ in range(num_records):
        client_id, client_name = random.choice(CLIENTS)
        inv_date = date(2025, 1, 1) + timedelta(days=random.randint(0, 364))
        amount = round(random.uniform(5000, 120000), 2)
        status = random.choice(["Paid", "Pending", "Overdue", "PAID", "pending"])  # inconsistent casing
        category = random.choice(SERVICE_TYPES)
        location = random.choice(REGIONS)
        # Introduce ~3% negative amounts (data entry errors)
        if random.random() < 0.03:
            amount = -amount
        # Introduce ~4% malformed dates
        if random.random() < 0.04:
            inv_date_str = inv_date.strftime("%m/%d/%Y")  # wrong format
        else:
            inv_date_str = inv_date.isoformat()
        rows.append([client_id, client_name, category, amount, inv_date_str, location, status])

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def generate_training_unit(filepath: str, num_records: int = 80) -> None:
    """Business Unit C: Training Programs — extra columns, duplicates, whitespace issues."""
    headers = ["client_id", "client_name", "service", "amount_usd", "date", "region", "payment_status", "notes"]
    rows = []
    for i in range(num_records):
        client_id, client_name = random.choice(CLIENTS)
        inv_date = date(2025, 1, 1) + timedelta(days=random.randint(0, 364))
        amount = round(random.uniform(1000, 45000), 2)
        status = random.choice(["paid", "pending", "overdue"])
        region = random.choice(REGIONS)
        notes = random.choice(["", "Quarterly review", "Renewal", "New engagement", "  ", "N/A"])
        # Add whitespace issues to client names (~8%)
        if random.random() < 0.08:
            client_name = f"  {client_name}  "
        row = [client_id, client_name, random.choice(SERVICE_TYPES), amount, inv_date.isoformat(), region, status, notes]
        rows.append(row)
        # Introduce ~5% exact duplicates
        if random.random() < 0.05:
            rows.append(row.copy())

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == "__main__":
    generate_consulting_unit(os.path.join(DATA_DIR, "unit_a_consulting.csv"))
    generate_analytics_unit(os.path.join(DATA_DIR, "unit_b_analytics.csv"))
    generate_training_unit(os.path.join(DATA_DIR, "unit_c_training.csv"))
    print("Sample data generated successfully in data/ directory.")
