"""Generate realistic synthetic timesheet data for testing the audit CLI.

Produces a CSV with intentional anomalies so the tool has something to flag:
  - entries exceeding 12 hours in a single day
  - missing days (gaps in a consultant's timesheet)
  - duplicate entries (same person, project, date)
  - weekend entries without prior approval flag
  - negative or zero hours
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

CONSULTANTS: list[dict[str, str]] = [
    {"employee_id": "EMP-1001", "name": "Anika Patel"},
    {"employee_id": "EMP-1002", "name": "Marcus Chen"},
    {"employee_id": "EMP-1003", "name": "Sofia Gutierrez"},
    {"employee_id": "EMP-1004", "name": "James Okafor"},
    {"employee_id": "EMP-1005", "name": "Lena Novak"},
]

PROJECTS: list[str] = [
    "PRJ-100-CRM-Migration",
    "PRJ-200-FinOps-Audit",
    "PRJ-300-Data-Platform",
    "PRJ-400-ERP-Integration",
    "PRJ-500-Compliance-Review",
]

TASK_CATEGORIES: list[str] = [
    "Analysis",
    "Development",
    "Client Meeting",
    "Internal Meeting",
    "Documentation",
    "Testing",
    "Code Review",
    "Deployment",
]


def _random_hours(normal: bool = True) -> float:
    """Return a plausible (or anomalous) number of hours."""
    if normal:
        return round(random.choice([4.0, 6.0, 7.5, 8.0, 8.0, 8.0, 8.5, 9.0]), 1)
    # Anomalous values
    return round(random.choice([-1.0, 0.0, 13.5, 16.0, 18.0, 24.0]), 1)


def generate_timesheets(
    start_date: date,
    end_date: date,
    output_path: Path,
    anomaly_rate: float = 0.08,
) -> int:
    """Write synthetic timesheet rows to *output_path* and return the row count."""
    rows: list[dict[str, str | float]] = []

    current = start_date
    while current <= end_date:
        is_weekend = current.weekday() >= 5

        for consultant in CONSULTANTS:
            # ~10 % chance a consultant skips a weekday (creates a gap anomaly)
            if not is_weekend and random.random() < 0.10:
                continue

            # Weekend work is rare but happens
            if is_weekend and random.random() > 0.15:
                continue

            project = random.choice(PROJECTS)
            task = random.choice(TASK_CATEGORIES)
            is_anomaly = random.random() < anomaly_rate
            hours = _random_hours(normal=not is_anomaly)
            approved = "N" if (is_weekend and random.random() < 0.5) else "Y"

            rows.append(
                {
                    "employee_id": consultant["employee_id"],
                    "employee_name": consultant["name"],
                    "date": current.isoformat(),
                    "project_code": project,
                    "task_category": task,
                    "hours": hours,
                    "approved": approved,
                    "notes": "",
                }
            )

            # ~5 % chance of a duplicate row (same person + date + project)
            if random.random() < 0.05:
                rows.append(rows[-1].copy())

        current += timedelta(days=1)

    random.shuffle(rows)

    fieldnames = [
        "employee_id",
        "employee_name",
        "date",
        "project_code",
        "task_category",
        "hours",
        "approved",
        "notes",
    ]

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    random.seed(42)
    out = Path(__file__).resolve().parent / "sample_timesheets.csv"
    n = generate_timesheets(
        start_date=date(2026, 1, 5),
        end_date=date(2026, 3, 15),
        output_path=out,
    )
    print(f"Wrote {n} timesheet rows → {out}")
