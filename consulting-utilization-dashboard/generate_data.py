"""
Generate synthetic consultant utilization data for dashboard demonstration.

Simulates a mid-size consulting firm with consultants across multiple practice
areas, generating weekly timesheet records over a 12-month period.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

RANDOM_SEED: int = 42
random.seed(RANDOM_SEED)

# --- Firm configuration ---
PRACTICE_AREAS: list[str] = [
    "Strategy & Operations",
    "Data & Analytics",
    "Technology Advisory",
    "Financial Consulting",
    "Risk & Compliance",
]

LEVELS: list[str] = ["Analyst", "Senior Analyst", "Consultant", "Senior Consultant", "Manager", "Director"]

LEVEL_BILL_RATES: dict[str, tuple[int, int]] = {
    "Analyst": (125, 175),
    "Senior Analyst": (175, 225),
    "Consultant": (225, 300),
    "Senior Consultant": (300, 375),
    "Manager": (375, 475),
    "Director": (475, 600),
}

LEVEL_TARGET_UTILIZATION: dict[str, float] = {
    "Analyst": 0.85,
    "Senior Analyst": 0.82,
    "Consultant": 0.78,
    "Senior Consultant": 0.72,
    "Manager": 0.60,
    "Director": 0.45,
}

FIRST_NAMES: list[str] = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery",
    "Cameron", "Dakota", "Skyler", "Reese", "Harper", "Emerson", "Rowan",
    "Blake", "Finley", "Sage", "Drew", "Kendall", "Parker", "Hayden",
    "Jamie", "Peyton", "Logan", "Elliot", "Tatum", "Charlie", "Phoenix", "Rory",
]

LAST_NAMES: list[str] = [
    "Chen", "Patel", "Rodriguez", "Kim", "O'Brien", "Nakamura", "Singh",
    "Martinez", "Williams", "Thompson", "Nguyen", "Garcia", "Lee", "Brown",
    "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas",
    "Jackson", "White", "Harris", "Martin", "Clark", "Lewis", "Robinson",
    "Walker", "Young",
]

CLIENT_NAMES: list[str] = [
    "Meridian Health Systems", "Atlas Financial Group", "Vertex Manufacturing",
    "Cascade Retail Holdings", "Pinnacle Energy Corp", "Bridgewater Logistics",
    "Summit Technology Inc", "Ironclad Insurance", "Clearview Telecom",
    "Pacific Coast Ventures", "Northstar Pharmaceuticals", "Greenfield Agriculture",
    "Horizon Media Group", "Sterling Automotive", "Crestline Hospitality",
]

PROJECT_TYPES: list[str] = [
    "Process Optimization", "Digital Transformation", "Cost Reduction",
    "Market Analysis", "System Implementation", "Compliance Audit",
    "Data Migration", "Organizational Restructuring", "Revenue Growth Strategy",
    "Risk Assessment", "Supply Chain Review", "Customer Experience Redesign",
]


def generate_consultants(n_consultants: int = 35) -> pd.DataFrame:
    """Generate a roster of consultants with assigned practice areas and levels."""
    consultants = []
    used_names: set[str] = set()

    for consultant_id in range(1, n_consultants + 1):
        while True:
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            full_name = f"{first} {last}"
            if full_name not in used_names:
                used_names.add(full_name)
                break

        level = random.choices(
            LEVELS,
            weights=[6, 5, 5, 4, 3, 2],  # More junior staff
            k=1,
        )[0]

        practice = random.choice(PRACTICE_AREAS)
        bill_low, bill_high = LEVEL_BILL_RATES[level]
        bill_rate = random.randint(bill_low, bill_high)
        start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 180))

        consultants.append({
            "consultant_id": f"C{consultant_id:03d}",
            "name": full_name,
            "level": level,
            "practice_area": practice,
            "bill_rate_usd": bill_rate,
            "hire_date": start_date.strftime("%Y-%m-%d"),
            "target_utilization": LEVEL_TARGET_UTILIZATION[level],
        })

    return pd.DataFrame(consultants)


def generate_projects(n_projects: int = 20) -> pd.DataFrame:
    """Generate active and completed client engagements."""
    projects = []

    for project_id in range(1, n_projects + 1):
        client = random.choice(CLIENT_NAMES)
        proj_type = random.choice(PROJECT_TYPES)
        start = datetime(2024, 4, 1) + timedelta(days=random.randint(0, 270))
        duration_weeks = random.randint(4, 26)
        end = start + timedelta(weeks=duration_weeks)
        status = "Completed" if end < datetime(2025, 3, 1) else "Active"

        projects.append({
            "project_id": f"P{project_id:03d}",
            "client": client,
            "project_type": proj_type,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "status": status,
            "practice_area": random.choice(PRACTICE_AREAS),
        })

    return pd.DataFrame(projects)


def generate_timesheets(
    consultants: pd.DataFrame,
    projects: pd.DataFrame,
    start_date: datetime = datetime(2024, 4, 1),
    end_date: datetime = datetime(2025, 3, 31),
) -> pd.DataFrame:
    """Generate weekly timesheet entries for each consultant."""
    records = []
    project_list = projects.to_dict("records")

    current = start_date
    while current <= end_date:
        week_end = current + timedelta(days=4)  # Mon-Fri

        for _, consultant in consultants.iterrows():
            hire_date = datetime.strptime(consultant["hire_date"], "%Y-%m-%d")
            if current < hire_date:
                continue

            target = consultant["target_utilization"]
            total_available_hours = 40.0

            # Simulate utilization with realistic variance
            base_utilization = target + random.gauss(0, 0.10)
            # Seasonal dip in December, spike in Q1
            month = current.month
            if month == 12:
                base_utilization -= 0.12
            elif month in (1, 2, 3):
                base_utilization += 0.05
            elif month in (7, 8):
                base_utilization -= 0.05  # Summer slowdown

            base_utilization = max(0.0, min(1.0, base_utilization))

            # PTO simulation (~10% chance any given week)
            pto_hours = 0.0
            if random.random() < 0.10:
                pto_hours = random.choice([8.0, 16.0, 24.0, 40.0])
                if pto_hours == 40.0:
                    # Full week off
                    records.append({
                        "consultant_id": consultant["consultant_id"],
                        "week_start": current.strftime("%Y-%m-%d"),
                        "week_end": week_end.strftime("%Y-%m-%d"),
                        "project_id": None,
                        "billable_hours": 0.0,
                        "internal_hours": 0.0,
                        "admin_hours": 0.0,
                        "pto_hours": 40.0,
                        "total_hours": 40.0,
                    })
                    continue

            available = total_available_hours - pto_hours
            billable = round(available * base_utilization, 1)

            # Assign billable hours to 1-2 projects
            eligible_projects = [
                p for p in project_list
                if (datetime.strptime(p["start_date"], "%Y-%m-%d") <= current
                    and datetime.strptime(p["end_date"], "%Y-%m-%d") >= current)
            ]

            if not eligible_projects:
                billable = 0.0
                eligible_projects = []

            n_projects_assigned = min(len(eligible_projects), random.choices([1, 2], weights=[7, 3], k=1)[0])
            assigned = random.sample(eligible_projects, n_projects_assigned) if eligible_projects else []

            remaining_billable = billable
            for i, proj in enumerate(assigned):
                if i == len(assigned) - 1:
                    proj_hours = remaining_billable
                else:
                    proj_hours = round(remaining_billable * random.uniform(0.4, 0.7), 1)
                    remaining_billable -= proj_hours

                internal_hours = round(random.uniform(1.0, 6.0), 1)
                admin_hours = round(random.uniform(0.5, 3.0), 1)

                records.append({
                    "consultant_id": consultant["consultant_id"],
                    "week_start": current.strftime("%Y-%m-%d"),
                    "week_end": week_end.strftime("%Y-%m-%d"),
                    "project_id": proj["project_id"],
                    "billable_hours": round(proj_hours, 1),
                    "internal_hours": internal_hours,
                    "admin_hours": admin_hours,
                    "pto_hours": pto_hours if i == 0 else 0.0,
                    "total_hours": round(proj_hours + internal_hours + admin_hours + (pto_hours if i == 0 else 0.0), 1),
                })

            # If no projects assigned but consultant is active
            if not assigned:
                internal_hours = round(random.uniform(8.0, 16.0), 1)
                admin_hours = round(random.uniform(2.0, 6.0), 1)
                records.append({
                    "consultant_id": consultant["consultant_id"],
                    "week_start": current.strftime("%Y-%m-%d"),
                    "week_end": week_end.strftime("%Y-%m-%d"),
                    "project_id": None,
                    "billable_hours": 0.0,
                    "internal_hours": internal_hours,
                    "admin_hours": admin_hours,
                    "pto_hours": pto_hours,
                    "total_hours": round(internal_hours + admin_hours + pto_hours, 1),
                })

        current += timedelta(weeks=1)

    return pd.DataFrame(records)


def main() -> None:
    """Generate all synthetic datasets and save to data/ directory."""
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    print("Generating consultant roster...")
    consultants = generate_consultants(35)
    consultants.to_csv(output_dir / "consultants.csv", index=False)
    print(f"  -> {len(consultants)} consultants")

    print("Generating project engagements...")
    projects = generate_projects(20)
    projects.to_csv(output_dir / "projects.csv", index=False)
    print(f"  -> {len(projects)} projects")

    print("Generating weekly timesheets...")
    timesheets = generate_timesheets(consultants, projects)
    timesheets.to_csv(output_dir / "timesheets.csv", index=False)
    print(f"  -> {len(timesheets)} timesheet records")

    # Summary stats for verification
    total_billable = timesheets["billable_hours"].sum()
    avg_utilization = timesheets.groupby("consultant_id")["billable_hours"].sum().mean()
    print(f"\nTotal billable hours: {total_billable:,.0f}")
    print(f"Avg billable hours per consultant: {avg_utilization:,.0f}")
    print("\nData generation complete.")


if __name__ == "__main__":
    main()
