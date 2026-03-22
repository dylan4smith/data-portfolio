"""Core audit logic for timesheet validation.

Each public function accepts a pandas DataFrame of timesheet entries and returns
a DataFrame of flagged issues.  The engine is deliberately decoupled from the
CLI so it can be reused in notebooks or pipeline scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Data-class for structured findings
# ---------------------------------------------------------------------------
@dataclass
class AuditFinding:
    """A single audit issue detected in the timesheet data."""

    rule_id: str
    severity: str  # "critical" | "warning" | "info"
    employee_id: str
    employee_name: str
    date: str
    description: str
    value: Optional[float] = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def load_timesheet(path: str) -> pd.DataFrame:
    """Read a timesheet CSV and coerce types."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
    return df


def check_excessive_hours(
    df: pd.DataFrame, threshold: float = 12.0
) -> list[AuditFinding]:
    """Flag any single entry where hours exceed *threshold*."""
    findings: list[AuditFinding] = []
    mask = df["hours"] > threshold
    for _, row in df[mask].iterrows():
        findings.append(
            AuditFinding(
                rule_id="EXCESS_HOURS",
                severity="critical",
                employee_id=row["employee_id"],
                employee_name=row["employee_name"],
                date=str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                description=f"Single entry of {row['hours']}h exceeds {threshold}h threshold",
                value=row["hours"],
            )
        )
    return findings


def check_negative_or_zero_hours(df: pd.DataFrame) -> list[AuditFinding]:
    """Flag entries with non-positive hours."""
    findings: list[AuditFinding] = []
    mask = df["hours"] <= 0
    for _, row in df[mask].iterrows():
        findings.append(
            AuditFinding(
                rule_id="INVALID_HOURS",
                severity="critical",
                employee_id=row["employee_id"],
                employee_name=row["employee_name"],
                date=str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                description=f"Non-positive hours entry: {row['hours']}h",
                value=row["hours"],
            )
        )
    return findings


def check_daily_total_exceeded(
    df: pd.DataFrame, max_daily: float = 16.0
) -> list[AuditFinding]:
    """Flag employees whose total hours for a single day exceed *max_daily*."""
    findings: list[AuditFinding] = []
    daily = (
        df.groupby(["employee_id", "employee_name", df["date"].dt.date])["hours"]
        .sum()
        .reset_index()
    )
    daily.columns = ["employee_id", "employee_name", "date", "total_hours"]
    over = daily[daily["total_hours"] > max_daily]
    for _, row in over.iterrows():
        findings.append(
            AuditFinding(
                rule_id="DAILY_TOTAL_EXCEEDED",
                severity="critical",
                employee_id=row["employee_id"],
                employee_name=row["employee_name"],
                date=str(row["date"]),
                description=f"Daily total of {row['total_hours']}h exceeds {max_daily}h cap",
                value=row["total_hours"],
            )
        )
    return findings


def check_duplicate_entries(df: pd.DataFrame) -> list[AuditFinding]:
    """Flag exact-duplicate rows (same employee, date, project, task, hours)."""
    findings: list[AuditFinding] = []
    dup_cols = ["employee_id", "date", "project_code", "task_category", "hours"]
    dups = df[df.duplicated(subset=dup_cols, keep="first")]
    for _, row in dups.iterrows():
        findings.append(
            AuditFinding(
                rule_id="DUPLICATE_ENTRY",
                severity="warning",
                employee_id=row["employee_id"],
                employee_name=row["employee_name"],
                date=str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                description=(
                    f"Duplicate entry on {row['project_code']} / "
                    f"{row['task_category']} ({row['hours']}h)"
                ),
                value=row["hours"],
            )
        )
    return findings


def check_unapproved_weekend_work(df: pd.DataFrame) -> list[AuditFinding]:
    """Flag weekend entries that are not marked as approved."""
    findings: list[AuditFinding] = []
    is_weekend = df["date"].dt.weekday >= 5
    is_unapproved = df["approved"].str.upper() == "N"
    mask = is_weekend & is_unapproved
    for _, row in df[mask].iterrows():
        day_name = row["date"].strftime("%A")
        findings.append(
            AuditFinding(
                rule_id="UNAPPROVED_WEEKEND",
                severity="warning",
                employee_id=row["employee_id"],
                employee_name=row["employee_name"],
                date=str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                description=f"Unapproved weekend work on {day_name} ({row['hours']}h)",
                value=row["hours"],
            )
        )
    return findings


def check_missing_weekdays(
    df: pd.DataFrame,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[AuditFinding]:
    """Flag weekdays where an employee has zero entries (potential gap)."""
    findings: list[AuditFinding] = []

    if start_date is None:
        start_date = df["date"].min().date()
    if end_date is None:
        end_date = df["date"].max().date()

    all_weekdays: list[date] = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            all_weekdays.append(current)
        current += timedelta(days=1)

    for emp_id in df["employee_id"].unique():
        emp_df = df[df["employee_id"] == emp_id]
        emp_name = emp_df["employee_name"].iloc[0]
        logged_dates = set(emp_df["date"].dt.date)
        missing = sorted(set(all_weekdays) - logged_dates)
        for d in missing:
            findings.append(
                AuditFinding(
                    rule_id="MISSING_WEEKDAY",
                    severity="info",
                    employee_id=emp_id,
                    employee_name=emp_name,
                    date=str(d),
                    description=f"No timesheet entry for {d.strftime('%A')} {d}",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_full_audit(
    df: pd.DataFrame,
    hours_threshold: float = 12.0,
    daily_max: float = 16.0,
) -> list[AuditFinding]:
    """Execute all audit checks and return a combined list of findings."""
    findings: list[AuditFinding] = []
    findings.extend(check_excessive_hours(df, threshold=hours_threshold))
    findings.extend(check_negative_or_zero_hours(df))
    findings.extend(check_daily_total_exceeded(df, max_daily=daily_max))
    findings.extend(check_duplicate_entries(df))
    findings.extend(check_unapproved_weekend_work(df))
    findings.extend(check_missing_weekdays(df))
    return findings


def findings_to_dataframe(findings: list[AuditFinding]) -> pd.DataFrame:
    """Convert a list of AuditFinding objects to a tidy DataFrame."""
    if not findings:
        return pd.DataFrame(
            columns=["rule_id", "severity", "employee_id", "employee_name", "date", "description", "value"]
        )
    return pd.DataFrame([f.__dict__ for f in findings])
