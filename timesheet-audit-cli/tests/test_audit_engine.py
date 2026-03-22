"""Unit tests for the timesheet audit engine."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from audit_engine import (
    AuditFinding,
    check_daily_total_exceeded,
    check_duplicate_entries,
    check_excessive_hours,
    check_missing_weekdays,
    check_negative_or_zero_hours,
    check_unapproved_weekend_work,
    findings_to_dataframe,
    run_full_audit,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_row(
    hours: float = 8.0,
    dt: str = "2026-01-12",
    emp_id: str = "EMP-1001",
    emp_name: str = "Test User",
    project: str = "PRJ-100",
    task: str = "Development",
    approved: str = "Y",
) -> dict:
    return {
        "employee_id": emp_id,
        "employee_name": emp_name,
        "date": pd.Timestamp(dt),
        "project_code": project,
        "task_category": task,
        "hours": hours,
        "approved": approved,
        "notes": "",
    }


def _df_from_rows(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ── check_excessive_hours ────────────────────────────────────────────────────

class TestExcessiveHours:
    def test_flags_entry_above_threshold(self) -> None:
        df = _df_from_rows(_make_row(hours=14.0))
        findings = check_excessive_hours(df, threshold=12.0)
        assert len(findings) == 1
        assert findings[0].rule_id == "EXCESS_HOURS"
        assert findings[0].severity == "critical"

    def test_does_not_flag_normal_entry(self) -> None:
        df = _df_from_rows(_make_row(hours=8.0))
        findings = check_excessive_hours(df, threshold=12.0)
        assert findings == []

    def test_custom_threshold(self) -> None:
        df = _df_from_rows(_make_row(hours=10.0))
        assert len(check_excessive_hours(df, threshold=9.0)) == 1
        assert len(check_excessive_hours(df, threshold=11.0)) == 0


# ── check_negative_or_zero_hours ─────────────────────────────────────────────

class TestNegativeOrZeroHours:
    def test_flags_negative(self) -> None:
        df = _df_from_rows(_make_row(hours=-2.0))
        findings = check_negative_or_zero_hours(df)
        assert len(findings) == 1
        assert findings[0].rule_id == "INVALID_HOURS"

    def test_flags_zero(self) -> None:
        df = _df_from_rows(_make_row(hours=0.0))
        assert len(check_negative_or_zero_hours(df)) == 1

    def test_positive_hours_pass(self) -> None:
        df = _df_from_rows(_make_row(hours=1.0))
        assert check_negative_or_zero_hours(df) == []


# ── check_daily_total_exceeded ───────────────────────────────────────────────

class TestDailyTotalExceeded:
    def test_flags_when_total_exceeds_cap(self) -> None:
        df = _df_from_rows(
            _make_row(hours=9.0, dt="2026-01-12", project="PRJ-100"),
            _make_row(hours=8.0, dt="2026-01-12", project="PRJ-200"),
        )
        findings = check_daily_total_exceeded(df, max_daily=16.0)
        assert len(findings) == 1
        assert findings[0].value == 17.0

    def test_separate_days_ok(self) -> None:
        df = _df_from_rows(
            _make_row(hours=9.0, dt="2026-01-12"),
            _make_row(hours=9.0, dt="2026-01-13"),
        )
        assert check_daily_total_exceeded(df, max_daily=16.0) == []


# ── check_duplicate_entries ──────────────────────────────────────────────────

class TestDuplicateEntries:
    def test_flags_exact_duplicate(self) -> None:
        row = _make_row()
        df = _df_from_rows(row, row.copy())
        findings = check_duplicate_entries(df)
        assert len(findings) == 1
        assert findings[0].rule_id == "DUPLICATE_ENTRY"

    def test_different_projects_not_flagged(self) -> None:
        df = _df_from_rows(
            _make_row(project="PRJ-100"),
            _make_row(project="PRJ-200"),
        )
        assert check_duplicate_entries(df) == []


# ── check_unapproved_weekend_work ────────────────────────────────────────────

class TestUnapprovedWeekendWork:
    def test_flags_unapproved_saturday(self) -> None:
        # 2026-01-10 is a Saturday
        df = _df_from_rows(_make_row(dt="2026-01-10", approved="N"))
        findings = check_unapproved_weekend_work(df)
        assert len(findings) == 1
        assert findings[0].rule_id == "UNAPPROVED_WEEKEND"

    def test_approved_weekend_ok(self) -> None:
        df = _df_from_rows(_make_row(dt="2026-01-10", approved="Y"))
        assert check_unapproved_weekend_work(df) == []

    def test_weekday_unapproved_not_flagged(self) -> None:
        # 2026-01-12 is a Monday
        df = _df_from_rows(_make_row(dt="2026-01-12", approved="N"))
        assert check_unapproved_weekend_work(df) == []


# ── check_missing_weekdays ───────────────────────────────────────────────────

class TestMissingWeekdays:
    def test_detects_gap(self) -> None:
        # Only log Monday, skip Tuesday–Friday
        df = _df_from_rows(_make_row(dt="2026-01-12"))  # Monday
        findings = check_missing_weekdays(
            df,
            start_date=date(2026, 1, 12),
            end_date=date(2026, 1, 16),  # Mon–Fri
        )
        assert len(findings) == 4  # Tue, Wed, Thu, Fri missing

    def test_no_gaps(self) -> None:
        rows = [_make_row(dt=f"2026-01-{12 + i}") for i in range(5)]
        df = _df_from_rows(*rows)
        findings = check_missing_weekdays(
            df, start_date=date(2026, 1, 12), end_date=date(2026, 1, 16)
        )
        assert findings == []


# ── findings_to_dataframe ────────────────────────────────────────────────────

class TestFindingsToDataframe:
    def test_empty_list_returns_empty_df(self) -> None:
        result = findings_to_dataframe([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_converts_findings(self) -> None:
        f = AuditFinding(
            rule_id="TEST",
            severity="info",
            employee_id="EMP-1",
            employee_name="Test",
            date="2026-01-12",
            description="test finding",
            value=5.0,
        )
        result = findings_to_dataframe([f])
        assert len(result) == 1
        assert result.iloc[0]["rule_id"] == "TEST"


# ── run_full_audit integration ───────────────────────────────────────────────

class TestRunFullAudit:
    def test_finds_multiple_issue_types(self) -> None:
        df = _df_from_rows(
            _make_row(hours=18.0, dt="2026-01-12"),       # excessive
            _make_row(hours=-1.0, dt="2026-01-13"),        # negative
            _make_row(dt="2026-01-10", approved="N"),      # unapproved weekend
        )
        findings = run_full_audit(df)
        rule_ids = {f.rule_id for f in findings}
        assert "EXCESS_HOURS" in rule_ids
        assert "INVALID_HOURS" in rule_ids
        assert "UNAPPROVED_WEEKEND" in rule_ids
