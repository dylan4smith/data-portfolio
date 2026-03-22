#!/usr/bin/env python3
"""Timesheet Audit CLI — validate consultant timesheets and surface anomalies.

Usage examples:
    python cli.py audit data/sample_timesheets.csv
    python cli.py audit data/sample_timesheets.csv --severity critical --format json
    python cli.py summary data/sample_timesheets.csv
    python cli.py audit data/sample_timesheets.csv --output reports/audit_findings.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from audit_engine import (
    findings_to_dataframe,
    load_timesheet,
    run_full_audit,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("timesheet-audit")


# ── CLI group ────────────────────────────────────────────────────────────────
@click.group()
@click.version_option(version="1.0.0", prog_name="timesheet-audit-cli")
def cli() -> None:
    """Timesheet Audit CLI — detect anomalies in consultant timesheets."""


# ── audit command ────────────────────────────────────────────────────────────
@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--hours-threshold",
    type=float,
    default=12.0,
    show_default=True,
    help="Flag single entries exceeding this many hours.",
)
@click.option(
    "--daily-max",
    type=float,
    default=16.0,
    show_default=True,
    help="Flag when a person's daily total exceeds this cap.",
)
@click.option(
    "--severity",
    type=click.Choice(["critical", "warning", "info", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Only show findings at this severity level (or all).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "csv", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format for findings.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Write findings to a file instead of stdout.",
)
def audit(
    file: str,
    hours_threshold: float,
    daily_max: float,
    severity: str,
    output_format: str,
    output: Optional[str],
) -> None:
    """Run all audit checks against a timesheet CSV and report findings."""
    logger.info("Loading timesheet from %s", file)
    df = load_timesheet(file)
    logger.info("Loaded %d entries across %d employees", len(df), df["employee_id"].nunique())

    findings = run_full_audit(df, hours_threshold=hours_threshold, daily_max=daily_max)
    results = findings_to_dataframe(findings)

    if severity != "all":
        results = results[results["severity"] == severity]

    results = results.sort_values(
        by=["severity", "employee_id", "date"],
        key=lambda col: col.map({"critical": 0, "warning": 1, "info": 2}) if col.name == "severity" else col,
    ).reset_index(drop=True)

    # ── summary counts ───────────────────────────────────────────────────
    counts = results["severity"].value_counts().to_dict()
    click.echo()
    click.secho("═══ Audit Summary ═══", fg="cyan", bold=True)
    click.echo(f"  Total findings : {len(results)}")
    click.echo(f"  Critical       : {counts.get('critical', 0)}")
    click.echo(f"  Warning        : {counts.get('warning', 0)}")
    click.echo(f"  Info           : {counts.get('info', 0)}")
    click.echo()

    if results.empty:
        click.secho("No findings to report — timesheet looks clean!", fg="green")
        return

    # ── output ───────────────────────────────────────────────────────────
    rendered = _render(results, output_format)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(rendered, encoding="utf-8")
        logger.info("Findings written to %s", output)
    else:
        click.echo(rendered)

    # Exit with non-zero if critical findings exist
    if counts.get("critical", 0) > 0:
        sys.exit(1)


# ── summary command ──────────────────────────────────────────────────────────
@cli.command()
@click.argument("file", type=click.Path(exists=True))
def summary(file: str) -> None:
    """Print a high-level summary of timesheet data (no audit)."""
    df = load_timesheet(file)

    click.echo()
    click.secho("═══ Timesheet Summary ═══", fg="cyan", bold=True)
    click.echo(f"  Period         : {df['date'].min().date()} → {df['date'].max().date()}")
    click.echo(f"  Total entries  : {len(df)}")
    click.echo(f"  Employees      : {df['employee_id'].nunique()}")
    click.echo(f"  Projects       : {df['project_code'].nunique()}")
    click.echo(f"  Total hours    : {df['hours'].sum():,.1f}")
    click.echo(f"  Avg hours/entry: {df['hours'].mean():.1f}")
    click.echo()

    click.secho("Hours by Employee:", bold=True)
    emp_hours = (
        df.groupby(["employee_id", "employee_name"])["hours"]
        .sum()
        .reset_index()
        .sort_values("hours", ascending=False)
    )
    for _, row in emp_hours.iterrows():
        click.echo(f"  {row['employee_name']:<22s}  {row['hours']:>8.1f}h")
    click.echo()

    click.secho("Hours by Project:", bold=True)
    proj_hours = (
        df.groupby("project_code")["hours"].sum().sort_values(ascending=False)
    )
    for proj, hrs in proj_hours.items():
        click.echo(f"  {proj:<30s}  {hrs:>8.1f}h")
    click.echo()


# ── helpers ──────────────────────────────────────────────────────────────────

def _render(results: pd.DataFrame, fmt: str) -> str:
    """Render findings DataFrame to the requested format string."""
    if fmt == "json":
        return results.to_json(orient="records", indent=2)
    if fmt == "csv":
        return results.to_csv(index=False)
    # Default: pretty table
    return results.to_string(index=False, max_colwidth=60)


if __name__ == "__main__":
    cli()
