"""
CSV Profiler — A CLI tool for automated data quality profiling of CSV datasets.

Generates comprehensive reports including column statistics, missing value analysis,
type inference, duplicate detection, and outlier flagging. Designed for data teams
who need quick, repeatable quality checks before loading data into pipelines.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    """Statistical profile for a single column."""

    name: str
    dtype_raw: str
    dtype_inferred: str
    total_count: int
    non_null_count: int
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    top_values: list[dict[str, Any]] = field(default_factory=list)
    # Numeric-only fields
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    outlier_count: Optional[int] = None
    outlier_pct: Optional[float] = None


@dataclass
class DatasetProfile:
    """Full profiling report for a CSV dataset."""

    file_path: str
    file_size_bytes: int
    row_count: int
    column_count: int
    duplicate_row_count: int
    duplicate_row_pct: float
    total_null_cells: int
    total_null_pct: float
    columns: list[ColumnProfile] = field(default_factory=list)
    profiled_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Profiling engine
# ---------------------------------------------------------------------------

def infer_semantic_type(series: pd.Series) -> str:
    """Infer a human-friendly semantic type for a pandas Series.

    Goes beyond raw dtype to detect dates, booleans stored as strings,
    identifiers, and categoricals.
    """
    if series.dropna().empty:
        return "empty"

    # Already numeric
    if pd.api.types.is_numeric_dtype(series):
        if set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            return "boolean"
        if pd.api.types.is_float_dtype(series):
            return "float"
        return "integer"

    # Try datetime
    try:
        pd.to_datetime(series.dropna(), format="mixed", dayfirst=False)
        return "datetime"
    except (ValueError, TypeError):
        pass

    # Boolean strings
    sample_lower = series.dropna().astype(str).str.lower()
    if sample_lower.isin(["true", "false", "yes", "no", "0", "1"]).all():
        return "boolean"

    # Cardinality heuristic — low cardinality ⇒ categorical
    nunique = series.nunique()
    n = len(series.dropna())
    if n > 0 and nunique / n <= 0.05:
        return "categorical"
    if n > 0 and nunique / n > 0.9:
        return "identifier"

    return "text"


def profile_column(series: pd.Series) -> ColumnProfile:
    """Build a :class:`ColumnProfile` for a single pandas Series."""
    total = len(series)
    non_null = int(series.notna().sum())
    null = total - non_null
    null_pct = round(null / total * 100, 2) if total > 0 else 0.0
    unique = int(series.nunique(dropna=True))
    unique_pct = round(unique / non_null * 100, 2) if non_null > 0 else 0.0

    top_values_series = series.value_counts(dropna=True).head(5)
    top_values = [
        {"value": str(val), "count": int(cnt)}
        for val, cnt in top_values_series.items()
    ]

    inferred = infer_semantic_type(series)

    col = ColumnProfile(
        name=series.name,
        dtype_raw=str(series.dtype),
        dtype_inferred=inferred,
        total_count=total,
        non_null_count=non_null,
        null_count=null,
        null_pct=null_pct,
        unique_count=unique,
        unique_pct=unique_pct,
        top_values=top_values,
    )

    # Numeric stats & outlier detection via IQR method
    # Skip boolean columns — quantile operations on bools raise in newer numpy
    if (pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
            and non_null > 0):
        col.mean = round(float(series.mean()), 4)
        col.median = round(float(series.median()), 4)
        col.std = round(float(series.std()), 4) if non_null > 1 else 0.0
        col.min_value = round(float(series.min()), 4)
        col.max_value = round(float(series.max()), 4)
        col.q1 = round(float(series.quantile(0.25)), 4)
        col.q3 = round(float(series.quantile(0.75)), 4)
        col.iqr = round(col.q3 - col.q1, 4)

        lower_fence = col.q1 - 1.5 * col.iqr
        upper_fence = col.q3 + 1.5 * col.iqr
        outliers = series.dropna()[(series.dropna() < lower_fence) | (series.dropna() > upper_fence)]
        col.outlier_count = int(len(outliers))
        col.outlier_pct = round(col.outlier_count / non_null * 100, 2)

    return col


def profile_dataset(df: pd.DataFrame, file_path: str, file_size: int) -> DatasetProfile:
    """Profile an entire DataFrame and return a :class:`DatasetProfile`."""
    row_count = len(df)
    col_count = len(df.columns)
    dup_count = int(df.duplicated().sum())
    dup_pct = round(dup_count / row_count * 100, 2) if row_count > 0 else 0.0
    total_nulls = int(df.isnull().sum().sum())
    total_cells = row_count * col_count
    total_null_pct = round(total_nulls / total_cells * 100, 2) if total_cells > 0 else 0.0

    column_profiles = [profile_column(df[col]) for col in df.columns]

    return DatasetProfile(
        file_path=file_path,
        file_size_bytes=file_size,
        row_count=row_count,
        column_count=col_count,
        duplicate_row_count=dup_count,
        duplicate_row_pct=dup_pct,
        total_null_cells=total_nulls,
        total_null_pct=total_null_pct,
        columns=column_profiles,
    )


# ---------------------------------------------------------------------------
# Report formatters
# ---------------------------------------------------------------------------

def format_text_report(profile: DatasetProfile) -> str:
    """Render the profile as a human-readable text report."""
    lines: list[str] = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("  CSV PROFILER — DATA QUALITY REPORT")
    lines.append(sep)
    lines.append(f"  File:           {profile.file_path}")
    lines.append(f"  Size:           {profile.file_size_bytes:,} bytes")
    lines.append(f"  Rows:           {profile.row_count:,}")
    lines.append(f"  Columns:        {profile.column_count}")
    lines.append(f"  Duplicate rows: {profile.duplicate_row_count:,} ({profile.duplicate_row_pct}%)")
    lines.append(f"  Missing cells:  {profile.total_null_cells:,} ({profile.total_null_pct}%)")
    lines.append(f"  Profiled at:    {profile.profiled_at}")
    lines.append(sep)

    for col in profile.columns:
        lines.append("")
        lines.append(f"  COLUMN: {col.name}")
        lines.append(f"  {'─' * 40}")
        lines.append(f"    Raw dtype:      {col.dtype_raw}")
        lines.append(f"    Inferred type:  {col.dtype_inferred}")
        lines.append(f"    Non-null:       {col.non_null_count:,} / {col.total_count:,}")
        lines.append(f"    Missing:        {col.null_count:,} ({col.null_pct}%)")
        lines.append(f"    Unique values:  {col.unique_count:,} ({col.unique_pct}%)")

        if col.mean is not None:
            lines.append(f"    Mean:           {col.mean}")
            lines.append(f"    Median:         {col.median}")
            lines.append(f"    Std dev:        {col.std}")
            lines.append(f"    Min / Max:      {col.min_value} / {col.max_value}")
            lines.append(f"    Q1 / Q3 / IQR:  {col.q1} / {col.q3} / {col.iqr}")
            lines.append(f"    Outliers (IQR): {col.outlier_count} ({col.outlier_pct}%)")

        if col.top_values:
            lines.append(f"    Top values:")
            for tv in col.top_values:
                lines.append(f"      • {tv['value']}  (n={tv['count']})")

    lines.append("")
    lines.append(sep)
    lines.append("  END OF REPORT")
    lines.append(sep)
    return "\n".join(lines)


def format_json_report(profile: DatasetProfile) -> str:
    """Render the profile as a JSON string."""
    return json.dumps(asdict(profile), indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o", "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Save report to a file instead of printing to stdout.",
)
@click.option(
    "-f", "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format for the profiling report.",
)
@click.option(
    "--coerce-numeric / --no-coerce-numeric",
    default=True,
    show_default=True,
    help="Attempt to coerce object columns to numeric types where possible.",
)
@click.option(
    "-d", "--delimiter",
    default=",",
    show_default=True,
    help="Column delimiter used in the CSV file.",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
def cli(
    csv_path: str,
    output: Optional[str],
    output_format: str,
    coerce_numeric: bool,
    delimiter: str,
    verbose: bool,
) -> None:
    """Profile a CSV file and generate a data quality report.

    CSV_PATH is the path to the CSV file you want to profile.

    \b
    Examples:
        csv-profiler data/sample_transactions.csv
        csv-profiler data/sales.csv -f json -o report.json
        csv-profiler data/export.tsv -d '\\t' --no-coerce-numeric
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    csv_file = Path(csv_path)
    logger.info("Reading %s …", csv_file.name)

    try:
        df = pd.read_csv(csv_file, delimiter=delimiter, low_memory=False)
    except Exception as exc:
        logger.error("Failed to read CSV: %s", exc)
        sys.exit(1)

    logger.info("Loaded %d rows × %d columns.", len(df), len(df.columns))

    if coerce_numeric:
        for col_name in df.select_dtypes(include=["object"]).columns:
            converted = pd.to_numeric(df[col_name], errors="coerce")
            # Only coerce if a meaningful fraction actually converted
            original_non_null = df[col_name].notna().sum()
            converted_non_null = converted.notna().sum()
            if original_non_null > 0 and converted_non_null / original_non_null >= 0.5:
                coerced_count = int((df[col_name].notna() & converted.isna()).sum())
                if coerced_count > 0:
                    logger.debug(
                        "Column '%s': coerced to numeric (%d values became NaN).",
                        col_name,
                        coerced_count,
                    )
                df[col_name] = converted

    logger.info("Profiling dataset …")
    profile = profile_dataset(df, str(csv_file), csv_file.stat().st_size)

    if output_format == "json":
        report = format_json_report(profile)
    else:
        report = format_text_report(profile)

    if output:
        out_path = Path(output)
        out_path.write_text(report, encoding="utf-8")
        logger.info("Report saved to %s", out_path)
    else:
        click.echo(report)

    # Summary warnings
    warnings: list[str] = []
    for col in profile.columns:
        if col.null_pct > 10:
            warnings.append(f"⚠  Column '{col.name}' has {col.null_pct}% missing values.")
        if col.outlier_count is not None and col.outlier_pct and col.outlier_pct > 5:
            warnings.append(f"⚠  Column '{col.name}' has {col.outlier_count} outliers ({col.outlier_pct}%).")

    if profile.duplicate_row_pct > 0:
        warnings.append(f"⚠  Dataset contains {profile.duplicate_row_count} duplicate rows ({profile.duplicate_row_pct}%).")

    if warnings:
        click.echo("")
        click.echo("Data Quality Warnings:")
        for w in warnings:
            click.echo(f"  {w}")


if __name__ == "__main__":
    cli()
