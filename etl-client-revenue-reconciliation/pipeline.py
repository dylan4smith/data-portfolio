"""
Multi-Source Client Revenue Reconciliation Pipeline

Ingests revenue data from three business units with different schemas and
data quality issues, normalizes to a unified schema, validates and cleans
records, loads into a SQLite analytical database, and produces a
reconciliation summary report.

Usage:
    python pipeline.py [--data-dir DATA_DIR] [--db-path DB_PATH] [--output-dir OUTPUT_DIR]
"""

import argparse
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UNIFIED_COLUMNS = [
    "client_id",
    "client_name",
    "service_type",
    "revenue",
    "invoice_date",
    "region",
    "payment_status",
    "source_unit",
]

VALID_STATUSES = {"paid", "pending", "overdue"}
VALID_REGIONS = {"West", "East", "Midwest", "South"}
VALID_SERVICES = {"Advisory", "Implementation", "Support", "Training", "Analytics"}


@dataclass
class PipelineStats:
    """Tracks record counts through each pipeline stage for auditing."""
    records_extracted: int = 0
    records_after_dedup: int = 0
    records_after_validation: int = 0
    records_loaded: int = 0
    records_rejected: int = 0
    negative_revenue_flagged: int = 0
    missing_regions_imputed: int = 0
    date_format_corrections: int = 0


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging(output_dir: str) -> logging.Logger:
    """Configure structured logging to both console and file."""
    logger = logging.getLogger("revenue_etl")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — DEBUG and above
    log_path = os.path.join(output_dir, "pipeline_run.log")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Extract: read and normalize schemas from each source
# ---------------------------------------------------------------------------

SCHEMA_MAP = {
    "unit_a_consulting.csv": {
        "rename": {
            "client_id": "client_id",
            "client_name": "client_name",
            "service_type": "service_type",
            "revenue": "revenue",
            "invoice_date": "invoice_date",
            "region": "region",
            "status": "payment_status",
        },
        "source_label": "Consulting",
    },
    "unit_b_analytics.csv": {
        "rename": {
            "ClientCode": "client_id",
            "Name": "client_name",
            "Category": "service_type",
            "Amount": "revenue",
            "Date": "invoice_date",
            "Location": "region",
            "PaymentStatus": "payment_status",
        },
        "source_label": "Analytics",
    },
    "unit_c_training.csv": {
        "rename": {
            "client_id": "client_id",
            "client_name": "client_name",
            "service": "service_type",
            "amount_usd": "revenue",
            "date": "invoice_date",
            "region": "region",
            "payment_status": "payment_status",
        },
        "drop": ["notes"],
        "source_label": "Training",
    },
}


def extract(data_dir: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Read all source CSVs and map them to the unified schema.

    Returns a single concatenated DataFrame with a `source_unit` column.
    """
    frames: list[pd.DataFrame] = []

    for filename, config in SCHEMA_MAP.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            logger.warning("Source file not found, skipping: %s", filepath)
            continue

        logger.info("Extracting: %s", filename)
        df = pd.read_csv(filepath, dtype=str)  # read everything as string first
        original_count = len(df)

        # Drop extra columns not in our mapping
        drop_cols = config.get("drop", [])
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # Rename to unified schema
        df = df.rename(columns=config["rename"])
        df["source_unit"] = config["source_label"]

        # Keep only unified columns
        df = df[[c for c in UNIFIED_COLUMNS if c in df.columns]]

        logger.info("  -> Extracted %d records from %s", original_count, filename)
        frames.append(df)

    if not frames:
        logger.error("No source files found in %s", data_dir)
        raise FileNotFoundError(f"No source CSVs found in {data_dir}")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total extracted records: %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Transform: clean, validate, deduplicate
# ---------------------------------------------------------------------------

def parse_date_flexible(date_str: str) -> Optional[str]:
    """Attempt to parse dates in ISO or US format, returning ISO string or None."""
    if not date_str or pd.isna(date_str):
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def transform(df: pd.DataFrame, logger: logging.Logger, stats: PipelineStats) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and validate the unified DataFrame.

    Returns:
        (clean_df, rejected_df) — valid records and rejected records with reasons.
    """
    stats.records_extracted = len(df)
    rejection_reasons: list[dict] = []

    # Work on an explicit copy to avoid SettingWithCopyWarning
    df = df.copy()

    # --- Strip whitespace from string columns ---
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # --- Deduplicate ---
    before_dedup = len(df)
    df = df.drop_duplicates()
    stats.records_after_dedup = len(df)
    dupes_removed = before_dedup - stats.records_after_dedup
    if dupes_removed > 0:
        logger.info("Removed %d duplicate records", dupes_removed)

    # --- Normalize payment_status ---
    df["payment_status"] = df["payment_status"].str.lower().str.strip()

    # --- Parse and validate dates ---
    original_dates = df["invoice_date"].copy()
    df["invoice_date"] = df["invoice_date"].apply(parse_date_flexible)
    format_fixes = ((original_dates != df["invoice_date"]) & df["invoice_date"].notna()).sum()
    stats.date_format_corrections = int(format_fixes)
    if format_fixes > 0:
        logger.info("Corrected %d date format inconsistencies", format_fixes)

    # --- Convert revenue to numeric ---
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    # --- Validation pass ---
    valid_mask = pd.Series(True, index=df.index)

    # Reject null client_id
    null_client = df["client_id"].isna() | (df["client_id"] == "")
    for idx in df[null_client].index:
        rejection_reasons.append({"index": idx, "reason": "missing_client_id", "record": df.loc[idx].to_dict()})
    valid_mask &= ~null_client

    # Reject null/unparseable dates
    null_dates = df["invoice_date"].isna()
    for idx in df[null_dates & valid_mask].index:
        rejection_reasons.append({"index": idx, "reason": "invalid_date", "record": df.loc[idx].to_dict()})
    valid_mask &= ~null_dates

    # Flag and reject negative revenue
    negative_rev = df["revenue"] < 0
    stats.negative_revenue_flagged = int(negative_rev.sum())
    for idx in df[negative_rev & valid_mask].index:
        rejection_reasons.append({"index": idx, "reason": "negative_revenue", "record": df.loc[idx].to_dict()})
    valid_mask &= ~negative_rev
    if stats.negative_revenue_flagged > 0:
        logger.warning("Flagged %d records with negative revenue", stats.negative_revenue_flagged)

    # Reject null revenue
    null_rev = df["revenue"].isna()
    for idx in df[null_rev & valid_mask].index:
        rejection_reasons.append({"index": idx, "reason": "null_revenue", "record": df.loc[idx].to_dict()})
    valid_mask &= ~null_rev

    # Reject invalid payment statuses
    invalid_status = ~df["payment_status"].isin(VALID_STATUSES)
    for idx in df[invalid_status & valid_mask].index:
        rejection_reasons.append({"index": idx, "reason": "invalid_status", "record": df.loc[idx].to_dict()})
    valid_mask &= ~invalid_status

    # --- Impute missing regions ---
    missing_region = df["region"].isna() | (df["region"] == "")
    stats.missing_regions_imputed = int((missing_region & valid_mask).sum())
    df.loc[missing_region, "region"] = "Unknown"
    if stats.missing_regions_imputed > 0:
        logger.info("Imputed %d missing regions as 'Unknown'", stats.missing_regions_imputed)

    # --- Split valid and rejected ---
    clean_df = df[valid_mask].copy().reset_index(drop=True)
    rejected_df = pd.DataFrame([r["record"] for r in rejection_reasons])
    if not rejected_df.empty:
        rejected_df["rejection_reason"] = [r["reason"] for r in rejection_reasons]

    stats.records_after_validation = len(clean_df)
    stats.records_rejected = len(rejection_reasons)

    logger.info(
        "Validation complete: %d valid, %d rejected",
        stats.records_after_validation,
        stats.records_rejected,
    )

    return clean_df, rejected_df


# ---------------------------------------------------------------------------
# Load: insert into SQLite
# ---------------------------------------------------------------------------

def load(df: pd.DataFrame, db_path: str, logger: logging.Logger, stats: PipelineStats) -> None:
    """Load validated records into a SQLite database."""
    logger.info("Loading %d records into SQLite at %s", len(df), db_path)

    con = sqlite3.connect(db_path)
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS revenue_records (
                client_id       TEXT,
                client_name     TEXT,
                service_type    TEXT,
                revenue         REAL,
                invoice_date    TEXT,
                region          TEXT,
                payment_status  TEXT,
                source_unit     TEXT,
                loaded_at       TEXT
            )
        """)

        # Clear previous run data to make pipeline idempotent
        con.execute("DELETE FROM revenue_records")

        # Add loaded_at timestamp
        load_df = df.copy()
        load_df["loaded_at"] = datetime.now().isoformat()

        load_df.to_sql("revenue_records", con, if_exists="append", index=False)
        con.commit()
        stats.records_loaded = len(df)

        logger.info("Successfully loaded %d records", stats.records_loaded)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Report: generate reconciliation summary
# ---------------------------------------------------------------------------

def generate_report(db_path: str, output_dir: str, stats: PipelineStats, logger: logging.Logger) -> str:
    """Query the loaded data and produce a reconciliation summary CSV and log."""
    con = sqlite3.connect(db_path)
    try:
        # Revenue by business unit
        by_unit = pd.read_sql_query("""
            SELECT source_unit,
                   COUNT(*) AS record_count,
                   ROUND(SUM(revenue), 2) AS total_revenue,
                   ROUND(AVG(revenue), 2) AS avg_revenue
            FROM revenue_records
            GROUP BY source_unit
            ORDER BY total_revenue DESC
        """, con)

        # Revenue by client
        by_client = pd.read_sql_query("""
            SELECT client_id, client_name,
                   COUNT(*) AS invoices,
                   ROUND(SUM(revenue), 2) AS total_revenue,
                   ROUND(AVG(revenue), 2) AS avg_invoice
            FROM revenue_records
            GROUP BY client_id, client_name
            ORDER BY total_revenue DESC
        """, con)

        # Revenue by payment status
        by_status = pd.read_sql_query("""
            SELECT payment_status,
                   COUNT(*) AS record_count,
                   ROUND(SUM(revenue), 2) AS total_revenue
            FROM revenue_records
            GROUP BY payment_status
            ORDER BY total_revenue DESC
        """, con)

        # Monthly trend
        monthly = pd.read_sql_query("""
            SELECT SUBSTR(invoice_date, 1, 7) AS month,
                   COUNT(*) AS invoices,
                   ROUND(SUM(revenue), 2) AS total_revenue
            FROM revenue_records
            GROUP BY month
            ORDER BY month
        """, con)

    finally:
        con.close()

    # Save summary CSVs
    by_unit.to_csv(os.path.join(output_dir, "summary_by_unit.csv"), index=False)
    by_client.to_csv(os.path.join(output_dir, "summary_by_client.csv"), index=False)
    by_status.to_csv(os.path.join(output_dir, "summary_by_status.csv"), index=False)
    monthly.to_csv(os.path.join(output_dir, "summary_monthly_trend.csv"), index=False)

    # Log the reconciliation report
    report_lines = [
        "",
        "=" * 60,
        "  RECONCILIATION SUMMARY REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "Pipeline Statistics:",
        f"  Records extracted:      {stats.records_extracted:>6}",
        f"  After deduplication:    {stats.records_after_dedup:>6}",
        f"  After validation:       {stats.records_after_validation:>6}",
        f"  Records loaded:         {stats.records_loaded:>6}",
        f"  Records rejected:       {stats.records_rejected:>6}",
        f"  Date formats corrected: {stats.date_format_corrections:>6}",
        f"  Negative rev flagged:   {stats.negative_revenue_flagged:>6}",
        f"  Missing regions imputed:{stats.missing_regions_imputed:>6}",
        "",
        "Revenue by Business Unit:",
    ]
    for _, row in by_unit.iterrows():
        report_lines.append(
            f"  {row['source_unit']:<15} | {row['record_count']:>5} records | ${row['total_revenue']:>12,.2f} total | ${row['avg_revenue']:>10,.2f} avg"
        )

    report_lines.append("")
    report_lines.append("Top 5 Clients by Revenue:")
    for _, row in by_client.head(5).iterrows():
        report_lines.append(
            f"  {row['client_name']:<35} | {row['invoices']:>3} invoices | ${row['total_revenue']:>12,.2f}"
        )

    report_lines.append("")
    report_lines.append("Revenue by Payment Status:")
    for _, row in by_status.iterrows():
        report_lines.append(
            f"  {row['payment_status']:<10} | {row['record_count']:>5} records | ${row['total_revenue']:>12,.2f}"
        )

    report_lines.append("")
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    logger.info(report_text)

    # Save report as text file
    report_path = os.path.join(output_dir, "reconciliation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    return report_text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(data_dir: str, db_path: str, output_dir: str) -> PipelineStats:
    """Execute the full ETL pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    logger = configure_logging(output_dir)
    stats = PipelineStats()

    logger.info("Starting revenue reconciliation ETL pipeline")
    logger.info("Data directory: %s", data_dir)
    logger.info("Database path:  %s", db_path)
    logger.info("Output directory: %s", output_dir)

    try:
        # Extract
        logger.info("--- EXTRACT PHASE ---")
        raw_df = extract(data_dir, logger)

        # Transform
        logger.info("--- TRANSFORM PHASE ---")
        clean_df, rejected_df = transform(raw_df, logger, stats)

        # Save rejected records for audit
        if not rejected_df.empty:
            rejected_path = os.path.join(output_dir, "rejected_records.csv")
            rejected_df.to_csv(rejected_path, index=False)
            logger.info("Rejected records saved to %s", rejected_path)

        # Load
        logger.info("--- LOAD PHASE ---")
        load(clean_df, db_path, logger, stats)

        # Report
        logger.info("--- REPORTING PHASE ---")
        generate_report(db_path, output_dir, stats, logger)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.exception("Pipeline failed: %s", str(e))
        raise

    return stats


def main() -> None:
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multi-Source Client Revenue Reconciliation ETL Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory containing source CSV files (default: ./data)",
    )
    parser.add_argument(
        "--db-path",
        default=os.path.join(os.path.dirname(__file__), "output", "revenue.sqlite"),
        help="Path for the SQLite database file (default: ./output/revenue.sqlite)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "output"),
        help="Directory for output files and reports (default: ./output)",
    )
    args = parser.parse_args()

    run_pipeline(args.data_dir, args.db_path, args.output_dir)


if __name__ == "__main__":
    main()
