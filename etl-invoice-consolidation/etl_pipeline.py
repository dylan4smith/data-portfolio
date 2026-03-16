"""
ETL Pipeline: Multi-Source Invoice Consolidation

Ingests invoice data from multiple vendors with different CSV schemas,
normalizes formats, validates data quality, and loads into a unified
SQLite database for downstream reporting.

Usage:
    python etl_pipeline.py --input-dir data/ --output-db output/invoices.db
    python etl_pipeline.py --input-dir data/ --output-db output/invoices.db --export-csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

UNIFIED_COLUMNS: list[str] = [
    "invoice_id",
    "vendor_name",
    "invoice_date",
    "due_date",
    "gross_amount_usd",
    "net_amount_usd",
    "tax_usd",
    "category",
    "payment_status",
    "source_file",
    "ingested_at",
]

PAYMENT_STATUS_MAP: dict[str, str] = {
    "paid": "Paid",
    "unpaid": "Pending",
    "pending": "Pending",
    "overdue": "Overdue",
    "yes": "Paid",
    "no": "Pending",
}

# Approximate exchange rates (USD) for currency normalization
EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "CAD": 0.74,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Quality Tracking
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Tracks data quality issues encountered during the ETL run."""

    total_records_read: int = 0
    records_loaded: int = 0
    records_rejected: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning(message)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        logger.error(message)

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  ETL Quality Report\n"
            f"{'='*60}\n"
            f"  Records read:     {self.total_records_read}\n"
            f"  Records loaded:   {self.records_loaded}\n"
            f"  Records rejected: {self.records_rejected}\n"
            f"  Warnings:         {len(self.warnings)}\n"
            f"  Errors:           {len(self.errors)}\n"
            f"{'='*60}\n"
        )


# ---------------------------------------------------------------------------
# Extraction Layer
# ---------------------------------------------------------------------------

def extract_csv(file_path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame, handling common encoding issues."""
    logger.info(f"Extracting data from {file_path.name}")
    try:
        df = pd.read_csv(file_path, encoding="utf-8", dtype=str)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        logger.info(f"  -> {len(df)} rows extracted")
        return df
    except Exception as exc:
        logger.error(f"Failed to read {file_path.name}: {exc}")
        raise


# ---------------------------------------------------------------------------
# Transformation Layer — Vendor-Specific Mappers
# ---------------------------------------------------------------------------

def transform_vendor_a(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    """Map Vendor A schema to the unified format."""
    records: list[dict] = []
    for _, row in df.iterrows():
        report.total_records_read += 1
        invoice_id = row.get("invoice_id", "")

        # Validate amount
        raw_amount = row.get("amount_usd", "")
        try:
            amount = float(raw_amount)
            if pd.isna(amount):
                raise ValueError("NaN value")
        except (ValueError, TypeError):
            report.add_warning(f"Vendor A {invoice_id}: missing or invalid amount '{raw_amount}' — skipping")
            report.records_rejected += 1
            continue

        if amount < 0:
            report.add_warning(f"Vendor A {invoice_id}: negative amount {amount} — likely a credit memo, skipping")
            report.records_rejected += 1
            continue

        # Parse dates
        invoice_date = _parse_date(row.get("invoice_date", ""), f"Vendor A {invoice_id}", report)
        due_date = _parse_date(row.get("due_date", ""), f"Vendor A {invoice_id}", report)

        # Normalize payment status
        status = _normalize_status(row.get("payment_status", ""), f"Vendor A {invoice_id}", report)

        records.append({
            "invoice_id": invoice_id,
            "vendor_name": row.get("vendor_name", "Unknown"),
            "invoice_date": invoice_date,
            "due_date": due_date,
            "gross_amount_usd": round(amount, 2),
            "net_amount_usd": round(amount, 2),  # Vendor A doesn't separate tax
            "tax_usd": 0.0,
            "category": row.get("category", "Uncategorized"),
            "payment_status": status,
        })

    return pd.DataFrame(records)


def transform_vendor_b(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    """Map Vendor B schema to the unified format."""
    records: list[dict] = []
    for _, row in df.iterrows():
        report.total_records_read += 1
        invoice_id = row.get("invoicenumber", "")

        # Validate amounts
        try:
            gross = float(row.get("total", ""))
            tax = float(row.get("tax", "0"))
            net = float(row.get("net", "0"))
        except (ValueError, TypeError):
            report.add_warning(f"Vendor B {invoice_id}: non-numeric amount — skipping")
            report.records_rejected += 1
            continue

        # Parse date (MM/DD/YYYY format)
        invoice_date = _parse_date(row.get("date", ""), f"Vendor B {invoice_id}", report, fmt="%m/%d/%Y")

        status = _normalize_status(row.get("status", ""), f"Vendor B {invoice_id}", report)

        records.append({
            "invoice_id": invoice_id,
            "vendor_name": row.get("vendor", "Unknown"),
            "invoice_date": invoice_date,
            "due_date": None,  # Vendor B doesn't provide due dates
            "gross_amount_usd": round(gross, 2),
            "net_amount_usd": round(net, 2),
            "tax_usd": round(tax, 2),
            "category": row.get("category", "Uncategorized"),
            "payment_status": status,
        })

    return pd.DataFrame(records)


def transform_vendor_c(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    """Map Vendor C schema to the unified format. Handles currency conversion."""
    records: list[dict] = []
    for _, row in df.iterrows():
        report.total_records_read += 1
        invoice_id = row.get("id", "")

        # Validate amount and convert currency
        try:
            raw_amount = float(row.get("gross_amount", ""))
        except (ValueError, TypeError):
            report.add_warning(f"Vendor C {invoice_id}: invalid amount — skipping")
            report.records_rejected += 1
            continue

        currency = row.get("currency", "USD").upper()
        rate = EXCHANGE_RATES.get(currency)
        if rate is None:
            report.add_warning(f"Vendor C {invoice_id}: unknown currency '{currency}' — skipping")
            report.records_rejected += 1
            continue

        amount_usd = round(raw_amount * rate, 2)
        if currency != "USD":
            logger.info(f"  Converted {raw_amount} {currency} -> {amount_usd} USD for {invoice_id}")

        invoice_date = _parse_date(row.get("date_issued", ""), f"Vendor C {invoice_id}", report)
        due_date = _parse_date(row.get("date_due", ""), f"Vendor C {invoice_id}", report)
        status = _normalize_status(row.get("paid", ""), f"Vendor C {invoice_id}", report)

        records.append({
            "invoice_id": invoice_id,
            "vendor_name": row.get("vendor", "Unknown"),
            "invoice_date": invoice_date,
            "due_date": due_date,
            "gross_amount_usd": amount_usd,
            "net_amount_usd": amount_usd,
            "tax_usd": 0.0,
            "category": row.get("expense_type", "Uncategorized"),
            "payment_status": status,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Shared Transformation Helpers
# ---------------------------------------------------------------------------

def _parse_date(
    raw_value: str,
    context: str,
    report: QualityReport,
    fmt: Optional[str] = None,
) -> Optional[str]:
    """Attempt to parse a date string into ISO format (YYYY-MM-DD)."""
    if not raw_value or pd.isna(raw_value):
        return None

    formats_to_try = [fmt] if fmt else ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]
    for date_fmt in formats_to_try:
        try:
            return datetime.strptime(raw_value.strip(), date_fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    report.add_warning(f"{context}: could not parse date '{raw_value}'")
    return None


def _normalize_status(raw_value: str, context: str, report: QualityReport) -> str:
    """Map vendor-specific payment statuses to a unified set."""
    if not raw_value or pd.isna(raw_value):
        report.add_warning(f"{context}: missing payment status — defaulting to 'Unknown'")
        return "Unknown"

    normalized = PAYMENT_STATUS_MAP.get(raw_value.strip().lower())
    if normalized is None:
        report.add_warning(f"{context}: unrecognized status '{raw_value}' — defaulting to 'Unknown'")
        return "Unknown"
    return normalized


# ---------------------------------------------------------------------------
# Vendor Router
# ---------------------------------------------------------------------------

VENDOR_TRANSFORMERS = {
    "vendor_a": transform_vendor_a,
    "vendor_b": transform_vendor_b,
    "vendor_c": transform_vendor_c,
}


def detect_vendor(file_path: Path) -> Optional[str]:
    """Detect the vendor based on the filename convention."""
    name_lower = file_path.stem.lower()
    for vendor_key in VENDOR_TRANSFORMERS:
        if vendor_key in name_lower:
            return vendor_key
    return None


# ---------------------------------------------------------------------------
# Load Layer
# ---------------------------------------------------------------------------

def load_to_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    """Load the consolidated DataFrame into a SQLite database."""
    logger.info(f"Loading {len(df)} records into {db_path}")

    con = sqlite3.connect(str(db_path))
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                invoice_id       TEXT PRIMARY KEY,
                vendor_name      TEXT,
                invoice_date     TEXT,
                due_date         TEXT,
                gross_amount_usd REAL,
                net_amount_usd   REAL,
                tax_usd          REAL,
                category         TEXT,
                payment_status   TEXT,
                source_file      TEXT,
                ingested_at      TEXT
            )
        """)

        # Clear and reload for idempotent re-runs
        con.execute("DELETE FROM invoices")

        df.to_sql("invoices", con, if_exists="replace", index=False)

        row_count = con.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
        logger.info(f"  -> {row_count} total records in database")

        # Create a summary view for quick reporting
        con.execute("DROP VIEW IF EXISTS invoice_summary")
        con.execute("""
            CREATE VIEW invoice_summary AS
            SELECT
                vendor_name,
                category,
                payment_status,
                COUNT(*)                        AS invoice_count,
                ROUND(SUM(gross_amount_usd), 2) AS total_gross_usd,
                ROUND(AVG(gross_amount_usd), 2) AS avg_invoice_usd,
                MIN(invoice_date)               AS earliest_invoice,
                MAX(invoice_date)               AS latest_invoice
            FROM invoices
            GROUP BY vendor_name, category, payment_status
            ORDER BY vendor_name, total_gross_usd DESC
        """)
        logger.info("  -> Created 'invoice_summary' view")
        con.commit()
    finally:
        con.close()


def export_consolidated_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    """Export the consolidated data as a CSV for non-technical stakeholders."""
    output_path = output_dir / "consolidated_invoices.csv"
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Exported consolidated CSV to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(input_dir: Path, output_db: Path, export_csv: bool = False) -> QualityReport:
    """Execute the full ETL pipeline: extract -> transform -> load."""
    report = QualityReport()
    ingestion_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        report.add_error(f"No CSV files found in {input_dir}")
        return report

    logger.info(f"Found {len(csv_files)} CSV file(s) in {input_dir}")

    all_transformed: list[pd.DataFrame] = []

    for csv_file in csv_files:
        vendor_key = detect_vendor(csv_file)
        if vendor_key is None:
            report.add_error(f"Could not detect vendor for {csv_file.name} — skipping file")
            continue

        transformer = VENDOR_TRANSFORMERS[vendor_key]

        # Extract
        raw_df = extract_csv(csv_file)

        # Transform
        transformed_df = transformer(raw_df, report)

        if transformed_df.empty:
            report.add_warning(f"No valid records from {csv_file.name}")
            continue

        # Add metadata columns
        transformed_df["source_file"] = csv_file.name
        transformed_df["ingested_at"] = ingestion_timestamp

        all_transformed.append(transformed_df)

    if not all_transformed:
        report.add_error("No records survived transformation — nothing to load")
        return report

    # Combine all vendor data
    consolidated_df = pd.concat(all_transformed, ignore_index=True)

    # Reorder columns to match the unified schema
    consolidated_df = consolidated_df[UNIFIED_COLUMNS]

    report.records_loaded = len(consolidated_df)

    # Load
    output_db.parent.mkdir(parents=True, exist_ok=True)
    load_to_sqlite(consolidated_df, output_db)

    # Optional CSV export
    if export_csv:
        export_consolidated_csv(consolidated_df, output_db.parent)

    return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-source invoice ETL pipeline — normalizes vendor CSVs into a unified SQLite database.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing vendor CSV files (default: data/)",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("output/invoices.db"),
        help="Path for the output SQLite database (default: output/invoices.db)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export consolidated data as CSV",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    logger.info("Starting invoice consolidation ETL pipeline")
    logger.info(f"  Input directory: {args.input_dir}")
    logger.info(f"  Output database: {args.output_db}")

    try:
        report = run_pipeline(args.input_dir, args.output_db, args.export_csv)
    except Exception as exc:
        logger.critical(f"Pipeline failed with unhandled exception: {exc}", exc_info=True)
        return 1

    print(report.summary())

    if report.errors:
        logger.error("Pipeline completed with errors — review log output")
        return 1

    logger.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
