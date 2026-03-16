# ETL Pipeline: Multi-Source Invoice Consolidation

## Business Problem

Mid-size companies that work with multiple vendors often receive invoice data in different formats — varying column names, date formats, currencies, and payment status labels. Manually reconciling these into a single source of truth is tedious, error-prone, and blocks timely financial reporting.

This pipeline automates the ingestion, normalization, and consolidation of invoice data from multiple vendor CSV sources into a unified DuckDB analytical database.

## Approach

The pipeline follows a classic **Extract → Transform → Load** pattern:

1. **Extract** — Reads CSV files from a configurable input directory, auto-detecting the vendor based on filename conventions.
2. **Transform** — Each vendor has a dedicated mapper that:
   - Normalizes column names to a unified schema
   - Parses dates across formats (`YYYY-MM-DD`, `MM/DD/YYYY`)
   - Converts foreign currencies to USD using configurable exchange rates
   - Standardizes payment status labels (`Paid`, `Pending`, `Overdue`)
   - Validates amounts (rejects nulls, negatives, non-numeric values)
3. **Load** — Writes the consolidated DataFrame into a DuckDB database with a pre-built summary view for reporting. Supports idempotent re-runs.

A data quality report is generated after every run showing records read, loaded, rejected, and any warnings or errors.

## Tech Stack

- **Python 3.10+**
- **pandas** — data manipulation and transformation
- **SQLite** — lightweight embedded database (zero dependencies, no server required)
- **argparse** — CLI interface
- **logging** — structured pipeline observability

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run with defaults (reads data/, writes to output/invoices.db)
python etl_pipeline.py

# Run with custom paths and CSV export
python etl_pipeline.py --input-dir data/ --output-db output/invoices.db --export-csv
```

## Sample Output

```
============================================================
  ETL Quality Report
============================================================
  Records read:     30
  Records loaded:   27
  Records rejected: 3
  Warnings:         3
  Errors:           0
============================================================
```

The pipeline intentionally includes dirty data (missing amounts, negative values, non-numeric entries, foreign currencies) to demonstrate robust error handling and data validation.

## Project Structure

```
etl-invoice-consolidation/
├── data/
│   ├── vendor_a_invoices.csv    # Standard US format
│   ├── vendor_b_invoices.csv    # Different columns, MM/DD/YYYY dates
│   └── vendor_c_invoices.csv    # Multi-currency, different status labels
├── output/
│   ├── invoices.db              # Unified SQLite analytical database
│   └── consolidated_invoices.csv # Optional CSV export
├── etl_pipeline.py              # Main pipeline script
├── requirements.txt
├── .gitignore
└── README.md
```

## Extending the Pipeline

To add a new vendor:

1. Create a `transform_vendor_x()` function following the existing pattern
2. Register it in the `VENDOR_TRANSFORMERS` dictionary
3. Name the input CSV with the vendor key (e.g., `vendor_x_invoices.csv`)
