# Multi-Source Client Revenue Reconciliation Pipeline

## Business Problem

Consulting firms that serve multiple business units often receive revenue data in different formats, with varying column names, date conventions, and data quality standards. Manually reconciling these sources is time-consuming and error-prone, leading to inaccurate financial reporting and delayed insights.

This pipeline automates the ingestion, validation, cleaning, and reconciliation of revenue data from three business units into a single analytical database, producing audit-ready summary reports.

## Approach

The pipeline follows a classic **Extract → Transform → Load** architecture:

1. **Extract**: Reads CSV exports from three business units, each with distinct schemas and column naming conventions, and normalizes them to a unified schema.
2. **Transform**: Applies data quality rules including deduplication, date format standardization, revenue validation (rejecting negatives), status normalization, and missing value imputation. Rejected records are saved separately for audit.
3. **Load**: Inserts validated records into a SQLite analytical database with idempotent writes (safe to re-run).
4. **Report**: Queries the database to produce reconciliation summaries by business unit, client, payment status, and monthly trend.

All steps include structured logging (console + file) and a `PipelineStats` audit trail tracking record counts through each stage.

## Tech Stack

- **Python 3.10+** — core language with type hints throughout
- **pandas** — data manipulation and schema normalization
- **SQLite** — lightweight analytical database (zero-dependency, no server required)
- **pytest** — unit and integration testing
- **argparse** — CLI interface
- **logging** — structured pipeline observability

## Project Structure

```
etl-client-revenue-reconciliation/
├── pipeline.py                  # Main ETL pipeline
├── generate_sample_data.py      # Synthetic data generator
├── requirements.txt             # Python dependencies
├── .gitignore
├── data/                        # Source CSV files (synthetic)
│   ├── unit_a_consulting.csv
│   ├── unit_b_analytics.csv
│   └── unit_c_training.csv
├── output/                      # Pipeline outputs (generated on run)
│   ├── revenue.sqlite
│   ├── reconciliation_report.txt
│   ├── rejected_records.csv
│   ├── summary_by_unit.csv
│   ├── summary_by_client.csv
│   ├── summary_by_status.csv
│   └── summary_monthly_trend.csv
└── tests/
    └── test_pipeline.py         # Unit and integration tests
```

## How to Run

### Setup

```bash
pip install -r requirements.txt
```

### Generate Sample Data

```bash
python generate_sample_data.py
```

### Run the Pipeline

```bash
python pipeline.py
```

With custom paths:

```bash
python pipeline.py --data-dir ./data --db-path ./output/revenue.sqlite --output-dir ./output
```

### Run Tests

```bash
pytest tests/ -v
```

## Sample Output

After running the pipeline, the reconciliation report includes:

```
Pipeline Statistics:
  Records extracted:         299
  After deduplication:       295
  After validation:          282
  Records loaded:            282
  Records rejected:           13
  Date formats corrected:      4
  Negative rev flagged:        3
  Missing regions imputed:     6

Revenue by Business Unit:
  Analytics       |    92 records | $  5,432,109.50 total | $   59,044.67 avg
  Consulting      |   118 records | $  4,987,234.80 total | $   42,264.70 avg
  Training        |    72 records | $  1,623,450.20 total | $   22,548.00 avg
```

## Data Quality Handling

| Issue | Action | Audit Trail |
|-------|--------|-------------|
| Duplicate records | Removed | Count logged |
| Negative revenue | Rejected | Saved to `rejected_records.csv` |
| Malformed dates | Parsed with flexible formats | Count logged |
| Unparseable dates | Rejected | Saved to `rejected_records.csv` |
| Missing regions | Imputed as "Unknown" | Count logged |
| Inconsistent status casing | Normalized to lowercase | Applied globally |
| Extra whitespace | Stripped from all string fields | Applied globally |
