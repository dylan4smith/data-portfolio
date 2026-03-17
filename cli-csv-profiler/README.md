# CSV Profiler

A Python CLI tool that generates automated data quality reports for CSV datasets. Designed for data teams and analysts who need quick, repeatable quality checks before loading data into warehouses or feeding it into analytical pipelines.

## Business Problem

When consulting teams receive raw data exports from clients, the first step is always the same: understand the shape, quality, and quirks of the data. Manually inspecting spreadsheets doesn't scale, and issues like missing values, type mismatches, and outliers can silently corrupt downstream analyses. CSV Profiler automates this gatekeeping step, producing a comprehensive quality report in seconds.

## Features

- **Column-level statistics** — counts, means, medians, standard deviations, quartiles for numeric columns
- **Missing value analysis** — per-column and dataset-wide null counts and percentages
- **Semantic type inference** — detects datetimes, booleans, categoricals, and identifiers beyond raw pandas dtypes
- **Outlier detection** — IQR-based flagging with configurable reporting
- **Duplicate row detection** — identifies exact row duplicates
- **Flexible output** — human-readable text or machine-parseable JSON
- **Numeric coercion** — optionally converts string columns that are mostly numeric, flagging values that couldn't convert
- **Custom delimiters** — supports TSV and other delimited formats

## Tech Stack

- Python 3.10+
- [click](https://click.palletsprojects.com/) — CLI framework
- [pandas](https://pandas.pydata.org/) — data loading and analysis
- [NumPy](https://numpy.org/) — numerical operations
- [pytest](https://docs.pytest.org/) — testing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic text report to stdout
python csv_profiler.py data/sample_transactions.csv

# JSON report saved to file
python csv_profiler.py data/sample_transactions.csv -f json -o report.json

# TSV file with verbose logging
python csv_profiler.py data/export.tsv -d '\t' -v

# Skip automatic numeric coercion
python csv_profiler.py data/messy_data.csv --no-coerce-numeric
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o, --output PATH` | Save report to file | stdout |
| `-f, --format [text\|json]` | Report format | `text` |
| `-d, --delimiter CHAR` | CSV delimiter | `,` |
| `--coerce-numeric / --no-coerce-numeric` | Auto-coerce numeric strings | enabled |
| `-v, --verbose` | DEBUG-level logging | off |

## Sample Output

```
========================================================================
  CSV PROFILER — DATA QUALITY REPORT
========================================================================
  File:           data/sample_transactions.csv
  Rows:           25
  Columns:        8
  Duplicate rows: 0 (0.0%)
  Missing cells:  4 (2.0%)
========================================================================

  COLUMN: amount
  ────────────────────────────────────────
    Inferred type:  float
    Non-null:       23 / 25
    Missing:        2 (8.0%)
    Mean:           1,195.63
    Outliers (IQR): 2 (8.7%)
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
cli-csv-profiler/
├── csv_profiler.py          # Main CLI application
├── data/
│   └── sample_transactions.csv  # Synthetic sample dataset
├── tests/
│   ├── __init__.py
│   └── test_profiler.py     # Unit + integration tests
├── requirements.txt
├── .gitignore
└── README.md
```
