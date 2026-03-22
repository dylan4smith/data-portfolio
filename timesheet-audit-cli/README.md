# Timesheet Audit CLI

A command-line tool that validates consultant timesheet data, detects anomalies, and generates audit reports. Built for operations and finance teams at professional services firms who need to catch billing errors before they reach clients.

## Business Problem

Consulting firms bill clients based on logged hours. Errors in timesheets — duplicate entries, impossible hour totals, unapproved weekend work, or missing days — lead to revenue leakage, compliance risk, and damaged client trust. Manual timesheet review doesn't scale beyond a handful of consultants.

This tool automates six categories of timesheet validation so a single ops analyst can audit hundreds of entries in seconds.

## Audit Rules

| Rule ID | Severity | Description |
|---|---|---|
| `EXCESS_HOURS` | Critical | Single entry exceeds configurable threshold (default 12h) |
| `INVALID_HOURS` | Critical | Entry with zero or negative hours |
| `DAILY_TOTAL_EXCEEDED` | Critical | Employee's total hours for one day exceed cap (default 16h) |
| `DUPLICATE_ENTRY` | Warning | Exact duplicate row (same person, date, project, task, hours) |
| `UNAPPROVED_WEEKEND` | Warning | Weekend work not marked as approved |
| `MISSING_WEEKDAY` | Info | Employee has no entries for a business day |

## Tech Stack

- **Python 3.9+**
- **click** — CLI framework with subcommands, options, and help text
- **pandas** — data loading, grouping, and aggregation
- **pytest** — unit and integration tests

## Project Structure

```
timesheet-audit-cli/
├── cli.py                  # CLI entry point (click commands)
├── audit_engine.py         # Core validation logic (decoupled from CLI)
├── requirements.txt
├── data/
│   ├── generate_sample_data.py   # Synthetic data generator
│   └── sample_timesheets.csv     # Pre-generated test data
├── tests/
│   ├── test_audit_engine.py      # Unit tests for each audit rule
│   └── test_cli.py               # Integration tests via CliRunner
└── reports/                # Output directory for saved findings
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run a full audit
python cli.py audit data/sample_timesheets.csv

# Filter by severity
python cli.py audit data/sample_timesheets.csv --severity critical

# Export findings as CSV
python cli.py audit data/sample_timesheets.csv --format csv --output reports/findings.csv

# Export findings as JSON
python cli.py audit data/sample_timesheets.csv --format json

# View timesheet summary (no audit)
python cli.py summary data/sample_timesheets.csv

# Customize thresholds
python cli.py audit data/sample_timesheets.csv --hours-threshold 10 --daily-max 14
```

## Running Tests

```bash
pytest tests/ -v
```

## Sample Output

```
═══ Audit Summary ═══
  Total findings : 87
  Critical       : 12
  Warning        : 18
  Info           : 57

rule_id              severity employee_id employee_name          date  ...
EXCESS_HOURS         critical    EMP-1002  Marcus Chen    2026-01-19  ...
INVALID_HOURS        critical    EMP-1003  Sofia Gutierrez 2026-02-04 ...
DAILY_TOTAL_EXCEEDED critical    EMP-1001  Anika Patel    2026-02-11  ...
...
```

## Design Decisions

- **Engine/CLI separation**: `audit_engine.py` is a pure-Python module with no CLI dependencies, making it reusable in Jupyter notebooks, Airflow DAGs, or API services.
- **Dataclass findings**: Each issue is an `AuditFinding` dataclass with structured fields, enabling downstream filtering and aggregation.
- **Non-zero exit code**: The `audit` command exits with code 1 when critical findings exist, making it CI/CD-friendly for automated gatekeeping.
- **Synthetic data generator**: Includes a seeded generator that produces realistic anomalies, so the tool can be demonstrated without real client data.
