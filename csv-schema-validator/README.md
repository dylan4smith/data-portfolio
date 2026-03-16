# CSV Schema Validator

A command-line tool that validates CSV files against user-defined JSON schemas. Designed for data teams that need to enforce data quality rules before ingesting files into pipelines, warehouses, or reporting systems.

## Business Problem

When consulting teams receive client data as CSV exports, files often contain subtle issues — missing fields, wrong data types, values outside expected ranges, or inconsistent formatting. These problems cascade through downstream pipelines and dashboards, causing hours of debugging. This tool catches issues at the gate, before bad data enters the system.

## Features

- **Column validation**: Verifies required columns exist and flags unexpected extras
- **Type checking**: Validates integer, float, boolean, date, and string types per column
- **Range enforcement**: Checks numeric min/max and string length constraints
- **Pattern matching**: Validates values against regex patterns (e.g., employee IDs, email addresses)
- **Allowed values**: Restricts columns to a predefined set of valid entries
- **Null handling**: Configurable nullable/non-nullable enforcement per column
- **Multiple output formats**: Human-readable text or machine-readable JSON
- **Error limiting**: Truncate output to the first N issues for large files
- **Exit codes**: Returns 0 for valid files, 1 for invalid — integrates cleanly into CI/CD and shell scripts

## Tech Stack

- Python 3.10+
- [Click](https://click.palletsprojects.com/) — CLI framework
- Standard library: `csv`, `json`, `re`, `logging`
- [pytest](https://docs.pytest.org/) — Testing

## How to Run

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Validate a CSV against a schema
python validator.py data/employees_valid.csv schemas/employees.json

# Output as JSON (for piping into other tools)
python validator.py data/employees_invalid.csv schemas/employees.json --format json

# Limit error output
python validator.py data/employees_invalid.csv schemas/employees.json --max-errors 5

# Verbose logging
python validator.py data/employees_valid.csv schemas/employees.json -v
```

### Defining a Schema

Schemas are JSON files that describe the expected structure of a CSV. Example:

```json
{
  "name": "employee_roster",
  "description": "Validates HR employee roster exports.",
  "allow_extra_columns": false,
  "columns": [
    {
      "name": "employee_id",
      "dtype": "string",
      "required": true,
      "nullable": false,
      "pattern": "^EMP-\\d{5}$"
    },
    {
      "name": "salary",
      "dtype": "float",
      "required": true,
      "min_value": 30000,
      "max_value": 500000
    }
  ]
}
```

Supported `dtype` values: `string`, `integer`, `float`, `boolean`, `date` (ISO 8601: YYYY-MM-DD).

### Running Tests

```bash
pytest tests/ -v
```

## Sample Output

**Valid file:**
```
Validation PASSED: data/employees_valid.csv
  Schema:   employee_roster
  Rows:     8
  Errors:   0
  Warnings: 0
```

**Invalid file:**
```
Validation FAILED: data/employees_invalid.csv
  Schema:   employee_roster
  Rows:     8
  Errors:   7
  Warnings: 0

Issues:
  [ERROR] row 3, col 'employee_id' — Value 'BAD_ID' in 'employee_id' does not match pattern '^EMP-\d{5}$'.
  [ERROR] row 4, col 'first_name' — Null/empty value in non-nullable column 'first_name'.
  [ERROR] row 5, col 'email' — Value 'not-an-email' in 'email' does not match pattern ...
  ...
```

## Project Structure

```
csv-schema-validator/
├── validator.py          # Main CLI application
├── schemas/
│   └── employees.json    # Example schema definition
├── data/
│   ├── employees_valid.csv
│   └── employees_invalid.csv
├── tests/
│   └── test_validator.py # pytest test suite
├── requirements.txt
├── .gitignore
└── README.md
```
