"""
CSV Schema Validator — Validate CSV files against user-defined schemas.

Ensures data quality by checking column presence, data types, required fields,
value ranges, allowed values, and regex patterns before data enters pipelines.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import click

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("csv-schema-validator")

# ---------------------------------------------------------------------------
# Data classes for schema definition
# ---------------------------------------------------------------------------

@dataclass
class ColumnRule:
    """Validation rules for a single CSV column."""

    name: str
    dtype: str = "string"  # string | integer | float | boolean | date
    required: bool = True
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[list[str]] = None
    pattern: Optional[str] = None  # regex pattern
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnRule":
        """Create a ColumnRule from a dictionary (parsed JSON)."""
        return cls(
            name=data["name"],
            dtype=data.get("dtype", "string"),
            required=data.get("required", True),
            nullable=data.get("nullable", False),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            allowed_values=data.get("allowed_values"),
            pattern=data.get("pattern"),
            min_length=data.get("min_length"),
            max_length=data.get("max_length"),
        )


@dataclass
class Schema:
    """A complete validation schema for a CSV file."""

    name: str
    description: str
    columns: list[ColumnRule]
    allow_extra_columns: bool = False

    @classmethod
    def from_file(cls, path: Path) -> "Schema":
        """Load a schema from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        columns = [ColumnRule.from_dict(col) for col in raw["columns"]]
        return cls(
            name=raw.get("name", path.stem),
            description=raw.get("description", ""),
            columns=columns,
            allow_extra_columns=raw.get("allow_extra_columns", False),
        )


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    """A single validation failure."""

    row: Optional[int]  # None for file-level issues
    column: Optional[str]
    message: str
    severity: str = "error"  # error | warning


@dataclass
class ValidationResult:
    """Aggregated result of validating a CSV against a schema."""

    file_path: str
    schema_name: str
    total_rows: int = 0
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(e.severity == "error" for e in self.errors)

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == "warning")


def _cast_value(value: str, dtype: str) -> tuple[Any, bool]:
    """Attempt to cast a string value to the target dtype.

    Returns (casted_value, success_bool).
    """
    if dtype == "string":
        return value, True
    if dtype == "integer":
        try:
            return int(value), True
        except ValueError:
            return value, False
    if dtype == "float":
        try:
            return float(value), True
        except ValueError:
            return value, False
    if dtype == "boolean":
        if value.lower() in ("true", "1", "yes"):
            return True, True
        if value.lower() in ("false", "0", "no"):
            return False, True
        return value, False
    if dtype == "date":
        # Accept ISO-8601 date format (YYYY-MM-DD)
        date_pattern = r"^\d{4}-\d{2}-\d{2}$"
        if re.match(date_pattern, value):
            return value, True
        return value, False
    return value, True


def validate_csv(csv_path: Path, schema: Schema) -> ValidationResult:
    """Validate a CSV file against the provided schema.

    Performs structural checks (column presence) and per-cell validation
    (type, nullability, ranges, patterns, allowed values).
    """
    result = ValidationResult(file_path=str(csv_path), schema_name=schema.name)
    expected_columns = {col.name for col in schema.columns}
    column_lookup: dict[str, ColumnRule] = {col.name: col for col in schema.columns}

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                result.errors.append(
                    ValidationError(row=None, column=None, message="CSV file is empty or has no header row.")
                )
                return result

            actual_columns = set(reader.fieldnames)

            # -- Column-level checks ----------------------------------------
            missing = expected_columns - actual_columns
            for col_name in sorted(missing):
                rule = column_lookup[col_name]
                if rule.required:
                    result.errors.append(
                        ValidationError(row=None, column=col_name, message=f"Required column '{col_name}' is missing.")
                    )
                else:
                    result.errors.append(
                        ValidationError(
                            row=None,
                            column=col_name,
                            message=f"Optional column '{col_name}' is missing.",
                            severity="warning",
                        )
                    )

            extra = actual_columns - expected_columns
            if extra and not schema.allow_extra_columns:
                for col_name in sorted(extra):
                    result.errors.append(
                        ValidationError(
                            row=None,
                            column=col_name,
                            message=f"Unexpected column '{col_name}' not defined in schema.",
                            severity="warning",
                        )
                    )

            # -- Row-level checks -------------------------------------------
            for row_idx, row in enumerate(reader, start=2):  # row 1 is header
                result.total_rows += 1
                for rule in schema.columns:
                    if rule.name not in actual_columns:
                        continue  # already reported as missing
                    raw_value = row.get(rule.name, "")
                    cell_value = raw_value.strip() if raw_value else ""

                    # Null / empty check
                    if cell_value == "":
                        if not rule.nullable:
                            result.errors.append(
                                ValidationError(
                                    row=row_idx,
                                    column=rule.name,
                                    message=f"Null/empty value in non-nullable column '{rule.name}'.",
                                )
                            )
                        continue  # skip further checks for empty cells

                    # Type check
                    casted, ok = _cast_value(cell_value, rule.dtype)
                    if not ok:
                        result.errors.append(
                            ValidationError(
                                row=row_idx,
                                column=rule.name,
                                message=(
                                    f"Type mismatch in '{rule.name}': "
                                    f"expected {rule.dtype}, got '{cell_value}'."
                                ),
                            )
                        )
                        continue  # don't run range / pattern checks on bad types

                    # Min / max value (numeric types)
                    if rule.dtype in ("integer", "float"):
                        numeric_val = float(casted)
                        if rule.min_value is not None and numeric_val < rule.min_value:
                            result.errors.append(
                                ValidationError(
                                    row=row_idx,
                                    column=rule.name,
                                    message=(
                                        f"Value {numeric_val} in '{rule.name}' "
                                        f"is below minimum {rule.min_value}."
                                    ),
                                )
                            )
                        if rule.max_value is not None and numeric_val > rule.max_value:
                            result.errors.append(
                                ValidationError(
                                    row=row_idx,
                                    column=rule.name,
                                    message=(
                                        f"Value {numeric_val} in '{rule.name}' "
                                        f"exceeds maximum {rule.max_value}."
                                    ),
                                )
                            )

                    # String length constraints
                    if rule.dtype == "string":
                        if rule.min_length is not None and len(cell_value) < rule.min_length:
                            result.errors.append(
                                ValidationError(
                                    row=row_idx,
                                    column=rule.name,
                                    message=(
                                        f"Value in '{rule.name}' has length {len(cell_value)}, "
                                        f"below minimum {rule.min_length}."
                                    ),
                                )
                            )
                        if rule.max_length is not None and len(cell_value) > rule.max_length:
                            result.errors.append(
                                ValidationError(
                                    row=row_idx,
                                    column=rule.name,
                                    message=(
                                        f"Value in '{rule.name}' has length {len(cell_value)}, "
                                        f"exceeds maximum {rule.max_length}."
                                    ),
                                )
                            )

                    # Allowed values
                    if rule.allowed_values is not None:
                        if str(casted) not in rule.allowed_values:
                            result.errors.append(
                                ValidationError(
                                    row=row_idx,
                                    column=rule.name,
                                    message=(
                                        f"Value '{casted}' in '{rule.name}' "
                                        f"is not in allowed values: {rule.allowed_values}."
                                    ),
                                )
                            )

                    # Regex pattern
                    if rule.pattern is not None:
                        if not re.fullmatch(rule.pattern, cell_value):
                            result.errors.append(
                                ValidationError(
                                    row=row_idx,
                                    column=rule.name,
                                    message=(
                                        f"Value '{cell_value}' in '{rule.name}' "
                                        f"does not match pattern '{rule.pattern}'."
                                    ),
                                )
                            )

    except FileNotFoundError:
        result.errors.append(
            ValidationError(row=None, column=None, message=f"File not found: {csv_path}")
        )
    except csv.Error as exc:
        result.errors.append(
            ValidationError(row=None, column=None, message=f"CSV parsing error: {exc}")
        )

    return result


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _format_text(result: ValidationResult) -> str:
    """Human-readable plain-text report."""
    lines: list[str] = []
    status = "PASSED" if result.is_valid else "FAILED"
    lines.append(f"Validation {status}: {result.file_path}")
    lines.append(f"  Schema:   {result.schema_name}")
    lines.append(f"  Rows:     {result.total_rows}")
    lines.append(f"  Errors:   {result.error_count}")
    lines.append(f"  Warnings: {result.warning_count}")

    if result.errors:
        lines.append("")
        lines.append("Issues:")
        for err in result.errors:
            loc = f"row {err.row}" if err.row else "file"
            col = f", col '{err.column}'" if err.column else ""
            lines.append(f"  [{err.severity.upper()}] {loc}{col} — {err.message}")

    return "\n".join(lines)


def _format_json(result: ValidationResult) -> str:
    """Machine-readable JSON report."""
    payload = {
        "file": result.file_path,
        "schema": result.schema_name,
        "valid": result.is_valid,
        "total_rows": result.total_rows,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "issues": [
            {
                "row": e.row,
                "column": e.column,
                "severity": e.severity,
                "message": e.message,
            }
            for e in result.errors
        ],
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("csv_file", type=click.Path(exists=True, path_type=Path))
@click.argument("schema_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format for the validation report.",
)
@click.option(
    "--max-errors", "max_errors",
    type=int,
    default=0,
    help="Stop after N errors (0 = report all).",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
def cli(csv_file: Path, schema_file: Path, output_format: str, max_errors: int, verbose: bool) -> None:
    """Validate a CSV file against a JSON schema definition.

    \b
    CSV_FILE     Path to the CSV file to validate.
    SCHEMA_FILE  Path to the JSON schema definition.

    \b
    Example:
        python validator.py data/employees.csv schemas/employees.json
        python validator.py data/employees.csv schemas/employees.json --format json
    """
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.debug("Loading schema from %s", schema_file)

    schema = Schema.from_file(schema_file)
    logger.debug("Schema '%s' loaded with %d column rules.", schema.name, len(schema.columns))

    logger.debug("Validating %s …", csv_file)
    result = validate_csv(csv_file, schema)

    # Truncate errors if requested
    if max_errors > 0 and len(result.errors) > max_errors:
        truncated = len(result.errors) - max_errors
        result.errors = result.errors[:max_errors]
        result.errors.append(
            ValidationError(
                row=None,
                column=None,
                message=f"… and {truncated} more issue(s) not shown (--max-errors={max_errors}).",
                severity="warning",
            )
        )

    # Output
    if output_format == "json":
        click.echo(_format_json(result))
    else:
        click.echo(_format_text(result))

    # Exit code: 0 = valid, 1 = invalid
    sys.exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    cli()
