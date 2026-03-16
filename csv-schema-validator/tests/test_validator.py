"""Unit tests for the CSV Schema Validator."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

# Adjust path so we can import from the parent package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validator import (
    ColumnRule,
    Schema,
    ValidationResult,
    _cast_value,
    cli,
    validate_csv,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def basic_schema(tmp_dir: Path) -> Path:
    """Write a minimal schema file and return its path."""
    schema = {
        "name": "test_schema",
        "description": "A minimal schema for tests.",
        "columns": [
            {"name": "id", "dtype": "integer", "required": True, "nullable": False, "min_value": 1},
            {"name": "name", "dtype": "string", "required": True, "nullable": False, "min_length": 1},
            {"name": "score", "dtype": "float", "required": True, "nullable": True, "min_value": 0, "max_value": 100},
        ],
    }
    path = tmp_dir / "schema.json"
    path.write_text(json.dumps(schema))
    return path


def _write_csv(path: Path, content: str) -> Path:
    """Helper to write CSV content to a file."""
    path.write_text(content.strip() + "\n")
    return path


# ── Unit tests: _cast_value ───────────────────────────────────────────────


class TestCastValue:
    def test_string_passthrough(self):
        val, ok = _cast_value("hello", "string")
        assert ok is True
        assert val == "hello"

    def test_integer_valid(self):
        val, ok = _cast_value("42", "integer")
        assert ok is True
        assert val == 42

    def test_integer_invalid(self):
        val, ok = _cast_value("abc", "integer")
        assert ok is False

    def test_float_valid(self):
        val, ok = _cast_value("3.14", "float")
        assert ok is True
        assert abs(val - 3.14) < 0.001

    def test_float_invalid(self):
        _, ok = _cast_value("not_a_number", "float")
        assert ok is False

    def test_boolean_true_variants(self):
        for v in ("true", "True", "1", "yes"):
            val, ok = _cast_value(v, "boolean")
            assert ok is True
            assert val is True

    def test_boolean_false_variants(self):
        for v in ("false", "False", "0", "no"):
            val, ok = _cast_value(v, "boolean")
            assert ok is True
            assert val is False

    def test_boolean_invalid(self):
        _, ok = _cast_value("maybe", "boolean")
        assert ok is False

    def test_date_valid(self):
        val, ok = _cast_value("2024-01-15", "date")
        assert ok is True

    def test_date_invalid(self):
        _, ok = _cast_value("Jan 15 2024", "date")
        assert ok is False


# ── Unit tests: ColumnRule ────────────────────────────────────────────────


class TestColumnRule:
    def test_from_dict_defaults(self):
        rule = ColumnRule.from_dict({"name": "test_col"})
        assert rule.name == "test_col"
        assert rule.dtype == "string"
        assert rule.required is True
        assert rule.nullable is False

    def test_from_dict_full(self):
        rule = ColumnRule.from_dict({
            "name": "amount",
            "dtype": "float",
            "required": True,
            "nullable": False,
            "min_value": 0,
            "max_value": 10000,
            "pattern": r"^\d+\.\d{2}$",
        })
        assert rule.dtype == "float"
        assert rule.max_value == 10000


# ── Unit tests: Schema loading ───────────────────────────────────────────


class TestSchema:
    def test_load_from_file(self, basic_schema: Path):
        schema = Schema.from_file(basic_schema)
        assert schema.name == "test_schema"
        assert len(schema.columns) == 3
        assert schema.columns[0].name == "id"


# ── Integration tests: validate_csv ──────────────────────────────────────


class TestValidateCSV:
    def test_valid_csv(self, tmp_dir: Path, basic_schema: Path):
        csv_path = _write_csv(
            tmp_dir / "good.csv",
            "id,name,score\n1,Alice,95.5\n2,Bob,87.0\n3,Carol,",
        )
        schema = Schema.from_file(basic_schema)
        result = validate_csv(csv_path, schema)
        assert result.is_valid
        assert result.total_rows == 3
        assert result.error_count == 0

    def test_missing_required_column(self, tmp_dir: Path, basic_schema: Path):
        csv_path = _write_csv(tmp_dir / "missing_col.csv", "id,score\n1,50.0")
        schema = Schema.from_file(basic_schema)
        result = validate_csv(csv_path, schema)
        assert not result.is_valid
        assert any("Required column 'name' is missing" in e.message for e in result.errors)

    def test_type_mismatch(self, tmp_dir: Path, basic_schema: Path):
        csv_path = _write_csv(tmp_dir / "bad_type.csv", "id,name,score\nabc,Alice,95.5")
        schema = Schema.from_file(basic_schema)
        result = validate_csv(csv_path, schema)
        assert not result.is_valid
        assert any("Type mismatch" in e.message for e in result.errors)

    def test_null_in_non_nullable(self, tmp_dir: Path, basic_schema: Path):
        csv_path = _write_csv(tmp_dir / "null.csv", "id,name,score\n1,,50.0")
        schema = Schema.from_file(basic_schema)
        result = validate_csv(csv_path, schema)
        assert not result.is_valid
        assert any("Null/empty" in e.message for e in result.errors)

    def test_value_below_min(self, tmp_dir: Path, basic_schema: Path):
        csv_path = _write_csv(tmp_dir / "low.csv", "id,name,score\n0,Alice,50.0")
        schema = Schema.from_file(basic_schema)
        result = validate_csv(csv_path, schema)
        assert not result.is_valid
        assert any("below minimum" in e.message for e in result.errors)

    def test_value_above_max(self, tmp_dir: Path, basic_schema: Path):
        csv_path = _write_csv(tmp_dir / "high.csv", "id,name,score\n1,Alice,150.0")
        schema = Schema.from_file(basic_schema)
        result = validate_csv(csv_path, schema)
        assert not result.is_valid
        assert any("exceeds maximum" in e.message for e in result.errors)

    def test_allowed_values(self, tmp_dir: Path):
        schema_data = {
            "name": "status_schema",
            "columns": [
                {"name": "status", "dtype": "string", "allowed_values": ["active", "inactive"]},
            ],
        }
        schema_path = tmp_dir / "schema.json"
        schema_path.write_text(json.dumps(schema_data))
        csv_path = _write_csv(tmp_dir / "bad_val.csv", "status\npending")
        schema = Schema.from_file(schema_path)
        result = validate_csv(csv_path, schema)
        assert not result.is_valid
        assert any("not in allowed values" in e.message for e in result.errors)

    def test_regex_pattern(self, tmp_dir: Path):
        schema_data = {
            "name": "code_schema",
            "columns": [
                {"name": "code", "dtype": "string", "pattern": r"^[A-Z]{3}-\d{3}$"},
            ],
        }
        schema_path = tmp_dir / "schema.json"
        schema_path.write_text(json.dumps(schema_data))
        csv_path = _write_csv(tmp_dir / "bad_pattern.csv", "code\nabc-123")
        schema = Schema.from_file(schema_path)
        result = validate_csv(csv_path, schema)
        assert not result.is_valid
        assert any("does not match pattern" in e.message for e in result.errors)

    def test_extra_columns_warning(self, tmp_dir: Path, basic_schema: Path):
        csv_path = _write_csv(tmp_dir / "extra.csv", "id,name,score,bonus\n1,Alice,50.0,yes")
        schema = Schema.from_file(basic_schema)
        result = validate_csv(csv_path, schema)
        assert result.is_valid  # warnings don't cause failure
        assert result.warning_count > 0

    def test_empty_file(self, tmp_dir: Path, basic_schema: Path):
        csv_path = _write_csv(tmp_dir / "empty.csv", "")
        schema = Schema.from_file(basic_schema)
        result = validate_csv(csv_path, schema)
        assert not result.is_valid
        assert any("empty" in e.message.lower() for e in result.errors)

    def test_file_not_found(self, tmp_dir: Path, basic_schema: Path):
        schema = Schema.from_file(basic_schema)
        result = validate_csv(tmp_dir / "nonexistent.csv", schema)
        assert not result.is_valid
        assert any("File not found" in e.message for e in result.errors)


# ── CLI tests ─────────────────────────────────────────────────────────────


class TestCLI:
    def test_cli_valid_file(self, tmp_dir: Path, basic_schema: Path):
        csv_path = _write_csv(tmp_dir / "good.csv", "id,name,score\n1,Alice,90.0")
        runner = CliRunner()
        result = runner.invoke(cli, [str(csv_path), str(basic_schema)])
        assert result.exit_code == 0
        assert "PASSED" in result.output

    def test_cli_invalid_file(self, tmp_dir: Path, basic_schema: Path):
        csv_path = _write_csv(tmp_dir / "bad.csv", "id,name,score\nabc,Alice,90.0")
        runner = CliRunner()
        result = runner.invoke(cli, [str(csv_path), str(basic_schema)])
        assert result.exit_code == 1
        assert "FAILED" in result.output

    def test_cli_json_format(self, tmp_dir: Path, basic_schema: Path):
        csv_path = _write_csv(tmp_dir / "good.csv", "id,name,score\n1,Alice,90.0")
        runner = CliRunner()
        result = runner.invoke(cli, [str(csv_path), str(basic_schema), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["valid"] is True

    def test_cli_max_errors(self, tmp_dir: Path, basic_schema: Path):
        # Multiple errors, but limit to 1
        csv_path = _write_csv(
            tmp_dir / "many_bad.csv",
            "id,name,score\nabc,,200\ndef,,300",
        )
        runner = CliRunner()
        result = runner.invoke(cli, [str(csv_path), str(basic_schema), "--max-errors", "1"])
        assert result.exit_code == 1
        assert "not shown" in result.output
