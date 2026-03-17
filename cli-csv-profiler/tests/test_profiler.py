"""
Unit tests for the csv_profiler module.

Covers type inference, column profiling, dataset profiling, report formatting,
and the CLI entry point.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from click.testing import CliRunner

from csv_profiler import (
    ColumnProfile,
    DatasetProfile,
    cli,
    format_json_report,
    format_text_report,
    infer_semantic_type,
    profile_column,
    profile_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Write a small CSV file and return its path."""
    content = textwrap.dedent("""\
        id,name,amount,region,active,created_at
        1,Alice,100.50,East,True,2025-01-01
        2,Bob,200.00,West,False,2025-01-02
        3,Charlie,,East,True,2025-01-03
        4,Alice,100.50,East,True,2025-01-01
        5,Dana,9999.99,West,False,2025-01-05
    """)
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(content, encoding="utf-8")
    return csv_file


@pytest.fixture()
def sample_df(sample_csv: Path) -> pd.DataFrame:
    return pd.read_csv(sample_csv)


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

class TestInferSemanticType:
    def test_integer_column(self) -> None:
        s = pd.Series([1, 2, 3, 4, 5], name="nums")
        assert infer_semantic_type(s) == "integer"

    def test_float_column(self) -> None:
        s = pd.Series([1.1, 2.2, 3.3], name="floats")
        assert infer_semantic_type(s) == "float"

    def test_boolean_numeric(self) -> None:
        s = pd.Series([0, 1, 1, 0, 1], name="flags")
        assert infer_semantic_type(s) == "boolean"

    def test_boolean_strings(self) -> None:
        s = pd.Series(["True", "False", "True", "False"], name="active")
        assert infer_semantic_type(s) == "boolean"

    def test_datetime_strings(self) -> None:
        s = pd.Series(["2025-01-01", "2025-02-15", "2025-03-20"], name="dates")
        assert infer_semantic_type(s) == "datetime"

    def test_categorical_low_cardinality(self) -> None:
        s = pd.Series(["A", "B", "A", "B", "A", "B"] * 20, name="cat")
        assert infer_semantic_type(s) == "categorical"

    def test_identifier_high_cardinality(self) -> None:
        s = pd.Series([f"ID-{i}" for i in range(100)], name="uid")
        assert infer_semantic_type(s) == "identifier"

    def test_empty_series(self) -> None:
        s = pd.Series([None, None, None], name="empty")
        assert infer_semantic_type(s) == "empty"


# ---------------------------------------------------------------------------
# Column profiling
# ---------------------------------------------------------------------------

class TestProfileColumn:
    def test_null_stats(self, sample_df: pd.DataFrame) -> None:
        col = profile_column(sample_df["amount"])
        assert col.null_count == 1
        assert col.non_null_count == 4

    def test_numeric_stats_present(self, sample_df: pd.DataFrame) -> None:
        col = profile_column(sample_df["amount"])
        assert col.mean is not None
        assert col.median is not None
        assert col.std is not None

    def test_outlier_detection(self, sample_df: pd.DataFrame) -> None:
        col = profile_column(sample_df["amount"])
        # 9999.99 should be flagged as outlier
        assert col.outlier_count is not None
        assert col.outlier_count >= 1

    def test_non_numeric_no_stats(self, sample_df: pd.DataFrame) -> None:
        col = profile_column(sample_df["name"])
        assert col.mean is None
        assert col.outlier_count is None

    def test_top_values(self, sample_df: pd.DataFrame) -> None:
        col = profile_column(sample_df["name"])
        assert len(col.top_values) > 0
        top_value_names = [tv["value"] for tv in col.top_values]
        assert "Alice" in top_value_names

    def test_unique_stats(self, sample_df: pd.DataFrame) -> None:
        col = profile_column(sample_df["region"])
        assert col.unique_count == 2  # East, West


# ---------------------------------------------------------------------------
# Dataset profiling
# ---------------------------------------------------------------------------

class TestProfileDataset:
    def test_row_and_column_counts(self, sample_df: pd.DataFrame) -> None:
        profile = profile_dataset(sample_df, "test.csv", 256)
        assert profile.row_count == 5
        assert profile.column_count == 6

    def test_duplicate_detection(self, sample_df: pd.DataFrame) -> None:
        profile = profile_dataset(sample_df, "test.csv", 256)
        # Row 1 and Row 4 are duplicates (Alice, 100.50, East, True, 2025-01-01)
        assert profile.duplicate_row_count >= 1

    def test_total_null_cells(self, sample_df: pd.DataFrame) -> None:
        profile = profile_dataset(sample_df, "test.csv", 256)
        assert profile.total_null_cells >= 1

    def test_columns_profiled(self, sample_df: pd.DataFrame) -> None:
        profile = profile_dataset(sample_df, "test.csv", 256)
        col_names = [c.name for c in profile.columns]
        assert "id" in col_names
        assert "amount" in col_names


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

class TestReportFormatting:
    def test_text_report_contains_header(self, sample_df: pd.DataFrame) -> None:
        profile = profile_dataset(sample_df, "test.csv", 256)
        report = format_text_report(profile)
        assert "CSV PROFILER" in report
        assert "test.csv" in report

    def test_json_report_is_valid(self, sample_df: pd.DataFrame) -> None:
        profile = profile_dataset(sample_df, "test.csv", 256)
        report = format_json_report(profile)
        data = json.loads(report)
        assert data["row_count"] == 5
        assert len(data["columns"]) == 6


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCLI:
    def test_basic_run(self, sample_csv: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, [str(sample_csv)])
        assert result.exit_code == 0
        assert "CSV PROFILER" in result.output

    def test_json_format(self, sample_csv: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, [str(sample_csv), "-f", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output.split("\nData Quality Warnings:")[0])
        assert "row_count" in data

    def test_output_to_file(self, sample_csv: Path, tmp_path: Path) -> None:
        out_file = tmp_path / "report.txt"
        runner = CliRunner()
        result = runner.invoke(cli, [str(sample_csv), "-o", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        assert "CSV PROFILER" in out_file.read_text()

    def test_nonexistent_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["/nonexistent/path.csv"])
        assert result.exit_code != 0

    def test_verbose_flag(self, sample_csv: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, [str(sample_csv), "-v"])
        assert result.exit_code == 0
