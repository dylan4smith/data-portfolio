"""
Unit tests for the revenue reconciliation ETL pipeline.
"""

import logging
import os
import tempfile

import pandas as pd
import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import (
    PipelineStats,
    extract,
    load,
    parse_date_flexible,
    run_pipeline,
    transform,
)
from generate_sample_data import (
    generate_analytics_unit,
    generate_consulting_unit,
    generate_training_unit,
)


@pytest.fixture
def sample_data_dir(tmp_path: str) -> str:
    """Generate fresh sample data in a temp directory."""
    data_dir = os.path.join(tmp_path, "data")
    os.makedirs(data_dir)
    generate_consulting_unit(os.path.join(data_dir, "unit_a_consulting.csv"), num_records=30)
    generate_analytics_unit(os.path.join(data_dir, "unit_b_analytics.csv"), num_records=25)
    generate_training_unit(os.path.join(data_dir, "unit_c_training.csv"), num_records=20)
    return data_dir


@pytest.fixture
def logger() -> logging.Logger:
    """Create a test logger."""
    log = logging.getLogger("test_etl")
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        log.addHandler(logging.StreamHandler())
    return log


class TestDateParsing:
    """Tests for flexible date parsing."""

    def test_iso_format(self) -> None:
        assert parse_date_flexible("2025-06-15") == "2025-06-15"

    def test_us_format(self) -> None:
        assert parse_date_flexible("06/15/2025") == "2025-06-15"

    def test_invalid_date(self) -> None:
        assert parse_date_flexible("not-a-date") is None

    def test_empty_string(self) -> None:
        assert parse_date_flexible("") is None

    def test_whitespace(self) -> None:
        assert parse_date_flexible("  2025-06-15  ") == "2025-06-15"


class TestExtract:
    """Tests for the extraction phase."""

    def test_extracts_all_sources(self, sample_data_dir: str, logger: logging.Logger) -> None:
        df = extract(sample_data_dir, logger)
        assert len(df) > 0
        assert "source_unit" in df.columns
        assert set(df["source_unit"].unique()) == {"Consulting", "Analytics", "Training"}

    def test_unified_schema(self, sample_data_dir: str, logger: logging.Logger) -> None:
        df = extract(sample_data_dir, logger)
        expected_cols = {"client_id", "client_name", "service_type", "revenue",
                         "invoice_date", "region", "payment_status", "source_unit"}
        assert set(df.columns) == expected_cols

    def test_missing_file_skipped(self, tmp_path: str, logger: logging.Logger) -> None:
        data_dir = os.path.join(tmp_path, "data")
        os.makedirs(data_dir)
        generate_consulting_unit(os.path.join(data_dir, "unit_a_consulting.csv"), num_records=10)
        # Only unit A exists — B and C should be skipped with warning
        df = extract(data_dir, logger)
        assert len(df) > 0
        assert df["source_unit"].unique().tolist() == ["Consulting"]

    def test_no_files_raises(self, tmp_path: str, logger: logging.Logger) -> None:
        empty_dir = os.path.join(tmp_path, "empty")
        os.makedirs(empty_dir)
        with pytest.raises(FileNotFoundError):
            extract(empty_dir, logger)


class TestTransform:
    """Tests for the transformation phase."""

    def test_removes_duplicates(self, logger: logging.Logger) -> None:
        df = pd.DataFrame({
            "client_id": ["C-1001", "C-1001"],
            "client_name": ["Acme", "Acme"],
            "service_type": ["Advisory", "Advisory"],
            "revenue": ["5000", "5000"],
            "invoice_date": ["2025-01-15", "2025-01-15"],
            "region": ["West", "West"],
            "payment_status": ["paid", "paid"],
            "source_unit": ["Consulting", "Consulting"],
        })
        stats = PipelineStats()
        clean, _ = transform(df, logger, stats)
        assert len(clean) == 1

    def test_rejects_negative_revenue(self, logger: logging.Logger) -> None:
        df = pd.DataFrame({
            "client_id": ["C-1001"],
            "client_name": ["Acme"],
            "service_type": ["Advisory"],
            "revenue": ["-5000"],
            "invoice_date": ["2025-01-15"],
            "region": ["West"],
            "payment_status": ["paid"],
            "source_unit": ["Consulting"],
        })
        stats = PipelineStats()
        clean, rejected = transform(df, logger, stats)
        assert len(clean) == 0
        assert stats.negative_revenue_flagged == 1

    def test_imputes_missing_region(self, logger: logging.Logger) -> None:
        df = pd.DataFrame({
            "client_id": ["C-1001"],
            "client_name": ["Acme"],
            "service_type": ["Advisory"],
            "revenue": ["5000"],
            "invoice_date": ["2025-01-15"],
            "region": [""],
            "payment_status": ["paid"],
            "source_unit": ["Consulting"],
        })
        stats = PipelineStats()
        clean, _ = transform(df, logger, stats)
        assert clean.iloc[0]["region"] == "Unknown"
        assert stats.missing_regions_imputed == 1

    def test_normalizes_status_casing(self, logger: logging.Logger) -> None:
        df = pd.DataFrame({
            "client_id": ["C-1001"],
            "client_name": ["Acme"],
            "service_type": ["Advisory"],
            "revenue": ["5000"],
            "invoice_date": ["2025-01-15"],
            "region": ["West"],
            "payment_status": ["PAID"],
            "source_unit": ["Consulting"],
        })
        stats = PipelineStats()
        clean, _ = transform(df, logger, stats)
        assert clean.iloc[0]["payment_status"] == "paid"


class TestEndToEnd:
    """Integration test for the full pipeline."""

    def test_full_pipeline(self, sample_data_dir: str, tmp_path: str) -> None:
        db_path = os.path.join(tmp_path, "test.sqlite")
        output_dir = os.path.join(tmp_path, "output")
        stats = run_pipeline(sample_data_dir, db_path, output_dir)

        assert stats.records_extracted > 0
        assert stats.records_loaded > 0
        assert stats.records_loaded <= stats.records_extracted
        assert os.path.exists(db_path)
        assert os.path.exists(os.path.join(output_dir, "reconciliation_report.txt"))
        assert os.path.exists(os.path.join(output_dir, "summary_by_unit.csv"))
        assert os.path.exists(os.path.join(output_dir, "summary_by_client.csv"))
