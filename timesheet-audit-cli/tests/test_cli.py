"""Integration tests for the CLI interface using Click's test runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from cli import cli

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "data" / "sample_timesheets.csv"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestAuditCommand:
    def test_runs_successfully(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["audit", str(SAMPLE_DATA)])
        assert "Audit Summary" in result.output
        # Should exit 1 because sample data has critical findings
        assert result.exit_code == 1

    def test_severity_filter(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["audit", str(SAMPLE_DATA), "--severity", "info"])
        assert "Audit Summary" in result.output
        # Info-only should exit 0 (no critical findings in filtered set)
        assert result.exit_code == 0

    def test_json_format(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["audit", str(SAMPLE_DATA), "--format", "json"]
        )
        assert "rule_id" in result.output or "No findings" in result.output

    def test_csv_format(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["audit", str(SAMPLE_DATA), "--format", "csv"]
        )
        assert "rule_id" in result.output or "No findings" in result.output

    def test_output_to_file(self, runner: CliRunner, tmp_path: Path) -> None:
        out_file = tmp_path / "findings.csv"
        result = runner.invoke(
            cli,
            ["audit", str(SAMPLE_DATA), "--format", "csv", "--output", str(out_file)],
        )
        assert out_file.exists()
        content = out_file.read_text()
        assert "rule_id" in content


class TestSummaryCommand:
    def test_runs_successfully(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["summary", str(SAMPLE_DATA)])
        assert result.exit_code == 0
        assert "Timesheet Summary" in result.output
        assert "Total entries" in result.output

    def test_shows_employees(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["summary", str(SAMPLE_DATA)])
        assert "Hours by Employee" in result.output

    def test_shows_projects(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["summary", str(SAMPLE_DATA)])
        assert "Hours by Project" in result.output


class TestVersionFlag:
    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert "1.0.0" in result.output
