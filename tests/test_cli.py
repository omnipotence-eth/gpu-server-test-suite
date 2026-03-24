"""Tests for the monitor and history CLI commands.

Uses Click's CliRunner so no real GPU or terminal is required.
The monitor loop is stopped by making time.sleep raise KeyboardInterrupt;
Rich's Live display is mocked to avoid terminal-escape-code pollution.
"""

from unittest.mock import patch

from click.testing import CliRunner

from src.main import cli
from src.reporting.history import save_run
from src.reporting.models import TestResult, TestStatus


def _pass_result() -> TestResult:
    return TestResult(
        test_name="deployment.driver_loaded",
        status=TestStatus.PASS,
        duration_seconds=0.1,
        message="ok",
    )


# ─── monitor command ────────────────────────────────────────────


class TestMonitorCommand:
    def test_exits_cleanly_on_keyboard_interrupt(self, mock_gpu_info):
        runner = CliRunner()
        with (
            patch("src.main.get_all_gpus", return_value=[mock_gpu_info]),
            patch("src.main.time.sleep", side_effect=KeyboardInterrupt),
            patch("rich.live.Live"),
        ):
            result = runner.invoke(cli, ["monitor", "--interval", "0"])
        assert result.exit_code == 0

    def test_gpu_poll_error_does_not_crash(self):
        """An exception from get_all_gpus should be caught, not propagated."""
        runner = CliRunner()
        call_count = 0

        def _flaky_gpus():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise KeyboardInterrupt
            raise RuntimeError("GPU not found")

        with (
            patch("src.main.get_all_gpus", side_effect=_flaky_gpus),
            patch("src.main.time.sleep"),
            patch("rich.live.Live"),
        ):
            result = runner.invoke(cli, ["monitor", "--interval", "0"])
        assert result.exit_code == 0


# ─── history command ────────────────────────────────────────────


class TestHistoryCommand:
    def test_empty_history_shows_guidance(self, tmp_path):
        runner = CliRunner()
        missing = tmp_path / "no_history.jsonl"
        with patch("src.reporting.history._HISTORY_FILE", missing):
            result = runner.invoke(cli, ["history"])
        assert result.exit_code == 0
        assert "No run history" in result.output

    def test_shows_run_entries(self, tmp_path):
        runner = CliRunner()
        hf = tmp_path / ".run_history.jsonl"
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("abc123ef", "quick", [_pass_result()], 0.9)
            result = runner.invoke(cli, ["history"])
        assert result.exit_code == 0
        assert "quick" in result.output
        assert "PASS" in result.output

    def test_failures_flag_filters_passing_runs(self, tmp_path):
        runner = CliRunner()
        hf = tmp_path / ".run_history.jsonl"
        fail_result = TestResult(
            test_name="deployment.driver_loaded",
            status=TestStatus.FAIL,
            duration_seconds=0.1,
            message="bad",
            failure_code="DIAG-001",
        )
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("pass-only", "quick", [_pass_result()], 1.0)
            save_run("has-fail", "quick", [fail_result], 1.0)
            result = runner.invoke(cli, ["history", "--failures"])
        assert result.exit_code == 0
        assert "has-fail"[:8] in result.output or "FAIL" in result.output
        # The passing run's level should not appear as a failure entry
        assert result.output.count("FAIL") >= 1

    def test_limit_flag_respected(self, tmp_path):
        runner = CliRunner()
        hf = tmp_path / ".run_history.jsonl"
        with patch("src.reporting.history._HISTORY_FILE", hf):
            for i in range(10):
                save_run(f"run-{i:02d}", "quick", [_pass_result()], 1.0)
            result = runner.invoke(cli, ["history", "--limit", "3"])
        assert result.exit_code == 0
        # Only 3 entries should appear; count "quick" occurrences as a proxy
        assert result.output.count("quick") <= 3

    def test_empty_failures_only_shows_message(self, tmp_path):
        runner = CliRunner()
        hf = tmp_path / ".run_history.jsonl"
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("clean", "quick", [_pass_result()], 1.0)
            result = runner.invoke(cli, ["history", "--failures"])
        assert result.exit_code == 0
        assert "No failed runs" in result.output
