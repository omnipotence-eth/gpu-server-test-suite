"""Tests for file-based run history (src/reporting/history.py).

Uses tmp_path to isolate every test from the real history file.
All tests patch _HISTORY_FILE so no writes reach reports/.run_history.jsonl.
"""

import json
from pathlib import Path
from unittest.mock import patch

from src.reporting.history import load_runs, save_run
from src.reporting.models import TestResult, TestStatus

# ─── Helpers ────────────────────────────────────────────────────


def _results(status: TestStatus = TestStatus.PASS, count: int = 2) -> list[TestResult]:
    return [
        TestResult(
            test_name=f"test.item.{i}",
            status=status,
            duration_seconds=0.1,
            message="ok",
        )
        for i in range(count)
    ]


def _history_file(tmp_path: Path) -> Path:
    return tmp_path / ".run_history.jsonl"


# ─── save_run ───────────────────────────────────────────────────


class TestSaveRun:
    def test_creates_file_on_first_save(self, tmp_path):
        hf = _history_file(tmp_path)
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("r1", "quick", _results(), 1.0)
        assert hf.exists()

    def test_appends_one_line_per_run(self, tmp_path):
        hf = _history_file(tmp_path)
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("r1", "quick", _results(), 1.0)
            save_run("r2", "medium", _results(), 2.0)
        lines = [l for l in hf.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_entry_contains_expected_fields(self, tmp_path):
        hf = _history_file(tmp_path)
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("run-xyz", "long", _results(TestStatus.PASS, 3), 7.5)
        entry = json.loads(hf.read_text())
        assert entry["run_id"] == "run-xyz"
        assert entry["run_level"] == "long"
        assert entry["total"] == 3
        assert entry["duration_s"] == 7.5
        assert "timestamp" in entry

    def test_failed_and_warned_counts_are_accurate(self, tmp_path):
        hf = _history_file(tmp_path)
        mixed = (
            _results(TestStatus.PASS, 2)
            + _results(TestStatus.FAIL, 1)
            + _results(TestStatus.WARN, 1)
            + _results(TestStatus.ERROR, 1)
        )
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("r", "q", mixed, 1.0)
        entry = json.loads(hf.read_text())
        assert entry["failed"] == 2   # FAIL + ERROR both count
        assert entry["warned"] == 1
        assert entry["total"] == 5

    def test_silent_on_unwritable_path(self):
        """save_run must not raise when the path cannot be created."""
        bad_path = Path("/nonexistent/deeply/nested/.run_history.jsonl")
        with patch("src.reporting.history._HISTORY_FILE", bad_path):
            save_run("r", "q", _results(), 1.0)  # must not raise


# ─── load_runs ──────────────────────────────────────────────────


class TestLoadRuns:
    def test_returns_empty_list_for_missing_file(self, tmp_path):
        missing = tmp_path / "does_not_exist.jsonl"
        with patch("src.reporting.history._HISTORY_FILE", missing):
            assert load_runs() == []

    def test_returns_entries_newest_first(self, tmp_path):
        hf = _history_file(tmp_path)
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("oldest", "quick", _results(), 1.0)
            save_run("newest", "medium", _results(), 2.0)
            runs = load_runs()
        assert runs[0]["run_id"] == "newest"
        assert runs[1]["run_id"] == "oldest"

    def test_failures_only_excludes_passing_runs(self, tmp_path):
        hf = _history_file(tmp_path)
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("pass-run", "quick", _results(TestStatus.PASS), 1.0)
            save_run("fail-run", "quick", _results(TestStatus.FAIL), 1.0)
            runs = load_runs(failures_only=True)
        assert len(runs) == 1
        assert runs[0]["run_id"] == "fail-run"

    def test_limit_caps_results(self, tmp_path):
        hf = _history_file(tmp_path)
        with patch("src.reporting.history._HISTORY_FILE", hf):
            for i in range(10):
                save_run(f"run-{i}", "quick", _results(), 1.0)
            runs = load_runs(limit=4)
        assert len(runs) == 4

    def test_skips_malformed_json_lines(self, tmp_path):
        hf = _history_file(tmp_path)
        hf.write_text('not-valid-json\n{"run_id": "good-run", "overall_status": "PASS"}\n')
        with patch("src.reporting.history._HISTORY_FILE", hf):
            runs = load_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "good-run"


# ─── overall_status logic ────────────────────────────────────────


class TestOverallStatus:
    def test_all_pass_is_pass(self, tmp_path):
        hf = _history_file(tmp_path)
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("r", "q", _results(TestStatus.PASS, 3), 1.0)
            runs = load_runs()
        assert runs[0]["overall_status"] == "PASS"

    def test_any_fail_overrides_warn(self, tmp_path):
        hf = _history_file(tmp_path)
        mixed = _results(TestStatus.WARN, 2) + _results(TestStatus.FAIL, 1)
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("r", "q", mixed, 1.0)
            runs = load_runs()
        assert runs[0]["overall_status"] == "FAIL"

    def test_warn_without_fail_is_warn(self, tmp_path):
        hf = _history_file(tmp_path)
        mixed = _results(TestStatus.PASS, 2) + _results(TestStatus.WARN, 1)
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("r", "q", mixed, 1.0)
            runs = load_runs()
        assert runs[0]["overall_status"] == "WARN"

    def test_error_status_counts_as_fail(self, tmp_path):
        hf = _history_file(tmp_path)
        with patch("src.reporting.history._HISTORY_FILE", hf):
            save_run("r", "q", _results(TestStatus.ERROR, 1), 1.0)
            runs = load_runs()
        assert runs[0]["overall_status"] == "FAIL"
