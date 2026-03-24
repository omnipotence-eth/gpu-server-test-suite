"""File-based diagnostic run history.

Saves and loads run summaries as newline-delimited JSON (JSONL) at
<project_root>/reports/.run_history.jsonl. One entry per line.

Kept intentionally simple: the full result data lives in JSON/JUnit
output files; history stores only the metadata needed for the CLI table.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.reporting.models import TestResult, TestStatus

logger = logging.getLogger(__name__)

_HISTORY_FILE = (
    Path(__file__).resolve().parent.parent.parent / "reports" / ".run_history.jsonl"
)
_MAX_ENTRIES = 500


def _overall_status(results: list[TestResult]) -> str:
    if any(r.status in (TestStatus.FAIL, TestStatus.ERROR) for r in results):
        return "FAIL"
    if any(r.status == TestStatus.WARN for r in results):
        return "WARN"
    return "PASS"


def save_run(
    run_id: str,
    run_level: str,
    results: list[TestResult],
    duration_seconds: float,
) -> None:
    """Append a run summary line to the history file.

    Silently logs errors rather than raising — history write failure
    should never abort a diagnostic run.
    """
    entry = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_level": run_level,
        "overall_status": _overall_status(results),
        "total": len(results),
        "failed": sum(
            1 for r in results if r.status in (TestStatus.FAIL, TestStatus.ERROR)
        ),
        "warned": sum(1 for r in results if r.status == TestStatus.WARN),
        "skipped": sum(1 for r in results if r.status == TestStatus.SKIP),
        "duration_s": round(duration_seconds, 2),
    }
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        logger.error("Failed to write run history to %s", _HISTORY_FILE, exc_info=True)


def load_runs(failures_only: bool = False, limit: int = 50) -> list[dict]:
    """Load recent run summaries from the history file.

    Returns entries in reverse-chronological order (newest first).
    """
    if not _HISTORY_FILE.exists():
        return []
    try:
        with _HISTORY_FILE.open(encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        logger.error("Failed to read run history from %s", _HISTORY_FILE, exc_info=True)
        return []

    runs: list[dict] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if failures_only and entry.get("overall_status") != "FAIL":
            continue
        runs.append(entry)
        if len(runs) >= limit:
            break

    return runs
