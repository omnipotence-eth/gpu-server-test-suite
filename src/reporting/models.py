"""Data models for diagnostic test results.

Defines the core data structures used across the entire test suite:
TestStatus enum, TestResult per-test output, and DiagnosticRun for
complete run aggregation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TestStatus(Enum):
    """Test outcome status codes — mirrors DCGM diagnostic result states."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class TestResult:
    """Result of a single diagnostic test execution.

    Attributes:
        test_name: Identifier matching the diagnostic module name.
        status: PASS, FAIL, WARN, SKIP, or ERROR.
        duration_seconds: Wall-clock execution time.
        message: Human-readable summary of result.
        details: Arbitrary key-value data (metrics, thresholds, measured values).
        timestamp: When the test completed.
        gpu_uuid: UUID of the GPU tested (empty for system-level tests).
        failure_code: Diagnostic failure code (e.g., "DIAG-003") if status is FAIL.
    """

    test_name: str
    status: TestStatus
    duration_seconds: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    gpu_uuid: str = ""
    failure_code: str = ""


@dataclass
class DiagnosticRun:
    """Aggregated results of a complete diagnostic run.

    Attributes:
        run_id: Unique UUID for this run.
        run_level: One of quick, medium, long, extended.
        hostname: Machine hostname.
        timestamp: When the run started.
        gpu_count: Number of GPUs detected.
        overall_status: Aggregate status — FAIL if any test failed.
        duration_seconds: Total wall-clock time for the entire run.
        results: List of individual TestResult objects.
        system_info: System inventory snapshot as dict.
        gpu_info: Per-GPU inventory snapshots as list of dicts.
    """

    run_id: str
    run_level: str
    hostname: str
    timestamp: datetime
    gpu_count: int
    overall_status: TestStatus
    duration_seconds: float
    results: list[TestResult] = field(default_factory=list)
    system_info: dict[str, Any] = field(default_factory=dict)
    gpu_info: list[dict[str, Any]] = field(default_factory=list)
