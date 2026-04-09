"""Fault injection for validating diagnostic failure handling.

Provides synthetic TestResult objects that simulate known GPU failure modes.
Use with the --inject-fault CLI flag to verify that the diagnostic pipeline
correctly detects, codes, and escalates each failure type.

Injected failure codes are intentionally offset from real codes so they
are easy to identify in CI logs and alert rules.
"""

from __future__ import annotations

from src.reporting.models import TestResult, TestStatus

_FAULT_TEMPLATES: dict[str, TestResult] = {
    "thermal": TestResult(
        test_name="fault_injection.thermal",
        status=TestStatus.FAIL,
        duration_seconds=0.01,
        message="[INJECTED] GPU temperature exceeded threshold: 95°C (limit 85°C)",
        failure_code="DIAG-FI-300",
        details={"injected": True, "simulated_temp_c": 95, "threshold_c": 85},
    ),
    "ecc": TestResult(
        test_name="fault_injection.ecc",
        status=TestStatus.FAIL,
        duration_seconds=0.01,
        message="[INJECTED] Double-bit ECC error detected: DBE count=1",
        failure_code="DIAG-FI-401",
        details={"injected": True, "simulated_dbe": 1},
    ),
    "pcie": TestResult(
        test_name="fault_injection.pcie",
        status=TestStatus.FAIL,
        duration_seconds=0.01,
        message="[INJECTED] PCIe link degraded: Gen4 x8 (expected x16)",
        failure_code="DIAG-FI-202",
        details={"injected": True, "simulated_link_width": 8, "expected_link_width": 16},
    ),
    "clock": TestResult(
        test_name="fault_injection.clock",
        status=TestStatus.FAIL,
        duration_seconds=0.01,
        message="[INJECTED] Clock throttle active: SW_THERMAL_SLOWDOWN",
        failure_code="DIAG-FI-501",
        details={"injected": True, "simulated_throttle_reason": "SW_THERMAL_SLOWDOWN"},
    ),
    "memory": TestResult(
        test_name="fault_injection.memory",
        status=TestStatus.FAIL,
        duration_seconds=0.01,
        message="[INJECTED] VRAM stress failure: memory error after 512 iterations",
        failure_code="DIAG-FI-102",
        details={"injected": True, "simulated_error_iteration": 512},
    ),
}

SUPPORTED_FAULTS: list[str] = list(_FAULT_TEMPLATES)


def inject_fault(fault_type: str) -> TestResult:
    """Return a synthetic FAIL TestResult for the specified fault type.

    Args:
        fault_type: One of 'thermal', 'ecc', 'pcie', 'clock', 'memory'.

    Returns:
        A TestResult with status FAIL and injected=True in details.
        Failure codes use the DIAG-FI-* prefix to distinguish injected
        faults from real diagnostic failures in logs and alert rules.

    Raises:
        ValueError: If fault_type is not a supported fault type.
    """
    if fault_type not in _FAULT_TEMPLATES:
        raise ValueError(f"Unknown fault type: {fault_type!r}. Supported: {SUPPORTED_FAULTS}")
    return _FAULT_TEMPLATES[fault_type]
