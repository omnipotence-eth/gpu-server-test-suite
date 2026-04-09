"""DCGM Level 2 — PCIe link validation diagnostic.

Validates PCIe link configuration against profile expectations,
checks for degraded links (gen/width mismatch), and monitors
replay counters for signal integrity issues.

This is the diagnostic test version — uses the inventory module's
PCIe topology detection and applies profile-based validation rules.
"""

import time
from typing import Any

from src.inventory.gpu_inventory import GPUInfo
from src.inventory.pcie_topology import PCIeInfo, get_pcie_topology
from src.reporting.models import TestResult, TestStatus


def _check_pcie_gen(
    pcie_infos: list[PCIeInfo],
    expected_gen: int,
) -> TestResult:
    """Verify PCIe link generation capability matches expected value.

    Validates against link_gen_max (hardware capability) rather than
    link_gen_current, because GPUs dynamically downshift PCIe link
    speed at idle for power saving (e.g., Gen5 -> Gen2). The max
    capability reflects the actual hardware slot negotiation.
    """
    start = time.time()
    mismatches = []
    for pcie in pcie_infos:
        if pcie.link_gen_max < expected_gen:
            mismatches.append(
                {
                    "gpu_index": pcie.gpu_index,
                    "current_gen": pcie.link_gen_current,
                    "max_gen": pcie.link_gen_max,
                    "expected_gen": expected_gen,
                }
            )

    details = {
        "expected_gen": expected_gen,
        "gpu_results": [
            {
                "gpu_index": p.gpu_index,
                "gen_current": p.link_gen_current,
                "gen_max": p.link_gen_max,
            }
            for p in pcie_infos
        ],
    }

    if mismatches:
        return TestResult(
            test_name="pcie_validation.link_gen",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message=f"PCIe gen capability degraded on {len(mismatches)} "
            f"GPU(s): expected Gen{expected_gen}",
            failure_code="DIAG-200",
            details={**details, "mismatches": mismatches},
        )
    return TestResult(
        test_name="pcie_validation.link_gen",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message=f"PCIe link generation OK: Gen{expected_gen} capable",
        details=details,
    )


def _check_pcie_width(
    pcie_infos: list[PCIeInfo],
    expected_width: int,
) -> TestResult:
    """Verify PCIe link width matches expected value from profile."""
    start = time.time()
    mismatches = []
    for pcie in pcie_infos:
        if pcie.link_width_current < expected_width:
            mismatches.append(
                {
                    "gpu_index": pcie.gpu_index,
                    "current_width": pcie.link_width_current,
                    "max_width": pcie.link_width_max,
                    "expected_width": expected_width,
                }
            )

    details = {
        "expected_width": expected_width,
        "gpu_results": [
            {
                "gpu_index": p.gpu_index,
                "width_current": p.link_width_current,
                "width_max": p.link_width_max,
            }
            for p in pcie_infos
        ],
    }

    if mismatches:
        return TestResult(
            test_name="pcie_validation.link_width",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message=f"PCIe width degraded on {len(mismatches)} GPU(s): expected x{expected_width}",
            failure_code="DIAG-201",
            details={**details, "mismatches": mismatches},
        )
    return TestResult(
        test_name="pcie_validation.link_width",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message=f"PCIe link width OK: x{expected_width}",
        details=details,
    )


def _check_pcie_replay(
    pcie_infos: list[PCIeInfo],
    max_replays: int,
) -> TestResult:
    """Check PCIe replay counters for signal integrity issues."""
    start = time.time()
    issues = []
    for pcie in pcie_infos:
        if pcie.replay_counter > max_replays:
            issues.append(
                {
                    "gpu_index": pcie.gpu_index,
                    "replay_counter": pcie.replay_counter,
                    "threshold": max_replays,
                }
            )

    details = {
        "threshold": max_replays,
        "gpu_results": [
            {"gpu_index": p.gpu_index, "replay_counter": p.replay_counter} for p in pcie_infos
        ],
    }

    if issues:
        return TestResult(
            test_name="pcie_validation.replay_counter",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=f"PCIe replay counter elevated on {len(issues)} GPU(s)",
            details={**details, "issues": issues},
        )
    return TestResult(
        test_name="pcie_validation.replay_counter",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message="PCIe replay counters OK",
        details=details,
    )


def _check_pcie_degradation_summary(pcie_infos: list[PCIeInfo]) -> TestResult:
    """Overall PCIe degradation summary across all GPUs."""
    start = time.time()
    degraded = [p for p in pcie_infos if p.is_degraded]

    details = {
        "total_gpus": len(pcie_infos),
        "degraded_count": len(degraded),
        "degraded_gpus": [
            {
                "gpu_index": p.gpu_index,
                "reason": p.degradation_reason,
            }
            for p in degraded
        ],
    }

    if degraded:
        return TestResult(
            test_name="pcie_validation.degradation_summary",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message=f"{len(degraded)}/{len(pcie_infos)} GPU(s) have degraded PCIe links",
            failure_code="DIAG-202",
            details=details,
        )
    return TestResult(
        test_name="pcie_validation.degradation_summary",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message=f"All {len(pcie_infos)} GPU(s) PCIe links operating at full speed",
        details=details,
    )


def run_pcie_validation(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute PCIe link validation diagnostic.

    Args:
        gpu_infos: List of detected GPUInfo objects.
        profile: GPU-specific profile dict with PCIe expectations.

    Returns:
        List of TestResult objects for each PCIe check.
    """
    expected_gen = profile.get("pcie_gen_expected", 4)
    expected_width = profile.get("pcie_width_expected", 16)
    thresholds = profile.get("thresholds", {})
    max_replays = thresholds.get("pcie_retransmit_max", 0)

    pcie_infos = get_pcie_topology(gpu_infos)

    return [
        _check_pcie_gen(pcie_infos, expected_gen),
        _check_pcie_width(pcie_infos, expected_width),
        _check_pcie_replay(pcie_infos, max_replays),
        _check_pcie_degradation_summary(pcie_infos),
    ]
