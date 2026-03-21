"""Pre-flight GPU health check — thermal and power baseline validation.

Runs before any stress tests to establish baseline readings and ensure
the GPU is in a healthy state to begin diagnostics. Equivalent to
DCGM's software/hardware pre-check before Level 2+.

Checks performed:
  - GPU temperature within safe operating range
  - Power draw within expected idle/baseline limits
  - VRAM available for testing
  - GPU clock sanity (not stuck at 0)
"""

import time
from typing import Any

import pynvml

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _check_temperature(gpu_infos: list[GPUInfo], profile: dict[str, Any]) -> TestResult:
    """Verify GPU temperatures are within safe operating range before testing."""
    start = time.time()
    thresholds = profile.get("thresholds", {})
    temp_warning = thresholds.get("temp_warning_c", 80)
    temp_critical = thresholds.get("temp_critical_c", 90)

    gpu_temps = []
    max_temp = 0
    any_critical = False
    any_warning = False

    for gpu in gpu_infos:
        temp = gpu.temperature_c
        max_temp = max(max_temp, temp)
        status = "OK"
        if temp >= temp_critical:
            any_critical = True
            status = "CRITICAL"
        elif temp >= temp_warning:
            any_warning = True
            status = "WARNING"
        gpu_temps.append({"index": gpu.index, "temp_c": temp, "status": status})

    if any_critical:
        return TestResult(
            test_name="health.temperature",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message=f"GPU temperature critical: {max_temp}C (limit: {temp_critical}C)",
            failure_code="DIAG-100",
            details={
                "gpu_temps": gpu_temps,
                "threshold_warning": temp_warning,
                "threshold_critical": temp_critical,
            },
        )
    if any_warning:
        return TestResult(
            test_name="health.temperature",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=f"GPU temperature elevated: {max_temp}C (warning: {temp_warning}C)",
            details={
                "gpu_temps": gpu_temps,
                "threshold_warning": temp_warning,
                "threshold_critical": temp_critical,
            },
        )
    return TestResult(
        test_name="health.temperature",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message=f"GPU temperature OK: {max_temp}C (limit: {temp_warning}C)",
        details={"gpu_temps": gpu_temps},
    )


def _check_power_baseline(gpu_infos: list[GPUInfo], profile: dict[str, Any]) -> TestResult:
    """Verify power draw is within baseline idle range."""
    start = time.time()
    tdp = profile.get("tdp_watts", 300)
    # Idle should be well under 50% of TDP
    idle_max = tdp * 0.5

    gpu_power = []
    max_power = 0.0
    any_high = False

    for gpu in gpu_infos:
        power = gpu.power_draw_w
        max_power = max(max_power, power)
        status = "OK"
        if power > idle_max:
            any_high = True
            status = "HIGH"
        gpu_power.append({
            "index": gpu.index,
            "power_w": power,
            "limit_w": gpu.power_limit_w,
            "status": status,
        })

    if any_high:
        return TestResult(
            test_name="health.power_baseline",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=f"Idle power unexpectedly high: {max_power:.1f}W (TDP: {tdp}W)",
            details={"gpu_power": gpu_power, "tdp_watts": tdp, "idle_max_watts": idle_max},
        )
    return TestResult(
        test_name="health.power_baseline",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message=f"Idle power OK: {max_power:.1f}W (TDP: {tdp}W)",
        details={"gpu_power": gpu_power, "tdp_watts": tdp},
    )


def _check_vram_available(gpu_infos: list[GPUInfo], profile: dict[str, Any]) -> TestResult:
    """Verify sufficient VRAM is available for diagnostic tests."""
    start = time.time()
    thresholds = profile.get("thresholds", {})
    alloc_pct = thresholds.get("vram_test_allocation_pct", 90)
    vram_total = profile.get("vram_total_mib", 0)

    gpu_vram = []
    any_insufficient = False

    for gpu in gpu_infos:
        free_pct = (gpu.vram_free_mib / gpu.vram_total_mib * 100) if gpu.vram_total_mib > 0 else 0
        needed_mib = int(gpu.vram_total_mib * alloc_pct / 100)
        sufficient = gpu.vram_free_mib >= needed_mib
        if not sufficient:
            any_insufficient = True
        gpu_vram.append({
            "index": gpu.index,
            "total_mib": gpu.vram_total_mib,
            "free_mib": gpu.vram_free_mib,
            "used_mib": gpu.vram_used_mib,
            "free_pct": round(free_pct, 1),
            "needed_mib": needed_mib,
            "sufficient": sufficient,
        })

    if any_insufficient:
        return TestResult(
            test_name="health.vram_available",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=f"Insufficient VRAM for {alloc_pct}% allocation test",
            details={"gpu_vram": gpu_vram, "allocation_pct": alloc_pct},
        )
    return TestResult(
        test_name="health.vram_available",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message=f"VRAM available for {alloc_pct}% allocation test",
        details={"gpu_vram": gpu_vram, "allocation_pct": alloc_pct},
    )


def _check_clocks_responsive(gpu_infos: list[GPUInfo]) -> TestResult:
    """Verify GPU clocks are reporting non-zero values (not stuck)."""
    start = time.time()
    gpu_clocks = []
    any_stuck = False

    for gpu in gpu_infos:
        stuck = gpu.clock_graphics_max_mhz == 0 and gpu.clock_memory_max_mhz == 0
        if stuck:
            any_stuck = True
        gpu_clocks.append({
            "index": gpu.index,
            "graphics_mhz": gpu.clock_graphics_mhz,
            "graphics_max_mhz": gpu.clock_graphics_max_mhz,
            "memory_mhz": gpu.clock_memory_mhz,
            "memory_max_mhz": gpu.clock_memory_max_mhz,
            "responsive": not stuck,
        })

    if any_stuck:
        return TestResult(
            test_name="health.clocks_responsive",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message="GPU clock reporting stuck at 0 — possible hardware issue",
            failure_code="DIAG-101",
            details={"gpu_clocks": gpu_clocks},
        )
    return TestResult(
        test_name="health.clocks_responsive",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message="GPU clocks reporting normally",
        details={"gpu_clocks": gpu_clocks},
    )


def run_gpu_health_checks(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute pre-flight GPU health checks.

    Args:
        gpu_infos: List of detected GPUInfo objects.
        profile: GPU-specific profile dict with thresholds.

    Returns:
        List of TestResult objects for each health check.
    """
    return [
        _check_temperature(gpu_infos, profile),
        _check_power_baseline(gpu_infos, profile),
        _check_vram_available(gpu_infos, profile),
        _check_clocks_responsive(gpu_infos),
    ]
