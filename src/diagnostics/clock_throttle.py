"""Clock throttling analysis — GPU performance degradation detection.

Detects and classifies GPU clock throttling reasons:
  - Thermal throttling (temperature limit reached)
  - Power brake (external power brake assertion)
  - HW slowdown (hardware thermal/power protection)
  - SW thermal slowdown (software thermal limit)
  - Sync boost (multi-GPU sync limiting clocks)
  - Application clock setting (user-set clock limit)

Uses NVML throttle reason bitmask for precise classification.
Mirrors DCGM's throttle monitoring for production reliability.
"""

import time
from typing import Any

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus

# NVML clock throttle reason bitmask values
# Defined here so we don't require pynvml at module import time
THROTTLE_REASONS = {
    0x0000000000000001: "GPU_IDLE",
    0x0000000000000002: "APPLICATIONS_CLOCKS_SETTING",
    0x0000000000000004: "SW_POWER_CAP",
    0x0000000000000008: "HW_SLOWDOWN",
    0x0000000000000010: "SYNC_BOOST",
    0x0000000000000020: "SW_THERMAL_SLOWDOWN",
    0x0000000000000040: "HW_THERMAL_SLOWDOWN",
    0x0000000000000080: "HW_POWER_BRAKE_SLOWDOWN",
    0x0000000000000100: "DISPLAY_CLOCK_SETTING",
}

# Reasons that indicate a real problem
PROBLEM_THROTTLE_BITS = {
    0x0000000000000008: "HW_SLOWDOWN",
    0x0000000000000020: "SW_THERMAL_SLOWDOWN",
    0x0000000000000040: "HW_THERMAL_SLOWDOWN",
    0x0000000000000080: "HW_POWER_BRAKE_SLOWDOWN",
}


def _get_throttle_reasons(gpu_index: int) -> dict:
    """Query current clock throttle reasons via pynvml.

    Returns dict with throttle bitmask and parsed reasons.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            # Get current throttle reasons
            try:
                throttle_reasons = (
                    pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(
                        handle
                    )
                )
            except (pynvml.NVMLError, AttributeError):
                throttle_reasons = 0

            # Get supported throttle reasons
            try:
                supported = (
                    pynvml.nvmlDeviceGetSupportedClocksThrottleReasons(
                        handle
                    )
                )
            except (pynvml.NVMLError, AttributeError):
                supported = 0xFFFFFFFF

            # Get current clocks for context
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(
                    handle, 0  # NVML_CLOCK_GRAPHICS
                )
            except pynvml.NVMLError:
                graphics_clock = 0

            try:
                max_graphics = pynvml.nvmlDeviceGetMaxClockInfo(
                    handle, 0
                )
            except pynvml.NVMLError:
                max_graphics = 0

            # Parse active throttle reasons
            active_reasons = []
            for bit, name in THROTTLE_REASONS.items():
                if throttle_reasons & bit:
                    active_reasons.append({
                        "bit": hex(bit),
                        "reason": name,
                        "is_problem": bit in PROBLEM_THROTTLE_BITS,
                    })

            return {
                "throttle_bitmask": hex(throttle_reasons),
                "supported_bitmask": hex(supported),
                "active_reasons": active_reasons,
                "graphics_clock_mhz": graphics_clock,
                "max_graphics_clock_mhz": max_graphics,
                "clock_reduction_pct": (
                    round(
                        (1 - graphics_clock / max_graphics) * 100, 1
                    )
                    if max_graphics > 0
                    else 0
                ),
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        return {"error": str(e), "active_reasons": []}


def _check_clock_throttling(
    gpu: GPUInfo,
    profile: dict[str, Any],
) -> TestResult:
    """Analyze clock throttling state for a single GPU."""
    start = time.time()

    throttle_data = _get_throttle_reasons(gpu.index)

    if "error" in throttle_data:
        return TestResult(
            test_name="telemetry.clock_throttle",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message=(
                f"Could not query throttle reasons: "
                f"{throttle_data['error']}"
            ),
            gpu_uuid=gpu.uuid,
            details=throttle_data,
        )

    active = throttle_data.get("active_reasons", [])
    problem_reasons = [r for r in active if r.get("is_problem")]

    details = {
        "gpu_index": gpu.index,
        **throttle_data,
        "problem_reasons": problem_reasons,
    }

    if problem_reasons:
        reason_names = [r["reason"] for r in problem_reasons]
        reduction = throttle_data.get("clock_reduction_pct", 0)
        return TestResult(
            test_name="telemetry.clock_throttle",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message=(
                f"GPU throttled: {', '.join(reason_names)} "
                f"(clock reduced {reduction}%)"
            ),
            failure_code="DIAG-910",
            gpu_uuid=gpu.uuid,
            details=details,
        )

    # Check if clocks are running well below max (non-idle)
    non_idle_active = [
        r for r in active if r["reason"] != "GPU_IDLE"
    ]
    if non_idle_active:
        reason_names = [r["reason"] for r in non_idle_active]
        return TestResult(
            test_name="telemetry.clock_throttle",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=(
                f"Clock limiting active: "
                f"{', '.join(reason_names)}"
            ),
            gpu_uuid=gpu.uuid,
            details=details,
        )

    return TestResult(
        test_name="telemetry.clock_throttle",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message="No problematic clock throttling detected",
        gpu_uuid=gpu.uuid,
        details=details,
    )


def run_clock_throttle_checks(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute clock throttle analysis on all GPUs."""
    return [
        _check_clock_throttling(gpu, profile) for gpu in gpu_infos
    ]
