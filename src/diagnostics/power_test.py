"""DCGM Level 3 — Targeted power draw test.

Drives the GPU to target power consumption level and verifies
it can sustain the load without thermal throttling or power
capping. Monitors power draw, temperature, and clocks during
the test.

Mirrors DCGM's Targeted Power Plugin.
"""

import time
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _run_power_stress(
    gpu: GPUInfo,
    target_pct: float,
    duration_seconds: int,
    tolerance_pct: float,
    profile: dict[str, Any],
) -> TestResult:
    """Run power-intensive workload and verify power draw reaches target."""
    start = time.time()

    if torch is None:
        return TestResult(
            test_name="power_test.sustained_power",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="PyTorch not available",
            gpu_uuid=gpu.uuid,
        )

    if not torch.cuda.is_available():
        return TestResult(
            test_name="power_test.sustained_power",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="CUDA not available",
            gpu_uuid=gpu.uuid,
        )

    try:
        device = torch.device(f"cuda:{gpu.index}")
        tdp = profile.get("tdp_watts", gpu.power_limit_w)
        target_watts = tdp * (target_pct / 100)
        min_watts = target_watts * (1 - tolerance_pct / 100)

        matrix_size = 4096
        try:
            matrices = []
            for _ in range(4):
                matrices.append(
                    torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
                )
        except torch.cuda.OutOfMemoryError:
            matrix_size = 2048
            matrices = []
            for _ in range(4):
                matrices.append(
                    torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
                )

        # Warm-up
        for _ in range(10):
            _ = torch.mm(matrices[0], matrices[1])
        torch.cuda.synchronize(device)

        power_samples = []
        temp_samples = []
        clock_samples = []
        iteration_count = 0

        stress_start = time.perf_counter()
        last_sample = stress_start

        while (time.perf_counter() - stress_start) < duration_seconds:
            _ = torch.mm(matrices[0], matrices[1])
            _ = torch.mm(matrices[2], matrices[3])
            torch.cuda.synchronize(device)
            iteration_count += 1

            now = time.perf_counter()
            if now - last_sample > 2.0:
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)

                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_w = power_mw / 1000.0
                    power_samples.append(
                        {
                            "time_s": round(now - stress_start, 1),
                            "power_w": round(power_w, 1),
                        }
                    )

                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temp_samples.append(
                        {
                            "time_s": round(now - stress_start, 1),
                            "temp_c": temp,
                        }
                    )

                    try:
                        clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                        clock_samples.append(
                            {
                                "time_s": round(now - stress_start, 1),
                                "clock_mhz": clock,
                            }
                        )
                    except Exception:
                        pass

                    pynvml.nvmlShutdown()
                except Exception:
                    pass
                last_sample = now

        stress_duration = time.perf_counter() - stress_start

        del matrices
        torch.cuda.empty_cache()

        if power_samples:
            avg_power = sum(s["power_w"] for s in power_samples) / len(power_samples)
            max_power = max(s["power_w"] for s in power_samples)
            min_power = min(s["power_w"] for s in power_samples)
        else:
            avg_power = 0
            max_power = 0
            min_power = 0

        max_temp = max(s["temp_c"] for s in temp_samples) if temp_samples else 0

        details = {
            "target_watts": round(target_watts, 1),
            "min_watts_threshold": round(min_watts, 1),
            "avg_power_w": round(avg_power, 1),
            "max_power_w": round(max_power, 1),
            "min_power_w": round(min_power, 1),
            "max_temp_c": max_temp,
            "duration_seconds": round(stress_duration, 2),
            "iterations": iteration_count,
            "power_samples": power_samples[-10:],
            "temp_samples": temp_samples[-10:],
            "clock_samples": clock_samples[-5:],
            "tdp_watts": tdp,
        }

        if not power_samples:
            return TestResult(
                test_name="power_test.sustained_power",
                status=TestStatus.WARN,
                duration_seconds=time.time() - start,
                message="Power monitoring not available — stress completed "
                f"({iteration_count} iterations, {stress_duration:.1f}s)",
                gpu_uuid=gpu.uuid,
                details=details,
            )

        thresholds = profile.get("thresholds", {})
        temp_critical = thresholds.get("temp_critical_c", 90)
        if max_temp >= temp_critical:
            return TestResult(
                test_name="power_test.sustained_power",
                status=TestStatus.FAIL,
                duration_seconds=time.time() - start,
                message=f"Thermal throttling detected: {max_temp}C (critical: {temp_critical}C)",
                failure_code="DIAG-801",
                gpu_uuid=gpu.uuid,
                details=details,
            )

        if avg_power < min_watts and avg_power > 0:
            return TestResult(
                test_name="power_test.sustained_power",
                status=TestStatus.WARN,
                duration_seconds=time.time() - start,
                message=f"Power below target: {avg_power:.0f}W avg "
                f"(target: {target_watts:.0f}W, min: {min_watts:.0f}W)",
                gpu_uuid=gpu.uuid,
                details=details,
            )

        return TestResult(
            test_name="power_test.sustained_power",
            status=TestStatus.PASS,
            duration_seconds=time.time() - start,
            message=f"Power test OK: {avg_power:.0f}W avg, "
            f"max temp {max_temp}C over {stress_duration:.0f}s",
            gpu_uuid=gpu.uuid,
            details=details,
        )

    except Exception as e:
        return TestResult(
            test_name="power_test.sustained_power",
            status=TestStatus.ERROR,
            duration_seconds=time.time() - start,
            message=f"Power test error: {e}",
            gpu_uuid=gpu.uuid,
        )


def run_power_test(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute targeted power draw test on all GPUs."""
    thresholds = profile.get("thresholds", {})
    target_pct = thresholds.get("power_target_pct", 90)
    duration = thresholds.get("power_duration_seconds", 30)
    tolerance = thresholds.get("power_tolerance_pct", 5)

    results = []
    for gpu in gpu_infos:
        results.append(_run_power_stress(gpu, target_pct, duration, tolerance, profile))

    return results
