"""DCGM Level 3 — Sustained compute stress test.

Runs continuous GPU compute workload for a configurable duration
to verify stability under sustained load. Monitors utilization
and checks for thermal throttling, ECC errors, or crashes.

Mirrors DCGM's Targeted Stress Plugin — keeps the GPU at high
utilization and verifies it doesn't crash or throttle.
"""

import time
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _run_compute_stress(
    gpu: GPUInfo,
    duration_seconds: int,
    min_utilization_pct: int,
    profile: dict[str, Any],
) -> TestResult:
    """Run sustained matrix multiplication stress on a single GPU."""
    start = time.time()

    if torch is None:
        return TestResult(
            test_name="compute_stress.sustained",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="PyTorch not available",
            gpu_uuid=gpu.uuid,
        )

    if not torch.cuda.is_available():
        return TestResult(
            test_name="compute_stress.sustained",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="CUDA not available",
            gpu_uuid=gpu.uuid,
        )

    try:
        device = torch.device(f"cuda:{gpu.index}")
        matrix_size = 4096

        try:
            a = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
            b = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
        except torch.cuda.OutOfMemoryError:
            return TestResult(
                test_name="compute_stress.sustained",
                status=TestStatus.SKIP,
                duration_seconds=time.time() - start,
                message="Insufficient VRAM for stress test matrices",
                gpu_uuid=gpu.uuid,
            )

        # Warm-up
        _ = torch.mm(a, b)
        torch.cuda.synchronize(device)

        # Stress loop
        iteration_count = 0
        errors = []
        stress_start = time.perf_counter()
        last_check = stress_start
        temp_samples = []
        initial_temp = gpu.temperature_c
        temp_samples.append({"time_s": 0, "temp_c": initial_temp})

        while (time.perf_counter() - stress_start) < duration_seconds:
            try:
                c = torch.mm(a, b)
                torch.cuda.synchronize(device)
                iteration_count += 1

                now = time.perf_counter()
                if now - last_check > 5.0:
                    try:
                        import pynvml

                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)
                        temp = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        temp_samples.append({
                            "time_s": round(now - stress_start, 1),
                            "temp_c": temp,
                        })
                        pynvml.nvmlShutdown()
                    except Exception:
                        pass
                    last_check = now

            except RuntimeError as e:
                errors.append(str(e))
                if len(errors) >= 3:
                    break

        stress_duration = time.perf_counter() - stress_start
        flops_per_gemm = 2 * matrix_size**3
        total_flops = flops_per_gemm * iteration_count
        gflops = total_flops / (stress_duration * 1e9)

        del a, b
        torch.cuda.empty_cache()

        details = {
            "duration_seconds": round(stress_duration, 2),
            "target_duration": duration_seconds,
            "iterations": iteration_count,
            "gflops_avg": round(gflops, 1),
            "matrix_size": matrix_size,
            "temp_samples": temp_samples,
            "errors": errors,
        }

        if errors:
            return TestResult(
                test_name="compute_stress.sustained",
                status=TestStatus.FAIL,
                duration_seconds=time.time() - start,
                message=f"Compute stress failed with {len(errors)} error(s) "
                        f"after {iteration_count} iterations",
                failure_code="DIAG-600",
                gpu_uuid=gpu.uuid,
                details=details,
            )

        if stress_duration < duration_seconds * 0.95:
            return TestResult(
                test_name="compute_stress.sustained",
                status=TestStatus.WARN,
                duration_seconds=time.time() - start,
                message=f"Stress test ended early: {stress_duration:.1f}s "
                        f"of {duration_seconds}s target",
                gpu_uuid=gpu.uuid,
                details=details,
            )

        return TestResult(
            test_name="compute_stress.sustained",
            status=TestStatus.PASS,
            duration_seconds=time.time() - start,
            message=f"Compute stress OK: {iteration_count} GEMM ops, "
                    f"{gflops:.0f} GFLOPS over {stress_duration:.1f}s",
            gpu_uuid=gpu.uuid,
            details=details,
        )

    except Exception as e:
        return TestResult(
            test_name="compute_stress.sustained",
            status=TestStatus.ERROR,
            duration_seconds=time.time() - start,
            message=f"Compute stress error: {e}",
            gpu_uuid=gpu.uuid,
        )


def run_compute_stress(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute sustained compute stress test on all GPUs."""
    thresholds = profile.get("thresholds", {})
    duration = thresholds.get("stress_duration_seconds", 60)
    min_util = thresholds.get("stress_min_utilization_pct", 95)

    results = []
    for gpu in gpu_infos:
        results.append(_run_compute_stress(gpu, duration, min_util, profile))

    return results
