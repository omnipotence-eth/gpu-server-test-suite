"""DCGM Level 3 — SM (Streaming Multiprocessor) performance validation.

Measures theoretical peak compute throughput and compares against
expected GFLOPS for the GPU model. Uses large matrix operations
to saturate all SMs and verifies the GPU achieves expected performance.
"""

import time
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _measure_sm_throughput(
    gpu: GPUInfo,
    target_gflops: float,
    tolerance_pct: float,
) -> TestResult:
    """Measure SM throughput using large GEMM operations."""
    start = time.time()

    if torch is None:
        return TestResult(
            test_name="sm_stress.throughput",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="PyTorch not available",
            gpu_uuid=gpu.uuid,
        )

    if not torch.cuda.is_available():
        return TestResult(
            test_name="sm_stress.throughput",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="CUDA not available",
            gpu_uuid=gpu.uuid,
        )

    try:
        device = torch.device(f"cuda:{gpu.index}")
        matrix_size = 8192

        try:
            a = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
            b = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
        except torch.cuda.OutOfMemoryError:
            matrix_size = 4096
            a = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
            b = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)

        flops_per_gemm = 2 * matrix_size**3

        # Warm-up (important for clock ramp-up)
        for _ in range(5):
            _ = torch.mm(a, b)
        torch.cuda.synchronize(device)

        iterations = 20
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = torch.mm(a, b)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        best_time = min(times)
        median_time = sorted(times)[len(times) // 2]

        peak_gflops = flops_per_gemm / (best_time * 1e9)
        median_gflops = flops_per_gemm / (median_time * 1e9)

        del a, b
        torch.cuda.empty_cache()

        min_acceptable = target_gflops * (1 - tolerance_pct / 100)

        details = {
            "peak_gflops": round(peak_gflops, 1),
            "median_gflops": round(median_gflops, 1),
            "target_gflops": target_gflops,
            "min_acceptable_gflops": round(min_acceptable, 1),
            "tolerance_pct": tolerance_pct,
            "matrix_size": matrix_size,
            "iterations": iterations,
            "best_time_ms": round(best_time * 1000, 2),
            "median_time_ms": round(median_time * 1000, 2),
        }

        if peak_gflops < min_acceptable:
            return TestResult(
                test_name="sm_stress.throughput",
                status=TestStatus.FAIL,
                duration_seconds=time.time() - start,
                message=f"SM throughput below target: {peak_gflops:.0f} GFLOPS "
                        f"(min: {min_acceptable:.0f} GFLOPS)",
                failure_code="DIAG-700",
                gpu_uuid=gpu.uuid,
                details=details,
            )

        if median_gflops < peak_gflops * 0.85:
            return TestResult(
                test_name="sm_stress.throughput",
                status=TestStatus.WARN,
                duration_seconds=time.time() - start,
                message=f"SM throughput inconsistent: peak {peak_gflops:.0f}, "
                        f"median {median_gflops:.0f} GFLOPS",
                gpu_uuid=gpu.uuid,
                details=details,
            )

        return TestResult(
            test_name="sm_stress.throughput",
            status=TestStatus.PASS,
            duration_seconds=time.time() - start,
            message=f"SM throughput OK: {peak_gflops:.0f} GFLOPS "
                    f"(target: {target_gflops:.0f} GFLOPS)",
            gpu_uuid=gpu.uuid,
            details=details,
        )

    except Exception as e:
        return TestResult(
            test_name="sm_stress.throughput",
            status=TestStatus.ERROR,
            duration_seconds=time.time() - start,
            message=f"SM throughput test error: {e}",
            gpu_uuid=gpu.uuid,
        )


def _measure_fp16_throughput(
    gpu: GPUInfo,
    target_gflops: float,
) -> TestResult:
    """Measure FP16/TensorCore throughput if available."""
    start = time.time()

    if torch is None:
        return TestResult(
            test_name="sm_stress.fp16_throughput",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="PyTorch not available",
            gpu_uuid=gpu.uuid,
        )

    if not torch.cuda.is_available():
        return TestResult(
            test_name="sm_stress.fp16_throughput",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="CUDA not available",
            gpu_uuid=gpu.uuid,
        )

    try:
        device = torch.device(f"cuda:{gpu.index}")

        major, minor = torch.cuda.get_device_capability(gpu.index)
        if major < 7:
            return TestResult(
                test_name="sm_stress.fp16_throughput",
                status=TestStatus.SKIP,
                duration_seconds=time.time() - start,
                message=f"FP16 tensor cores not available (CC {major}.{minor})",
                gpu_uuid=gpu.uuid,
                details={"compute_capability": f"{major}.{minor}"},
            )

        matrix_size = 8192
        try:
            a = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
            b = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
        except torch.cuda.OutOfMemoryError:
            matrix_size = 4096
            a = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
            b = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)

        flops_per_gemm = 2 * matrix_size**3

        for _ in range(5):
            _ = torch.mm(a, b)
        torch.cuda.synchronize(device)

        iterations = 20
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = torch.mm(a, b)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        best_time = min(times)
        peak_gflops = flops_per_gemm / (best_time * 1e9)

        del a, b
        torch.cuda.empty_cache()

        fp16_target = target_gflops * 2

        details = {
            "peak_gflops": round(peak_gflops, 1),
            "fp16_target_gflops": round(fp16_target, 1),
            "matrix_size": matrix_size,
            "compute_capability": f"{major}.{minor}",
            "best_time_ms": round(best_time * 1000, 2),
        }

        return TestResult(
            test_name="sm_stress.fp16_throughput",
            status=TestStatus.PASS,
            duration_seconds=time.time() - start,
            message=f"FP16 throughput: {peak_gflops:.0f} GFLOPS",
            gpu_uuid=gpu.uuid,
            details=details,
        )

    except Exception as e:
        return TestResult(
            test_name="sm_stress.fp16_throughput",
            status=TestStatus.ERROR,
            duration_seconds=time.time() - start,
            message=f"FP16 throughput test error: {e}",
            gpu_uuid=gpu.uuid,
        )


def run_sm_stress(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute SM performance validation on all GPUs."""
    thresholds = profile.get("thresholds", {})
    target_gflops = thresholds.get("sm_target_gflops", 20000)
    tolerance_pct = thresholds.get("sm_tolerance_pct", 10)

    results = []
    for gpu in gpu_infos:
        results.append(_measure_sm_throughput(gpu, target_gflops, tolerance_pct))
        results.append(_measure_fp16_throughput(gpu, target_gflops))

    return results
