"""DCGM Level 3 — GPU memory (framebuffer) bandwidth measurement.

Measures on-device memory bandwidth by performing large-scale
tensor operations that are memory-bound (copy, add, scale).
This tests the HBM/GDDR bandwidth path — not the PCIe bus.

RTX 5070 Ti (GDDR7): ~504 GB/s theoretical
H100 SXM (HBM3):     ~3.35 TB/s theoretical
A100 80GB (HBM2e):    ~2.0 TB/s theoretical
"""

import time
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _measure_memory_bandwidth(
    gpu: GPUInfo,
    min_gibs: float,
    test_size_mib: int = 512,
    iterations: int = 20,
) -> TestResult:
    """Measure GPU memory bandwidth using STREAM-like copy kernel.

    Performs element-wise copy: B = A (read A, write B).
    Bandwidth = 2 * data_size / time (read + write).
    """
    start = time.time()

    if torch is None:
        return TestResult(
            test_name="memory_bandwidth.stream_copy",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="PyTorch not available",
            gpu_uuid=gpu.uuid,
        )

    if not torch.cuda.is_available():
        return TestResult(
            test_name="memory_bandwidth.stream_copy",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="CUDA not available",
            gpu_uuid=gpu.uuid,
        )

    try:
        device = torch.device(f"cuda:{gpu.index}")
        num_elements = (test_size_mib * 1024 * 1024) // 4
        data_bytes = num_elements * 4

        try:
            a = torch.randn(num_elements, dtype=torch.float32, device=device)
            b = torch.empty(num_elements, dtype=torch.float32, device=device)
        except torch.cuda.OutOfMemoryError:
            return TestResult(
                test_name="memory_bandwidth.stream_copy",
                status=TestStatus.SKIP,
                duration_seconds=time.time() - start,
                message=f"Insufficient VRAM for {test_size_mib * 2} MiB test buffers",
                gpu_uuid=gpu.uuid,
            )

        # Warm-up
        b.copy_(a)
        torch.cuda.synchronize(device)

        # Timed iterations
        total_time = 0.0
        for _ in range(iterations):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            b.copy_(a)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            total_time += t1 - t0

        avg_time = total_time / iterations
        bandwidth_gibs = (2 * data_bytes / (1024**3)) / avg_time

        del a, b
        torch.cuda.empty_cache()

        details = {
            "bandwidth_gibs": round(bandwidth_gibs, 2),
            "min_gibs": min_gibs,
            "test_size_mib": test_size_mib,
            "iterations": iterations,
            "avg_time_ms": round(avg_time * 1000, 3),
            "operation": "stream_copy",
        }

        if bandwidth_gibs < min_gibs:
            return TestResult(
                test_name="memory_bandwidth.stream_copy",
                status=TestStatus.FAIL,
                duration_seconds=time.time() - start,
                message=f"Memory bandwidth low: {bandwidth_gibs:.1f} GiB/s (min: {min_gibs} GiB/s)",
                failure_code="DIAG-500",
                gpu_uuid=gpu.uuid,
                details=details,
            )
        return TestResult(
            test_name="memory_bandwidth.stream_copy",
            status=TestStatus.PASS,
            duration_seconds=time.time() - start,
            message=f"Memory bandwidth OK: {bandwidth_gibs:.1f} GiB/s (min: {min_gibs} GiB/s)",
            gpu_uuid=gpu.uuid,
            details=details,
        )

    except Exception as e:
        return TestResult(
            test_name="memory_bandwidth.stream_copy",
            status=TestStatus.ERROR,
            duration_seconds=time.time() - start,
            message=f"Memory bandwidth test error: {e}",
            gpu_uuid=gpu.uuid,
        )


def _measure_memory_bandwidth_triad(
    gpu: GPUInfo,
    min_gibs: float,
    test_size_mib: int = 512,
    iterations: int = 20,
) -> TestResult:
    """Measure GPU memory bandwidth using STREAM triad: A = B + scalar * C.

    Bandwidth = 3 * data_size / time (2 reads + 1 write).
    """
    start = time.time()

    if torch is None:
        return TestResult(
            test_name="memory_bandwidth.stream_triad",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="PyTorch not available",
            gpu_uuid=gpu.uuid,
        )

    if not torch.cuda.is_available():
        return TestResult(
            test_name="memory_bandwidth.stream_triad",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="CUDA not available",
            gpu_uuid=gpu.uuid,
        )

    try:
        device = torch.device(f"cuda:{gpu.index}")
        num_elements = (test_size_mib * 1024 * 1024) // 4
        data_bytes = num_elements * 4
        scalar = 2.0

        try:
            a = torch.empty(num_elements, dtype=torch.float32, device=device)
            b = torch.randn(num_elements, dtype=torch.float32, device=device)
            c = torch.randn(num_elements, dtype=torch.float32, device=device)
        except torch.cuda.OutOfMemoryError:
            return TestResult(
                test_name="memory_bandwidth.stream_triad",
                status=TestStatus.SKIP,
                duration_seconds=time.time() - start,
                message=f"Insufficient VRAM for {test_size_mib * 3} MiB test buffers",
                gpu_uuid=gpu.uuid,
            )

        # Warm-up
        torch.add(b, c, alpha=scalar, out=a)
        torch.cuda.synchronize(device)

        total_time = 0.0
        for _ in range(iterations):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            torch.add(b, c, alpha=scalar, out=a)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            total_time += t1 - t0

        avg_time = total_time / iterations
        bandwidth_gibs = (3 * data_bytes / (1024**3)) / avg_time

        del a, b, c
        torch.cuda.empty_cache()

        details = {
            "bandwidth_gibs": round(bandwidth_gibs, 2),
            "min_gibs": min_gibs,
            "test_size_mib": test_size_mib,
            "iterations": iterations,
            "avg_time_ms": round(avg_time * 1000, 3),
            "operation": "stream_triad",
        }

        if bandwidth_gibs < min_gibs:
            return TestResult(
                test_name="memory_bandwidth.stream_triad",
                status=TestStatus.FAIL,
                duration_seconds=time.time() - start,
                message=f"Memory triad bandwidth low: {bandwidth_gibs:.1f} GiB/s "
                f"(min: {min_gibs} GiB/s)",
                failure_code="DIAG-501",
                gpu_uuid=gpu.uuid,
                details=details,
            )
        return TestResult(
            test_name="memory_bandwidth.stream_triad",
            status=TestStatus.PASS,
            duration_seconds=time.time() - start,
            message=f"Memory triad bandwidth OK: {bandwidth_gibs:.1f} GiB/s "
            f"(min: {min_gibs} GiB/s)",
            gpu_uuid=gpu.uuid,
            details=details,
        )

    except Exception as e:
        return TestResult(
            test_name="memory_bandwidth.stream_triad",
            status=TestStatus.ERROR,
            duration_seconds=time.time() - start,
            message=f"Memory triad test error: {e}",
            gpu_uuid=gpu.uuid,
        )


def run_memory_bandwidth(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute GPU memory bandwidth tests on all GPUs."""
    thresholds = profile.get("thresholds", {})
    min_gibs = thresholds.get("memory_bandwidth_min_gibs", 400.0)

    results = []
    for gpu in gpu_infos:
        results.append(_measure_memory_bandwidth(gpu, min_gibs))
        results.append(_measure_memory_bandwidth_triad(gpu, min_gibs))

    return results
