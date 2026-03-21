"""DCGM Level 3 — PCIe bandwidth measurement.

Measures Host-to-Device (H2D) and Device-to-Host (D2H) PCIe transfer
rates using large tensor copies. Compares against profile thresholds
to detect bandwidth degradation.

PCIe Gen4 x16 theoretical max: ~25 GiB/s (31.5 GB/s)
Practical achievable: ~22-24 GiB/s with DMA overhead
"""

import time
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _measure_h2d_bandwidth(
    gpu: GPUInfo,
    min_gibs: float,
    transfer_size_mib: int = 256,
    iterations: int = 10,
) -> TestResult:
    """Measure Host-to-Device PCIe transfer bandwidth."""
    start = time.time()

    if torch is None:
        return TestResult(
            test_name="pcie_bandwidth.h2d",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="PyTorch not available",
            gpu_uuid=gpu.uuid,
        )

    if not torch.cuda.is_available():
        return TestResult(
            test_name="pcie_bandwidth.h2d",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="CUDA not available",
            gpu_uuid=gpu.uuid,
        )

    try:
        device = torch.device(f"cuda:{gpu.index}")
        num_elements = (transfer_size_mib * 1024 * 1024) // 4
        transfer_bytes = num_elements * 4

        host_tensor = torch.randn(num_elements, dtype=torch.float32, pin_memory=True)

        # Warm-up
        _ = host_tensor.to(device, non_blocking=False)
        torch.cuda.synchronize(device)

        # Timed transfers
        total_time = 0.0
        for _ in range(iterations):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = host_tensor.to(device, non_blocking=False)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            total_time += (t1 - t0)

        avg_time = total_time / iterations
        bandwidth_gibs = (transfer_bytes / (1024**3)) / avg_time

        del host_tensor
        torch.cuda.empty_cache()

        details = {
            "bandwidth_gibs": round(bandwidth_gibs, 2),
            "min_gibs": min_gibs,
            "transfer_size_mib": transfer_size_mib,
            "iterations": iterations,
            "avg_time_ms": round(avg_time * 1000, 2),
        }

        if bandwidth_gibs < min_gibs:
            return TestResult(
                test_name="pcie_bandwidth.h2d",
                status=TestStatus.FAIL,
                duration_seconds=time.time() - start,
                message=f"H2D bandwidth low: {bandwidth_gibs:.2f} GiB/s "
                        f"(min: {min_gibs} GiB/s)",
                failure_code="DIAG-400",
                gpu_uuid=gpu.uuid,
                details=details,
            )
        return TestResult(
            test_name="pcie_bandwidth.h2d",
            status=TestStatus.PASS,
            duration_seconds=time.time() - start,
            message=f"H2D bandwidth OK: {bandwidth_gibs:.2f} GiB/s "
                    f"(min: {min_gibs} GiB/s)",
            gpu_uuid=gpu.uuid,
            details=details,
        )

    except Exception as e:
        return TestResult(
            test_name="pcie_bandwidth.h2d",
            status=TestStatus.ERROR,
            duration_seconds=time.time() - start,
            message=f"H2D bandwidth test error: {e}",
            gpu_uuid=gpu.uuid,
        )


def _measure_d2h_bandwidth(
    gpu: GPUInfo,
    min_gibs: float,
    transfer_size_mib: int = 256,
    iterations: int = 10,
) -> TestResult:
    """Measure Device-to-Host PCIe transfer bandwidth."""
    start = time.time()

    if torch is None:
        return TestResult(
            test_name="pcie_bandwidth.d2h",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="PyTorch not available",
            gpu_uuid=gpu.uuid,
        )

    if not torch.cuda.is_available():
        return TestResult(
            test_name="pcie_bandwidth.d2h",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="CUDA not available",
            gpu_uuid=gpu.uuid,
        )

    try:
        device = torch.device(f"cuda:{gpu.index}")
        num_elements = (transfer_size_mib * 1024 * 1024) // 4
        transfer_bytes = num_elements * 4

        device_tensor = torch.randn(num_elements, dtype=torch.float32, device=device)

        # Warm-up
        _ = device_tensor.to("cpu", non_blocking=False)
        torch.cuda.synchronize(device)

        # Timed transfers
        total_time = 0.0
        for _ in range(iterations):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = device_tensor.to("cpu", non_blocking=False)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            total_time += (t1 - t0)

        avg_time = total_time / iterations
        bandwidth_gibs = (transfer_bytes / (1024**3)) / avg_time

        del device_tensor
        torch.cuda.empty_cache()

        details = {
            "bandwidth_gibs": round(bandwidth_gibs, 2),
            "min_gibs": min_gibs,
            "transfer_size_mib": transfer_size_mib,
            "iterations": iterations,
            "avg_time_ms": round(avg_time * 1000, 2),
        }

        if bandwidth_gibs < min_gibs:
            return TestResult(
                test_name="pcie_bandwidth.d2h",
                status=TestStatus.FAIL,
                duration_seconds=time.time() - start,
                message=f"D2H bandwidth low: {bandwidth_gibs:.2f} GiB/s "
                        f"(min: {min_gibs} GiB/s)",
                failure_code="DIAG-401",
                gpu_uuid=gpu.uuid,
                details=details,
            )
        return TestResult(
            test_name="pcie_bandwidth.d2h",
            status=TestStatus.PASS,
            duration_seconds=time.time() - start,
            message=f"D2H bandwidth OK: {bandwidth_gibs:.2f} GiB/s "
                    f"(min: {min_gibs} GiB/s)",
            gpu_uuid=gpu.uuid,
            details=details,
        )

    except Exception as e:
        return TestResult(
            test_name="pcie_bandwidth.d2h",
            status=TestStatus.ERROR,
            duration_seconds=time.time() - start,
            message=f"D2H bandwidth test error: {e}",
            gpu_uuid=gpu.uuid,
        )


def run_pcie_bandwidth(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute PCIe bandwidth tests on all GPUs.

    Args:
        gpu_infos: List of detected GPUInfo objects.
        profile: GPU-specific profile dict with bandwidth thresholds.

    Returns:
        List of TestResult objects for H2D and D2H bandwidth tests.
    """
    thresholds = profile.get("thresholds", {})
    h2d_min = thresholds.get("pcie_h2d_min_gibs", 20.0)
    d2h_min = thresholds.get("pcie_d2h_min_gibs", 20.0)

    results = []
    for gpu in gpu_infos:
        results.append(_measure_h2d_bandwidth(gpu, h2d_min))
        results.append(_measure_d2h_bandwidth(gpu, d2h_min))

    return results
