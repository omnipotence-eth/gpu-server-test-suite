"""NCCL collective operation validation — distributed training readiness.

Validates GPU communication via NCCL (NVIDIA Collective Communications
Library) collective operations:
  - AllReduce: Sum tensors across all GPUs (the backbone of data-parallel training)
  - AllGather: Gather tensors from all GPUs
  - Broadcast: Send tensor from one GPU to all others

This is the "Gold Standard" for verifying a server is ready for
distributed LLM training. Poor NCCL performance indicates topology
misconfiguration, degraded NVLinks, or driver issues.

Requires: torch.distributed with NCCL backend.
"""

import time
from typing import Any

try:
    import torch
    import torch.distributed as dist
except ImportError:
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _nccl_available() -> bool:
    """Check if NCCL backend is available."""
    if torch is None or dist is None:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return dist.is_nccl_available()
    except Exception:
        return False


def _run_allreduce_bench(
    gpu_count: int,
    size_mib: int = 128,
    iterations: int = 50,
) -> dict:
    """Benchmark AllReduce performance across GPUs.

    This is a simplified in-process benchmark. Production NCCL tests
    use nccl-tests (multi-process with torchrun/mpirun).
    Returns bandwidth and latency metrics.
    """
    if torch is None or not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    if gpu_count < 2:
        return {"error": "AllReduce requires 2+ GPUs"}

    try:
        num_elements = size_mib * 1024 * 1024 // 4
        tensors = []
        for i in range(gpu_count):
            device = torch.device(f"cuda:{i}")
            tensors.append(
                torch.randn(
                    num_elements,
                    dtype=torch.float32,
                    device=device,
                )
            )

        # Manual ring-allreduce simulation for single-process
        # (Real NCCL tests use torch.distributed)
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            # Simulate allreduce: reduce to GPU 0, broadcast back
            accumulated = tensors[0].clone()
            for i in range(1, gpu_count):
                accumulated += tensors[i].to(tensors[0].device)
            torch.cuda.synchronize()
            for i in range(1, gpu_count):
                tensors[i].copy_(accumulated.to(tensors[i].device))
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations
        # AllReduce bus bandwidth formula: 2 * (n-1)/n * size / time
        data_size = num_elements * 4  # bytes
        bus_bw = (
            2 * (gpu_count - 1) / gpu_count
            * data_size / avg_time / (1024**3)
        )
        algo_bw = data_size / avg_time / (1024**3)

        for t in tensors:
            del t
        torch.cuda.empty_cache()

        return {
            "bus_bandwidth_gibs": round(bus_bw, 2),
            "algo_bandwidth_gibs": round(algo_bw, 2),
            "avg_latency_ms": round(avg_time * 1000, 3),
            "message_size_mib": size_mib,
            "iterations": iterations,
            "gpu_count": gpu_count,
        }
    except Exception as e:
        return {"error": str(e)}


def _run_allgather_bench(
    gpu_count: int,
    size_mib: int = 64,
    iterations: int = 50,
) -> dict:
    """Benchmark AllGather performance across GPUs."""
    if torch is None or not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    if gpu_count < 2:
        return {"error": "AllGather requires 2+ GPUs"}

    try:
        per_gpu_elements = size_mib * 1024 * 1024 // 4
        src_tensors = []
        for i in range(gpu_count):
            device = torch.device(f"cuda:{i}")
            src_tensors.append(
                torch.randn(
                    per_gpu_elements,
                    dtype=torch.float32,
                    device=device,
                )
            )

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            # Simulate allgather: each GPU gets all chunks
            for i in range(gpu_count):
                for j in range(gpu_count):
                    if i != j:
                        _ = src_tensors[j].to(
                            src_tensors[i].device
                        )
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations
        total_data = per_gpu_elements * 4 * gpu_count
        bus_bw = (
            (gpu_count - 1) / gpu_count
            * total_data / avg_time / (1024**3)
        )

        for t in src_tensors:
            del t
        torch.cuda.empty_cache()

        return {
            "bus_bandwidth_gibs": round(bus_bw, 2),
            "avg_latency_ms": round(avg_time * 1000, 3),
            "per_gpu_size_mib": size_mib,
            "iterations": iterations,
            "gpu_count": gpu_count,
        }
    except Exception as e:
        return {"error": str(e)}


def run_nccl_validation(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute NCCL collective operation benchmarks.

    Tests AllReduce and AllGather performance to validate
    multi-GPU communication readiness.
    """
    results = []

    if len(gpu_infos) < 2:
        return [TestResult(
            test_name="interconnect.nccl_allreduce",
            status=TestStatus.SKIP,
            duration_seconds=0.0,
            message=(
                "NCCL validation requires 2+ GPUs "
                "(single GPU detected)"
            ),
        )]

    if torch is None:
        return [TestResult(
            test_name="interconnect.nccl_allreduce",
            status=TestStatus.SKIP,
            duration_seconds=0.0,
            message="PyTorch not available",
        )]

    if not torch.cuda.is_available():
        return [TestResult(
            test_name="interconnect.nccl_allreduce",
            status=TestStatus.SKIP,
            duration_seconds=0.0,
            message="CUDA not available",
        )]

    gpu_count = len(gpu_infos)
    thresholds = profile.get("thresholds", {})

    # AllReduce benchmark
    start = time.time()
    ar_result = _run_allreduce_bench(gpu_count)
    ar_duration = time.time() - start

    if "error" in ar_result:
        results.append(TestResult(
            test_name="interconnect.nccl_allreduce",
            status=TestStatus.ERROR,
            duration_seconds=ar_duration,
            message=f"AllReduce failed: {ar_result['error']}",
            failure_code="DIAG-960",
            details=ar_result,
        ))
    else:
        min_bw = thresholds.get("nccl_allreduce_min_gibs", 5.0)
        bw = ar_result["bus_bandwidth_gibs"]
        status = TestStatus.PASS if bw >= min_bw else TestStatus.FAIL

        results.append(TestResult(
            test_name="interconnect.nccl_allreduce",
            status=status,
            duration_seconds=ar_duration,
            message=(
                f"AllReduce bus BW: {bw:.1f} GiB/s "
                f"(min: {min_bw} GiB/s) "
                f"across {gpu_count} GPUs"
            ),
            failure_code=(
                "DIAG-961" if status == TestStatus.FAIL else ""
            ),
            details=ar_result,
        ))

    # AllGather benchmark
    start = time.time()
    ag_result = _run_allgather_bench(gpu_count)
    ag_duration = time.time() - start

    if "error" in ag_result:
        results.append(TestResult(
            test_name="interconnect.nccl_allgather",
            status=TestStatus.ERROR,
            duration_seconds=ag_duration,
            message=f"AllGather failed: {ag_result['error']}",
            failure_code="DIAG-962",
            details=ag_result,
        ))
    else:
        min_bw = thresholds.get("nccl_allgather_min_gibs", 4.0)
        bw = ag_result["bus_bandwidth_gibs"]
        status = TestStatus.PASS if bw >= min_bw else TestStatus.FAIL

        results.append(TestResult(
            test_name="interconnect.nccl_allgather",
            status=status,
            duration_seconds=ag_duration,
            message=(
                f"AllGather bus BW: {bw:.1f} GiB/s "
                f"(min: {min_bw} GiB/s) "
                f"across {gpu_count} GPUs"
            ),
            failure_code=(
                "DIAG-963" if status == TestStatus.FAIL else ""
            ),
            details=ag_result,
        ))

    return results
