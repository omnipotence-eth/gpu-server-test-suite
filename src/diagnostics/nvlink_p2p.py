"""NVLink / P2P bandwidth validation — inter-GPU communication test.

Tests Peer-to-Peer (P2P) bandwidth between GPU pairs to detect:
  - Degraded NVLink bridges (e.g., 900 GB/s → 50 GB/s)
  - P2P disabled by BIOS/driver configuration
  - Asymmetric link performance (one direction slower)

In multi-GPU servers (HGX, DGX), NVLink is the primary interconnect.
A single degraded bridge can bottleneck an entire training job.
Falls back to PCIe P2P measurement on systems without NVLink.
"""

import time
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _check_p2p_access(gpu_a: int, gpu_b: int) -> dict:
    """Check if P2P access is enabled between two GPUs."""
    if torch is None or not torch.cuda.is_available():
        return {"supported": False, "reason": "CUDA not available"}

    try:
        can_access = torch.cuda.can_device_access_peer(gpu_a, gpu_b)
        return {"supported": can_access, "reason": "OK" if can_access else "P2P not enabled"}
    except Exception as e:
        return {"supported": False, "reason": str(e)}


def _measure_p2p_bandwidth(
    src_idx: int,
    dst_idx: int,
    size_mib: int = 256,
    iterations: int = 20,
) -> dict:
    """Measure P2P transfer bandwidth between two GPUs.

    Args:
        src_idx: Source GPU index.
        dst_idx: Destination GPU index.
        size_mib: Transfer size in MiB.
        iterations: Number of transfer iterations for averaging.

    Returns:
        Dict with bandwidth_gibs, latency_us, and transfer details.
    """
    if torch is None:
        return {"error": "PyTorch not available"}
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    try:
        num_elements = size_mib * 1024 * 1024 // 4  # float32
        src_device = torch.device(f"cuda:{src_idx}")
        dst_device = torch.device(f"cuda:{dst_idx}")

        src_tensor = torch.randn(
            num_elements, dtype=torch.float32, device=src_device
        )

        # Warm up
        dst_tensor = src_tensor.to(dst_device)
        torch.cuda.synchronize()
        del dst_tensor

        # Timed transfers
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            dst_tensor = src_tensor.to(dst_device)
            torch.cuda.synchronize()
            del dst_tensor
        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations
        size_bytes = num_elements * 4
        bandwidth_gibs = (
            size_bytes / avg_time / (1024**3)
        )

        del src_tensor
        torch.cuda.empty_cache()

        return {
            "bandwidth_gibs": round(bandwidth_gibs, 2),
            "avg_latency_ms": round(avg_time * 1000, 3),
            "transfer_size_mib": size_mib,
            "iterations": iterations,
        }
    except Exception as e:
        return {"error": str(e)}


def _test_gpu_pair(
    gpu_a: GPUInfo,
    gpu_b: GPUInfo,
    profile: dict[str, Any],
) -> list[TestResult]:
    """Test P2P bandwidth between a GPU pair (both directions)."""
    results = []
    start = time.time()

    # Check P2P access
    access = _check_p2p_access(gpu_a.index, gpu_b.index)

    if not access["supported"]:
        results.append(TestResult(
            test_name="interconnect.p2p_access",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=(
                f"P2P not available GPU {gpu_a.index} → "
                f"GPU {gpu_b.index}: {access['reason']}"
            ),
            details={
                "src_gpu": gpu_a.index,
                "dst_gpu": gpu_b.index,
                **access,
            },
        ))
        return results

    thresholds = profile.get("thresholds", {})
    nvlink_expected = profile.get("nvlink_expected", False)
    min_bw = thresholds.get(
        "nvlink_min_bw_gibs",
        thresholds.get("pcie_h2d_min_gibs", 10.0),
    )

    # Forward direction: A → B
    fwd_start = time.time()
    fwd = _measure_p2p_bandwidth(gpu_a.index, gpu_b.index)
    fwd_time = time.time() - fwd_start

    if "error" in fwd:
        results.append(TestResult(
            test_name="interconnect.p2p_bandwidth",
            status=TestStatus.ERROR,
            duration_seconds=fwd_time,
            message=(
                f"P2P measurement failed GPU {gpu_a.index} → "
                f"GPU {gpu_b.index}: {fwd['error']}"
            ),
            failure_code="DIAG-950",
        ))
        return results

    fwd_bw = fwd["bandwidth_gibs"]
    fwd_status = TestStatus.PASS if fwd_bw >= min_bw else TestStatus.FAIL

    link_type = "NVLink" if nvlink_expected else "PCIe P2P"
    results.append(TestResult(
        test_name="interconnect.p2p_bandwidth",
        status=fwd_status,
        duration_seconds=fwd_time,
        message=(
            f"{link_type} GPU {gpu_a.index} → GPU {gpu_b.index}: "
            f"{fwd_bw:.1f} GiB/s "
            f"(min: {min_bw} GiB/s)"
        ),
        failure_code="DIAG-951" if fwd_status == TestStatus.FAIL else "",
        details={
            "direction": f"{gpu_a.index}->{gpu_b.index}",
            "link_type": link_type,
            **fwd,
        },
    ))

    # Reverse direction: B → A
    rev_start = time.time()
    rev = _measure_p2p_bandwidth(gpu_b.index, gpu_a.index)
    rev_time = time.time() - rev_start

    if "error" not in rev:
        rev_bw = rev["bandwidth_gibs"]
        rev_status = (
            TestStatus.PASS if rev_bw >= min_bw
            else TestStatus.FAIL
        )

        # Check for asymmetry (>20% difference)
        asymmetry_pct = (
            abs(fwd_bw - rev_bw) / max(fwd_bw, rev_bw) * 100
        )
        if asymmetry_pct > 20 and rev_status == TestStatus.PASS:
            rev_status = TestStatus.WARN

        results.append(TestResult(
            test_name="interconnect.p2p_bandwidth",
            status=rev_status,
            duration_seconds=rev_time,
            message=(
                f"{link_type} GPU {gpu_b.index} → "
                f"GPU {gpu_a.index}: "
                f"{rev_bw:.1f} GiB/s "
                f"(asymmetry: {asymmetry_pct:.1f}%)"
            ),
            failure_code=(
                "DIAG-951" if rev_status == TestStatus.FAIL else ""
            ),
            details={
                "direction": f"{gpu_b.index}->{gpu_a.index}",
                "link_type": link_type,
                "asymmetry_pct": round(asymmetry_pct, 1),
                **rev,
            },
        ))

    return results


def run_nvlink_p2p(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute P2P bandwidth tests across all GPU pairs.

    For single-GPU systems, reports SKIP.
    For multi-GPU, tests every unique pair in both directions.
    """
    if len(gpu_infos) < 2:
        return [TestResult(
            test_name="interconnect.p2p_bandwidth",
            status=TestStatus.SKIP,
            duration_seconds=0.0,
            message="P2P test requires 2+ GPUs (single GPU detected)",
        )]

    results = []
    for i, gpu_a in enumerate(gpu_infos):
        for gpu_b in gpu_infos[i + 1:]:
            pair_results = _test_gpu_pair(gpu_a, gpu_b, profile)
            results.extend(pair_results)

    return results
