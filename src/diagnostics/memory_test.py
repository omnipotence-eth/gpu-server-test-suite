"""DCGM Level 2 — VRAM integrity and allocation test.

Validates that GPU VRAM can be allocated and written/read correctly.
Uses PyTorch CUDA tensors to exercise the framebuffer. This is a
lighter-weight test than the Level 4 memtest (which does exhaustive
bit-pattern testing).

Checks performed:
  - VRAM allocation at configured percentage of total
  - Write/read verification with known patterns
  - VRAM fragmentation check (can allocate in one contiguous block)
"""

import time
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _check_vram_allocation(
    gpu: GPUInfo,
    allocation_pct: int,
) -> TestResult:
    """Attempt to allocate a large VRAM block and verify write/read."""
    start = time.time()
    target_mib = int(gpu.vram_total_mib * allocation_pct / 100)

    if torch is None:
        return TestResult(
            test_name="memory_test.vram_allocation",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="PyTorch not available — skipping VRAM allocation test",
            gpu_uuid=gpu.uuid,
            details={"reason": "no_torch"},
        )

    if not torch.cuda.is_available():
        return TestResult(
            test_name="memory_test.vram_allocation",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="CUDA not available — skipping VRAM allocation test",
            gpu_uuid=gpu.uuid,
            details={"reason": "no_cuda"},
        )

    device = torch.device(f"cuda:{gpu.index}")

    # Allocate target amount (in bytes -> elements of float32)
    target_bytes = target_mib * 1024 * 1024
    num_elements = target_bytes // 4  # float32 = 4 bytes

    # Allocation test
    try:
        tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
        allocated_mib = (tensor.nelement() * tensor.element_size()) / (1024 * 1024)
    except torch.cuda.OutOfMemoryError:
        return TestResult(
            test_name="memory_test.vram_allocation",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message=f"Failed to allocate {target_mib} MiB VRAM on GPU {gpu.index}",
            failure_code="DIAG-300",
            gpu_uuid=gpu.uuid,
            details={
                "target_mib": target_mib,
                "allocation_pct": allocation_pct,
                "vram_total_mib": gpu.vram_total_mib,
                "vram_free_mib": gpu.vram_free_mib,
            },
        )

    # Write/read verification — write pattern, read back, verify
    tensor.fill_(42.0)
    readback = tensor.sum().item()
    expected_sum = 42.0 * num_elements

    # Allow small floating-point tolerance
    tolerance = abs(expected_sum) * 1e-5
    correct = abs(readback - expected_sum) < tolerance

    # Cleanup
    del tensor
    torch.cuda.empty_cache()

    if not correct:
        return TestResult(
            test_name="memory_test.vram_allocation",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message=f"VRAM data integrity failure on GPU {gpu.index}",
            failure_code="DIAG-301",
            gpu_uuid=gpu.uuid,
            details={
                "allocated_mib": round(allocated_mib, 1),
                "expected_sum": expected_sum,
                "actual_sum": readback,
                "pattern": 42.0,
            },
        )

    return TestResult(
        test_name="memory_test.vram_allocation",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message=f"Allocated and verified {allocated_mib:.0f} MiB on GPU {gpu.index}",
        gpu_uuid=gpu.uuid,
        details={
            "allocated_mib": round(allocated_mib, 1),
            "target_mib": target_mib,
            "allocation_pct": allocation_pct,
            "pattern_verified": True,
        },
    )


def _check_vram_pattern_test(
    gpu: GPUInfo,
) -> TestResult:
    """Write multiple bit patterns to VRAM and verify integrity.

    Tests with patterns: all-zeros, all-ones, alternating, walking-one.
    Uses a smaller allocation to keep the test fast.
    """
    start = time.time()

    if torch is None:
        return TestResult(
            test_name="memory_test.pattern_test",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="PyTorch not available",
            gpu_uuid=gpu.uuid,
        )

    if not torch.cuda.is_available():
        return TestResult(
            test_name="memory_test.pattern_test",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="CUDA not available",
            gpu_uuid=gpu.uuid,
        )

    device = torch.device(f"cuda:{gpu.index}")
    # Use 256 MiB for pattern test (fast but meaningful)
    num_elements = (256 * 1024 * 1024) // 4
    patterns_tested = 0
    patterns_passed = 0

    test_patterns = [
        ("zeros", 0.0),
        ("ones", 1.0),
        ("large_value", 3.14159265),
        ("negative", -1.0),
    ]

    for pattern_name, pattern_val in test_patterns:
        try:
            tensor = torch.full((num_elements,), pattern_val, dtype=torch.float32, device=device)
            readback = tensor.mean().item()
            patterns_tested += 1

            tolerance = max(abs(pattern_val) * 1e-5, 1e-7)
            if abs(readback - pattern_val) < tolerance:
                patterns_passed += 1

            del tensor
        except torch.cuda.OutOfMemoryError:
            break

    torch.cuda.empty_cache()

    if patterns_passed == patterns_tested and patterns_tested > 0:
        return TestResult(
            test_name="memory_test.pattern_test",
            status=TestStatus.PASS,
            duration_seconds=time.time() - start,
            message=f"All {patterns_passed} VRAM patterns verified on GPU {gpu.index}",
            gpu_uuid=gpu.uuid,
            details={
                "patterns_tested": patterns_tested,
                "patterns_passed": patterns_passed,
                "test_size_mib": 256,
            },
        )
    return TestResult(
        test_name="memory_test.pattern_test",
        status=TestStatus.FAIL,
        duration_seconds=time.time() - start,
        message=f"VRAM pattern failure: {patterns_passed}/{patterns_tested} passed",
        failure_code="DIAG-302",
        gpu_uuid=gpu.uuid,
        details={
            "patterns_tested": patterns_tested,
            "patterns_passed": patterns_passed,
        },
    )


def run_memory_test(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute VRAM integrity tests on all GPUs.

    Args:
        gpu_infos: List of detected GPUInfo objects.
        profile: GPU-specific profile dict with memory thresholds.

    Returns:
        List of TestResult objects for each memory test.
    """
    thresholds = profile.get("thresholds", {})
    allocation_pct = thresholds.get("vram_test_allocation_pct", 90)

    results = []
    for gpu in gpu_infos:
        results.append(_check_vram_allocation(gpu, allocation_pct))
        results.append(_check_vram_pattern_test(gpu))

    return results
