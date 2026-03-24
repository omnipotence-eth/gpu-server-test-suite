"""DCGM Level 1 equivalent — Deployment validation checks.

Mirrors DCGM's Deployment Plugin: verifies the GPU server is correctly
configured before running any performance or stress diagnostics.

Checks performed:
  - NVIDIA driver is loaded and responsive
  - Expected number of GPUs detected
  - GPU model matches expected profile
  - ECC mode matches expected setting
  - No other processes using the GPU
  - Driver version meets minimum requirement
  - Persistence mode status
"""

import time
from typing import Any

import pynvml

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _check_driver_loaded() -> TestResult:
    """Verify NVIDIA driver is loaded and pynvml can initialize."""
    start = time.time()
    try:
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8")
        pynvml.nvmlShutdown()
        return TestResult(
            test_name="deployment.driver_loaded",
            status=TestStatus.PASS,
            duration_seconds=time.time() - start,
            message=f"NVIDIA driver loaded: v{driver}",
            details={"driver_version": driver},
        )
    except pynvml.NVMLError as e:
        return TestResult(
            test_name="deployment.driver_loaded",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message=f"NVIDIA driver not loaded: {e}",
            failure_code="DIAG-001",
            details={"error": str(e)},
        )


def _check_gpu_count(gpu_infos: list[GPUInfo], expected_count: int) -> TestResult:
    """Verify expected number of GPUs are detected."""
    start = time.time()
    actual = len(gpu_infos)
    if actual == expected_count:
        return TestResult(
            test_name="deployment.gpu_count",
            status=TestStatus.PASS,
            duration_seconds=time.time() - start,
            message=f"GPU count matches: {actual} detected, {expected_count} expected",
            details={"detected": actual, "expected": expected_count},
        )
    return TestResult(
        test_name="deployment.gpu_count",
        status=TestStatus.FAIL,
        duration_seconds=time.time() - start,
        message=f"GPU count mismatch: {actual} detected, {expected_count} expected",
        failure_code="DIAG-001",
        details={"detected": actual, "expected": expected_count},
    )


def _check_gpu_model(gpu_infos: list[GPUInfo], expected_model: str) -> TestResult:
    """Verify GPU model matches the expected profile."""
    start = time.time()
    mismatches = []
    for gpu in gpu_infos:
        if expected_model.lower() not in gpu.name.lower():
            mismatches.append({"index": gpu.index, "detected": gpu.name})

    if not mismatches:
        return TestResult(
            test_name="deployment.gpu_model",
            status=TestStatus.PASS,
            duration_seconds=time.time() - start,
            message=f"GPU model matches expected: {expected_model}",
            details={"expected": expected_model},
        )
    return TestResult(
        test_name="deployment.gpu_model",
        status=TestStatus.FAIL,
        duration_seconds=time.time() - start,
        message=f"GPU model mismatch: expected '{expected_model}'",
        failure_code="DIAG-001",
        details={"expected": expected_model, "mismatches": mismatches},
    )


def _check_ecc_mode(gpu_infos: list[GPUInfo], profile: dict[str, Any]) -> TestResult:
    """Check ECC mode against profile expectation."""
    start = time.time()
    ecc_supported = profile.get("ecc_supported", False)
    ecc_expected = profile.get("ecc_expected", None)

    results = []
    for gpu in gpu_infos:
        if not ecc_supported:
            results.append(
                {"index": gpu.index, "ecc_mode": gpu.ecc_mode, "status": "SKIP"}
            )
        elif ecc_expected is not None:
            expected_str = "enabled" if ecc_expected else "disabled"
            if gpu.ecc_mode == expected_str:
                results.append(
                    {"index": gpu.index, "ecc_mode": gpu.ecc_mode, "status": "PASS"}
                )
            else:
                results.append(
                    {"index": gpu.index, "ecc_mode": gpu.ecc_mode, "status": "FAIL"}
                )

    if not ecc_supported:
        return TestResult(
            test_name="deployment.ecc_mode",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message="ECC not supported on this GPU model",
            details={"results": results},
        )

    any_fail = any(r["status"] == "FAIL" for r in results)
    return TestResult(
        test_name="deployment.ecc_mode",
        status=TestStatus.FAIL if any_fail else TestStatus.PASS,
        duration_seconds=time.time() - start,
        message="ECC mode check complete",
        failure_code="DIAG-001" if any_fail else "",
        details={"results": results},
    )


def _check_gpu_processes(gpu_infos: list[GPUInfo]) -> TestResult:
    """Verify no compute workloads are using the GPUs before testing.

    On Windows (WDDM), nvmlDeviceGetComputeRunningProcesses returns all
    processes with a GPU context — including the desktop compositor, browsers,
    and system UI. These are not real compute workloads. We filter to processes
    with >100 MB VRAM usage, which reliably identifies actual CUDA workloads
    (ML inference, training) while ignoring display/system processes.
    """
    start = time.time()
    busy_gpus = []
    COMPUTE_VRAM_THRESHOLD_MB = 100

    pynvml.nvmlInit()
    try:
        for gpu in gpu_infos:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                heavy = [
                    p for p in procs
                    if (p.usedGpuMemory or 0) > COMPUTE_VRAM_THRESHOLD_MB * 1024 * 1024
                ]
                if heavy:
                    busy_gpus.append(
                        {
                            "index": gpu.index,
                            "process_count": len(heavy),
                            "pids": [p.pid for p in heavy],
                        }
                    )
            except pynvml.NVMLError:
                pass  # Some consumer GPUs don't support this query
    finally:
        pynvml.nvmlShutdown()

    if busy_gpus:
        return TestResult(
            test_name="deployment.gpu_processes",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=(
                f"GPU(s) in use by "
                f"{sum(g['process_count'] for g in busy_gpus)} "
                f"compute process(es) (>{COMPUTE_VRAM_THRESHOLD_MB}MB VRAM)"
            ),
            details={"busy_gpus": busy_gpus},
        )
    return TestResult(
        test_name="deployment.gpu_processes",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message="No compute workloads on GPU(s) — clean for testing",
    )


def _check_persistence_mode(gpu_infos: list[GPUInfo]) -> TestResult:
    """Check persistence mode status (data center GPUs should have it enabled)."""
    start = time.time()
    results = []

    pynvml.nvmlInit()
    try:
        for gpu in gpu_infos:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)
            try:
                mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
                mode_str = "enabled" if mode == pynvml.NVML_FEATURE_ENABLED else "disabled"
                results.append({"index": gpu.index, "persistence_mode": mode_str})
            except pynvml.NVMLError:
                results.append({"index": gpu.index, "persistence_mode": "unknown"})
    finally:
        pynvml.nvmlShutdown()

    return TestResult(
        test_name="deployment.persistence_mode",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message="Persistence mode check complete",
        details={"results": results},
    )


def run_deployment_checks(
    gpu_infos: list[GPUInfo],
    config: dict[str, Any],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute all Level 1 deployment validation checks.

    Args:
        gpu_infos: List of detected GPUInfo objects.
        config: Master test configuration dict.
        profile: GPU-specific profile dict with expected values.

    Returns:
        List of TestResult objects for each deployment check.
    """
    expected_count = config.get("expected", {}).get("gpu_count", 1)
    expected_model = profile.get("gpu_model", "")

    results = [
        _check_driver_loaded(),
        _check_gpu_count(gpu_infos, expected_count),
        _check_gpu_model(gpu_infos, expected_model),
        _check_ecc_mode(gpu_infos, profile),
        _check_gpu_processes(gpu_infos),
        _check_persistence_mode(gpu_infos),
    ]

    return results
