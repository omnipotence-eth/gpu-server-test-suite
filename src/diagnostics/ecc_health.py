"""ECC memory health — Single/Double Bit Error tracking.

Monitors ECC (Error Correcting Code) memory health:
  - SBE (Single Bit Errors): Correctable — tracked as aggregate rate
  - DBE (Double Bit Errors): Uncorrectable — triggers immediate FAIL
  - Retired pages: Pages taken offline due to persistent errors
  - Row remapping: Hardware-level error mitigation (Ampere+)

In production clusters, rising SBE counts trigger proactive node drain
before a DBE crashes a training job. This mirrors DCGM's ECC policy engine.
"""

import time
from typing import Any

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _query_ecc_counters(gpu_index: int) -> dict:
    """Query ECC error counters and retired page info via pynvml."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            # Volatile ECC errors (since last driver load)
            vol_sbe = 0
            vol_dbe = 0
            try:
                vol_sbe = (
                    pynvml.nvmlDeviceGetTotalEccErrors(
                        handle,
                        pynvml.NVML_SINGLE_BIT_ECC,
                        pynvml.NVML_VOLATILE_ECC,
                    )
                )
            except (pynvml.NVMLError, AttributeError):
                vol_sbe = -1  # Not supported

            try:
                vol_dbe = (
                    pynvml.nvmlDeviceGetTotalEccErrors(
                        handle,
                        pynvml.NVML_DOUBLE_BIT_ECC,
                        pynvml.NVML_VOLATILE_ECC,
                    )
                )
            except (pynvml.NVMLError, AttributeError):
                vol_dbe = -1

            # Aggregate ECC errors (lifetime)
            agg_sbe = 0
            agg_dbe = 0
            try:
                agg_sbe = (
                    pynvml.nvmlDeviceGetTotalEccErrors(
                        handle,
                        pynvml.NVML_SINGLE_BIT_ECC,
                        pynvml.NVML_AGGREGATE_ECC,
                    )
                )
            except (pynvml.NVMLError, AttributeError):
                agg_sbe = -1

            try:
                agg_dbe = (
                    pynvml.nvmlDeviceGetTotalEccErrors(
                        handle,
                        pynvml.NVML_DOUBLE_BIT_ECC,
                        pynvml.NVML_AGGREGATE_ECC,
                    )
                )
            except (pynvml.NVMLError, AttributeError):
                agg_dbe = -1

            # Retired pages
            retired_sbe = 0
            retired_dbe = 0
            try:
                retired_sbe = (
                    pynvml.nvmlDeviceGetRetiredPages(
                        handle,
                        pynvml.NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
                    )
                )
                if isinstance(retired_sbe, (list, tuple)):
                    retired_sbe = len(retired_sbe)
            except (pynvml.NVMLError, AttributeError):
                retired_sbe = -1

            try:
                retired_dbe = (
                    pynvml.nvmlDeviceGetRetiredPages(
                        handle,
                        pynvml.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR,
                    )
                )
                if isinstance(retired_dbe, (list, tuple)):
                    retired_dbe = len(retired_dbe)
            except (pynvml.NVMLError, AttributeError):
                retired_dbe = -1

            # Pending retired pages
            pending_retirement = False
            try:
                pending = (
                    pynvml.nvmlDeviceGetRetiredPages_v2(handle)
                )
                if pending:
                    pending_retirement = True
            except (pynvml.NVMLError, AttributeError):
                pass

            # Row remapper (Ampere+)
            remapped_rows = {}
            try:
                remapped = pynvml.nvmlDeviceGetRemappedRows(handle)
                correctable, uncorrectable, pending_remap, failure = (
                    remapped
                )
                remapped_rows = {
                    "correctable": correctable,
                    "uncorrectable": uncorrectable,
                    "pending": pending_remap,
                    "failure": failure,
                }
            except (pynvml.NVMLError, AttributeError):
                pass

            return {
                "ecc_supported": True,
                "volatile": {
                    "sbe": vol_sbe,
                    "dbe": vol_dbe,
                },
                "aggregate": {
                    "sbe": agg_sbe,
                    "dbe": agg_dbe,
                },
                "retired_pages": {
                    "sbe_caused": retired_sbe,
                    "dbe_caused": retired_dbe,
                    "pending_retirement": pending_retirement,
                },
                "remapped_rows": remapped_rows,
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        return {"ecc_supported": False, "error": str(e)}


def _check_ecc_health(
    gpu: GPUInfo,
    profile: dict[str, Any],
) -> TestResult:
    """Analyze ECC health for a single GPU."""
    start = time.time()

    # Check if ECC is supported on this GPU
    if gpu.ecc_mode == "not_supported":
        ecc_supported = profile.get("ecc_supported", False)
        if not ecc_supported:
            return TestResult(
                test_name="telemetry.ecc_health",
                status=TestStatus.SKIP,
                duration_seconds=time.time() - start,
                message=(
                    "ECC not supported on this GPU "
                    f"({gpu.name})"
                ),
                gpu_uuid=gpu.uuid,
                details={"ecc_mode": gpu.ecc_mode},
            )

    counters = _query_ecc_counters(gpu.index)

    if not counters.get("ecc_supported"):
        return TestResult(
            test_name="telemetry.ecc_health",
            status=TestStatus.SKIP,
            duration_seconds=time.time() - start,
            message=(
                f"ECC query failed: "
                f"{counters.get('error', 'unknown')}"
            ),
            gpu_uuid=gpu.uuid,
            details=counters,
        )

    details = {
        "gpu_index": gpu.index,
        "ecc_mode": gpu.ecc_mode,
        **counters,
    }

    thresholds = profile.get("thresholds", {})
    sbe_warn_threshold = thresholds.get("ecc_sbe_warn_count", 10)

    # Check for DBE (uncorrectable) — always critical
    vol_dbe = counters["volatile"]["dbe"]
    agg_dbe = counters["aggregate"]["dbe"]
    if vol_dbe > 0 or agg_dbe > 0:
        return TestResult(
            test_name="telemetry.ecc_health",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message=(
                f"Double-bit ECC errors: "
                f"volatile={vol_dbe}, aggregate={agg_dbe} "
                f"— node drain required"
            ),
            failure_code="DIAG-920",
            gpu_uuid=gpu.uuid,
            details=details,
        )

    # Check remapped rows failure
    remapped = counters.get("remapped_rows", {})
    if remapped.get("failure"):
        return TestResult(
            test_name="telemetry.ecc_health",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message="Row remapping failure — GPU replacement needed",
            failure_code="DIAG-921",
            gpu_uuid=gpu.uuid,
            details=details,
        )

    # Check for rising SBE count
    vol_sbe = counters["volatile"]["sbe"]
    agg_sbe = counters["aggregate"]["sbe"]
    if vol_sbe > sbe_warn_threshold or agg_sbe > sbe_warn_threshold:
        return TestResult(
            test_name="telemetry.ecc_health",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=(
                f"Elevated SBE count: "
                f"volatile={vol_sbe}, aggregate={agg_sbe} "
                f"(threshold: {sbe_warn_threshold}) "
                f"— proactive drain recommended"
            ),
            gpu_uuid=gpu.uuid,
            details=details,
        )

    # Check retired pages
    retired = counters.get("retired_pages", {})
    if retired.get("pending_retirement"):
        return TestResult(
            test_name="telemetry.ecc_health",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=(
                "Pages pending retirement — reboot recommended"
            ),
            gpu_uuid=gpu.uuid,
            details=details,
        )

    return TestResult(
        test_name="telemetry.ecc_health",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message=(
            f"ECC healthy: SBE={vol_sbe}, DBE={vol_dbe}, "
            f"retired pages: "
            f"{retired.get('sbe_caused', 0) + retired.get('dbe_caused', 0)}"
        ),
        gpu_uuid=gpu.uuid,
        details=details,
    )


def run_ecc_health_checks(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute ECC health analysis on all GPUs."""
    return [
        _check_ecc_health(gpu, profile) for gpu in gpu_infos
    ]
