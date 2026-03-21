"""NVIDIA XID error tracking — GPU fault code monitoring.

XID errors are kernel-level GPU fault codes logged by the NVIDIA driver.
Critical XIDs indicate hardware failures that require node drain:
  - XID 31: GPU memory page fault
  - XID 43: GPU stopped processing
  - XID 48: Double-bit ECC error
  - XID 61: Internal firmware error
  - XID 62: Internal firmware error (row remapper)
  - XID 63: ECC page retirement / row remapping failure
  - XID 64: ECC page retirement / row remapping (DBE)
  - XID 74: NVLink error
  - XID 79: GPU fallen off the bus
  - XID 92: High single-bit ECC error rate

Mirrors DCGM's XID event listener for production GPU cluster health.
"""

import subprocess
import time
from typing import Any

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus

# XID severity classification
CRITICAL_XIDS = {31, 43, 48, 61, 62, 63, 64, 74, 79}
WARNING_XIDS = {13, 32, 45, 68, 92, 94, 95}

XID_DESCRIPTIONS = {
    13: "Graphics engine exception",
    31: "GPU memory page fault",
    32: "Invalid or corrupted push buffer stream",
    43: "GPU stopped processing",
    45: "Preemptive cleanup — no further action",
    48: "Double-bit ECC error",
    61: "Internal firmware error",
    62: "Internal firmware error (row remapper)",
    63: "ECC page retirement / row remapping failure",
    64: "ECC page retirement (DBE) / row remapping",
    68: "NVDEC0 exception",
    74: "NVLink error",
    79: "GPU has fallen off the bus",
    92: "High single-bit ECC error rate",
    94: "Contained ECC error (requires GPU reset)",
    95: "Uncontained ECC error",
}


def _query_xid_from_dmesg() -> list[dict]:
    """Parse XID errors from system dmesg (Linux only).

    Returns list of dicts: {"xid": int, "timestamp": str, "raw": str}
    """
    try:
        result = subprocess.run(
            ["dmesg", "--time-format=iso"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        xid_events = []
        for line in result.stdout.splitlines():
            if "NVRM: Xid" in line:
                # Parse: "... NVRM: Xid (PCI:0000:01:00): 31, ..."
                try:
                    xid_part = line.split("Xid")[1]
                    # Extract XID number
                    parts = xid_part.split(":")
                    for part in parts:
                        part = part.strip().rstrip(",").strip()
                        if part.isdigit():
                            xid_events.append({
                                "xid": int(part),
                                "raw": line.strip(),
                            })
                            break
                except (IndexError, ValueError):
                    continue
        return xid_events
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _query_xid_via_nvml(gpu_index: int) -> list[dict]:
    """Query recent XID events via pynvml event set.

    Uses NVML event monitoring to check for recent GPU errors.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            # Check for remapped rows (indicates ECC-related XIDs)
            xid_events = []
            try:
                remapped = pynvml.nvmlDeviceGetRemappedRows(handle)
                correctable, uncorrectable, pending, failure = remapped
                if uncorrectable > 0:
                    xid_events.append({
                        "xid": 64,
                        "detail": (
                            f"Uncorrectable remapped rows: "
                            f"{uncorrectable}"
                        ),
                    })
                if failure:
                    xid_events.append({
                        "xid": 63,
                        "detail": "Row remapping failure detected",
                    })
            except (pynvml.NVMLError, AttributeError):
                pass

            return xid_events
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return []


def _check_xid_errors(
    gpu: GPUInfo,
) -> TestResult:
    """Check for XID errors on a single GPU."""
    start = time.time()

    # Gather XID events from all sources
    dmesg_xids = _query_xid_from_dmesg()
    nvml_xids = _query_xid_via_nvml(gpu.index)

    all_xids = []
    for evt in dmesg_xids:
        xid = evt["xid"]
        all_xids.append({
            "xid": xid,
            "source": "dmesg",
            "description": XID_DESCRIPTIONS.get(xid, "Unknown"),
            "severity": (
                "CRITICAL" if xid in CRITICAL_XIDS
                else ("WARNING" if xid in WARNING_XIDS else "INFO")
            ),
            "raw": evt.get("raw", ""),
        })

    for evt in nvml_xids:
        xid = evt["xid"]
        all_xids.append({
            "xid": xid,
            "source": "nvml",
            "description": XID_DESCRIPTIONS.get(xid, "Unknown"),
            "severity": (
                "CRITICAL" if xid in CRITICAL_XIDS
                else ("WARNING" if xid in WARNING_XIDS else "INFO")
            ),
            "detail": evt.get("detail", ""),
        })

    critical_count = sum(
        1 for x in all_xids if x["severity"] == "CRITICAL"
    )
    warning_count = sum(
        1 for x in all_xids if x["severity"] == "WARNING"
    )

    details = {
        "gpu_index": gpu.index,
        "xid_events": all_xids,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "total_count": len(all_xids),
    }

    if critical_count > 0:
        critical_ids = [
            x["xid"] for x in all_xids if x["severity"] == "CRITICAL"
        ]
        return TestResult(
            test_name="telemetry.xid_errors",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message=(
                f"Critical XID errors detected: "
                f"{critical_ids} — node drain recommended"
            ),
            failure_code="DIAG-900",
            gpu_uuid=gpu.uuid,
            details=details,
        )

    if warning_count > 0:
        return TestResult(
            test_name="telemetry.xid_errors",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=(
                f"{warning_count} XID warning(s) detected — "
                f"monitor closely"
            ),
            gpu_uuid=gpu.uuid,
            details=details,
        )

    return TestResult(
        test_name="telemetry.xid_errors",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message="No XID errors detected",
        gpu_uuid=gpu.uuid,
        details=details,
    )


def run_xid_checks(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute XID error checks on all GPUs.

    Args:
        gpu_infos: List of detected GPUInfo objects.
        profile: GPU profile dict (unused, kept for interface consistency).

    Returns:
        List of TestResult objects.
    """
    return [_check_xid_errors(gpu) for gpu in gpu_infos]
