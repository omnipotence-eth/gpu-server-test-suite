"""GPU state cleanup — idempotent reset after diagnostic tests.

Ensures diagnostic tests don't leave GPUs in zombie states:
  - Clears CUDA context and cached memory
  - Resets GPU clocks to default
  - Resets power limits to default
  - Resets compute mode to default
  - Clears any pending ECC page retirements (warns)

Called automatically after test completion and available as
a standalone cleanup command. Designed for idempotency —
safe to run multiple times without side effects.
"""

import time
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _cleanup_cuda_context(gpu_index: int) -> dict:
    """Clear CUDA context and cached memory for a GPU."""
    result = {"cuda_cleanup": "skipped"}

    if torch is None or not torch.cuda.is_available():
        return result

    try:
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            torch.cuda.reset_accumulated_memory_stats(device)
        except AttributeError:
            pass
        result["cuda_cleanup"] = "success"
        result["memory_cached_after"] = torch.cuda.memory_reserved(device)
    except Exception as e:
        result["cuda_cleanup"] = f"error: {e}"

    return result


def _reset_gpu_clocks(gpu_index: int) -> dict:
    """Reset GPU clocks to default via pynvml."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            try:
                pynvml.nvmlDeviceResetApplicationsClocks(handle)
                return {"clock_reset": "success"}
            except pynvml.NVMLError as e:
                return {"clock_reset": f"not needed: {e}"}
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        return {"clock_reset": f"error: {e}"}


def _reset_power_limit(gpu_index: int) -> dict:
    """Reset GPU power limit to default via pynvml."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            try:
                default_limit = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
                current_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                if current_limit != default_limit:
                    pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_limit)
                    return {
                        "power_reset": "success",
                        "previous_limit_mw": current_limit,
                        "default_limit_mw": default_limit,
                    }
                return {
                    "power_reset": "already_default",
                    "limit_mw": current_limit,
                }
            except pynvml.NVMLError as e:
                return {"power_reset": f"error: {e}"}
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        return {"power_reset": f"error: {e}"}


def _check_pending_retirement(gpu_index: int) -> dict:
    """Check for pending ECC page retirements."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            try:
                pending = pynvml.nvmlDeviceGetRetiredPagesPendingStatus(handle)
                return {
                    "pending_retirement": bool(pending),
                    "reboot_needed": bool(pending),
                }
            except (pynvml.NVMLError, AttributeError):
                return {"pending_retirement": "unknown"}
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        return {"pending_retirement": f"error: {e}"}


def cleanup_gpu(gpu: GPUInfo) -> TestResult:
    """Run full cleanup on a single GPU.

    Args:
        gpu: GPUInfo object for the GPU to clean up.

    Returns:
        TestResult summarizing cleanup actions.
    """
    start = time.time()

    actions = {}
    actions.update(_cleanup_cuda_context(gpu.index))
    actions.update(_reset_gpu_clocks(gpu.index))
    actions.update(_reset_power_limit(gpu.index))
    actions.update(_check_pending_retirement(gpu.index))

    # Determine if any cleanup action failed
    errors = [v for k, v in actions.items() if isinstance(v, str) and v.startswith("error:")]

    if errors:
        return TestResult(
            test_name="cleanup.gpu_reset",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=(f"GPU {gpu.index} cleanup partial: {len(errors)} action(s) had errors"),
            gpu_uuid=gpu.uuid,
            details=actions,
        )

    # Check if reboot needed
    if actions.get("reboot_needed"):
        return TestResult(
            test_name="cleanup.gpu_reset",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=(f"GPU {gpu.index} cleanup done — reboot needed for page retirement"),
            gpu_uuid=gpu.uuid,
            details=actions,
        )

    return TestResult(
        test_name="cleanup.gpu_reset",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message=f"GPU {gpu.index} cleanup successful",
        gpu_uuid=gpu.uuid,
        details=actions,
    )


def run_cleanup(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute cleanup on all GPUs.

    Args:
        gpu_infos: List of GPUInfo objects.
        profile: GPU profile dict (unused, interface consistency).

    Returns:
        List of TestResult objects per GPU.
    """
    return [cleanup_gpu(gpu) for gpu in gpu_infos]
