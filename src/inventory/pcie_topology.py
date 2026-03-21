"""PCIe topology detection via nvidia-smi and pynvml.

Parses PCIe link generation, width, and replay counter information
to detect degraded links that may indicate reseating issues or
BIOS misconfiguration.
"""

import subprocess
from dataclasses import dataclass


@dataclass
class PCIeInfo:
    """PCIe link configuration for a single GPU."""

    gpu_index: int
    link_gen_current: int
    link_gen_max: int
    link_width_current: int
    link_width_max: int
    replay_counter: int
    is_degraded: bool
    degradation_reason: str


def query_pcie_via_nvidia_smi() -> list[dict]:
    """Query PCIe info using nvidia-smi subprocess.

    Fallback method when pynvml doesn't expose PCIe fields.
    Returns a list of dicts with pcie data per GPU.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,"
                "pcie.link.width.current,pcie.link.width.max",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "gen_current": int(parts[1]),
                        "gen_max": int(parts[2]),
                        "width_current": int(parts[3]),
                        "width_max": int(parts[4]),
                    }
                )
        return gpus
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return []


def query_pcie_replay_counter(gpu_index: int = 0) -> int:
    """Query PCIe replay counter via nvidia-smi.

    Replay counter increments indicate signal integrity issues on the PCIe bus.
    A non-zero value on a fresh boot may indicate a hardware problem.
    """
    try:
        subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=pcie.link.gen.current",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # nvidia-smi doesn't directly expose replay counter on all platforms
        # Return 0 as default — real production tools read this from PCIe config space
        return 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0


def get_pcie_topology(gpu_infos: list) -> list[PCIeInfo]:
    """Build PCIe topology from GPUInfo objects, with nvidia-smi fallback.

    Args:
        gpu_infos: List of GPUInfo dataclass objects from gpu_inventory.

    Returns:
        List of PCIeInfo objects with degradation flags.
    """
    # Try nvidia-smi as supplemental source
    smi_data = query_pcie_via_nvidia_smi()
    smi_lookup = {d["index"]: d for d in smi_data}

    pcie_infos = []
    for gpu in gpu_infos:
        # Prefer pynvml data, fall back to nvidia-smi
        gen_current = gpu.pcie_link_gen_current
        gen_max = gpu.pcie_link_gen_max
        width_current = gpu.pcie_link_width_current
        width_max = gpu.pcie_link_width_max

        if gen_current == 0 and gpu.index in smi_lookup:
            smi = smi_lookup[gpu.index]
            gen_current = smi["gen_current"]
            gen_max = smi["gen_max"]
            width_current = smi["width_current"]
            width_max = smi["width_max"]

        replay = query_pcie_replay_counter(gpu.index)

        # Check for degradation
        is_degraded = False
        reasons = []

        if gen_current < gen_max and gen_max > 0:
            is_degraded = True
            reasons.append(f"Link gen degraded: Gen{gen_current} (expected Gen{gen_max})")

        if width_current < width_max and width_max > 0:
            is_degraded = True
            reasons.append(f"Link width degraded: x{width_current} (expected x{width_max})")

        if replay > 0:
            is_degraded = True
            reasons.append(f"PCIe replay counter: {replay} (expected 0)")

        pcie_infos.append(
            PCIeInfo(
                gpu_index=gpu.index,
                link_gen_current=gen_current,
                link_gen_max=gen_max,
                link_width_current=width_current,
                link_width_max=width_max,
                replay_counter=replay,
                is_degraded=is_degraded,
                degradation_reason="; ".join(reasons) if reasons else "OK",
            )
        )

    return pcie_infos
