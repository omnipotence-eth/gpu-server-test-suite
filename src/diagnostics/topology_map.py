"""GPU topology mapping — PCIe root complex, NUMA, and NVLink detection.

Automatically detects the physical interconnect topology:
  - PCIe root complex / PLX switch grouping
  - NUMA node affinity (critical for CPU↔GPU data paths)
  - NVLink mesh connectivity (which GPUs are directly connected)
  - GPU affinity matrix (P2P capability between all pairs)

Understanding topology is critical for optimal workload placement.
A GPU on the wrong NUMA node can lose 30-40% memory bandwidth.
"""

import subprocess
import time
from typing import Any

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


def _query_nvidia_topo() -> str:
    """Run nvidia-smi topo -m to get the GPU topology matrix."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _parse_topo_matrix(topo_output: str) -> dict:
    """Parse nvidia-smi topo -m output into structured data.

    Topology legend:
      X   = Self
      SYS = Cross-socket (QPI/UPI)
      NODE = Same NUMA node
      PHB = Same PCIe host bridge
      PXB = Same PCIe switch (PLX)
      PIX = Same PCIe root complex
      NV#  = NVLink connection (#=link count)
    """
    if not topo_output.strip():
        return {"parsed": False, "reason": "No topology data"}

    lines = topo_output.strip().splitlines()
    if len(lines) < 2:
        return {"parsed": False, "reason": "Insufficient data"}

    # Find header line (contains GPU0, GPU1, etc.)
    header_idx = -1
    for i, line in enumerate(lines):
        if "GPU0" in line or "GPU 0" in line:
            header_idx = i
            break

    if header_idx < 0:
        return {"parsed": False, "reason": "Header not found"}

    headers = lines[header_idx].split()
    gpu_names = [
        h for h in headers if h.startswith("GPU")
    ]

    matrix = {}
    for line in lines[header_idx + 1:]:
        parts = line.split()
        if not parts or not parts[0].startswith("GPU"):
            continue
        src = parts[0]
        connections = {}
        for j, gpu_name in enumerate(gpu_names):
            if j + 1 < len(parts):
                connections[gpu_name] = parts[j + 1]
        matrix[src] = connections

    # Detect NVLink connections
    nvlink_pairs = []
    for src, conns in matrix.items():
        for dst, link_type in conns.items():
            if link_type.startswith("NV"):
                nvlink_pairs.append({
                    "src": src,
                    "dst": dst,
                    "link_type": link_type,
                })

    return {
        "parsed": True,
        "gpu_count": len(gpu_names),
        "matrix": matrix,
        "nvlink_pairs": nvlink_pairs,
        "has_nvlink": len(nvlink_pairs) > 0,
    }


def _query_numa_affinity() -> dict[int, int]:
    """Query NUMA node affinity for each GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,gpu_bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}

        affinity = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                gpu_idx = int(parts[0])
                bus_id = parts[1]
                # Try to read NUMA node from sysfs
                try:
                    numa_path = (
                        f"/sys/bus/pci/devices/{bus_id}/numa_node"
                    )
                    with open(numa_path) as f:
                        numa_node = int(f.read().strip())
                    affinity[gpu_idx] = numa_node
                except (FileNotFoundError, ValueError, OSError):
                    affinity[gpu_idx] = -1
        return affinity
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}


def _check_topology(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> TestResult:
    """Map and validate GPU topology."""
    start = time.time()

    topo_raw = _query_nvidia_topo()
    topo_parsed = _parse_topo_matrix(topo_raw)
    numa_affinity = _query_numa_affinity()

    details = {
        "gpu_count": len(gpu_infos),
        "topology": topo_parsed,
        "numa_affinity": numa_affinity,
        "raw_topo": topo_raw[:2000] if topo_raw else "",
    }

    nvlink_expected = profile.get("nvlink_expected", False)

    if not topo_parsed.get("parsed"):
        return TestResult(
            test_name="interconnect.topology_map",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=(
                f"Topology detection limited: "
                f"{topo_parsed.get('reason', 'unknown')}"
            ),
            details=details,
        )

    # Validate NVLink if expected
    if nvlink_expected and not topo_parsed.get("has_nvlink"):
        return TestResult(
            test_name="interconnect.topology_map",
            status=TestStatus.FAIL,
            duration_seconds=time.time() - start,
            message=(
                "NVLink expected but not detected in topology"
            ),
            failure_code="DIAG-970",
            details=details,
        )

    # Check NUMA alignment (all GPUs should be same NUMA on
    # single-socket, or evenly split on dual-socket)
    numa_nodes = set(numa_affinity.values())
    numa_warning = False
    if len(numa_nodes) > 1 and -1 not in numa_nodes:
        # Multiple NUMA nodes — check if balanced
        from collections import Counter

        counts = Counter(numa_affinity.values())
        if len(set(counts.values())) > 1:
            numa_warning = True

    if numa_warning:
        return TestResult(
            test_name="interconnect.topology_map",
            status=TestStatus.WARN,
            duration_seconds=time.time() - start,
            message=(
                f"Unbalanced NUMA affinity: {dict(numa_affinity)} "
                f"— may impact memory bandwidth"
            ),
            details=details,
        )

    nvlink_info = ""
    if topo_parsed.get("has_nvlink"):
        n = len(topo_parsed["nvlink_pairs"])
        nvlink_info = f", {n} NVLink connections"

    return TestResult(
        test_name="interconnect.topology_map",
        status=TestStatus.PASS,
        duration_seconds=time.time() - start,
        message=(
            f"Topology mapped: {len(gpu_infos)} GPU(s)"
            f"{nvlink_info}"
        ),
        details=details,
    )


def run_topology_map(
    gpu_infos: list[GPUInfo],
    profile: dict[str, Any],
) -> list[TestResult]:
    """Execute GPU topology mapping and validation."""
    return [_check_topology(gpu_infos, profile)]
