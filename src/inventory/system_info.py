"""Host system inventory — CPU, RAM, OS, driver, CUDA version.

Collects non-GPU system information for diagnostic report context
and production traceability.
"""

import platform
import socket
from dataclasses import dataclass
from datetime import datetime, timezone

import psutil


@dataclass
class SystemInfo:
    """Host system hardware and software inventory."""

    hostname: str
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    ram_total_gib: float
    os_name: str
    os_version: str
    kernel_version: str
    driver_version: str
    cuda_version: str
    timestamp: str


def get_system_info(driver_version: str = "", cuda_version: str = "") -> SystemInfo:
    """Collect host system inventory.

    Args:
        driver_version: NVIDIA driver version (passed from GPU inventory).
        cuda_version: CUDA version string (passed from GPU inventory).

    Returns:
        SystemInfo dataclass with all host details.
    """
    # CPU info
    cpu_model = platform.processor() or "Unknown"

    # On Windows, platform.processor() returns a meaningful string.
    # On Linux, it may be empty — try /proc/cpuinfo.
    if not cpu_model or cpu_model == "Unknown":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.strip().startswith("model name"):
                        cpu_model = line.split(":")[1].strip()
                        break
        except (FileNotFoundError, PermissionError):
            cpu_model = platform.processor() or "Unknown"

    cpu_cores = psutil.cpu_count(logical=False) or 0
    cpu_threads = psutil.cpu_count(logical=True) or 0

    # RAM
    ram = psutil.virtual_memory()
    ram_total_gib = round(ram.total / (1024**3), 1)

    # OS
    os_name = f"{platform.system()} {platform.release()}"
    os_version = platform.version()
    kernel_version = platform.release()

    # Timestamp
    timestamp = datetime.now(timezone.utc).isoformat()

    return SystemInfo(
        hostname=socket.gethostname(),
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        ram_total_gib=ram_total_gib,
        os_name=os_name,
        os_version=os_version,
        kernel_version=kernel_version,
        driver_version=driver_version,
        cuda_version=cuda_version,
        timestamp=timestamp,
    )
