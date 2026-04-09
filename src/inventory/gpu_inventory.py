"""GPU inventory detection via pynvml.

Queries NVIDIA Management Library for all detected GPUs and returns
structured GPUInfo dataclass objects with hardware properties, thermal
state, power draw, and memory configuration.

This module wraps the same NVML C library that nvidia-smi and DCGM use.
"""

from dataclasses import dataclass, field

import pynvml


@dataclass
class GPUInfo:
    """Structured GPU hardware inventory data."""

    index: int
    name: str
    uuid: str
    serial: str
    vram_total_mib: int
    vram_free_mib: int
    vram_used_mib: int
    driver_version: str
    cuda_version: str
    ecc_mode: str  # "enabled", "disabled", "not_supported"
    temperature_c: int
    power_draw_w: float
    power_limit_w: float
    power_default_limit_w: float
    compute_capability: str
    pstate: str
    clock_graphics_mhz: int
    clock_graphics_max_mhz: int
    clock_memory_mhz: int
    clock_memory_max_mhz: int
    pcie_link_gen_current: int = 0
    pcie_link_gen_max: int = 0
    pcie_link_width_current: int = 0
    pcie_link_width_max: int = 0
    extra: dict = field(default_factory=dict)


def _safe_query(func, *args, default=None):
    """Call a pynvml function, returning default on any error."""
    try:
        return func(*args)
    except pynvml.NVMLError:
        return default


def get_gpu_count() -> int:
    """Return the number of NVIDIA GPUs detected by the driver."""
    pynvml.nvmlInit()
    try:
        return pynvml.nvmlDeviceGetCount()
    finally:
        pynvml.nvmlShutdown()


def get_gpu_info(index: int = 0) -> GPUInfo:
    """Query detailed hardware info for a single GPU by index."""
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)

        # Basic identification
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        uuid = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode("utf-8")

        serial = _safe_query(pynvml.nvmlDeviceGetSerial, handle, default="N/A")
        if isinstance(serial, bytes):
            serial = serial.decode("utf-8")

        # Driver and CUDA version
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode("utf-8")

        cuda_version_raw = _safe_query(pynvml.nvmlSystemGetCudaDriverVersion_v2, default=0)
        cuda_major = cuda_version_raw // 1000 if cuda_version_raw else 0
        cuda_minor = (cuda_version_raw % 1000) // 10 if cuda_version_raw else 0
        cuda_version = f"{cuda_major}.{cuda_minor}"

        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_total_mib = mem_info.total // (1024 * 1024)
        vram_free_mib = mem_info.free // (1024 * 1024)
        vram_used_mib = mem_info.used // (1024 * 1024)

        # ECC mode
        ecc_mode = "not_supported"
        try:
            ecc_current, _ = pynvml.nvmlDeviceGetEccMode(handle)
            ecc_mode = "enabled" if ecc_current == pynvml.NVML_FEATURE_ENABLED else "disabled"
        except pynvml.NVMLError:
            ecc_mode = "not_supported"

        # Thermal
        temperature_c = _safe_query(
            pynvml.nvmlDeviceGetTemperature,
            handle,
            pynvml.NVML_TEMPERATURE_GPU,
            default=0,
        )

        # Power
        power_draw_mw = _safe_query(pynvml.nvmlDeviceGetPowerUsage, handle, default=0)
        power_draw_w = power_draw_mw / 1000.0

        power_limit_mw = _safe_query(pynvml.nvmlDeviceGetPowerManagementLimit, handle, default=0)
        power_limit_w = power_limit_mw / 1000.0

        power_default_mw = _safe_query(
            pynvml.nvmlDeviceGetPowerManagementDefaultLimit, handle, default=0
        )
        power_default_limit_w = power_default_mw / 1000.0

        # Compute capability
        major = _safe_query(
            lambda h: pynvml.nvmlDeviceGetCudaComputeCapability(h)[0], handle, default=0
        )
        minor = _safe_query(
            lambda h: pynvml.nvmlDeviceGetCudaComputeCapability(h)[1], handle, default=0
        )
        compute_capability = f"{major}.{minor}"

        # P-state
        pstate_raw = _safe_query(pynvml.nvmlDeviceGetPerformanceState, handle, default=0)
        pstate = f"P{pstate_raw}"

        # Clocks
        clock_graphics_mhz = _safe_query(
            pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_GRAPHICS, default=0
        )
        clock_graphics_max_mhz = _safe_query(
            pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_GRAPHICS, default=0
        )
        clock_memory_mhz = _safe_query(
            pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM, default=0
        )
        clock_memory_max_mhz = _safe_query(
            pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_MEM, default=0
        )

        # PCIe link info via pynvml
        pcie_gen_current = _safe_query(
            pynvml.nvmlDeviceGetCurrPcieLinkGeneration, handle, default=0
        )
        pcie_gen_max = _safe_query(pynvml.nvmlDeviceGetMaxPcieLinkGeneration, handle, default=0)
        pcie_width_current = _safe_query(pynvml.nvmlDeviceGetCurrPcieLinkWidth, handle, default=0)
        pcie_width_max = _safe_query(pynvml.nvmlDeviceGetMaxPcieLinkWidth, handle, default=0)

        return GPUInfo(
            index=index,
            name=name,
            uuid=uuid,
            serial=serial,
            vram_total_mib=vram_total_mib,
            vram_free_mib=vram_free_mib,
            vram_used_mib=vram_used_mib,
            driver_version=driver_version,
            cuda_version=cuda_version,
            ecc_mode=ecc_mode,
            temperature_c=temperature_c,
            power_draw_w=power_draw_w,
            power_limit_w=power_limit_w,
            power_default_limit_w=power_default_limit_w,
            compute_capability=compute_capability,
            pstate=pstate,
            clock_graphics_mhz=clock_graphics_mhz,
            clock_graphics_max_mhz=clock_graphics_max_mhz,
            clock_memory_mhz=clock_memory_mhz,
            clock_memory_max_mhz=clock_memory_max_mhz,
            pcie_link_gen_current=pcie_gen_current,
            pcie_link_gen_max=pcie_gen_max,
            pcie_link_width_current=pcie_width_current,
            pcie_link_width_max=pcie_width_max,
        )
    finally:
        pynvml.nvmlShutdown()


def get_all_gpus() -> list[GPUInfo]:
    """Detect and query all NVIDIA GPUs in the system."""
    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
    finally:
        pynvml.nvmlShutdown()

    return [get_gpu_info(i) for i in range(count)]
