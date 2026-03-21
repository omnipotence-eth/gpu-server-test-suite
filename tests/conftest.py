"""Shared test fixtures — mock pynvml for CI without a GPU.

All pynvml calls are patched with realistic mock data so that
pytest runs in GitHub Actions (no NVIDIA GPU required).
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.inventory.gpu_inventory import GPUInfo

# ─── Realistic Mock GPU Data ────────────────────────────────────

MOCK_GPU_INFO = GPUInfo(
    index=0,
    name="NVIDIA GeForce RTX 5070 Ti",
    uuid="GPU-12345678-abcd-efgh-ijkl-123456789012",
    serial="N/A",
    vram_total_mib=16384,
    vram_free_mib=15800,
    vram_used_mib=584,
    driver_version="572.16",
    cuda_version="12.8",
    ecc_mode="not_supported",
    temperature_c=42,
    power_draw_w=35.0,
    power_limit_w=300.0,
    power_default_limit_w=300.0,
    compute_capability="12.0",
    pstate="P8",
    clock_graphics_mhz=210,
    clock_graphics_max_mhz=2632,
    clock_memory_mhz=1188,
    clock_memory_max_mhz=1500,
    pcie_link_gen_current=4,
    pcie_link_gen_max=4,
    pcie_link_width_current=16,
    pcie_link_width_max=16,
)

MOCK_GPU_INFO_2 = GPUInfo(
    index=1,
    name="NVIDIA GeForce RTX 5070 Ti",
    uuid="GPU-87654321-dcba-hgfe-lkji-210987654321",
    serial="N/A",
    vram_total_mib=16384,
    vram_free_mib=15800,
    vram_used_mib=584,
    driver_version="572.16",
    cuda_version="12.8",
    ecc_mode="not_supported",
    temperature_c=40,
    power_draw_w=33.0,
    power_limit_w=300.0,
    power_default_limit_w=300.0,
    compute_capability="12.0",
    pstate="P8",
    clock_graphics_mhz=210,
    clock_graphics_max_mhz=2632,
    clock_memory_mhz=1188,
    clock_memory_max_mhz=1500,
    pcie_link_gen_current=4,
    pcie_link_gen_max=4,
    pcie_link_width_current=16,
    pcie_link_width_max=16,
)

MOCK_GPU_INFO_ECC = GPUInfo(
    index=0,
    name="NVIDIA A100 80GB PCIe",
    uuid="GPU-A100-abcd-efgh-ijkl-123456789012",
    serial="1234567890",
    vram_total_mib=81920,
    vram_free_mib=80000,
    vram_used_mib=1920,
    driver_version="535.104",
    cuda_version="12.2",
    ecc_mode="enabled",
    temperature_c=38,
    power_draw_w=55.0,
    power_limit_w=300.0,
    power_default_limit_w=300.0,
    compute_capability="8.0",
    pstate="P0",
    clock_graphics_mhz=1410,
    clock_graphics_max_mhz=1410,
    clock_memory_mhz=1215,
    clock_memory_max_mhz=1215,
    pcie_link_gen_current=4,
    pcie_link_gen_max=4,
    pcie_link_width_current=16,
    pcie_link_width_max=16,
)

MOCK_GPU_INFO_DEGRADED = GPUInfo(
    index=0,
    name="NVIDIA GeForce RTX 5070 Ti",
    uuid="GPU-12345678-abcd-efgh-ijkl-123456789012",
    serial="N/A",
    vram_total_mib=16384,
    vram_free_mib=15800,
    vram_used_mib=584,
    driver_version="572.16",
    cuda_version="12.8",
    ecc_mode="not_supported",
    temperature_c=42,
    power_draw_w=35.0,
    power_limit_w=300.0,
    power_default_limit_w=300.0,
    compute_capability="12.0",
    pstate="P8",
    clock_graphics_mhz=210,
    clock_graphics_max_mhz=2632,
    clock_memory_mhz=1188,
    clock_memory_max_mhz=1500,
    pcie_link_gen_current=3,   # Degraded: Gen3 instead of Gen4
    pcie_link_gen_max=4,
    pcie_link_width_current=8,  # Degraded: x8 instead of x16
    pcie_link_width_max=16,
)


@pytest.fixture
def mock_gpu_info():
    """Provide a healthy mock GPUInfo for testing."""
    return MOCK_GPU_INFO


@pytest.fixture
def mock_gpu_info_2():
    """Provide a second mock GPUInfo for multi-GPU testing."""
    return MOCK_GPU_INFO_2


@pytest.fixture
def mock_gpu_info_ecc():
    """Provide an A100 mock GPUInfo with ECC for testing."""
    return MOCK_GPU_INFO_ECC


@pytest.fixture
def mock_gpu_info_degraded():
    """Provide a degraded-PCIe mock GPUInfo for testing."""
    return MOCK_GPU_INFO_DEGRADED


@pytest.fixture
def mock_config():
    """Provide a test configuration dict."""
    return {
        "gpu_profile": "rtx_5070ti",
        "expected": {"gpu_count": 1},
        "run_levels": {
            "quick": ["deployment"],
            "medium": [
                "deployment",
                "gpu_health",
                "pcie_validation",
                "memory_test",
                "xid_errors",
                "clock_throttle",
                "ecc_health",
            ],
            "long": [
                "deployment",
                "gpu_health",
                "pcie_validation",
                "memory_test",
                "xid_errors",
                "clock_throttle",
                "ecc_health",
                "topology_map",
                "pcie_bandwidth",
                "memory_bandwidth",
                "compute_stress",
                "sm_stress",
                "power_test",
                "nvlink_p2p",
            ],
            "extended": [
                "deployment",
                "gpu_health",
                "pcie_validation",
                "memory_test",
                "xid_errors",
                "clock_throttle",
                "ecc_health",
                "topology_map",
                "pcie_bandwidth",
                "memory_bandwidth",
                "compute_stress",
                "sm_stress",
                "power_test",
                "nvlink_p2p",
                "nccl_validation",
                "memtest",
            ],
        },
    }


@pytest.fixture
def mock_profile():
    """Provide a GPU profile dict for testing."""
    return {
        "gpu_model": "NVIDIA GeForce RTX 5070 Ti",
        "vram_total_mib": 16384,
        "pcie_gen_expected": 4,
        "pcie_width_expected": 16,
        "ecc_supported": False,
        "nvlink_expected": False,
        "tdp_watts": 300,
        "thresholds": {
            "temp_warning_c": 75,
            "temp_critical_c": 83,
            "pcie_h2d_min_gibs": 20.0,
            "pcie_d2h_min_gibs": 20.0,
            "pcie_retransmit_max": 0,
            "vram_test_allocation_pct": 90,
            "memory_bandwidth_min_gibs": 400.0,
            "stress_duration_seconds": 60,
            "stress_min_utilization_pct": 95,
            "sm_target_gflops": 20000,
            "sm_tolerance_pct": 10,
            "power_target_pct": 90,
            "power_duration_seconds": 30,
            "power_tolerance_pct": 5,
            "ecc_sbe_warn_count": 10,
            "nvlink_min_bw_gibs": 10.0,
            "nccl_allreduce_min_gibs": 5.0,
            "nccl_allgather_min_gibs": 4.0,
        },
    }


@pytest.fixture
def mock_profile_ecc():
    """Provide an A100 profile with ECC for testing."""
    return {
        "gpu_model": "NVIDIA A100 80GB PCIe",
        "vram_total_mib": 81920,
        "ecc_supported": True,
        "nvlink_expected": False,
        "tdp_watts": 300,
        "thresholds": {
            "ecc_sbe_warn_count": 10,
        },
    }


@pytest.fixture
def mock_profile_nvlink():
    """Provide an H100 profile expecting NVLink."""
    return {
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "nvlink_expected": True,
        "thresholds": {
            "nvlink_min_bw_gibs": 100.0,
            "nccl_allreduce_min_gibs": 200.0,
            "nccl_allgather_min_gibs": 150.0,
        },
    }


# ─── pynvml Mock Fixture ────────────────────────────────────────


@dataclass
class MockMemInfo:
    total: int = 16384 * 1024 * 1024
    free: int = 15800 * 1024 * 1024
    used: int = 584 * 1024 * 1024


@pytest.fixture
def mock_pynvml():
    """Patch pynvml globally with realistic mock responses."""
    with patch("src.inventory.gpu_inventory.pynvml") as mock_nvml:
        mock_handle = MagicMock()

        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlShutdown.return_value = None
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = (
            mock_handle
        )
        mock_nvml.nvmlDeviceGetName.return_value = (
            "NVIDIA GeForce RTX 5070 Ti"
        )
        mock_nvml.nvmlDeviceGetUUID.return_value = (
            "GPU-12345678-abcd-efgh-ijkl-123456789012"
        )
        mock_nvml.nvmlDeviceGetSerial.return_value = "N/A"
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "572.16"
        mock_nvml.nvmlSystemGetCudaDriverVersion_v2.return_value = (
            12080
        )
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = (
            MockMemInfo()
        )
        mock_nvml.nvmlDeviceGetEccMode.side_effect = (
            mock_nvml.NVMLError("Not supported")
        )
        mock_nvml.nvmlDeviceGetTemperature.return_value = 42
        mock_nvml.nvmlDeviceGetPowerUsage.return_value = 35000
        mock_nvml.nvmlDeviceGetPowerManagementLimit.return_value = (
            300000
        )
        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = (
            300000
        )
        mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (
            (12, 0)
        )
        mock_nvml.nvmlDeviceGetPerformanceState.return_value = 8
        mock_nvml.nvmlDeviceGetClockInfo.return_value = 210
        mock_nvml.nvmlDeviceGetMaxClockInfo.return_value = 2632
        mock_nvml.nvmlDeviceGetCurrPcieLinkGeneration.return_value = 4
        mock_nvml.nvmlDeviceGetMaxPcieLinkGeneration.return_value = 4
        mock_nvml.nvmlDeviceGetCurrPcieLinkWidth.return_value = 16
        mock_nvml.nvmlDeviceGetMaxPcieLinkWidth.return_value = 16

        mock_nvml.NVML_TEMPERATURE_GPU = 0
        mock_nvml.NVML_CLOCK_GRAPHICS = 0
        mock_nvml.NVML_CLOCK_MEM = 1
        mock_nvml.NVML_FEATURE_ENABLED = 1
        mock_nvml.NVMLError = Exception

        yield mock_nvml
