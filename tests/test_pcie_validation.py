"""Tests for PCIe topology detection and degradation flagging."""

import pytest

from src.inventory.pcie_topology import PCIeInfo, get_pcie_topology
from tests.conftest import MOCK_GPU_INFO, MOCK_GPU_INFO_DEGRADED


class TestPCIeTopology:
    """Test PCIe link validation logic."""

    def test_healthy_pcie_not_degraded(self, mock_gpu_info):
        pcie_infos = get_pcie_topology([mock_gpu_info])
        assert len(pcie_infos) == 1
        assert pcie_infos[0].is_degraded is False
        assert pcie_infos[0].link_gen_current == 4
        assert pcie_infos[0].link_width_current == 16

    def test_degraded_gen_detected(self, mock_gpu_info_degraded):
        pcie_infos = get_pcie_topology([mock_gpu_info_degraded])
        assert pcie_infos[0].is_degraded is True
        assert "gen degraded" in pcie_infos[0].degradation_reason.lower()

    def test_degraded_width_detected(self, mock_gpu_info_degraded):
        pcie_infos = get_pcie_topology([mock_gpu_info_degraded])
        assert "width degraded" in pcie_infos[0].degradation_reason.lower()

    def test_pcie_info_dataclass(self):
        info = PCIeInfo(
            gpu_index=0,
            link_gen_current=4,
            link_gen_max=4,
            link_width_current=16,
            link_width_max=16,
            replay_counter=0,
            is_degraded=False,
            degradation_reason="OK",
        )
        assert info.gpu_index == 0
        assert info.is_degraded is False
