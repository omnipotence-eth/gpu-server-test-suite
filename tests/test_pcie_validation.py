"""Tests for PCIe topology detection and degradation flagging."""

from src.inventory.pcie_topology import PCIeInfo, get_pcie_topology


class TestPCIeTopology:
    """Test PCIe link validation logic."""

    def test_healthy_pcie_not_degraded(self, mock_gpu_info):
        pcie_infos = get_pcie_topology([mock_gpu_info])
        assert len(pcie_infos) == 1
        assert pcie_infos[0].is_degraded is False
        assert pcie_infos[0].link_gen_current == 4
        assert pcie_infos[0].link_width_current == 16

    def test_idle_gen_downshift_not_flagged(self, mock_gpu_info_degraded):
        """Gen downshift at idle is normal power-saving; not flagged as degraded."""
        pcie_infos = get_pcie_topology([mock_gpu_info_degraded])
        # is_degraded is True due to width, not gen
        assert "width degraded" in pcie_infos[0].degradation_reason.lower()
        assert "gen degraded" not in pcie_infos[0].degradation_reason.lower()

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
