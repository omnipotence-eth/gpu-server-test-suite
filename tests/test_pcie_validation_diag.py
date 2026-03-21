"""Tests for PCIe link validation diagnostic module."""

from unittest.mock import patch

import pytest

from src.diagnostics.pcie_validation import (
    _check_pcie_degradation_summary,
    _check_pcie_gen,
    _check_pcie_replay,
    _check_pcie_width,
    run_pcie_validation,
)
from src.inventory.pcie_topology import PCIeInfo
from src.reporting.models import TestStatus
from tests.conftest import MOCK_GPU_INFO, MOCK_GPU_INFO_DEGRADED


# ─── Mock PCIe Data ────────────────────────────────────────────────────────

HEALTHY_PCIE = PCIeInfo(
    gpu_index=0,
    link_gen_current=4,
    link_gen_max=4,
    link_width_current=16,
    link_width_max=16,
    replay_counter=0,
    is_degraded=False,
    degradation_reason="OK",
)

DEGRADED_PCIE = PCIeInfo(
    gpu_index=0,
    link_gen_current=3,
    link_gen_max=4,
    link_width_current=8,
    link_width_max=16,
    replay_counter=5,
    is_degraded=True,
    degradation_reason="Link gen degraded; Link width degraded",
)


class TestPCIeGenCheck:
    """Test PCIe link generation validation."""

    def test_gen4_matches_pass(self):
        result = _check_pcie_gen([HEALTHY_PCIE], expected_gen=4)
        assert result.status == TestStatus.PASS

    def test_gen3_below_gen4_fail(self):
        result = _check_pcie_gen([DEGRADED_PCIE], expected_gen=4)
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-200"

    def test_gen5_exceeds_gen4_pass(self):
        gen5 = PCIeInfo(
            gpu_index=0, link_gen_current=5, link_gen_max=5,
            link_width_current=16, link_width_max=16,
            replay_counter=0, is_degraded=False, degradation_reason="OK",
        )
        result = _check_pcie_gen([gen5], expected_gen=4)
        assert result.status == TestStatus.PASS


class TestPCIeWidthCheck:
    """Test PCIe link width validation."""

    def test_x16_matches_pass(self):
        result = _check_pcie_width([HEALTHY_PCIE], expected_width=16)
        assert result.status == TestStatus.PASS

    def test_x8_below_x16_fail(self):
        result = _check_pcie_width([DEGRADED_PCIE], expected_width=16)
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-201"


class TestPCIeReplayCheck:
    """Test PCIe replay counter validation."""

    def test_zero_replay_pass(self):
        result = _check_pcie_replay([HEALTHY_PCIE], max_replays=0)
        assert result.status == TestStatus.PASS

    def test_elevated_replay_warn(self):
        result = _check_pcie_replay([DEGRADED_PCIE], max_replays=0)
        assert result.status == TestStatus.WARN


class TestPCIeDegradationSummary:
    """Test overall degradation summary."""

    def test_no_degradation_pass(self):
        result = _check_pcie_degradation_summary([HEALTHY_PCIE])
        assert result.status == TestStatus.PASS

    def test_degraded_link_fail(self):
        result = _check_pcie_degradation_summary([DEGRADED_PCIE])
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-202"

    def test_mixed_gpus(self):
        healthy2 = PCIeInfo(
            gpu_index=1, link_gen_current=4, link_gen_max=4,
            link_width_current=16, link_width_max=16,
            replay_counter=0, is_degraded=False, degradation_reason="OK",
        )
        result = _check_pcie_degradation_summary([DEGRADED_PCIE, healthy2])
        assert result.status == TestStatus.FAIL
        assert "1/2" in result.message


class TestPCIeValidationSuite:
    """Integration test for full PCIe validation."""

    @patch("src.diagnostics.pcie_validation.get_pcie_topology")
    def test_healthy_pcie_all_pass(self, mock_topo, mock_gpu_info, mock_profile):
        mock_topo.return_value = [HEALTHY_PCIE]
        results = run_pcie_validation([mock_gpu_info], mock_profile)
        assert len(results) == 4
        assert all(r.status == TestStatus.PASS for r in results)

    @patch("src.diagnostics.pcie_validation.get_pcie_topology")
    def test_degraded_pcie_has_failures(self, mock_topo, mock_gpu_info, mock_profile):
        mock_topo.return_value = [DEGRADED_PCIE]
        results = run_pcie_validation([mock_gpu_info], mock_profile)
        fail_count = sum(1 for r in results if r.status == TestStatus.FAIL)
        assert fail_count >= 2  # gen, width, and degradation summary should fail

    @patch("src.diagnostics.pcie_validation.get_pcie_topology")
    def test_result_names(self, mock_topo, mock_gpu_info, mock_profile):
        mock_topo.return_value = [HEALTHY_PCIE]
        results = run_pcie_validation([mock_gpu_info], mock_profile)
        names = [r.test_name for r in results]
        assert "pcie_validation.link_gen" in names
        assert "pcie_validation.link_width" in names
        assert "pcie_validation.replay_counter" in names
        assert "pcie_validation.degradation_summary" in names
