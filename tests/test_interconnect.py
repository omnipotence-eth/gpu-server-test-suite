"""Tests for interconnect: NVLink P2P, NCCL validation, topology mapping."""

from unittest.mock import MagicMock, patch

from src.diagnostics.nccl_validation import run_nccl_validation
from src.diagnostics.nvlink_p2p import (
    _check_p2p_access,
    run_nvlink_p2p,
)
from src.diagnostics.topology_map import (
    _parse_topo_matrix,
    run_topology_map,
)
from src.reporting.models import TestStatus
from tests.conftest import MOCK_GPU_INFO, MOCK_GPU_INFO_2


def _mock_no_cuda():
    m = MagicMock()
    m.cuda.is_available.return_value = False
    return m


class TestNVLinkP2P:
    """Test NVLink / P2P bandwidth module."""

    def test_skip_single_gpu(self, mock_profile):
        results = run_nvlink_p2p([MOCK_GPU_INFO], mock_profile)
        assert len(results) == 1
        assert results[0].status == TestStatus.SKIP
        assert "single GPU" in results[0].message

    def test_skip_no_torch(self, mock_profile):
        with patch("src.diagnostics.nvlink_p2p.torch", None):
            result = _check_p2p_access(0, 1)
        assert not result["supported"]

    def test_skip_no_cuda(self, mock_profile):
        with patch(
            "src.diagnostics.nvlink_p2p.torch", _mock_no_cuda(),
        ):
            result = _check_p2p_access(0, 1)
        assert not result["supported"]

    @patch("src.diagnostics.nvlink_p2p._test_gpu_pair")
    def test_multi_gpu_pairs(self, mock_pair, mock_profile):
        from src.reporting.models import TestResult

        mock_pair.return_value = [TestResult(
            test_name="interconnect.p2p_bandwidth",
            status=TestStatus.PASS,
            duration_seconds=0.1,
            message="OK",
        )]
        results = run_nvlink_p2p(
            [MOCK_GPU_INFO, MOCK_GPU_INFO_2], mock_profile,
        )
        assert len(results) >= 1
        mock_pair.assert_called_once()


class TestNCCLValidation:
    """Test NCCL collective operation validation."""

    def test_skip_single_gpu(self, mock_profile):
        results = run_nccl_validation(
            [MOCK_GPU_INFO], mock_profile,
        )
        assert len(results) == 1
        assert results[0].status == TestStatus.SKIP

    def test_skip_no_torch(self, mock_profile):
        with patch("src.diagnostics.nccl_validation.torch", None):
            results = run_nccl_validation(
                [MOCK_GPU_INFO, MOCK_GPU_INFO_2], mock_profile,
            )
        assert len(results) == 1
        assert results[0].status == TestStatus.SKIP

    def test_skip_no_cuda(self, mock_profile):
        with patch(
            "src.diagnostics.nccl_validation.torch",
            _mock_no_cuda(),
        ):
            results = run_nccl_validation(
                [MOCK_GPU_INFO, MOCK_GPU_INFO_2], mock_profile,
            )
        assert len(results) == 1
        assert results[0].status == TestStatus.SKIP


class TestTopologyMap:
    """Test GPU topology mapping module."""

    @patch("src.diagnostics.topology_map._query_numa_affinity")
    @patch("src.diagnostics.topology_map._query_nvidia_topo")
    def test_single_gpu_topology(
        self, mock_topo, mock_numa, mock_profile,
    ):
        mock_topo.return_value = ""
        mock_numa.return_value = {}
        results = run_topology_map(
            [MOCK_GPU_INFO], mock_profile,
        )
        assert len(results) == 1

    def test_parse_empty_topo(self):
        result = _parse_topo_matrix("")
        assert not result["parsed"]

    def test_parse_topo_matrix(self):
        topo_output = (
            "\tGPU0\tGPU1\n"
            "GPU0\t X\t PIX\n"
            "GPU1\t PIX\t X\n"
        )
        result = _parse_topo_matrix(topo_output)
        assert result["parsed"]
        assert result["gpu_count"] == 2
        assert not result["has_nvlink"]

    def test_parse_nvlink_topo(self):
        topo_output = (
            "\tGPU0\tGPU1\n"
            "GPU0\t X\t NV12\n"
            "GPU1\t NV12\t X\n"
        )
        result = _parse_topo_matrix(topo_output)
        assert result["parsed"]
        assert result["has_nvlink"]
        assert len(result["nvlink_pairs"]) == 2

    @patch("src.diagnostics.topology_map._query_numa_affinity")
    @patch("src.diagnostics.topology_map._query_nvidia_topo")
    def test_nvlink_expected_but_missing(
        self, mock_topo, mock_numa,
    ):
        mock_topo.return_value = (
            "\tGPU0\tGPU1\n"
            "GPU0\t X\t PIX\n"
            "GPU1\t PIX\t X\n"
        )
        mock_numa.return_value = {0: 0, 1: 0}
        profile = {"nvlink_expected": True}
        results = run_topology_map(
            [MOCK_GPU_INFO, MOCK_GPU_INFO_2], profile,
        )
        assert results[0].status == TestStatus.FAIL
        assert results[0].failure_code == "DIAG-970"
