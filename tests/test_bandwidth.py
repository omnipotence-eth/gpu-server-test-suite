"""Tests for PCIe bandwidth and memory bandwidth diagnostics.

Tests patch the module-level torch reference to run without a real GPU.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.diagnostics.pcie_bandwidth import (
    _measure_d2h_bandwidth,
    _measure_h2d_bandwidth,
    run_pcie_bandwidth,
)
from src.diagnostics.memory_bandwidth import (
    _measure_memory_bandwidth,
    _measure_memory_bandwidth_triad,
    run_memory_bandwidth,
)
from src.reporting.models import TestStatus
from tests.conftest import MOCK_GPU_INFO


def _mock_no_cuda():
    """Create a mock torch with CUDA unavailable."""
    m = MagicMock()
    m.cuda.is_available.return_value = False
    return m


class TestPCIeBandwidthH2D:
    """Test Host-to-Device bandwidth measurement."""

    def test_h2d_skip_no_cuda(self):
        with patch("src.diagnostics.pcie_bandwidth.torch", _mock_no_cuda()):
            result = _measure_h2d_bandwidth(MOCK_GPU_INFO, min_gibs=20.0)
        assert result.status == TestStatus.SKIP

    def test_h2d_skip_no_torch(self):
        with patch("src.diagnostics.pcie_bandwidth.torch", None):
            result = _measure_h2d_bandwidth(MOCK_GPU_INFO, min_gibs=20.0)
        assert result.status == TestStatus.SKIP

    def test_h2d_result_has_correct_name(self):
        with patch("src.diagnostics.pcie_bandwidth.torch", _mock_no_cuda()):
            result = _measure_h2d_bandwidth(MOCK_GPU_INFO, min_gibs=20.0)
        assert result.test_name == "pcie_bandwidth.h2d"


class TestPCIeBandwidthD2H:
    """Test Device-to-Host bandwidth measurement."""

    def test_d2h_skip_no_cuda(self):
        with patch("src.diagnostics.pcie_bandwidth.torch", _mock_no_cuda()):
            result = _measure_d2h_bandwidth(MOCK_GPU_INFO, min_gibs=20.0)
        assert result.status == TestStatus.SKIP

    def test_d2h_result_has_correct_name(self):
        with patch("src.diagnostics.pcie_bandwidth.torch", _mock_no_cuda()):
            result = _measure_d2h_bandwidth(MOCK_GPU_INFO, min_gibs=20.0)
        assert result.test_name == "pcie_bandwidth.d2h"


class TestPCIeBandwidthSuite:
    """Integration test for PCIe bandwidth module."""

    def test_produces_two_results_per_gpu(self, mock_gpu_info, mock_profile):
        with patch("src.diagnostics.pcie_bandwidth.torch", _mock_no_cuda()):
            results = run_pcie_bandwidth([mock_gpu_info], mock_profile)
        assert len(results) == 2  # H2D + D2H per GPU

    def test_gpu_uuid_set(self, mock_gpu_info, mock_profile):
        with patch("src.diagnostics.pcie_bandwidth.torch", _mock_no_cuda()):
            results = run_pcie_bandwidth([mock_gpu_info], mock_profile)
        for r in results:
            assert r.gpu_uuid == mock_gpu_info.uuid


class TestMemoryBandwidthCopy:
    """Test STREAM copy memory bandwidth measurement."""

    def test_copy_skip_no_cuda(self):
        with patch("src.diagnostics.memory_bandwidth.torch", _mock_no_cuda()):
            result = _measure_memory_bandwidth(MOCK_GPU_INFO, min_gibs=400.0)
        assert result.status == TestStatus.SKIP
        assert result.test_name == "memory_bandwidth.stream_copy"

    def test_copy_skip_no_torch(self):
        with patch("src.diagnostics.memory_bandwidth.torch", None):
            result = _measure_memory_bandwidth(MOCK_GPU_INFO, min_gibs=400.0)
        assert result.status == TestStatus.SKIP


class TestMemoryBandwidthTriad:
    """Test STREAM triad memory bandwidth measurement."""

    def test_triad_skip_no_cuda(self):
        with patch("src.diagnostics.memory_bandwidth.torch", _mock_no_cuda()):
            result = _measure_memory_bandwidth_triad(MOCK_GPU_INFO, min_gibs=400.0)
        assert result.status == TestStatus.SKIP
        assert result.test_name == "memory_bandwidth.stream_triad"


class TestMemoryBandwidthSuite:
    """Integration test for memory bandwidth module."""

    def test_produces_two_results_per_gpu(self, mock_gpu_info, mock_profile):
        with patch("src.diagnostics.memory_bandwidth.torch", _mock_no_cuda()):
            results = run_memory_bandwidth([mock_gpu_info], mock_profile)
        assert len(results) == 2

    def test_result_names(self, mock_gpu_info, mock_profile):
        with patch("src.diagnostics.memory_bandwidth.torch", _mock_no_cuda()):
            results = run_memory_bandwidth([mock_gpu_info], mock_profile)
        names = [r.test_name for r in results]
        assert "memory_bandwidth.stream_copy" in names
        assert "memory_bandwidth.stream_triad" in names
