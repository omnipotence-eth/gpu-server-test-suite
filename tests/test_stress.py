"""Tests for compute stress, SM stress, and power test diagnostics.

Tests patch the module-level torch reference to run without a real GPU.
"""

from unittest.mock import MagicMock, patch

from src.diagnostics.compute_stress import _run_compute_stress, run_compute_stress
from src.diagnostics.power_test import _run_power_stress, run_power_test
from src.diagnostics.sm_stress import (
    _measure_fp16_throughput,
    _measure_sm_throughput,
    run_sm_stress,
)
from src.reporting.models import TestStatus
from tests.conftest import MOCK_GPU_INFO


def _mock_no_cuda():
    m = MagicMock()
    m.cuda.is_available.return_value = False
    return m


class TestComputeStress:
    """Test sustained compute stress module."""

    def test_skip_no_cuda(self):
        with patch("src.diagnostics.compute_stress.torch", _mock_no_cuda()):
            result = _run_compute_stress(
                MOCK_GPU_INFO,
                duration_seconds=5,
                min_utilization_pct=95,
                profile={},
            )
        assert result.status == TestStatus.SKIP
        assert result.test_name == "compute_stress.sustained"

    def test_skip_no_torch(self):
        with patch("src.diagnostics.compute_stress.torch", None):
            result = _run_compute_stress(MOCK_GPU_INFO, 5, 95, {})
        assert result.status == TestStatus.SKIP

    def test_suite_produces_results(self, mock_gpu_info, mock_profile):
        with patch("src.diagnostics.compute_stress.torch", _mock_no_cuda()):
            results = run_compute_stress([mock_gpu_info], mock_profile)
        assert len(results) == 1
        assert results[0].gpu_uuid == mock_gpu_info.uuid


class TestSMStress:
    """Test SM throughput validation module."""

    def test_throughput_skip_no_cuda(self):
        with patch("src.diagnostics.sm_stress.torch", _mock_no_cuda()):
            result = _measure_sm_throughput(MOCK_GPU_INFO, target_gflops=20000, tolerance_pct=10)
        assert result.status == TestStatus.SKIP
        assert result.test_name == "sm_stress.throughput"

    def test_fp16_skip_no_cuda(self):
        with patch("src.diagnostics.sm_stress.torch", _mock_no_cuda()):
            result = _measure_fp16_throughput(MOCK_GPU_INFO, target_gflops=20000)
        assert result.status == TestStatus.SKIP
        assert result.test_name == "sm_stress.fp16_throughput"

    def test_suite_produces_results_per_gpu(self, mock_gpu_info, mock_profile):
        with patch("src.diagnostics.sm_stress.torch", _mock_no_cuda()):
            results = run_sm_stress([mock_gpu_info], mock_profile)
        assert len(results) == 2  # throughput + fp16 per GPU


class TestPowerTest:
    """Test targeted power draw module."""

    def test_skip_no_cuda(self):
        with patch("src.diagnostics.power_test.torch", _mock_no_cuda()):
            result = _run_power_stress(
                MOCK_GPU_INFO,
                target_pct=90,
                duration_seconds=5,
                tolerance_pct=5,
                profile={"tdp_watts": 300},
            )
        assert result.status == TestStatus.SKIP
        assert result.test_name == "power_test.sustained_power"

    def test_skip_no_torch(self):
        with patch("src.diagnostics.power_test.torch", None):
            result = _run_power_stress(MOCK_GPU_INFO, 90, 5, 5, {"tdp_watts": 300})
        assert result.status == TestStatus.SKIP

    def test_suite_produces_results(self, mock_gpu_info, mock_profile):
        with patch("src.diagnostics.power_test.torch", _mock_no_cuda()):
            results = run_power_test([mock_gpu_info], mock_profile)
        assert len(results) == 1


class TestStressResultAttributes:
    """Verify common attributes across all stress test results."""

    def test_compute_stress_has_gpu_uuid(self):
        with patch("src.diagnostics.compute_stress.torch", _mock_no_cuda()):
            result = _run_compute_stress(MOCK_GPU_INFO, 5, 95, {})
        assert result.gpu_uuid == MOCK_GPU_INFO.uuid

    def test_sm_stress_has_gpu_uuid(self):
        with patch("src.diagnostics.sm_stress.torch", _mock_no_cuda()):
            result = _measure_sm_throughput(MOCK_GPU_INFO, 20000, 10)
        assert result.gpu_uuid == MOCK_GPU_INFO.uuid

    def test_power_test_has_gpu_uuid(self):
        with patch("src.diagnostics.power_test.torch", _mock_no_cuda()):
            result = _run_power_stress(MOCK_GPU_INFO, 90, 5, 5, {"tdp_watts": 300})
        assert result.gpu_uuid == MOCK_GPU_INFO.uuid
