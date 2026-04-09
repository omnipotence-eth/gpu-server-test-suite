"""Tests for DCGM Level 1 deployment checks.

Validates that the deployment diagnostic module correctly identifies
pass/fail conditions using mocked GPU data — no real GPU required.
"""

from unittest.mock import MagicMock, patch

from src.diagnostics.deployment import (
    _check_driver_loaded,
    _check_ecc_mode,
    _check_gpu_count,
    _check_gpu_model,
    run_deployment_checks,
)
from src.reporting.models import TestStatus


class TestDriverLoaded:
    """Test driver detection logic."""

    @patch("src.diagnostics.deployment.pynvml")
    def test_driver_loaded_pass(self, mock_nvml):
        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "572.16"
        mock_nvml.nvmlShutdown.return_value = None

        result = _check_driver_loaded()
        assert result.status == TestStatus.PASS
        assert "572.16" in result.message

    @patch("src.diagnostics.deployment.pynvml")
    def test_driver_not_loaded_fail(self, mock_nvml):
        mock_nvml.NVMLError = Exception
        mock_nvml.nvmlInit.side_effect = Exception("Driver not loaded")

        result = _check_driver_loaded()
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-001"


class TestGPUCount:
    """Test GPU count validation."""

    def test_correct_count_pass(self, mock_gpu_info):
        result = _check_gpu_count([mock_gpu_info], expected_count=1)
        assert result.status == TestStatus.PASS

    def test_wrong_count_fail(self, mock_gpu_info):
        result = _check_gpu_count([mock_gpu_info], expected_count=8)
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-002"
        assert "8 expected" in result.message

    def test_zero_gpus_fail(self):
        result = _check_gpu_count([], expected_count=1)
        assert result.status == TestStatus.FAIL


class TestGPUModel:
    """Test GPU model matching."""

    def test_correct_model_pass(self, mock_gpu_info):
        result = _check_gpu_model([mock_gpu_info], "RTX 5070 Ti")
        assert result.status == TestStatus.PASS

    def test_wrong_model_fail(self, mock_gpu_info):
        result = _check_gpu_model([mock_gpu_info], "H100 80GB")
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-003"

    def test_case_insensitive_match(self, mock_gpu_info):
        result = _check_gpu_model([mock_gpu_info], "rtx 5070 ti")
        assert result.status == TestStatus.PASS


class TestECCMode:
    """Test ECC mode validation."""

    def test_ecc_not_supported_skip(self, mock_gpu_info):
        profile = {"ecc_supported": False}
        result = _check_ecc_mode([mock_gpu_info], profile)
        assert result.status == TestStatus.SKIP

    def test_ecc_enabled_pass(self, mock_gpu_info):
        gpu = mock_gpu_info
        # Create a copy with ECC enabled
        from dataclasses import replace

        gpu_ecc = replace(gpu, ecc_mode="enabled")
        profile = {"ecc_supported": True, "ecc_expected": True}
        result = _check_ecc_mode([gpu_ecc], profile)
        assert result.status == TestStatus.PASS

    def test_ecc_wrong_mode_fail(self, mock_gpu_info):
        from dataclasses import replace

        gpu_ecc = replace(mock_gpu_info, ecc_mode="disabled")
        profile = {"ecc_supported": True, "ecc_expected": True}
        result = _check_ecc_mode([gpu_ecc], profile)
        assert result.status == TestStatus.FAIL


class TestDeploymentSuite:
    """Integration test for the full deployment check suite."""

    @patch("src.diagnostics.deployment.pynvml")
    def test_full_deployment_pass(self, mock_nvml, mock_gpu_info, mock_config, mock_profile):
        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlShutdown.return_value = None
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "572.16"
        mock_nvml.NVMLError = Exception

        mock_handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        mock_nvml.nvmlDeviceGetPersistenceMode.return_value = 0
        mock_nvml.NVML_FEATURE_ENABLED = 1

        results = run_deployment_checks([mock_gpu_info], mock_config, mock_profile)

        assert len(results) >= 5
        # Driver check should pass
        driver_result = next(r for r in results if "driver" in r.test_name)
        assert driver_result.status == TestStatus.PASS

        # GPU count should pass
        count_result = next(r for r in results if "count" in r.test_name)
        assert count_result.status == TestStatus.PASS

        # GPU model should pass
        model_result = next(r for r in results if "model" in r.test_name)
        assert model_result.status == TestStatus.PASS
