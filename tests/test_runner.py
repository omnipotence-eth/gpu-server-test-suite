"""Tests for the TestRunner orchestrator."""

from unittest.mock import MagicMock, patch

from src.reporting.models import DiagnosticRun, TestStatus
from src.reporting.test_runner import TestRunner


class TestRunnerRegistration:
    """Test that TestRunner registers all expected tests."""

    @patch("src.diagnostics.deployment.pynvml")
    def test_all_tests_registered(
        self, mock_nvml, mock_gpu_info, mock_config, mock_profile,
    ):
        mock_nvml.NVMLError = Exception
        runner = TestRunner(
            [mock_gpu_info], mock_config, mock_profile,
        )
        expected = [
            "deployment", "gpu_health", "pcie_validation",
            "memory_test", "pcie_bandwidth", "memory_bandwidth",
            "compute_stress", "sm_stress", "power_test",
            "xid_errors", "clock_throttle", "ecc_health",
            "nvlink_p2p", "nccl_validation", "topology_map",
            "cleanup",
        ]
        for test_name in expected:
            assert test_name in runner.available_tests

    @patch("src.diagnostics.deployment.pynvml")
    def test_available_tests_count(
        self, mock_nvml, mock_gpu_info, mock_config, mock_profile,
    ):
        mock_nvml.NVMLError = Exception
        runner = TestRunner(
            [mock_gpu_info], mock_config, mock_profile,
        )
        # 9 original + 6 new + 1 cleanup = 16
        assert len(runner.available_tests) == 16


class TestRunnerLevels:
    """Test run level test selection."""

    @patch("src.diagnostics.deployment.pynvml")
    def test_quick_level_tests(
        self, mock_nvml, mock_gpu_info, mock_config, mock_profile,
    ):
        mock_nvml.NVMLError = Exception
        runner = TestRunner(
            [mock_gpu_info], mock_config, mock_profile,
        )
        tests = runner.get_tests_for_level("quick")
        assert tests == ["deployment"]

    @patch("src.diagnostics.deployment.pynvml")
    def test_medium_level_tests(
        self, mock_nvml, mock_gpu_info, mock_config, mock_profile,
    ):
        mock_nvml.NVMLError = Exception
        runner = TestRunner(
            [mock_gpu_info], mock_config, mock_profile,
        )
        tests = runner.get_tests_for_level("medium")
        assert "deployment" in tests
        assert "gpu_health" in tests
        assert "pcie_validation" in tests
        assert "xid_errors" in tests
        assert "clock_throttle" in tests
        assert "ecc_health" in tests

    @patch("src.diagnostics.deployment.pynvml")
    def test_long_level_includes_interconnect(
        self, mock_nvml, mock_gpu_info, mock_config, mock_profile,
    ):
        mock_nvml.NVMLError = Exception
        runner = TestRunner(
            [mock_gpu_info], mock_config, mock_profile,
        )
        tests = runner.get_tests_for_level("long")
        assert "topology_map" in tests
        assert "nvlink_p2p" in tests

    @patch("src.diagnostics.deployment.pynvml")
    def test_unknown_level_empty(
        self, mock_nvml, mock_gpu_info, mock_config, mock_profile,
    ):
        mock_nvml.NVMLError = Exception
        runner = TestRunner(
            [mock_gpu_info], mock_config, mock_profile,
        )
        tests = runner.get_tests_for_level("nonexistent")
        assert tests == []


class TestRunnerExecution:
    """Test test execution logic."""

    @patch("src.diagnostics.deployment.pynvml")
    def test_unregistered_test_skips(
        self, mock_nvml, mock_gpu_info, mock_config, mock_profile,
    ):
        mock_nvml.NVMLError = Exception
        runner = TestRunner(
            [mock_gpu_info], mock_config, mock_profile,
        )
        results = runner.run_single_test("nonexistent_test")
        assert len(results) == 1
        assert results[0].status == TestStatus.SKIP

    @patch("src.diagnostics.deployment.pynvml")
    def test_run_level_returns_diagnostic_run(
        self, mock_nvml, mock_gpu_info, mock_config, mock_profile,
    ):
        mock_nvml.NVMLError = Exception
        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlShutdown.return_value = None
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "572.16"
        handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = handle
        mock_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        mock_nvml.nvmlDeviceGetPersistenceMode.return_value = 0
        mock_nvml.NVML_FEATURE_ENABLED = 1

        runner = TestRunner(
            [mock_gpu_info], mock_config, mock_profile,
        )
        run = runner.run_level("quick")

        assert isinstance(run, DiagnosticRun)
        assert run.run_level == "quick"
        assert run.gpu_count == 1
        assert len(run.results) >= 1
        assert run.run_id  # Should have a UUID

    @patch("src.diagnostics.deployment.pynvml")
    def test_run_level_overall_status(
        self, mock_nvml, mock_gpu_info, mock_config, mock_profile,
    ):
        mock_nvml.NVMLError = Exception
        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlShutdown.return_value = None
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "572.16"
        handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = handle
        mock_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        mock_nvml.nvmlDeviceGetPersistenceMode.return_value = 0
        mock_nvml.NVML_FEATURE_ENABLED = 1

        runner = TestRunner(
            [mock_gpu_info], mock_config, mock_profile,
        )
        run = runner.run_level("quick")
        assert run.overall_status in (
            TestStatus.PASS, TestStatus.WARN,
        )


class TestRunnerPreflight:
    """Test pre-flight health check integration."""

    @patch("src.diagnostics.deployment.pynvml")
    def test_preflight_runs_health_first(
        self, mock_nvml, mock_gpu_info, mock_config, mock_profile,
    ):
        mock_nvml.NVMLError = Exception
        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlShutdown.return_value = None
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "572.16"
        handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = handle
        mock_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        mock_nvml.nvmlDeviceGetPersistenceMode.return_value = 0
        mock_nvml.NVML_FEATURE_ENABLED = 1

        runner = TestRunner(
            [mock_gpu_info], mock_config, mock_profile,
        )
        run = runner.run_with_preflight("quick")

        health_results = [
            r for r in run.results
            if r.test_name.startswith("health.")
        ]
        assert len(health_results) >= 1

    @patch("src.diagnostics.deployment.pynvml")
    def test_preflight_gpu_info(
        self, mock_nvml, mock_gpu_info, mock_config, mock_profile,
    ):
        mock_nvml.NVMLError = Exception
        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlShutdown.return_value = None
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "572.16"
        handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = handle
        mock_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        mock_nvml.nvmlDeviceGetPersistenceMode.return_value = 0
        mock_nvml.NVML_FEATURE_ENABLED = 1

        runner = TestRunner(
            [mock_gpu_info], mock_config, mock_profile,
        )
        run = runner.run_with_preflight("quick")

        assert len(run.gpu_info) == 1
        assert run.gpu_info[0]["name"] == mock_gpu_info.name


class TestRunnerCleanup:
    """Test cleanup integration in test runner."""

    @patch("src.diagnostics.deployment.pynvml")
    def test_run_with_cleanup(
        self, mock_nvml, mock_gpu_info, mock_config, mock_profile,
    ):
        mock_nvml.NVMLError = Exception
        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlShutdown.return_value = None
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "572.16"
        handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = handle
        mock_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        mock_nvml.nvmlDeviceGetPersistenceMode.return_value = 0
        mock_nvml.NVML_FEATURE_ENABLED = 1

        runner = TestRunner(
            [mock_gpu_info], mock_config, mock_profile,
        )
        run = runner.run_with_cleanup("quick")

        cleanup_results = [
            r for r in run.results
            if r.test_name.startswith("cleanup.")
        ]
        assert len(cleanup_results) >= 1
