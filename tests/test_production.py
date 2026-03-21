"""Tests for production integration: JUnit XML, Prometheus, GPU cleanup."""

from unittest.mock import patch
from xml.etree import ElementTree as ET

from src.diagnostics.gpu_cleanup import cleanup_gpu, run_cleanup
from src.reporting.junit_xml import (
    results_to_junit_xml,
    write_junit_xml,
)
from src.reporting.models import TestResult, TestStatus
from src.reporting.prometheus import MetricsStore
from tests.conftest import MOCK_GPU_INFO


def _make_results():
    """Create a sample list of TestResults for testing."""
    return [
        TestResult(
            test_name="health.temperature",
            status=TestStatus.PASS,
            duration_seconds=0.01,
            message="GPU temperature OK: 42C",
            gpu_uuid="GPU-1234",
        ),
        TestResult(
            test_name="health.power",
            status=TestStatus.FAIL,
            duration_seconds=0.02,
            message="Power too high",
            failure_code="DIAG-100",
        ),
        TestResult(
            test_name="memory.test",
            status=TestStatus.SKIP,
            duration_seconds=0.0,
            message="CUDA not available",
        ),
        TestResult(
            test_name="pcie.validation",
            status=TestStatus.WARN,
            duration_seconds=0.05,
            message="PCIe degraded",
        ),
        TestResult(
            test_name="stress.compute",
            status=TestStatus.ERROR,
            duration_seconds=1.0,
            message="Test crashed",
            failure_code="DIAG-ERR",
        ),
    ]


class TestJUnitXML:
    """Test JUnit XML output generation."""

    def test_basic_xml_output(self):
        results = _make_results()
        xml_str = results_to_junit_xml(results)
        assert '<?xml' in xml_str
        assert 'testsuite' in xml_str
        assert 'tests="5"' in xml_str

    def test_failure_count(self):
        results = _make_results()
        xml_str = results_to_junit_xml(results)
        assert 'failures="1"' in xml_str

    def test_error_count(self):
        results = _make_results()
        xml_str = results_to_junit_xml(results)
        assert 'errors="1"' in xml_str

    def test_skipped_count(self):
        results = _make_results()
        xml_str = results_to_junit_xml(results)
        assert 'skipped="1"' in xml_str

    def test_pass_has_no_children(self):
        results = [TestResult(
            test_name="test.pass",
            status=TestStatus.PASS,
            duration_seconds=0.01,
            message="OK",
        )]
        xml_str = results_to_junit_xml(results)
        root = ET.fromstring(xml_str)
        tc = root.find(".//testcase")
        assert tc is not None
        # PASS should have no failure/error/skipped children
        assert tc.find("failure") is None
        assert tc.find("error") is None
        assert tc.find("skipped") is None

    def test_fail_has_failure_element(self):
        results = [TestResult(
            test_name="test.fail",
            status=TestStatus.FAIL,
            duration_seconds=0.01,
            message="Failed check",
            failure_code="DIAG-001",
        )]
        xml_str = results_to_junit_xml(results)
        root = ET.fromstring(xml_str)
        failure = root.find(".//failure")
        assert failure is not None
        assert failure.get("message") == "Failed check"

    def test_warn_uses_system_out(self):
        results = [TestResult(
            test_name="test.warn",
            status=TestStatus.WARN,
            duration_seconds=0.01,
            message="Warning message",
        )]
        xml_str = results_to_junit_xml(results)
        root = ET.fromstring(xml_str)
        sysout = root.find(".//system-out")
        assert sysout is not None
        assert "WARNING" in sysout.text

    def test_gpu_uuid_as_property(self):
        results = [TestResult(
            test_name="test.gpu",
            status=TestStatus.PASS,
            duration_seconds=0.01,
            message="OK",
            gpu_uuid="GPU-1234",
        )]
        xml_str = results_to_junit_xml(results)
        root = ET.fromstring(xml_str)
        prop = root.find(".//property[@name='gpu_uuid']")
        assert prop is not None
        assert prop.get("value") == "GPU-1234"

    def test_write_junit_file(self, tmp_path):
        results = _make_results()
        output = tmp_path / "results.xml"
        write_junit_xml(results, str(output))
        assert output.exists()
        content = output.read_text()
        assert 'testsuite' in content

    def test_suite_name_custom(self):
        results = _make_results()
        xml_str = results_to_junit_xml(
            results, suite_name="my_suite",
        )
        assert 'name="my_suite"' in xml_str


class TestPrometheus:
    """Test Prometheus metrics exporter."""

    def test_metrics_store_empty(self):
        store = MetricsStore()
        output = store.format_prometheus()
        assert "gpu_diagnostic_run_total 0" in output

    def test_metrics_store_gpu_metrics(self):
        store = MetricsStore()
        store.update_gpu_metrics([MOCK_GPU_INFO])
        output = store.format_prometheus()
        assert 'gpu_temperature_celsius{gpu="0"}' in output
        assert "42" in output

    def test_metrics_store_power(self):
        store = MetricsStore()
        store.update_gpu_metrics([MOCK_GPU_INFO])
        output = store.format_prometheus()
        assert "gpu_power_draw_watts" in output
        assert "35.0" in output

    def test_metrics_store_vram(self):
        store = MetricsStore()
        store.update_gpu_metrics([MOCK_GPU_INFO])
        output = store.format_prometheus()
        assert "gpu_memory_used_mib" in output
        assert "584" in output

    def test_metrics_store_test_results(self):
        store = MetricsStore()
        results = [TestResult(
            test_name="health.temp",
            status=TestStatus.PASS,
            duration_seconds=0.01,
            message="OK",
        )]
        store.update_test_results(results)
        output = store.format_prometheus()
        assert "gpu_diagnostic_status" in output
        assert 'test="health.temp"' in output
        assert "gpu_diagnostic_run_total 1" in output

    def test_metrics_store_fail_status(self):
        store = MetricsStore()
        results = [TestResult(
            test_name="health.fail",
            status=TestStatus.FAIL,
            duration_seconds=0.01,
            message="Failed",
        )]
        store.update_test_results(results)
        output = store.format_prometheus()
        assert 'gpu_diagnostic_status{test="health.fail"} 0' in output

    def test_metrics_store_increment_run_count(self):
        store = MetricsStore()
        store.update_test_results([])
        store.update_test_results([])
        store.update_test_results([])
        output = store.format_prometheus()
        assert "gpu_diagnostic_run_total 3" in output

    def test_prometheus_format_headers(self):
        store = MetricsStore()
        output = store.format_prometheus()
        assert "# HELP" in output
        assert "# TYPE" in output


class TestGPUCleanup:
    """Test GPU state cleanup module."""

    @patch("src.diagnostics.gpu_cleanup._check_pending_retirement")
    @patch("src.diagnostics.gpu_cleanup._reset_power_limit")
    @patch("src.diagnostics.gpu_cleanup._reset_gpu_clocks")
    @patch("src.diagnostics.gpu_cleanup._cleanup_cuda_context")
    def test_cleanup_success(
        self, mock_cuda, mock_clocks, mock_power, mock_retire,
    ):
        mock_cuda.return_value = {"cuda_cleanup": "success"}
        mock_clocks.return_value = {"clock_reset": "success"}
        mock_power.return_value = {
            "power_reset": "already_default",
        }
        mock_retire.return_value = {
            "pending_retirement": False,
        }
        result = cleanup_gpu(MOCK_GPU_INFO)
        assert result.status == TestStatus.PASS
        assert result.test_name == "cleanup.gpu_reset"

    @patch("src.diagnostics.gpu_cleanup._check_pending_retirement")
    @patch("src.diagnostics.gpu_cleanup._reset_power_limit")
    @patch("src.diagnostics.gpu_cleanup._reset_gpu_clocks")
    @patch("src.diagnostics.gpu_cleanup._cleanup_cuda_context")
    def test_cleanup_with_errors(
        self, mock_cuda, mock_clocks, mock_power, mock_retire,
    ):
        mock_cuda.return_value = {
            "cuda_cleanup": "error: no device",
        }
        mock_clocks.return_value = {"clock_reset": "success"}
        mock_power.return_value = {
            "power_reset": "already_default",
        }
        mock_retire.return_value = {
            "pending_retirement": False,
        }
        result = cleanup_gpu(MOCK_GPU_INFO)
        assert result.status == TestStatus.WARN

    @patch("src.diagnostics.gpu_cleanup._check_pending_retirement")
    @patch("src.diagnostics.gpu_cleanup._reset_power_limit")
    @patch("src.diagnostics.gpu_cleanup._reset_gpu_clocks")
    @patch("src.diagnostics.gpu_cleanup._cleanup_cuda_context")
    def test_cleanup_reboot_needed(
        self, mock_cuda, mock_clocks, mock_power, mock_retire,
    ):
        mock_cuda.return_value = {"cuda_cleanup": "success"}
        mock_clocks.return_value = {"clock_reset": "success"}
        mock_power.return_value = {
            "power_reset": "already_default",
        }
        mock_retire.return_value = {
            "pending_retirement": True,
            "reboot_needed": True,
        }
        result = cleanup_gpu(MOCK_GPU_INFO)
        assert result.status == TestStatus.WARN
        assert "reboot" in result.message.lower()

    @patch("src.diagnostics.gpu_cleanup._check_pending_retirement")
    @patch("src.diagnostics.gpu_cleanup._reset_power_limit")
    @patch("src.diagnostics.gpu_cleanup._reset_gpu_clocks")
    @patch("src.diagnostics.gpu_cleanup._cleanup_cuda_context")
    def test_run_cleanup_multi_gpu(
        self, mock_cuda, mock_clocks, mock_power, mock_retire,
    ):
        mock_cuda.return_value = {"cuda_cleanup": "success"}
        mock_clocks.return_value = {"clock_reset": "success"}
        mock_power.return_value = {
            "power_reset": "already_default",
        }
        mock_retire.return_value = {
            "pending_retirement": False,
        }
        results = run_cleanup(
            [MOCK_GPU_INFO, MOCK_GPU_INFO], {},
        )
        assert len(results) == 2

    def test_cleanup_no_cuda(self):
        with patch("src.diagnostics.gpu_cleanup.torch", None):
            from src.diagnostics.gpu_cleanup import (
                _cleanup_cuda_context,
            )

            result = _cleanup_cuda_context(0)
        assert result["cuda_cleanup"] == "skipped"
