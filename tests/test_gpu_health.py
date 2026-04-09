"""Tests for pre-flight GPU health checks."""

from dataclasses import replace

from src.diagnostics.gpu_health import (
    _check_clocks_responsive,
    _check_power_baseline,
    _check_temperature,
    _check_vram_available,
    run_gpu_health_checks,
)
from src.reporting.models import TestStatus


class TestTemperatureCheck:
    """Test GPU temperature validation logic."""

    def test_healthy_temp_pass(self, mock_gpu_info, mock_profile):
        result = _check_temperature([mock_gpu_info], mock_profile)
        assert result.status == TestStatus.PASS
        assert "42" in result.message or "OK" in result.message

    def test_warning_temp(self, mock_gpu_info, mock_profile):
        hot_gpu = replace(mock_gpu_info, temperature_c=78)
        result = _check_temperature([hot_gpu], mock_profile)
        assert result.status == TestStatus.WARN

    def test_critical_temp_fail(self, mock_gpu_info, mock_profile):
        burning_gpu = replace(mock_gpu_info, temperature_c=85)
        result = _check_temperature([burning_gpu], mock_profile)
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-100"

    def test_multiple_gpus_one_hot(self, mock_gpu_info, mock_profile):
        hot_gpu = replace(mock_gpu_info, index=1, temperature_c=85)
        result = _check_temperature([mock_gpu_info, hot_gpu], mock_profile)
        assert result.status == TestStatus.FAIL


class TestPowerBaseline:
    """Test idle power draw validation."""

    def test_idle_power_pass(self, mock_gpu_info, mock_profile):
        result = _check_power_baseline([mock_gpu_info], mock_profile)
        assert result.status == TestStatus.PASS
        assert "35" in result.message or "OK" in result.message

    def test_high_idle_power_warn(self, mock_gpu_info, mock_profile):
        busy_gpu = replace(mock_gpu_info, power_draw_w=200.0)
        result = _check_power_baseline([busy_gpu], mock_profile)
        assert result.status == TestStatus.WARN


class TestVRAMAvailable:
    """Test VRAM availability check."""

    def test_sufficient_vram_pass(self, mock_gpu_info, mock_profile):
        result = _check_vram_available([mock_gpu_info], mock_profile)
        assert result.status == TestStatus.PASS

    def test_insufficient_vram_warn(self, mock_gpu_info, mock_profile):
        low_vram = replace(mock_gpu_info, vram_free_mib=1000)
        result = _check_vram_available([low_vram], mock_profile)
        assert result.status == TestStatus.WARN


class TestClocksResponsive:
    """Test GPU clock sanity check."""

    def test_clocks_normal_pass(self, mock_gpu_info):
        result = _check_clocks_responsive([mock_gpu_info])
        assert result.status == TestStatus.PASS

    def test_clocks_stuck_fail(self, mock_gpu_info):
        stuck_gpu = replace(mock_gpu_info, clock_graphics_max_mhz=0, clock_memory_max_mhz=0)
        result = _check_clocks_responsive([stuck_gpu])
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-101"


class TestHealthSuite:
    """Integration test for full health check suite."""

    def test_all_healthy(self, mock_gpu_info, mock_profile):
        results = run_gpu_health_checks([mock_gpu_info], mock_profile)
        assert len(results) == 4
        assert all(r.status == TestStatus.PASS for r in results)

    def test_result_names(self, mock_gpu_info, mock_profile):
        results = run_gpu_health_checks([mock_gpu_info], mock_profile)
        names = [r.test_name for r in results]
        assert "health.temperature" in names
        assert "health.power_baseline" in names
        assert "health.vram_available" in names
        assert "health.clocks_responsive" in names
