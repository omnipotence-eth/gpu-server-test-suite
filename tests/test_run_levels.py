"""Tests for run level configuration and test selection.

Verifies that each diagnostic run level selects the correct set of
tests, matching the DCGM run level architecture.
"""

from pathlib import Path

import pytest
import yaml


class TestRunLevelConfig:
    """Verify run level definitions in test_config.yaml."""

    @pytest.fixture
    def config(self):
        config_path = Path(__file__).parent.parent / "config" / "test_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_quick_level_has_deployment_only(self, config):
        quick = config["run_levels"]["quick"]
        assert quick == ["deployment"]

    def test_medium_level_includes_pcie_and_memory(self, config):
        medium = config["run_levels"]["medium"]
        assert "deployment" in medium
        assert "pcie_validation" in medium
        assert "memory_test" in medium
        assert "xid_errors" in medium
        assert "clock_throttle" in medium
        assert "ecc_health" in medium
        assert len(medium) == 7

    def test_long_level_includes_stress_tests(self, config):
        long_level = config["run_levels"]["long"]
        assert "deployment" in long_level
        assert "compute_stress" in long_level
        assert "sm_stress" in long_level
        assert "power_test" in long_level
        assert "pcie_bandwidth" in long_level
        assert "memory_bandwidth" in long_level
        assert "memtest" not in long_level  # memtest is Level 4 only

    def test_extended_level_includes_nccl(self, config):
        extended = config["run_levels"]["extended"]
        assert "nccl_validation" in extended
        assert "memtest" not in extended  # module not yet implemented
        # Extended should include everything from long
        long_level = config["run_levels"]["long"]
        for test in long_level:
            assert test in extended

    def test_levels_are_cumulative(self, config):
        """Each level should be a superset of the previous level."""
        quick = set(config["run_levels"]["quick"])
        medium = set(config["run_levels"]["medium"])
        long_level = set(config["run_levels"]["long"])
        extended = set(config["run_levels"]["extended"])

        assert quick.issubset(medium)
        assert medium.issubset(long_level)
        assert long_level.issubset(extended)


class TestRunLevelSelection:
    """Verify correct tests are selected for each run level via config."""

    def test_run_level_from_fixture(self, mock_config):
        """Verify mock_config fixture has consistent run levels."""
        assert len(mock_config["run_levels"]["quick"]) == 1
        assert len(mock_config["run_levels"]["medium"]) == 7
        assert len(mock_config["run_levels"]["long"]) == 14
        assert len(mock_config["run_levels"]["extended"]) == 15

    def test_expected_gpu_count(self, mock_config):
        assert mock_config["expected"]["gpu_count"] == 1


class TestGPUProfile:
    """Verify GPU profile YAML structure."""

    @pytest.fixture
    def profile(self):
        profile_path = Path(__file__).parent.parent / "config" / "profiles" / "rtx_5070ti.yaml"
        with open(profile_path) as f:
            return yaml.safe_load(f)

    def test_profile_has_gpu_model(self, profile):
        assert "gpu_model" in profile
        assert "5070" in profile["gpu_model"]

    def test_profile_has_thresholds(self, profile):
        assert "thresholds" in profile
        thresholds = profile["thresholds"]
        assert "temp_warning_c" in thresholds
        assert "temp_critical_c" in thresholds
        assert "pcie_h2d_min_gibs" in thresholds
        assert "stress_duration_seconds" in thresholds

    def test_profile_thermal_ordering(self, profile):
        t = profile["thresholds"]
        assert t["temp_warning_c"] < t["temp_critical_c"] < t["temp_shutdown_c"]

    def test_profile_pcie_config(self, profile):
        assert profile["pcie_gen_expected"] == 4
        assert profile["pcie_width_expected"] == 16

    def test_h100_profile_exists(self):
        profile_path = Path(__file__).parent.parent / "config" / "profiles" / "h100_sxm.yaml"
        assert profile_path.exists()
        with open(profile_path) as f:
            h100 = yaml.safe_load(f)
        assert h100["ecc_supported"] is True
        assert h100["pcie_gen_expected"] == 5
