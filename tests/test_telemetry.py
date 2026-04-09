"""Tests for advanced telemetry: XID errors, clock throttle, ECC health."""

from unittest.mock import patch

from src.diagnostics.clock_throttle import (
    _check_clock_throttling,
    run_clock_throttle_checks,
)
from src.diagnostics.ecc_health import (
    _check_ecc_health,
    run_ecc_health_checks,
)
from src.diagnostics.xid_errors import (
    _XID_RE,
    CRITICAL_XIDS,
    WARNING_XIDS,
    _check_xid_errors,
    run_xid_checks,
)
from src.reporting.models import TestStatus
from tests.conftest import MOCK_GPU_INFO, MOCK_GPU_INFO_ECC


class TestXIDErrors:
    """Test XID error tracking module."""

    @patch("src.diagnostics.xid_errors._query_xid_via_nvml")
    @patch("src.diagnostics.xid_errors._query_xid_from_dmesg")
    def test_no_xid_errors(self, mock_dmesg, mock_nvml):
        mock_dmesg.return_value = []
        mock_nvml.return_value = []
        result = _check_xid_errors(MOCK_GPU_INFO)
        assert result.status == TestStatus.PASS
        assert result.test_name == "telemetry.xid_errors"

    @patch("src.diagnostics.xid_errors._query_xid_via_nvml")
    @patch("src.diagnostics.xid_errors._query_xid_from_dmesg")
    def test_critical_xid_detected(self, mock_dmesg, mock_nvml):
        mock_dmesg.return_value = [
            {"xid": 79, "raw": "NVRM: Xid: 79, GPU off bus"},
        ]
        mock_nvml.return_value = []
        result = _check_xid_errors(MOCK_GPU_INFO)
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-900"
        assert "79" in result.message

    @patch("src.diagnostics.xid_errors._query_xid_via_nvml")
    @patch("src.diagnostics.xid_errors._query_xid_from_dmesg")
    def test_warning_xid_detected(self, mock_dmesg, mock_nvml):
        mock_dmesg.return_value = [
            {"xid": 92, "raw": "NVRM: Xid: 92, SBE rate"},
        ]
        mock_nvml.return_value = []
        result = _check_xid_errors(MOCK_GPU_INFO)
        assert result.status == TestStatus.WARN

    @patch("src.diagnostics.xid_errors._query_xid_via_nvml")
    @patch("src.diagnostics.xid_errors._query_xid_from_dmesg")
    def test_nvml_xid_source(self, mock_dmesg, mock_nvml):
        mock_dmesg.return_value = []
        mock_nvml.return_value = [
            {"xid": 64, "detail": "Uncorrectable remapped rows: 1"},
        ]
        result = _check_xid_errors(MOCK_GPU_INFO)
        assert result.status == TestStatus.FAIL
        assert result.details["total_count"] == 1

    @patch("src.diagnostics.xid_errors._query_xid_via_nvml")
    @patch("src.diagnostics.xid_errors._query_xid_from_dmesg")
    def test_run_xid_checks_multi_gpu(
        self,
        mock_dmesg,
        mock_nvml,
    ):
        mock_dmesg.return_value = []
        mock_nvml.return_value = []
        results = run_xid_checks(
            [MOCK_GPU_INFO, MOCK_GPU_INFO],
            {},
        )
        assert len(results) == 2

    def test_xid_classification(self):
        """Verify XID severity classification constants."""
        assert 79 in CRITICAL_XIDS  # GPU off bus
        assert 31 in CRITICAL_XIDS  # Memory page fault
        assert 92 in WARNING_XIDS  # SBE rate

    def test_dmesg_regex_parses_xid_not_pci_address(self):
        """Regex must extract the XID code, not PCI address bytes.

        Line format: "NVRM: Xid (PCI:0000:01:00): 31, pid=..."
        Old split-on-colon logic matched "0000" (PCI address) instead
        of "31" (the actual XID). The regex anchors after the closing
        paren to avoid this.
        """
        line = "2026-03-23T00:00:00 kernel: NVRM: Xid (PCI:0000:01:00): 31, pid=12345"
        match = _XID_RE.search(line)
        assert match is not None
        assert int(match.group(1)) == 31

    def test_dmesg_regex_parses_three_digit_xid(self):
        """Regex handles three-digit XID codes correctly."""
        line = "kernel: NVRM: Xid (PCI:0000:03:00): 119, pid=999"
        match = _XID_RE.search(line)
        assert match is not None
        assert int(match.group(1)) == 119

    def test_dmesg_regex_no_match_on_unrelated_line(self):
        """Regex returns None for lines without an XID event."""
        line = "kernel: NVRM: some other driver message: 00:01:00"
        assert _XID_RE.search(line) is None


class TestClockThrottle:
    """Test clock throttling analysis module."""

    @patch("src.diagnostics.clock_throttle._get_throttle_reasons")
    def test_no_throttling(self, mock_reasons):
        mock_reasons.return_value = {
            "throttle_bitmask": "0x0",
            "supported_bitmask": "0xff",
            "active_reasons": [],
            "graphics_clock_mhz": 2632,
            "max_graphics_clock_mhz": 2632,
            "clock_reduction_pct": 0,
        }
        result = _check_clock_throttling(MOCK_GPU_INFO, {})
        assert result.status == TestStatus.PASS

    @patch("src.diagnostics.clock_throttle._get_throttle_reasons")
    def test_thermal_throttling(self, mock_reasons):
        mock_reasons.return_value = {
            "throttle_bitmask": "0x40",
            "supported_bitmask": "0xff",
            "active_reasons": [
                {
                    "bit": "0x40",
                    "reason": "HW_THERMAL_SLOWDOWN",
                    "is_problem": True,
                }
            ],
            "graphics_clock_mhz": 1800,
            "max_graphics_clock_mhz": 2632,
            "clock_reduction_pct": 31.6,
        }
        result = _check_clock_throttling(MOCK_GPU_INFO, {})
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-910"
        assert "HW_THERMAL_SLOWDOWN" in result.message

    @patch("src.diagnostics.clock_throttle._get_throttle_reasons")
    def test_power_brake(self, mock_reasons):
        mock_reasons.return_value = {
            "throttle_bitmask": "0x80",
            "supported_bitmask": "0xff",
            "active_reasons": [
                {
                    "bit": "0x80",
                    "reason": "HW_POWER_BRAKE_SLOWDOWN",
                    "is_problem": True,
                }
            ],
            "graphics_clock_mhz": 2000,
            "max_graphics_clock_mhz": 2632,
            "clock_reduction_pct": 24.0,
        }
        result = _check_clock_throttling(MOCK_GPU_INFO, {})
        assert result.status == TestStatus.FAIL
        assert "POWER_BRAKE" in result.message

    @patch("src.diagnostics.clock_throttle._get_throttle_reasons")
    def test_app_clock_limiting(self, mock_reasons):
        # APPLICATIONS_CLOCKS_SETTING is a normal operating state — an app
        # requested specific clocks (common after compute workloads). It is
        # not a hardware problem and should not trigger WARN.
        mock_reasons.return_value = {
            "throttle_bitmask": "0x2",
            "supported_bitmask": "0xff",
            "active_reasons": [
                {
                    "bit": "0x2",
                    "reason": "APPLICATIONS_CLOCKS_SETTING",
                    "is_problem": False,
                }
            ],
            "graphics_clock_mhz": 2100,
            "max_graphics_clock_mhz": 2632,
            "clock_reduction_pct": 20.2,
        }
        result = _check_clock_throttling(MOCK_GPU_INFO, {})
        assert result.status == TestStatus.PASS

    @patch("src.diagnostics.clock_throttle._get_throttle_reasons")
    def test_skip_on_error(self, mock_reasons):
        mock_reasons.return_value = {
            "error": "NVML not available",
            "active_reasons": [],
        }
        result = _check_clock_throttling(MOCK_GPU_INFO, {})
        assert result.status == TestStatus.SKIP

    @patch("src.diagnostics.clock_throttle._get_throttle_reasons")
    def test_run_multi_gpu(self, mock_reasons):
        mock_reasons.return_value = {
            "throttle_bitmask": "0x0",
            "supported_bitmask": "0xff",
            "active_reasons": [],
            "graphics_clock_mhz": 2632,
            "max_graphics_clock_mhz": 2632,
            "clock_reduction_pct": 0,
        }
        results = run_clock_throttle_checks(
            [MOCK_GPU_INFO, MOCK_GPU_INFO],
            {},
        )
        assert len(results) == 2


class TestECCHealth:
    """Test ECC memory health module."""

    def test_skip_ecc_not_supported(self):
        result = _check_ecc_health(
            MOCK_GPU_INFO,
            {"ecc_supported": False},
        )
        assert result.status == TestStatus.SKIP

    @patch("src.diagnostics.ecc_health._query_ecc_counters")
    def test_ecc_healthy(self, mock_counters):
        mock_counters.return_value = {
            "ecc_supported": True,
            "volatile": {"sbe": 0, "dbe": 0},
            "aggregate": {"sbe": 0, "dbe": 0},
            "retired_pages": {
                "sbe_caused": 0,
                "dbe_caused": 0,
                "pending_retirement": False,
            },
            "remapped_rows": {},
        }
        result = _check_ecc_health(
            MOCK_GPU_INFO_ECC,
            {"ecc_supported": True, "thresholds": {}},
        )
        assert result.status == TestStatus.PASS

    @patch("src.diagnostics.ecc_health._query_ecc_counters")
    def test_dbe_detected(self, mock_counters):
        mock_counters.return_value = {
            "ecc_supported": True,
            "volatile": {"sbe": 0, "dbe": 3},
            "aggregate": {"sbe": 0, "dbe": 3},
            "retired_pages": {
                "sbe_caused": 0,
                "dbe_caused": 0,
                "pending_retirement": False,
            },
            "remapped_rows": {},
        }
        result = _check_ecc_health(
            MOCK_GPU_INFO_ECC,
            {"ecc_supported": True, "thresholds": {}},
        )
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-920"
        assert "Double-bit" in result.message

    @patch("src.diagnostics.ecc_health._query_ecc_counters")
    def test_high_sbe_count(self, mock_counters):
        mock_counters.return_value = {
            "ecc_supported": True,
            "volatile": {"sbe": 25, "dbe": 0},
            "aggregate": {"sbe": 25, "dbe": 0},
            "retired_pages": {
                "sbe_caused": 2,
                "dbe_caused": 0,
                "pending_retirement": False,
            },
            "remapped_rows": {},
        }
        result = _check_ecc_health(
            MOCK_GPU_INFO_ECC,
            {
                "ecc_supported": True,
                "thresholds": {"ecc_sbe_warn_count": 10},
            },
        )
        assert result.status == TestStatus.WARN
        assert "SBE" in result.message

    @patch("src.diagnostics.ecc_health._query_ecc_counters")
    def test_row_remapping_failure(self, mock_counters):
        mock_counters.return_value = {
            "ecc_supported": True,
            "volatile": {"sbe": 0, "dbe": 0},
            "aggregate": {"sbe": 0, "dbe": 0},
            "retired_pages": {
                "sbe_caused": 0,
                "dbe_caused": 0,
                "pending_retirement": False,
            },
            "remapped_rows": {"failure": True},
        }
        result = _check_ecc_health(
            MOCK_GPU_INFO_ECC,
            {"ecc_supported": True, "thresholds": {}},
        )
        assert result.status == TestStatus.FAIL
        assert result.failure_code == "DIAG-921"

    @patch("src.diagnostics.ecc_health._query_ecc_counters")
    def test_pending_retirement(self, mock_counters):
        mock_counters.return_value = {
            "ecc_supported": True,
            "volatile": {"sbe": 2, "dbe": 0},
            "aggregate": {"sbe": 2, "dbe": 0},
            "retired_pages": {
                "sbe_caused": 1,
                "dbe_caused": 0,
                "pending_retirement": True,
            },
            "remapped_rows": {},
        }
        result = _check_ecc_health(
            MOCK_GPU_INFO_ECC,
            {"ecc_supported": True, "thresholds": {}},
        )
        assert result.status == TestStatus.WARN
        assert "reboot" in result.message.lower()

    @patch("src.diagnostics.ecc_health._query_ecc_counters")
    def test_run_multi_gpu(self, mock_counters):
        mock_counters.return_value = {
            "ecc_supported": True,
            "volatile": {"sbe": 0, "dbe": 0},
            "aggregate": {"sbe": 0, "dbe": 0},
            "retired_pages": {
                "sbe_caused": 0,
                "dbe_caused": 0,
                "pending_retirement": False,
            },
            "remapped_rows": {},
        }
        results = run_ecc_health_checks(
            [MOCK_GPU_INFO_ECC, MOCK_GPU_INFO_ECC],
            {"ecc_supported": True, "thresholds": {}},
        )
        assert len(results) == 2
