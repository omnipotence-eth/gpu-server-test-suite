"""Tests for the fault injection module.

Validates that each fault type returns a correctly-structured synthetic
TestResult and that codes are unique and distinct from real diagnostic codes.
"""

import pytest

from src.fault_injection import SUPPORTED_FAULTS, inject_fault
from src.reporting.models import TestStatus


class TestSupportedFaults:
    def test_five_fault_types(self):
        assert set(SUPPORTED_FAULTS) == {"thermal", "ecc", "pcie", "clock", "memory"}

    def test_invalid_fault_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown fault type"):
            inject_fault("not_a_fault")

    def test_invalid_fault_message_lists_valid_types(self):
        with pytest.raises(ValueError, match="thermal"):
            inject_fault("bogus")


class TestInjectFaultResult:
    @pytest.mark.parametrize("fault_type", SUPPORTED_FAULTS)
    def test_status_is_fail(self, fault_type):
        assert inject_fault(fault_type).status == TestStatus.FAIL

    @pytest.mark.parametrize("fault_type", SUPPORTED_FAULTS)
    def test_failure_code_has_fi_prefix(self, fault_type):
        code = inject_fault(fault_type).failure_code
        assert code.startswith("DIAG-FI-"), (
            f"{fault_type!r} code {code!r} does not start with DIAG-FI-"
        )

    @pytest.mark.parametrize("fault_type", SUPPORTED_FAULTS)
    def test_details_has_injected_flag(self, fault_type):
        assert inject_fault(fault_type).details.get("injected") is True

    @pytest.mark.parametrize("fault_type", SUPPORTED_FAULTS)
    def test_message_has_injected_prefix(self, fault_type):
        assert "[INJECTED]" in inject_fault(fault_type).message

    @pytest.mark.parametrize("fault_type", SUPPORTED_FAULTS)
    def test_test_name_matches_fault_type(self, fault_type):
        result = inject_fault(fault_type)
        assert fault_type in result.test_name


class TestFaultCodeIntegrity:
    def test_all_codes_are_unique(self):
        codes = [inject_fault(ft).failure_code for ft in SUPPORTED_FAULTS]
        assert len(codes) == len(set(codes)), "Duplicate failure codes across fault types"

    def test_codes_do_not_collide_with_real_diag_codes(self):
        real_prefixes = {
            "DIAG-001",
            "DIAG-002",
            "DIAG-003",
            "DIAG-004",
            "DIAG-100",
            "DIAG-200",
            "DIAG-300",
            "DIAG-400",
            "DIAG-401",
            "DIAG-500",
            "DIAG-600",
        }
        for fault_type in SUPPORTED_FAULTS:
            code = inject_fault(fault_type).failure_code
            assert code not in real_prefixes, (
                f"Injected code {code!r} collides with real diagnostic code"
            )
