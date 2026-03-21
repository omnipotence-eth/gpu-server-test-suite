"""Test runner orchestrator — manages diagnostic test execution.

Provides centralized test discovery, execution ordering, and
result aggregation. Handles the test registry, pre-flight checks,
burn-in mode, and fault injection routing.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import DiagnosticRun, TestResult, TestStatus


class TestRunner:
    """Orchestrates diagnostic test execution across run levels.

    Manages the test registry, executes tests in order, handles
    pre-flight health checks, burn-in overrides, and aggregates
    results into a DiagnosticRun report.
    """

    def __init__(
        self,
        gpu_infos: list[GPUInfo],
        config: dict[str, Any],
        profile: dict[str, Any],
    ):
        self.gpu_infos = gpu_infos
        self.config = config
        self.profile = profile
        self._registry: dict[str, Callable[[], list[TestResult]]] = {}
        self._register_tests()

    def _register_tests(self):
        """Register all available diagnostic test modules."""
        from src.diagnostics.clock_throttle import (
            run_clock_throttle_checks,
        )
        from src.diagnostics.compute_stress import run_compute_stress
        from src.diagnostics.deployment import run_deployment_checks
        from src.diagnostics.ecc_health import run_ecc_health_checks
        from src.diagnostics.gpu_cleanup import run_cleanup
        from src.diagnostics.gpu_health import run_gpu_health_checks
        from src.diagnostics.memory_bandwidth import (
            run_memory_bandwidth,
        )
        from src.diagnostics.memory_test import run_memory_test
        from src.diagnostics.nccl_validation import (
            run_nccl_validation,
        )
        from src.diagnostics.nvlink_p2p import run_nvlink_p2p
        from src.diagnostics.pcie_bandwidth import run_pcie_bandwidth
        from src.diagnostics.pcie_validation import run_pcie_validation
        from src.diagnostics.power_test import run_power_test
        from src.diagnostics.sm_stress import run_sm_stress
        from src.diagnostics.topology_map import run_topology_map
        from src.diagnostics.xid_errors import run_xid_checks

        gpu = self.gpu_infos
        cfg = self.config
        prof = self.profile

        self._registry = {
            # Level 1 — Deployment
            "deployment": lambda: run_deployment_checks(
                gpu, cfg, prof
            ),
            # Level 2 — Validation
            "gpu_health": lambda: run_gpu_health_checks(gpu, prof),
            "pcie_validation": lambda: run_pcie_validation(
                gpu, prof
            ),
            "memory_test": lambda: run_memory_test(gpu, prof),
            # Level 3 — Stress
            "pcie_bandwidth": lambda: run_pcie_bandwidth(gpu, prof),
            "memory_bandwidth": lambda: run_memory_bandwidth(
                gpu, prof
            ),
            "compute_stress": lambda: run_compute_stress(gpu, prof),
            "sm_stress": lambda: run_sm_stress(gpu, prof),
            "power_test": lambda: run_power_test(gpu, prof),
            # Advanced Telemetry
            "xid_errors": lambda: run_xid_checks(gpu, prof),
            "clock_throttle": lambda: run_clock_throttle_checks(
                gpu, prof
            ),
            "ecc_health": lambda: run_ecc_health_checks(gpu, prof),
            # Interconnect & Topology
            "nvlink_p2p": lambda: run_nvlink_p2p(gpu, prof),
            "nccl_validation": lambda: run_nccl_validation(
                gpu, prof
            ),
            "topology_map": lambda: run_topology_map(gpu, prof),
            # Cleanup
            "cleanup": lambda: run_cleanup(gpu, prof),
        }

    @property
    def available_tests(self) -> list[str]:
        """Return list of registered test names."""
        return list(self._registry.keys())

    def get_tests_for_level(self, level: str) -> list[str]:
        """Get ordered list of test names for a given run level."""
        run_levels = self.config.get("run_levels", {})
        return run_levels.get(level, [])

    def run_single_test(self, test_name: str) -> list[TestResult]:
        """Execute a single named test.

        Args:
            test_name: Name matching a registered test module.

        Returns:
            List of TestResult objects from the test.
        """
        if test_name not in self._registry:
            return [
                TestResult(
                    test_name=test_name,
                    status=TestStatus.SKIP,
                    duration_seconds=0.0,
                    message=(
                        f"Test module '{test_name}' not registered"
                    ),
                )
            ]

        try:
            return self._registry[test_name]()
        except Exception as e:
            return [
                TestResult(
                    test_name=test_name,
                    status=TestStatus.ERROR,
                    duration_seconds=0.0,
                    message=f"Test crashed: {e}",
                    failure_code="DIAG-ERR",
                )
            ]

    def _build_diagnostic_run(
        self, level, all_results, run_id, run_duration,
    ) -> DiagnosticRun:
        """Build a DiagnosticRun from collected results."""
        import socket

        any_fail = any(
            r.status in (TestStatus.FAIL, TestStatus.ERROR)
            for r in all_results
        )
        any_warn = any(
            r.status == TestStatus.WARN for r in all_results
        )

        if any_fail:
            overall = TestStatus.FAIL
        elif any_warn:
            overall = TestStatus.WARN
        else:
            overall = TestStatus.PASS

        return DiagnosticRun(
            run_id=run_id,
            run_level=level,
            hostname=socket.gethostname(),
            timestamp=datetime.now(timezone.utc),
            gpu_count=len(self.gpu_infos),
            overall_status=overall,
            duration_seconds=run_duration,
            results=all_results,
            system_info={},
            gpu_info=[
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "uuid": gpu.uuid,
                    "vram_total_mib": gpu.vram_total_mib,
                }
                for gpu in self.gpu_infos
            ],
        )

    def run_level(self, level: str) -> DiagnosticRun:
        """Execute all tests for a given run level.

        Args:
            level: One of quick, medium, long, extended.

        Returns:
            DiagnosticRun with aggregated results.
        """
        run_id = str(uuid.uuid4())
        run_start = time.time()
        tests_to_run = self.get_tests_for_level(level)

        all_results = []
        for test_name in tests_to_run:
            results = self.run_single_test(test_name)
            all_results.extend(results)

        run_duration = time.time() - run_start
        return self._build_diagnostic_run(
            level, all_results, run_id, run_duration,
        )

    def run_with_preflight(self, level: str) -> DiagnosticRun:
        """Execute tests with automatic pre-flight health check.

        Always runs gpu_health checks first. If any critical health
        check fails, aborts remaining tests.

        Args:
            level: Run level to execute.

        Returns:
            DiagnosticRun with aggregated results.
        """
        run_id = str(uuid.uuid4())
        run_start = time.time()
        all_results = []

        # Pre-flight health check
        health_results = self.run_single_test("gpu_health")
        all_results.extend(health_results)

        # Abort if any health check FAILed
        health_failed = any(
            r.status == TestStatus.FAIL for r in health_results
        )
        if health_failed:
            all_results.append(
                TestResult(
                    test_name="preflight.abort",
                    status=TestStatus.SKIP,
                    duration_seconds=0.0,
                    message=(
                        "Remaining tests skipped — "
                        "pre-flight health check failed"
                    ),
                )
            )
        else:
            # Run remaining tests for the level
            tests_to_run = self.get_tests_for_level(level)
            for test_name in tests_to_run:
                if test_name == "gpu_health":
                    continue  # Already ran
                results = self.run_single_test(test_name)
                all_results.extend(results)

        run_duration = time.time() - run_start
        return self._build_diagnostic_run(
            level, all_results, run_id, run_duration,
        )

    def run_with_cleanup(self, level: str) -> DiagnosticRun:
        """Execute tests with automatic cleanup afterwards.

        Runs the full level, then cleans up GPU state.

        Args:
            level: Run level to execute.

        Returns:
            DiagnosticRun with aggregated results (incl. cleanup).
        """
        run_id = str(uuid.uuid4())
        run_start = time.time()
        all_results = []

        # Run all level tests
        tests_to_run = self.get_tests_for_level(level)
        for test_name in tests_to_run:
            results = self.run_single_test(test_name)
            all_results.extend(results)

        # Cleanup
        cleanup_results = self.run_single_test("cleanup")
        all_results.extend(cleanup_results)

        run_duration = time.time() - run_start
        return self._build_diagnostic_run(
            level, all_results, run_id, run_duration,
        )
