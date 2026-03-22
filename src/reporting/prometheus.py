"""Prometheus metrics exporter — GPU diagnostic metrics for Grafana.

Exposes GPU diagnostic results as Prometheus metrics via an HTTP
endpoint. Enables real-time monitoring dashboards and alerting
through standard observability infrastructure.

Metrics exposed:
  - gpu_diagnostic_status (gauge): 1=pass, 0=fail per test
  - gpu_diagnostic_duration_seconds (gauge): test execution time
  - gpu_temperature_celsius (gauge): current GPU temperature
  - gpu_power_draw_watts (gauge): current power consumption
  - gpu_memory_used_mib (gauge): VRAM usage
  - gpu_ecc_sbe_total (counter): single-bit ECC errors
  - gpu_ecc_dbe_total (counter): double-bit ECC errors
  - gpu_pcie_bandwidth_gibs (gauge): measured PCIe bandwidth
  - gpu_memory_bandwidth_gibs (gauge): measured memory bandwidth
  - gpu_diagnostic_run_total (counter): total diagnostic runs

Uses Python's built-in http.server — no external deps required.
For production, front with nginx or use the prometheus_client lib.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus


class MetricsStore:
    """Thread-safe store for Prometheus metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._gpu_metrics: dict[int, dict[str, float]] = {}
        self._test_results: list[TestResult] = []
        self._run_count = 0
        self._last_run_timestamp = 0.0

    def update_gpu_metrics(
        self, gpu_infos: list[GPUInfo],
    ) -> None:
        """Update GPU hardware metrics from inventory."""
        with self._lock:
            for gpu in gpu_infos:
                self._gpu_metrics[gpu.index] = {
                    "temperature_c": gpu.temperature_c,
                    "power_draw_w": gpu.power_draw_w,
                    "power_limit_w": gpu.power_limit_w,
                    "vram_total_mib": gpu.vram_total_mib,
                    "vram_used_mib": gpu.vram_used_mib,
                    "vram_free_mib": gpu.vram_free_mib,
                    "clock_graphics_mhz": gpu.clock_graphics_mhz,
                    "clock_memory_mhz": gpu.clock_memory_mhz,
                }

    def update_test_results(
        self, results: list[TestResult],
    ) -> None:
        """Update with latest diagnostic test results."""
        with self._lock:
            self._test_results = list(results)
            self._run_count += 1
            self._last_run_timestamp = time.time()

    def format_prometheus(self) -> str:
        """Format all metrics in Prometheus exposition format."""
        with self._lock:
            lines = []

            # GPU hardware metrics
            lines.append(
                "# HELP gpu_temperature_celsius "
                "Current GPU temperature"
            )
            lines.append(
                "# TYPE gpu_temperature_celsius gauge"
            )
            for idx, m in self._gpu_metrics.items():
                lines.append(
                    f'gpu_temperature_celsius'
                    f'{{gpu="{idx}"}} '
                    f'{m["temperature_c"]}'
                )

            lines.append(
                "# HELP gpu_power_draw_watts "
                "Current GPU power draw"
            )
            lines.append(
                "# TYPE gpu_power_draw_watts gauge"
            )
            for idx, m in self._gpu_metrics.items():
                lines.append(
                    f'gpu_power_draw_watts'
                    f'{{gpu="{idx}"}} '
                    f'{m["power_draw_w"]}'
                )

            lines.append(
                "# HELP gpu_memory_used_mib "
                "GPU VRAM usage in MiB"
            )
            lines.append(
                "# TYPE gpu_memory_used_mib gauge"
            )
            for idx, m in self._gpu_metrics.items():
                lines.append(
                    f'gpu_memory_used_mib'
                    f'{{gpu="{idx}"}} '
                    f'{m["vram_used_mib"]}'
                )

            lines.append(
                "# HELP gpu_clock_graphics_mhz "
                "GPU graphics clock in MHz"
            )
            lines.append(
                "# TYPE gpu_clock_graphics_mhz gauge"
            )
            for idx, m in self._gpu_metrics.items():
                lines.append(
                    f'gpu_clock_graphics_mhz'
                    f'{{gpu="{idx}"}} '
                    f'{m["clock_graphics_mhz"]}'
                )

            # Diagnostic test results
            lines.append(
                "# HELP gpu_diagnostic_status "
                "Diagnostic test status (1=pass, 0=fail)"
            )
            lines.append(
                "# TYPE gpu_diagnostic_status gauge"
            )
            for r in self._test_results:
                val = 1 if r.status == TestStatus.PASS else 0
                gpu_label = (
                    f',gpu_uuid="{r.gpu_uuid}"'
                    if r.gpu_uuid else ""
                )
                lines.append(
                    f'gpu_diagnostic_status'
                    f'{{test="{r.test_name}"'
                    f'{gpu_label}}} {val}'
                )

            lines.append(
                "# HELP gpu_diagnostic_duration_seconds "
                "Diagnostic test duration"
            )
            lines.append(
                "# TYPE gpu_diagnostic_duration_seconds gauge"
            )
            for r in self._test_results:
                lines.append(
                    f'gpu_diagnostic_duration_seconds'
                    f'{{test="{r.test_name}"}} '
                    f'{r.duration_seconds:.3f}'
                )

            # Run counter
            lines.append(
                "# HELP gpu_diagnostic_run_total "
                "Total diagnostic runs"
            )
            lines.append(
                "# TYPE gpu_diagnostic_run_total counter"
            )
            lines.append(
                f"gpu_diagnostic_run_total "
                f"{self._run_count}"
            )

            lines.append(
                "# HELP gpu_diagnostic_last_run_timestamp "
                "Timestamp of last diagnostic run"
            )
            lines.append(
                "# TYPE gpu_diagnostic_last_run_timestamp gauge"
            )
            lines.append(
                f"gpu_diagnostic_last_run_timestamp "
                f"{self._last_run_timestamp:.0f}"
            )

            return "\n".join(lines) + "\n"


# Global metrics store
_metrics_store = MetricsStore()


def get_metrics_store() -> MetricsStore:
    """Get the global MetricsStore singleton."""
    return _metrics_store


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus /metrics endpoint."""

    def _send_cors_headers(self):
        """Add CORS headers for dashboard access."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/metrics":
            body = _metrics_store.format_prometheus()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/plain; version=0.0.4; charset=utf-8",
            )
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(
                json.dumps({"status": "ok"}).encode("utf-8")
            )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default request logging."""
        pass


def start_metrics_server(
    port: int = 9835,
    daemon: bool = True,
) -> HTTPServer:
    """Start Prometheus metrics HTTP server.

    Args:
        port: HTTP port to listen on (default 9835).
        daemon: Run as daemon thread (exits with main process).

    Returns:
        HTTPServer instance.
    """
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    thread = threading.Thread(
        target=server.serve_forever,
        daemon=daemon,
    )
    thread.start()
    return server
