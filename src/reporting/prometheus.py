"""Prometheus metrics exporter — GPU diagnostic metrics for Grafana.

Exposes GPU diagnostic results as Prometheus metrics via an HTTP
endpoint. Enables real-time monitoring dashboards and alerting
through standard observability infrastructure.

Metrics exposed:
  - gpu_temperature_celsius (gauge): current GPU temperature
  - gpu_power_draw_watts (gauge): current power consumption
  - gpu_memory_used_mib (gauge): VRAM usage
  - gpu_clock_graphics_mhz (gauge): graphics clock frequency
  - gpu_ecc_sbe_total (gauge): volatile single-bit ECC errors
  - gpu_ecc_dbe_total (gauge): volatile double-bit ECC errors
  - gpu_diagnostic_status (gauge): 1=pass, 0=fail, 2=warn, 3=skip
  - gpu_diagnostic_duration_seconds (gauge): test execution time
  - gpu_diagnostic_run_total (counter): total diagnostic runs
  - gpu_diagnostic_last_run_timestamp (gauge): last run unix timestamp

Uses prometheus_client for correct exposition format and label handling.
Front the /metrics endpoint with nginx or a load balancer for production.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)

from src.inventory.gpu_inventory import GPUInfo
from src.reporting.models import TestResult, TestStatus

_STATUS_VALUES = {
    TestStatus.PASS: 1,
    TestStatus.FAIL: 0,
    TestStatus.WARN: 2,
    TestStatus.SKIP: 3,
    TestStatus.ERROR: 0,
}


class MetricsStore:
    """Thread-safe Prometheus metrics store backed by prometheus_client."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registry = CollectorRegistry()

        self._temp = Gauge(
            "gpu_temperature_celsius", "Current GPU temperature", ["gpu"],
            registry=self._registry,
        )
        self._power = Gauge(
            "gpu_power_draw_watts", "Current GPU power draw", ["gpu"],
            registry=self._registry,
        )
        self._mem_used = Gauge(
            "gpu_memory_used_mib", "GPU VRAM usage in MiB", ["gpu"],
            registry=self._registry,
        )
        self._clock = Gauge(
            "gpu_clock_graphics_mhz", "GPU graphics clock in MHz", ["gpu"],
            registry=self._registry,
        )
        self._ecc_sbe = Gauge(
            "gpu_ecc_sbe_total", "Volatile single-bit ECC error count", ["gpu_uuid"],
            registry=self._registry,
        )
        self._ecc_dbe = Gauge(
            "gpu_ecc_dbe_total", "Volatile double-bit ECC error count", ["gpu_uuid"],
            registry=self._registry,
        )
        self._diag_status = Gauge(
            "gpu_diagnostic_status",
            "Diagnostic test status (1=pass, 0=fail, 2=warn, 3=skip)",
            ["test", "gpu_uuid"],
            registry=self._registry,
        )
        self._diag_duration = Gauge(
            "gpu_diagnostic_duration_seconds", "Diagnostic test duration", ["test"],
            registry=self._registry,
        )
        # Named "gpu_diagnostic_run" so prometheus_client appends "_total"
        # yielding the metric name "gpu_diagnostic_run_total" in output.
        self._run_total = Counter(
            "gpu_diagnostic_run", "Total diagnostic runs",
            registry=self._registry,
        )
        self._last_run = Gauge(
            "gpu_diagnostic_last_run_timestamp", "Timestamp of last diagnostic run",
            registry=self._registry,
        )

    def update_gpu_metrics(self, gpu_infos: list[GPUInfo]) -> None:
        """Update GPU hardware metrics from inventory."""
        with self._lock:
            for gpu in gpu_infos:
                label = str(gpu.index)
                self._temp.labels(gpu=label).set(gpu.temperature_c)
                self._power.labels(gpu=label).set(gpu.power_draw_w)
                self._mem_used.labels(gpu=label).set(gpu.vram_used_mib)
                self._clock.labels(gpu=label).set(gpu.clock_graphics_mhz)

    def update_test_results(self, results: list[TestResult]) -> None:
        """Update with latest diagnostic test results.

        Increments the run counter, stamps the last-run timestamp, updates
        per-test status/duration gauges, and extracts ECC volatile error
        counts from ecc_health results.
        """
        with self._lock:
            self._run_total.inc()
            self._last_run.set(time.time())
            for r in results:
                gpu_label = r.gpu_uuid or ""
                self._diag_status.labels(
                    test=r.test_name, gpu_uuid=gpu_label
                ).set(_STATUS_VALUES.get(r.status, 0))
                self._diag_duration.labels(test=r.test_name).set(r.duration_seconds)

                if "ecc_health" in r.test_name and r.gpu_uuid and r.details:
                    volatile = r.details.get("volatile", {})
                    self._ecc_sbe.labels(gpu_uuid=r.gpu_uuid).set(
                        volatile.get("sbe", 0)
                    )
                    self._ecc_dbe.labels(gpu_uuid=r.gpu_uuid).set(
                        volatile.get("dbe", 0)
                    )

    def generate_latest(self) -> bytes:
        """Return Prometheus exposition format bytes."""
        return generate_latest(self._registry)

    def format_prometheus(self) -> str:
        """Return Prometheus exposition format as a UTF-8 string."""
        return self.generate_latest().decode("utf-8")


# Global metrics store singleton
_metrics_store = MetricsStore()


def get_metrics_store() -> MetricsStore:
    """Return the global MetricsStore singleton."""
    return _metrics_store


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus /metrics endpoint."""

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = _metrics_store.generate_latest()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:  # noqa: A002
        """Suppress default request logging."""


def start_metrics_server(
    port: int = 9835,
    host: str = "127.0.0.1",
    daemon: bool = True,
) -> HTTPServer:
    """Start Prometheus metrics HTTP server.

    Args:
        port: HTTP port to listen on (default 9835).
        host: Interface to bind (default 127.0.0.1). Pass "0.0.0.0"
              only when running inside a container where external
              Prometheus scraping is required.
        daemon: Run as daemon thread (exits with main process).

    Returns:
        HTTPServer instance.
    """
    server = HTTPServer((host, port), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=daemon)
    thread.start()
    return server
