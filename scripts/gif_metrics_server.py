"""Temporary metrics server that progressively injects test results.

Used to capture dashboard screenshots for the README GIF.
Starts a real metrics server with live GPU data, then injects
test results one at a time with delays to simulate a running diagnostic.
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reporting.prometheus import get_metrics_store, start_metrics_server
from src.reporting.models import TestResult, TestStatus
from src.inventory.gpu_inventory import get_all_gpus

# Get real GPU data
gpus = get_all_gpus()
store = get_metrics_store()
store.update_gpu_metrics(gpus)

# Start server
server = start_metrics_server(port=9835, daemon=True)
print("Metrics server started on :9835")
print("Dashboard: http://localhost:8080/dashboard.html")

# Simulated test results that mirror a real medium-level run
test_sequence = [
    ("deployment.driver_loaded", TestStatus.PASS, 0.02, "NVIDIA driver loaded: v595.71"),
    ("deployment.gpu_count", TestStatus.PASS, 0.00, "GPU count matches: 1 detected, 1 expected"),
    ("deployment.gpu_model", TestStatus.PASS, 0.00, "GPU model matches: NVIDIA GeForce RTX 5070 Ti"),
    ("deployment.ecc_mode", TestStatus.SKIP, 0.00, "ECC not supported on this GPU model"),
    ("deployment.gpu_processes", TestStatus.WARN, 0.02, "GPU(s) in use by 21 process(es)"),
    ("deployment.persistence_mode", TestStatus.PASS, 0.01, "Persistence mode check complete"),
    ("gpu_health.temperature", TestStatus.PASS, 0.01, "GPU 0: 36C (limit: 90C)"),
    ("gpu_health.power", TestStatus.PASS, 0.01, "GPU 0: 40.6W (limit: 300W)"),
    ("gpu_health.vram_usage", TestStatus.PASS, 0.00, "GPU 0: 1319/16384 MiB (8.1%)"),
    ("pcie_validation.gen_check", TestStatus.PASS, 0.00, "All GPUs at expected PCIe gen"),
    ("pcie_validation.width_check", TestStatus.PASS, 0.00, "All GPUs at x16 width"),
    ("pcie_validation.replay_counters", TestStatus.PASS, 0.00, "PCIe replay counters OK"),
    ("memory_test.vram_allocation", TestStatus.PASS, 0.39, "Allocated and verified 14672 MiB on GPU 0"),
    ("memory_test.pattern_test", TestStatus.PASS, 0.01, "All 4 VRAM patterns verified on GPU 0"),
    ("telemetry.xid_errors", TestStatus.PASS, 0.01, "No XID errors detected"),
    ("telemetry.clock_throttle", TestStatus.WARN, 0.02, "Clock limiting active: SW_POWER_CAP"),
    ("telemetry.ecc_health", TestStatus.SKIP, 0.00, "ECC not supported on this GPU"),
    ("cleanup.gpu_reset", TestStatus.PASS, 0.05, "GPU 0 cleanup successful"),
]

# Phase 1: Show idle state (no test results) for 8 seconds
print("Phase 1: Idle state (8s)...")
for i in range(4):
    gpus = get_all_gpus()
    store.update_gpu_metrics(gpus)
    time.sleep(2)

# Phase 2: Inject test results progressively
print("Phase 2: Injecting test results...")
results = []
for name, status, duration, message in test_sequence:
    result = TestResult(
        test_name=name,
        status=status,
        duration_seconds=duration,
        message=message,
    )
    results.append(result)
    store.update_test_results(results)
    # Also refresh GPU metrics
    gpus = get_all_gpus()
    store.update_gpu_metrics(gpus)
    print(f"  + {name}: {status.value}")
    time.sleep(2)

# Phase 3: Hold final state
print("Phase 3: Final state - holding for 30s...")
for i in range(15):
    gpus = get_all_gpus()
    store.update_gpu_metrics(gpus)
    time.sleep(2)

print("Done.")
server.shutdown()
