"""CLI entry point for GPU Server Diagnostic Test Suite.

Provides the command-line interface modeled on DCGM's `dcgmi diag` tool.
Supports run levels (quick/medium/long/extended), individual test selection,
inventory display, health monitoring, burn-in mode, Prometheus metrics,
and JUnit XML output for CI/CD integration.

Usage:
    python -m src.main inventory
    python -m src.main diag --level quick
    python -m src.main diag --level long --output json
    python -m src.main diag --level long --output junit --junit-file results.xml
    python -m src.main diag --test deployment
    python -m src.main diag --mode burnin --duration 3600
    python -m src.main diag --level long --inject-fault thermal
    python -m src.main cleanup
    python -m src.main metrics --port 9835
"""

import json
import sys
import time
import uuid
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.inventory.gpu_inventory import get_all_gpus
from src.inventory.pcie_topology import get_pcie_topology
from src.inventory.system_info import get_system_info
from src.reporting.models import TestResult, TestStatus

console = Console()

# Resolve config paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config() -> dict:
    """Load master test configuration from config/test_config.yaml."""
    config_path = CONFIG_DIR / "test_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_profile(profile_name: str) -> dict:
    """Load GPU-specific profile from config/profiles/{name}.yaml."""
    profile_path = CONFIG_DIR / "profiles" / f"{profile_name}.yaml"
    with open(profile_path) as f:
        return yaml.safe_load(f)


def _status_color(status: TestStatus) -> str:
    """Map TestStatus to rich color string."""
    return {
        TestStatus.PASS: "green",
        TestStatus.FAIL: "red",
        TestStatus.WARN: "yellow",
        TestStatus.SKIP: "dim",
        TestStatus.ERROR: "bold red",
    }.get(status, "white")


def _print_inventory(gpu_infos, system_info, pcie_infos):
    """Print formatted system and GPU inventory using rich."""
    # System info panel
    sys_table = Table(show_header=False, box=None, padding=(0, 2))
    sys_table.add_column("Field", style="bold cyan")
    sys_table.add_column("Value")
    sys_table.add_row("Hostname", system_info.hostname)
    cpu_str = (
        f"{system_info.cpu_model} "
        f"({system_info.cpu_cores}C/{system_info.cpu_threads}T)"
    )
    sys_table.add_row("CPU", cpu_str)
    sys_table.add_row("RAM", f"{system_info.ram_total_gib} GiB")
    sys_table.add_row("OS", system_info.os_name)
    sys_table.add_row("Driver", system_info.driver_version)
    sys_table.add_row("CUDA", system_info.cuda_version)
    sys_table.add_row("Timestamp", system_info.timestamp)
    console.print(
        Panel(sys_table, title="System Inventory", border_style="blue")
    )

    # Per-GPU panels
    for gpu in gpu_infos:
        gpu_table = Table(
            show_header=False, box=None, padding=(0, 2)
        )
        gpu_table.add_column("Field", style="bold cyan")
        gpu_table.add_column("Value")
        gpu_table.add_row("Name", gpu.name)
        gpu_table.add_row("UUID", gpu.uuid)
        gpu_table.add_row("Serial", gpu.serial)
        vram_str = (
            f"{gpu.vram_total_mib} MiB "
            f"({gpu.vram_free_mib} MiB free)"
        )
        gpu_table.add_row("VRAM", vram_str)
        gpu_table.add_row("ECC", gpu.ecc_mode)
        gpu_table.add_row("Temperature", f"{gpu.temperature_c} C")
        pwr_str = (
            f"{gpu.power_draw_w:.1f} W / "
            f"{gpu.power_limit_w:.1f} W limit"
        )
        gpu_table.add_row("Power", pwr_str)
        gpu_table.add_row("Compute Cap.", gpu.compute_capability)
        gpu_table.add_row("P-State", gpu.pstate)
        gfx = (
            f"{gpu.clock_graphics_mhz} / "
            f"{gpu.clock_graphics_max_mhz} MHz"
        )
        mem = (
            f"{gpu.clock_memory_mhz} / "
            f"{gpu.clock_memory_max_mhz} MHz"
        )
        gpu_table.add_row("Clocks (Graphics)", gfx)
        gpu_table.add_row("Clocks (Memory)", mem)

        # PCIe info
        pcie = next(
            (p for p in pcie_infos if p.gpu_index == gpu.index),
            None,
        )
        if pcie:
            pcie_str = (
                f"Gen{pcie.link_gen_current} "
                f"x{pcie.link_width_current}"
            )
            if pcie.is_degraded:
                pcie_str += (
                    f" [red](DEGRADED: "
                    f"{pcie.degradation_reason})[/red]"
                )
            else:
                pcie_str += (
                    f" (max: Gen{pcie.link_gen_max} "
                    f"x{pcie.link_width_max})"
                )
            gpu_table.add_row("PCIe Link", pcie_str)

        console.print(
            Panel(
                gpu_table,
                title=f"GPU {gpu.index}",
                border_style="green",
            )
        )


def _print_results_table(results, run_level, duration):
    """Print formatted test results summary table."""
    table = Table(
        title=f"Diagnostic Results — Level: {run_level}",
        show_lines=True,
    )
    table.add_column("Test", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Message")
    table.add_column("Code", justify="center")

    for r in results:
        color = _status_color(r.status)
        table.add_row(
            r.test_name,
            f"[{color}]{r.status.value}[/{color}]",
            f"{r.duration_seconds:.2f}s",
            r.message,
            r.failure_code or "-",
        )

    console.print(table)

    # Overall verdict
    any_fail = any(
        r.status == TestStatus.FAIL for r in results
    )
    any_error = any(
        r.status == TestStatus.ERROR for r in results
    )
    any_warn = any(
        r.status == TestStatus.WARN for r in results
    )

    if any_fail or any_error:
        overall = TestStatus.FAIL
    elif any_warn:
        overall = TestStatus.WARN
    else:
        overall = TestStatus.PASS

    color = _status_color(overall)
    console.print(
        Panel(
            f"[bold {color}]{overall.value}[/bold {color}]",
            title="Overall Verdict",
            border_style=color,
            width=40,
        )
    )
    console.print(f"Total duration: {duration:.2f}s")
    return overall


def _build_test_registry(gpu_infos, config, profile):
    """Build the full test registry mapping names to runners."""
    from src.diagnostics.clock_throttle import run_clock_throttle_checks
    from src.diagnostics.compute_stress import run_compute_stress
    from src.diagnostics.deployment import run_deployment_checks
    from src.diagnostics.ecc_health import run_ecc_health_checks
    from src.diagnostics.gpu_cleanup import run_cleanup
    from src.diagnostics.gpu_health import run_gpu_health_checks
    from src.diagnostics.memory_bandwidth import run_memory_bandwidth
    from src.diagnostics.memory_test import run_memory_test
    from src.diagnostics.nccl_validation import run_nccl_validation
    from src.diagnostics.nvlink_p2p import run_nvlink_p2p
    from src.diagnostics.pcie_bandwidth import run_pcie_bandwidth
    from src.diagnostics.pcie_validation import run_pcie_validation
    from src.diagnostics.power_test import run_power_test
    from src.diagnostics.sm_stress import run_sm_stress
    from src.diagnostics.topology_map import run_topology_map
    from src.diagnostics.xid_errors import run_xid_checks

    return {
        # Level 1 — Quick / Deployment
        "deployment": lambda: run_deployment_checks(
            gpu_infos, config, profile
        ),
        # Level 2 — Medium / Validation
        "gpu_health": lambda: run_gpu_health_checks(
            gpu_infos, profile
        ),
        "pcie_validation": lambda: run_pcie_validation(
            gpu_infos, profile
        ),
        "memory_test": lambda: run_memory_test(
            gpu_infos, profile
        ),
        # Level 3 — Long / Stress
        "pcie_bandwidth": lambda: run_pcie_bandwidth(
            gpu_infos, profile
        ),
        "memory_bandwidth": lambda: run_memory_bandwidth(
            gpu_infos, profile
        ),
        "compute_stress": lambda: run_compute_stress(
            gpu_infos, profile
        ),
        "sm_stress": lambda: run_sm_stress(gpu_infos, profile),
        "power_test": lambda: run_power_test(gpu_infos, profile),
        # Advanced Telemetry
        "xid_errors": lambda: run_xid_checks(
            gpu_infos, profile
        ),
        "clock_throttle": lambda: run_clock_throttle_checks(
            gpu_infos, profile
        ),
        "ecc_health": lambda: run_ecc_health_checks(
            gpu_infos, profile
        ),
        # Interconnect & Topology
        "nvlink_p2p": lambda: run_nvlink_p2p(
            gpu_infos, profile
        ),
        "nccl_validation": lambda: run_nccl_validation(
            gpu_infos, profile
        ),
        "topology_map": lambda: run_topology_map(
            gpu_infos, profile
        ),
        # Cleanup
        "cleanup": lambda: run_cleanup(gpu_infos, profile),
    }


def _output_results(
    all_results, run_level, run_duration, output_format,
    junit_file=None,
):
    """Handle output formatting: text, json, or junit."""
    if output_format == "json":
        output = {
            "run_id": str(uuid.uuid4()),
            "run_level": run_level,
            "duration_seconds": run_duration,
            "results": [
                {
                    "test_name": r.test_name,
                    "status": r.status.value,
                    "duration_seconds": r.duration_seconds,
                    "message": r.message,
                    "failure_code": r.failure_code,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in all_results
            ],
        }
        click.echo(json.dumps(output, indent=2))
        return None

    if output_format == "junit":
        from src.reporting.junit_xml import (
            results_to_junit_xml,
            write_junit_xml,
        )

        if junit_file:
            write_junit_xml(all_results, junit_file)
            console.print(
                f"[green]JUnit XML written to: {junit_file}[/green]"
            )
        else:
            xml_str = results_to_junit_xml(all_results)
            click.echo(xml_str)
        return None

    # Default: text
    overall = _print_results_table(
        all_results, run_level, run_duration
    )
    return overall


# ─── CLI Commands ────────────────────────────────────────────────


@click.group()
def cli():
    """GPU Server Diagnostic Test Suite — modeled on NVIDIA DCGM."""
    pass


@cli.command()
def inventory():
    """Print system and GPU inventory."""
    console.print(
        "[bold blue]Collecting system inventory...[/bold blue]"
    )

    try:
        gpu_infos = get_all_gpus()
    except Exception as e:
        console.print(f"[red]Failed to detect GPUs: {e}[/red]")
        sys.exit(1)

    if not gpu_infos:
        console.print("[red]No NVIDIA GPUs detected.[/red]")
        sys.exit(1)

    system_info = get_system_info(
        driver_version=gpu_infos[0].driver_version,
        cuda_version=gpu_infos[0].cuda_version,
    )
    pcie_infos = get_pcie_topology(gpu_infos)

    _print_inventory(gpu_infos, system_info, pcie_infos)


@cli.command()
@click.option(
    "--level",
    type=click.Choice(["quick", "medium", "long", "extended"]),
    default=None,
    help="Diagnostic run level (mirrors DCGM -r 1/2/3/4).",
)
@click.option(
    "--test",
    "test_name",
    type=str,
    default=None,
    help="Run a single named test (e.g., deployment, xid_errors).",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["text", "json", "junit"]),
    default="text",
    help="Output format.",
)
@click.option(
    "--junit-file",
    "junit_file",
    type=str,
    default=None,
    help="Write JUnit XML to file (requires --output junit).",
)
@click.option(
    "--mode",
    type=click.Choice(["preflight", "standard", "burnin"]),
    default="standard",
    help="Execution mode: preflight (60s), standard, burnin (extended).",
)
@click.option(
    "--duration",
    type=int,
    default=None,
    help="Override stress test duration in seconds (burn-in mode).",
)
@click.option(
    "--inject-fault",
    "inject_fault",
    type=click.Choice([
        "thermal", "ecc", "pcie", "clock", "memory",
    ]),
    default=None,
    help="Inject a simulated fault for testing failure handling.",
)
@click.option(
    "--cleanup/--no-cleanup",
    default=True,
    help="Run GPU cleanup after tests (default: enabled).",
)
@click.option(
    "--metrics-port",
    type=int,
    default=None,
    help="Start Prometheus metrics server on this port.",
)
def diag(
    level, test_name, output_format, junit_file,
    mode, duration, inject_fault, cleanup, metrics_port,
):
    """Run GPU diagnostic tests."""
    if not level and not test_name:
        console.print(
            "[red]Specify --level or --test. "
            "Use --help for options.[/red]"
        )
        sys.exit(1)

    config = load_config()
    profile_name = config.get("gpu_profile", "rtx_5070ti")
    profile = load_profile(profile_name)

    # Override stress duration for burn-in mode
    if mode == "burnin" and duration:
        profile.setdefault("thresholds", {})
        profile["thresholds"]["stress_duration_seconds"] = duration
        profile["thresholds"]["power_duration_seconds"] = duration
    elif mode == "preflight":
        profile.setdefault("thresholds", {})
        profile["thresholds"]["stress_duration_seconds"] = 30
        profile["thresholds"]["power_duration_seconds"] = 15

    # Determine which tests to run
    if test_name:
        tests_to_run = [test_name]
        run_level = f"single:{test_name}"
    else:
        run_levels = config.get("run_levels", {})
        tests_to_run = run_levels.get(level, [])
        run_level = level

    if not tests_to_run:
        console.print(
            f"[red]No tests defined for level '{level}'.[/red]"
        )
        sys.exit(1)

    # Collect GPU info
    try:
        gpu_infos = get_all_gpus()
    except Exception as e:
        console.print(f"[red]Failed to detect GPUs: {e}[/red]")
        sys.exit(1)

    # Start Prometheus metrics server if requested
    if metrics_port:
        from src.reporting.prometheus import (
            get_metrics_store,
            start_metrics_server,
        )

        start_metrics_server(port=metrics_port)
        get_metrics_store().update_gpu_metrics(gpu_infos)
        console.print(
            f"[green]Prometheus metrics: "
            f"http://localhost:{metrics_port}/metrics[/green]"
        )

    if inject_fault:
        console.print(
            f"[yellow]Fault injection active: "
            f"{inject_fault}[/yellow]"
        )

    mode_label = ""
    if mode == "burnin":
        dur = duration or "default"
        mode_label = f" [BURN-IN: {dur}s]"
    elif mode == "preflight":
        mode_label = " [PRE-FLIGHT]"

    console.print(
        f"[bold blue]Running diagnostic level: {run_level} "
        f"({len(tests_to_run)} test(s)){mode_label}[/bold blue]"
    )
    console.print(
        f"Profile: {profile_name} | GPU(s): {len(gpu_infos)}\n"
    )

    # Build test registry
    test_registry = _build_test_registry(
        gpu_infos, config, profile
    )

    all_results = []
    run_start = time.time()

    for test in tests_to_run:
        if test in test_registry:
            try:
                results = test_registry[test]()
                all_results.extend(results)
            except Exception as e:
                all_results.append(
                    TestResult(
                        test_name=test,
                        status=TestStatus.ERROR,
                        duration_seconds=0.0,
                        message=f"Test crashed: {e}",
                        failure_code="DIAG-ERR",
                    )
                )
                console.print(
                    f"[red]Test '{test}' crashed: {e}[/red]"
                )
        else:
            all_results.append(
                TestResult(
                    test_name=test,
                    status=TestStatus.SKIP,
                    duration_seconds=0.0,
                    message="Test module not yet implemented",
                )
            )
            console.print(
                f"[dim]Skipping '{test}' — "
                f"not yet implemented[/dim]"
            )

    # Automatic GPU cleanup after tests
    if cleanup and mode != "preflight":
        try:
            from src.diagnostics.gpu_cleanup import run_cleanup

            cleanup_results = run_cleanup(gpu_infos, profile)
            all_results.extend(cleanup_results)
        except Exception as e:
            console.print(
                f"[yellow]Cleanup warning: {e}[/yellow]"
            )

    run_duration = time.time() - run_start

    # Update Prometheus metrics if server is running
    if metrics_port:
        get_metrics_store().update_test_results(all_results)

    # Output results
    overall = _output_results(
        all_results, run_level, run_duration,
        output_format, junit_file,
    )

    # Exit code: 0 for PASS/WARN, 1 for FAIL/ERROR
    if overall in (TestStatus.FAIL, TestStatus.ERROR):
        sys.exit(1)


@cli.command(name="cleanup")
def cleanup_cmd():
    """Reset GPUs to clean state (clocks, power, CUDA context)."""
    console.print(
        "[bold blue]Running GPU cleanup...[/bold blue]"
    )

    try:
        gpu_infos = get_all_gpus()
    except Exception as e:
        console.print(f"[red]Failed to detect GPUs: {e}[/red]")
        sys.exit(1)

    from src.diagnostics.gpu_cleanup import run_cleanup

    results = run_cleanup(gpu_infos, {})

    for r in results:
        color = _status_color(r.status)
        console.print(
            f"  [{color}]{r.status.value}[/{color}] {r.message}"
        )


@cli.command()
@click.option(
    "--port", default=9835, help="HTTP port for /metrics endpoint.",
)
def metrics(port):
    """Start Prometheus metrics exporter (HTTP /metrics endpoint)."""
    from src.reporting.prometheus import (
        get_metrics_store,
        start_metrics_server,
    )

    console.print(
        f"[bold blue]Starting Prometheus metrics server "
        f"on port {port}...[/bold blue]"
    )

    try:
        gpu_infos = get_all_gpus()
        get_metrics_store().update_gpu_metrics(gpu_infos)
    except Exception as e:
        console.print(
            f"[yellow]GPU detection failed: {e} — "
            f"serving empty metrics[/yellow]"
        )

    server = start_metrics_server(port=port, daemon=False)
    console.print(
        f"[green]Metrics available at: "
        f"http://localhost:{port}/metrics[/green]"
    )
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    try:
        import threading

        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
        console.print("\n[dim]Metrics server stopped.[/dim]")


@cli.command()
@click.option(
    "--interval", default=5,
    help="Polling interval in seconds.",
)
def monitor(interval):
    """Start background GPU health monitoring."""
    console.print(
        f"[bold blue]Health monitoring — interval: "
        f"{interval}s (Ctrl+C to stop)[/bold blue]"
    )
    console.print(
        "[dim]Phase 3 implementation — health_daemon.py[/dim]"
    )
    console.print(
        "[yellow]Monitor not yet implemented. "
        "Coming in Phase 3.[/yellow]"
    )


@cli.command()
@click.option(
    "--failures", is_flag=True,
    help="Show only failed runs.",
)
def history(failures):
    """Show recent diagnostic run history from database."""
    console.print(
        "[dim]Phase 4 implementation — database queries[/dim]"
    )
    console.print(
        "[yellow]History not yet implemented. "
        "Coming in Phase 4.[/yellow]"
    )


if __name__ == "__main__":
    cli()
