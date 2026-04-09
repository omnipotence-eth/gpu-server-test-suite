# GPU Server Test Suite — Claude Code Project Guide

Production GPU validation framework modeled on NVIDIA DCGM architecture.
Read this before making any changes.

---

## What This Is

A CLI-driven diagnostic suite that validates GPU server health across 16 modules: PCIe bandwidth,
VRAM integrity, ECC error rates, clock throttling, NVLink P2P, SM stress, power limits, and more.
Ships a Prometheus metrics exporter and Grafana dashboards. Designed for RTX 5070 Ti (Blackwell sm_120)
but parameterizable for any NVIDIA GPU.

Portfolio value: proves ML infrastructure engineering, GPU fleet management, and production monitoring
skills to hiring managers — especially at companies running GPU clusters (Capital One, AT&T, NVIDIA partners).

---

## Repo Layout

```
gpu-server-test-suite/
├── src/
│   ├── main.py              # Click CLI entry point — `gpu-diag` command
│   ├── diagnostics/         # 16 diagnostic modules (one per hardware test)
│   │   ├── clock_throttle.py
│   │   ├── compute_stress.py
│   │   ├── ecc_health.py
│   │   ├── gpu_health.py
│   │   ├── memory_bandwidth.py
│   │   ├── memory_test.py
│   │   ├── nccl_validation.py
│   │   ├── nvlink_p2p.py
│   │   ├── pcie_bandwidth.py
│   │   ├── pcie_validation.py
│   │   ├── power_test.py
│   │   ├── sm_stress.py
│   │   ├── topology_map.py
│   │   ├── xid_errors.py
│   │   ├── deployment.py
│   │   └── fault_injection/
│   ├── monitoring/          # Prometheus metrics exporter
│   ├── database/            # SQLAlchemy models for test history (PostgreSQL)
│   ├── inventory/           # Hardware inventory and profiling
│   └── reporting/           # JUnit XML + Rich CLI output
├── tests/                   # 16 test files — one per diagnostic module
├── config/                  # YAML hardware profiles (RTX 5070 Ti profile included)
├── docs/                    # Architecture, deployment, usage guides
├── reports/                 # Generated test reports (gitignored)
├── docker-compose.yml       # Prometheus + Grafana monitoring stack
├── pyproject.toml           # v1.0.0 — setuptools, Python 3.11+
├── ROADMAP.md               # Planned features
└── README.md
```

---

## Running

```bash
# Prerequisites: Python 3.11+, NVIDIA drivers, pynvml

# Install (standard pip, not uv — this project uses setuptools)
pip install -e ".[dev]"

# Run all diagnostics
gpu-diag run --all

# Run specific module
gpu-diag run --test pcie_bandwidth

# Quick health check
gpu-diag health

# Start monitoring stack (Prometheus + Grafana)
docker-compose up -d

# Run with RTX 5070 Ti hardware profile
gpu-diag run --profile config/rtx5070ti.yaml
```

---

## Testing

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing -v

# Specific test
pytest tests/test_pcie_validation.py -v
```

**Test conventions**:
- Each diagnostic module has a corresponding `test_<module>.py`
- Hardware tests use `unittest.mock.patch("pynvml.*")` — real NVIDIA hardware not required for unit tests
- `conftest.py` provides shared GPU mock fixtures
- Use `pytest.mark.integration` for tests that require real hardware

---

## Key Files

| File | Why it matters |
|------|---------------|
| `src/main.py` | CLI entry — all commands defined here via Click |
| `src/diagnostics/gpu_health.py` | Core GPU health check — most comprehensive diagnostic |
| `src/monitoring/` | Prometheus exporter — metrics exposed at `:9090/metrics` |
| `src/database/` | SQLAlchemy models for test run history + trend analysis |
| `config/rtx5070ti.yaml` | RTX 5070 Ti hardware profile — thresholds for Blackwell sm_120 |
| `docker-compose.yml` | Prometheus + Grafana stack for local monitoring dashboard |

---

## Hardware Context

- **Primary target**: RTX 5070 Ti (16GB VRAM, Blackwell sm_120, CUDA 12.8)
- **pynvml**: primary interface to NVML (NVIDIA Management Library)
- **Clock throttle module**: detects thermal/power throttling conditions
- **ECC module**: single-bit vs double-bit error tracking
- **PCIe bandwidth**: measured via DMA transfers, compared to Gen 5 x16 theoretical max
- **NVLink/P2P**: multi-GPU topology validation (N/A for single RTX 5070 Ti)

---

## Observability Stack

| Component | Port | Purpose |
|-----------|------|---------|
| Prometheus | 9090 | Metrics scraping |
| Grafana | 3000 | Dashboard visualization |
| PostgreSQL | 5432 | Test run history (SQLAlchemy) |

Start with: `docker-compose up -d`
Default Grafana login: `admin / admin`

---

## IMPORTANT Rules

- **Never commit real hardware profiles with proprietary threshold data** — keep to config/
- **Tests must not require real GPU hardware** — mock pynvml in all unit tests
- **pyproject.toml uses setuptools, not uv** — install with `pip install -e .` not `uv sync`
- No CI/CD configured yet — add `.github/workflows/ci.yml` before any public PR
- No CHANGELOG.md yet — add before tagging a release

---

## How Claude Code Should Approach Changes

1. Read the existing diagnostic module before adding a new one — all modules follow the same pattern
2. New diagnostics: implement `run()` → `validate()` → return `DiagnosticResult` dataclass
3. New CLI commands: add to `src/main.py` via Click, not a new entry point
4. Always add a corresponding test file with mocked pynvml
5. Hardware thresholds live in `config/*.yaml` — never hardcode in diagnostic code
6. This project uses `setuptools`, not `uv` — do not add `uv.lock` or change build backend
