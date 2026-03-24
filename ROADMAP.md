# Project Roadmap

## Completed ✅

- [x] Core diagnostic framework (16 tests across 6 categories)
- [x] Multi-level run modes (quick/medium/long/extended)
- [x] Prometheus metrics exporter — rewritten to use `prometheus_client` library for correct exposition format
- [x] Docker Compose containerization (gpu-diag + Prometheus + Grafana)
- [x] Auto-provisioned Grafana dashboards and alerting rules
- [x] Hardware profiles for common GPUs (RTX 5070 Ti, A100 80GB, H100 SXM)
- [x] JUnit XML output for CI/CD integration
- [x] GitHub Actions CI pipeline
- [x] 157 unit tests passing, ruff linting clean
- [x] `PytestCollectionWarning` suppression via `pyproject.toml`
- [x] **Fault injection framework** — 5 synthetic fault types (`thermal`, `ecc`, `pcie`, `clock`, `memory`) via `--inject-fault`; failure codes use `DIAG-FI-*` prefix to distinguish from real faults in alert rules
- [x] **Live GPU health monitor** — `monitor` command with Rich Live display; configurable poll interval; temperature colour-coded green/yellow/red at 75°C/85°C
- [x] **File-based run history** — `diag` appends JSONL summaries to `reports/.run_history.jsonl`; `history` command renders a table with `--failures` and `--limit` filters
- [x] **Unique deployment failure codes** — DIAG-001 (driver), DIAG-002 (count), DIAG-003 (model), DIAG-004 (ECC); previously all shared DIAG-001
- [x] **pynvml session context manager** — `deployment.py` opens a single nvml session for the full deployment check run instead of redundant per-check init/shutdown pairs

---

## Planned Enhancements

### Phase 1: High-Value Tests (Priority: High)

**Thermal Profiling Test**
- Map GPU temperature vs load curve
- Identify thermal throttling threshold
- Validate fan curve behaviour (if applicable)
- Detect thermal interface material degradation
- Useful for preventive maintenance

**Burn-in Stress Enhancements**
- Continuous memory stress with pattern verification during burn-in
- Per-iteration thermal and power monitoring
- Automatic abort with diagnostic report on sustained throttle

**Estimated effort:** 30–45 minutes

---

### Phase 2: Performance (Priority: Medium)

**Test Runner Parallelisation**
- Current: ~0.62s for medium level (sequential)
- Target: <0.3s (parallel execution of independent tests)
- Bottleneck: independent modules can run concurrently via `threading`

**Memory Usage Optimisation**
- Profile large VRAM allocation tests
- Reduce intermediate buffer allocations in bandwidth measurements

---

### Phase 3: Documentation & Case Study (Priority: Medium)

**Architectural Design Document**
- Deep dive into DCGM-inspired design decisions
- Test categorisation rationale and profile-based configuration approach
- Multi-level diagnostic strategy

**Technical Case Study**
- Real-world deployment scenarios (Kubernetes, Slurm)
- Monitoring SLA examples
- Cost–benefit comparison vs commercial solutions (DCGM, MIG manager)

---

### Phase 4: Advanced Features (Priority: Low)

**NCCL Collective Operations Benchmark** *(current impl is simulated)*
- Current: in-process P2P simulation — measures PCIe bandwidth, does not invoke the NCCL library or initialise `torch.distributed`
- Planned: replace with multi-process `torch.distributed` (NCCL backend) via `torchrun`, exercising real ring-allreduce over NVLink/PCIe
- Requires a 2+ GPU node

**Database Persistence**
- SQLAlchemy schema for historical results (dependencies already in place)
- Trend analysis — temperature rise over time, ECC error rate
- Anomaly detection via baseline comparison
- Replace or supplement current JSONL history with queryable storage

**Kubernetes Integration**
- CRD for GPU diagnostic jobs
- Prometheus `ServiceMonitor` for scrape config
- Automated node health checks via `DaemonSet`

**Daemon-based Health Monitoring**
- Persistent background process polling GPU telemetry
- Alerting on threshold breaches without a full diagnostic run
- Structured `monitoring/` module (currently a placeholder)

---

## Known Limitations

- Single-GPU diagnostics only (no multi-GPU collective ops yet — NCCL is simulated)
- Database persistence not yet active (file-based JSONL history is available)
- Daemon-based background monitoring not yet implemented (`monitor` command polls on demand)
- Limited to NVIDIA GPUs (AMD/Intel not supported)

---

## Success Criteria

| Goal                   | Current            | Target            |
|------------------------|--------------------|-------------------|
| Diagnostic modules     | 16                 | 18–20             |
| Unit test coverage     | 157 tests          | 175+ tests        |
| Test runtime (medium)  | ~5s                | <3s               |
| Documentation          | README + roadmap   | + architecture doc |
| Production deployments | Example config     | Multiple documented|

---

**Last Updated:** March 2026
**Maintained By:** Tremayne Timms
**Status:** Active Development
