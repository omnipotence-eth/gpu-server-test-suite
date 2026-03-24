# Project Roadmap

## Completed ✅

- [x] Core diagnostic framework (17 tests across 6 categories)
- [x] Multi-level run modes (quick/medium/long/extended)
- [x] Prometheus metrics exporter with CORS support
- [x] Docker Compose containerization (gpu-diag + Prometheus + Grafana)
- [x] Auto-provisioned Grafana dashboards and alerting rules
- [x] Live HTML dashboard with real-time metrics
- [x] Hardware profiles for common GPUs
- [x] Fault injection framework for testing
- [x] JUnit XML output for CI/CD integration
- [x] GitHub Actions CI pipeline
- [x] Professional documentation (README, technical paper, GIF demo)
- [x] All 154 unit tests passing with ruff linting clean
- [x] Suppressed `PytestCollectionWarning` for `TestStatus`/`TestResult` production classes via `pyproject.toml` `filterwarnings`

## Planned Enhancements

### Phase 1: High-Value Tests (Priority: High)

**Burn-in Stress Test**
- Sustained compute load for configurable duration (1h, 8h, 24h)
- Continuous memory stress with pattern verification
- Thermal and power monitoring throughout
- Early indicator of latent hardware defects
- Industry standard for GPU qualification

**Thermal Profiling Test**
- Map GPU temperature vs load curve
- Identify thermal throttling threshold
- Validate fan curve behavior (if applicable)
- Detect thermal interface material degradation
- Useful for preventive maintenance

**Estimated effort:** 30-45 minutes | ~10-12k tokens

### Phase 2: Performance Optimization (Priority: Medium)

**Test Runner Parallelization**
- Current: ~0.62s for medium level (sequential)
- Target: <0.3s (parallel test execution)
- Bottleneck: Independent tests can run concurrently
- Use threading/multiprocessing for compute tests

**Memory Usage Optimization**
- Profile large VRAM allocation tests
- Optimize bandwidth measurement algorithms
- Reduce intermediate buffer allocations

**Estimated effort:** 60-90 minutes | ~15-20k tokens

### Phase 3: Documentation & Case Study (Priority: Medium)

**Architectural Design Document**
- Deep dive into DCGM-inspired design decisions
- Test categorization rationale
- Why specific metrics over others
- Profile-based configuration approach
- Multi-level diagnostic strategy

**Technical Case Study**
- Real-world deployment scenarios
- Integration with Kubernetes/Slurm clusters
- Monitoring SLA examples
- Cost-benefit vs commercial solutions (DCGM, MIG manager)

**Estimated effort:** 45-60 minutes | ~12-18k tokens

### Phase 4: Advanced Features (Priority: Low)

**NCCL Collective Operations Benchmark**
- All-reduce, all-gather, broadcast latency measurement
- Multi-GPU synchronization validation
- Network throughput characterization

**NVLink Bandwidth Measurement**
- P2P throughput profiling
- Compare vs PCIe bandwidth
- Detect link degradation

**Database Persistence**
- SQLAlchemy schema for historical results
- Trend analysis (temperature rise over time, etc.)
- Anomaly detection via baseline comparison

**Kubernetes Integration**
- CRD for GPU diagnostic jobs
- Prometheus ServiceMonitor for scrape config
- Automated node health checks via DaemonSet

**Estimated effort:** 2-3 weeks (out of scope for single sprint)

## Known Limitations

- Single-GPU diagnostics only (no multi-GPU collective ops yet)
- No persistent result storage (in-memory metrics only)
- Dashboard requires manual metrics server (not auto-scaled)
- Limited to NVIDIA GPUs (AMD/Intel not supported)

## Success Criteria

| Goal | Current | Target |
|------|---------|--------|
| Diagnostic modules | 16 | 18-20 |
| Unit test coverage | 154 tests | 170+ tests |
| Test runtime (medium) | ~5s | <3s |
| Documentation | README + paper | + case study |
| Production deployments | Example | Multiple documented |

---

**Last Updated:** March 2026
**Maintained By:** Tremayne Timms
**Status:** Active Development
