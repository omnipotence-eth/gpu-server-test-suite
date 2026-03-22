"""Generate professional technical paper: GPU Server Diagnostic Framework."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas as pdfcanvas


# ─── Color Palette ──────────────────────────────────────────────
NVIDIA_GREEN = HexColor("#76B900")
DARK_BG = HexColor("#1a1a2e")
ACCENT_BLUE = HexColor("#0f3460")
ACCENT_TEAL = HexColor("#16213e")
HEADER_GREEN = HexColor("#4a7c10")
LIGHT_GRAY = HexColor("#f5f5f5")
MED_GRAY = HexColor("#e0e0e0")
DARK_GRAY = HexColor("#333333")
CODE_BG = HexColor("#f8f8f8")
TABLE_HEADER = HexColor("#2c3e50")
TABLE_ALT = HexColor("#ecf0f1")
PASS_GREEN = HexColor("#27ae60")
FAIL_RED = HexColor("#c0392b")
WARN_AMBER = HexColor("#f39c12")


# ─── Custom Styles ──────────────────────────────────────────────
def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="PaperTitle",
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=6,
        textColor=DARK_GRAY,
    ))
    styles.add(ParagraphStyle(
        name="PaperSubtitle",
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        spaceAfter=4,
        textColor=HexColor("#666666"),
    ))
    styles.add(ParagraphStyle(
        name="AuthorLine",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=2,
        textColor=HexColor("#555555"),
    ))
    styles.add(ParagraphStyle(
        name="SectionHead",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=18,
        spaceAfter=8,
        textColor=ACCENT_BLUE,
        borderWidth=0,
        borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        name="SubSection",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        spaceBefore=12,
        spaceAfter=6,
        textColor=DARK_GRAY,
    ))
    styles.add(ParagraphStyle(
        name="BodyText2",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        textColor=DARK_GRAY,
    ))
    styles.add(ParagraphStyle(
        name="BulletItem",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        leftIndent=20,
        spaceAfter=3,
        textColor=DARK_GRAY,
    ))
    styles.add(ParagraphStyle(
        name="CodeBlock",
        fontName="Courier",
        fontSize=8,
        leading=11,
        leftIndent=12,
        spaceAfter=6,
        backColor=CODE_BG,
        textColor=DARK_GRAY,
    ))
    styles.add(ParagraphStyle(
        name="TableHeader",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=white,
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        alignment=TA_LEFT,
        textColor=DARK_GRAY,
    ))
    styles.add(ParagraphStyle(
        name="TableCellCenter",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=DARK_GRAY,
    ))
    styles.add(ParagraphStyle(
        name="Caption",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        spaceAfter=10,
        textColor=HexColor("#666666"),
    ))
    styles.add(ParagraphStyle(
        name="FooterStyle",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=HexColor("#999999"),
    ))
    styles.add(ParagraphStyle(
        name="Abstract",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        alignment=TA_JUSTIFY,
        leftIndent=30,
        rightIndent=30,
        spaceAfter=8,
        textColor=DARK_GRAY,
    ))
    return styles


# ─── Page Template ──────────────────────────────────────────────
def header_footer(canvas, doc):
    canvas.saveState()
    # Header line
    canvas.setStrokeColor(NVIDIA_GREEN)
    canvas.setLineWidth(1.5)
    canvas.line(
        doc.leftMargin, letter[1] - 45,
        letter[0] - doc.rightMargin, letter[1] - 45,
    )
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(HexColor("#888888"))
    canvas.drawString(
        doc.leftMargin, letter[1] - 40,
        "GPU Server Diagnostic Test Suite"
    )
    canvas.drawRightString(
        letter[0] - doc.rightMargin, letter[1] - 40,
        "Technical Reference"
    )
    # Footer
    canvas.setStrokeColor(MED_GRAY)
    canvas.setLineWidth(0.5)
    canvas.line(
        doc.leftMargin, 45,
        letter[0] - doc.rightMargin, 45,
    )
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#999999"))
    canvas.drawCentredString(
        letter[0] / 2, 32,
        f"Page {doc.page}"
    )
    canvas.restoreState()


# ─── Table Builder ──────────────────────────────────────────────
def make_table(headers, rows, col_widths=None):
    s = build_styles()
    header_cells = [
        Paragraph(h, s["TableHeader"]) for h in headers
    ]
    data = [header_cells]
    for row in rows:
        data.append([
            Paragraph(str(cell), s["TableCell"]) if i == 0
            else Paragraph(str(cell), s["TableCellCenter"])
            for i, cell in enumerate(row)
        ])

    if col_widths is None:
        col_widths = [
            (letter[0] - 1.4 * inch) / len(headers)
        ] * len(headers)

    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(
                ("BACKGROUND", (0, i), (-1, i), TABLE_ALT)
            )
    t.setStyle(TableStyle(style_cmds))
    return t


# ─── Document Content ───────────────────────────────────────────
def build_document():
    doc = SimpleDocTemplate(
        "C:\\Users\\ttimm\\Desktop\\gpu-server-test-suite\\docs\\"
        "GPU_Server_Diagnostic_Framework.pdf",
        pagesize=letter,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
    )

    s = build_styles()
    story = []
    W = letter[0] - 1.4 * inch  # Usable width

    # ─── Title Page ─────────────────────────────────────────
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph(
        "GPU Server Diagnostic Framework",
        s["PaperTitle"],
    ))
    story.append(Paragraph(
        "A Production-Grade Reliability Testing Architecture<br/>"
        "for NVIDIA GPU Infrastructure",
        s["PaperSubtitle"],
    ))
    story.append(Spacer(1, 24))
    story.append(HRFlowable(
        width="40%", thickness=1.5, color=NVIDIA_GREEN,
        spaceAfter=20, spaceBefore=0,
    ))
    story.append(Paragraph(
        "Tremayne Timms",
        s["AuthorLine"],
    ))
    story.append(Paragraph(
        "GPU Reliability Engineering",
        ParagraphStyle(
            "tmp", parent=s["AuthorLine"],
            fontSize=9, textColor=HexColor("#777777"),
        ),
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "March 2026",
        ParagraphStyle(
            "tmp2", parent=s["AuthorLine"],
            fontSize=9, textColor=HexColor("#999999"),
        ),
    ))
    story.append(Spacer(1, 0.8 * inch))

    # Abstract
    story.append(Paragraph("Abstract", ParagraphStyle(
        "abshead", parent=s["SubSection"], alignment=TA_CENTER,
        spaceBefore=0,
    )))
    story.append(Paragraph(
        "This paper presents a comprehensive GPU diagnostic framework "
        "that systematically validates server-class GPU infrastructure "
        "across five critical dimensions: hardware deployment verification, "
        "advanced telemetry analysis, interconnect topology validation, "
        "sustained workload stress testing, and production observability "
        "integration. The framework implements 16 diagnostic modules "
        "producing 153 automated test assertions, with structured output "
        "for CI/CD pipelines (JUnit XML) and real-time monitoring "
        "(Prometheus metrics). The entire stack is containerized via "
        "Docker Compose with a three-service architecture (diagnostics, "
        "Prometheus, Grafana) providing a production-ready deployment "
        "model. Architecturally modeled on NVIDIA's Data Center GPU "
        "Manager (DCGM), the system supports four run levels from "
        "60-second pre-flight checks to 24-hour burn-in validation, "
        "covering the full lifecycle of GPU fleet reliability engineering.",
        s["Abstract"],
    ))

    story.append(Spacer(1, 0.4 * inch))

    # Key stats box
    stats_data = [
        ["Diagnostic Modules", "Test Assertions",
         "Lines of Code", "Run Levels"],
        ["16", "153", "7,868", "4"],
    ]
    stats_table = Table(
        stats_data,
        colWidths=[W / 4] * 4,
    )
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 18),
        ("TEXTCOLOR", (0, 1), (-1, 1), NVIDIA_GREEN),
        ("BACKGROUND", (0, 1), (-1, 1), LIGHT_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(stats_table)
    story.append(PageBreak())

    # ─── Section 1: Introduction ────────────────────────────
    story.append(Paragraph("1. Introduction", s["SectionHead"]))
    story.append(Paragraph(
        "Modern GPU clusters powering large-scale AI training represent "
        "significant capital investment where undetected hardware "
        "degradation directly translates to wasted compute-hours and "
        "failed training runs. A single degraded NVLink bridge can "
        "reduce inter-GPU bandwidth from 900 GB/s to 50 GB/s without "
        "triggering a hard failure. A rising single-bit ECC error rate "
        "often precedes an uncorrectable double-bit error that crashes "
        "a multi-day training job.",
        s["BodyText2"],
    ))
    story.append(Paragraph(
        "This framework addresses the gap between basic GPU monitoring "
        "and production-grade reliability engineering. It implements the "
        "diagnostic methodology established by NVIDIA's DCGM while "
        "extending it with CI/CD-native output formats, Prometheus "
        "observability integration, and idempotent GPU state management "
        "suitable for automated fleet operations.",
        s["BodyText2"],
    ))

    story.append(Paragraph(
        "1.1 Design Principles", s["SubSection"],
    ))
    principles = [
        "<b>Configuration-driven thresholds.</b> All pass/fail criteria "
        "are defined in per-GPU YAML profiles (RTX 5070 Ti, A100, H100), "
        "not hardcoded. Deploying to a new GPU SKU requires only a new "
        "profile file.",
        "<b>Fail-fast with structured diagnostics.</b> Every failure "
        "returns a unique diagnostic code (DIAG-001 through DIAG-970), "
        "a human-readable message, and a machine-parseable details dict "
        "enabling automated triage.",
        "<b>CI/CD-native.</b> Tests run without a GPU via comprehensive "
        "mocking (153 assertions in CI). Production runs produce JUnit "
        "XML for pipeline integration and Prometheus metrics for "
        "Grafana dashboards.",
        "<b>Idempotent execution.</b> Automatic GPU cleanup after every "
        "run resets CUDA context, application clocks, and power limits. "
        "A crashed test never leaves the GPU in a zombie state.",
    ]
    for p in principles:
        story.append(Paragraph(
            f"\u2022  {p}", s["BulletItem"],
        ))

    # ─── Section 2: Architecture ────────────────────────────
    story.append(Paragraph("2. Architecture", s["SectionHead"]))
    story.append(Paragraph(
        "The framework follows a layered architecture with clear "
        "separation between hardware abstraction, diagnostic logic, "
        "orchestration, and output formatting.",
        s["BodyText2"],
    ))

    arch_table = make_table(
        ["Layer", "Modules", "Responsibility"],
        [
            ["Inventory", "gpu_inventory, pcie_topology,\nsystem_info",
             "Hardware detection via pynvml/nvidia-smi"],
            ["Diagnostics", "16 test modules",
             "Validation, stress testing, telemetry"],
            ["Orchestration", "test_runner, main CLI",
             "Run levels, preflight gates, burn-in"],
            ["Reporting", "models, junit_xml, prometheus",
             "Structured output and observability"],
            ["Configuration", "test_config.yaml, profiles/",
             "Thresholds, run levels, fleet config"],
        ],
        col_widths=[W * 0.18, W * 0.35, W * 0.47],
    )
    story.append(arch_table)
    story.append(Paragraph(
        "Table 1. Framework architecture layers.",
        s["Caption"],
    ))

    story.append(Paragraph(
        "2.1 Run Level Hierarchy", s["SubSection"],
    ))
    story.append(Paragraph(
        "Mirroring DCGM's four diagnostic levels, each subsequent level "
        "is a strict superset of the previous, enabling operators to "
        "select appropriate depth for the operational context.",
        s["BodyText2"],
    ))

    level_table = make_table(
        ["Level", "Tests", "Duration", "Use Case"],
        [
            ["Quick (L1)", "1", "< 5s", "Deployment validation"],
            ["Medium (L2)", "7", "< 30s",
             "Pre-job health + telemetry"],
            ["Long (L3)", "14", "2-5 min",
             "Full stress + interconnect"],
            ["Extended (L4)", "16", "10+ min",
             "NCCL collectives + memtest"],
        ],
        col_widths=[W * 0.18, W * 0.10, W * 0.17, W * 0.55],
    )
    story.append(level_table)
    story.append(Paragraph(
        "Table 2. Diagnostic run levels.",
        s["Caption"],
    ))

    # ─── Section 3: Diagnostic Modules ──────────────────────
    story.append(Paragraph(
        "3. Diagnostic Modules", s["SectionHead"],
    ))

    story.append(Paragraph(
        "3.1 Deployment Validation (Level 1)", s["SubSection"],
    ))
    story.append(Paragraph(
        "Verifies that the GPU software stack is correctly installed "
        "and the hardware matches the expected configuration. Checks "
        "include driver load status, GPU count and model verification "
        "against the profile, ECC mode validation, running process "
        "detection, and persistence mode status. This mirrors DCGM's "
        "Level 1 software deployment check.",
        s["BodyText2"],
    ))

    story.append(Paragraph(
        "3.2 Advanced Telemetry", s["SubSection"],
    ))
    story.append(Paragraph(
        "Three modules provide DCGM-level GPU health telemetry that "
        "goes beyond basic temperature and utilization monitoring.",
        s["BodyText2"],
    ))

    telem_table = make_table(
        ["Module", "Monitors", "Failure Codes"],
        [
            ["XID Error Tracking",
             "Kernel-level GPU faults from dmesg\n"
             "and NVML (XIDs 31, 43, 48, 61, 74, 79)",
             "DIAG-900"],
            ["Clock Throttle Analysis",
             "NVML throttle reason bitmask:\n"
             "thermal, power brake, HW slowdown",
             "DIAG-910"],
            ["ECC Memory Health",
             "SBE/DBE counters, retired pages,\n"
             "row remapping status (Ampere+)",
             "DIAG-920/921"],
        ],
        col_widths=[W * 0.25, W * 0.45, W * 0.30],
    )
    story.append(telem_table)
    story.append(Paragraph(
        "Table 3. Advanced telemetry modules.",
        s["Caption"],
    ))

    story.append(Paragraph(
        "<b>XID Error Classification.</b> NVIDIA XID errors are the "
        "primary signal for GPU hardware failure in production clusters. "
        "The framework classifies XIDs into three severity tiers: "
        "critical (XID 79: GPU fallen off bus, XID 48: double-bit ECC), "
        "warning (XID 92: high SBE rate), and informational. Critical "
        "XIDs trigger an immediate FAIL with a node-drain recommendation.",
        s["BodyText2"],
    ))
    story.append(Paragraph(
        "<b>ECC Policy Engine.</b> For ECC-capable GPUs (A100, H100), "
        "the framework tracks volatile and aggregate error counters, "
        "retired page counts, and row remapping status. The policy "
        "mirrors production practice: rising SBE counts above a "
        "configurable threshold trigger proactive drain before an "
        "uncorrectable DBE crashes a training job. Row remapping failure "
        "indicates the GPU has exhausted its hardware error mitigation "
        "capacity and requires replacement.",
        s["BodyText2"],
    ))

    story.append(Paragraph(
        "3.3 Interconnect and Topology Validation", s["SubSection"],
    ))
    story.append(Paragraph(
        "In multi-GPU servers, the communication fabric is frequently "
        "the performance bottleneck. Three modules validate the "
        "interconnect layer.",
        s["BodyText2"],
    ))

    inter_table = make_table(
        ["Module", "Method", "Detects"],
        [
            ["NVLink/P2P Bandwidth",
             "Bidirectional tensor transfers\n"
             "between GPU pairs via PyTorch",
             "Degraded NVLink bridges,\n"
             "asymmetric links (>20% delta)"],
            ["NCCL Validation",
             "AllReduce and AllGather\n"
             "collective benchmarks",
             "Topology misconfiguration,\n"
             "driver-level comm failures"],
            ["Topology Mapping",
             "nvidia-smi topo parsing,\n"
             "sysfs NUMA node affinity",
             "Missing NVLink, PLX switch\n"
             "grouping, NUMA imbalance"],
        ],
        col_widths=[W * 0.25, W * 0.37, W * 0.38],
    )
    story.append(inter_table)
    story.append(Paragraph(
        "Table 4. Interconnect validation modules.",
        s["Caption"],
    ))

    story.append(Paragraph(
        "The NCCL validation module implements a single-process "
        "ring-allreduce simulation that measures bus bandwidth using "
        "the standard formula: 2(n-1)/n * message_size / latency. "
        "While production NCCL benchmarks use multi-process execution "
        "via torchrun, this approach provides a reliable signal for "
        "detecting degraded interconnects without requiring distributed "
        "process coordination infrastructure.",
        s["BodyText2"],
    ))

    story.append(PageBreak())

    story.append(Paragraph(
        "3.4 Stress Testing (Level 3)", s["SubSection"],
    ))
    story.append(Paragraph(
        "Five modules exercise the GPU under sustained load to detect "
        "intermittent hardware failures that only manifest under thermal "
        "and electrical stress.",
        s["BodyText2"],
    ))

    stress_table = make_table(
        ["Module", "Workload", "Validates"],
        [
            ["Compute Stress",
             "Sustained 4096x4096 FP32 GEMM\n"
             "with 5s thermal sampling",
             "Stability under sustained\n"
             "compute load"],
            ["SM Stress",
             "FP32 peak GFLOPS (8192x8192)\n"
             "+ FP16 tensor core throughput",
             "SM throughput vs. spec"],
            ["Memory Bandwidth",
             "STREAM Copy (B=A) and\n"
             "STREAM Triad (A=B+s*C)",
             "HBM/GDDR bandwidth"],
            ["PCIe Bandwidth",
             "Pinned-memory H2D/D2H\n"
             "transfers with iteration avg",
             "PCIe link performance"],
            ["Power Test",
             "Multi-matrix stress targeting\n"
             "90% TDP with 2s power sampling",
             "Power delivery stability"],
        ],
        col_widths=[W * 0.22, W * 0.40, W * 0.38],
    )
    story.append(stress_table)
    story.append(Paragraph(
        "Table 5. Stress testing modules.",
        s["Caption"],
    ))

    # ─── Section 4: Production Integration ──────────────────
    story.append(Paragraph(
        "4. Production Integration", s["SectionHead"],
    ))

    story.append(Paragraph(
        "4.1 Structured Output Formats", s["SubSection"],
    ))
    story.append(Paragraph(
        "The framework produces three output formats to integrate with "
        "standard infrastructure tooling.",
        s["BodyText2"],
    ))

    output_table = make_table(
        ["Format", "Consumer", "Content"],
        [
            ["Rich Text", "Terminal operator",
             "Color-coded table with overall verdict"],
            ["JSON", "API consumers, Slurm",
             "Full results with run_id, timestamps,\n"
             "per-test details"],
            ["JUnit XML", "Jenkins, GitHub Actions,\nGitLab CI",
             "Standard test report with failure/\n"
             "error/skip/pass per testcase"],
        ],
        col_widths=[W * 0.17, W * 0.30, W * 0.53],
    )
    story.append(output_table)
    story.append(Paragraph(
        "Table 6. Output format mapping.",
        s["Caption"],
    ))

    story.append(Paragraph(
        "4.2 Prometheus Observability", s["SubSection"],
    ))
    story.append(Paragraph(
        "An embedded HTTP server exposes GPU metrics in Prometheus "
        "exposition format at /metrics, enabling direct integration "
        "with Grafana dashboards without external dependencies.",
        s["BodyText2"],
    ))

    metrics_items = [
        "<b>gpu_temperature_celsius</b> - Per-GPU temperature (gauge)",
        "<b>gpu_power_draw_watts</b> - Per-GPU power consumption (gauge)",
        "<b>gpu_memory_used_mib</b> - Per-GPU VRAM usage (gauge)",
        "<b>gpu_diagnostic_status</b> - Per-test pass/fail (gauge, 1/0)",
        "<b>gpu_diagnostic_duration_seconds</b> - Per-test timing (gauge)",
        "<b>gpu_diagnostic_run_total</b> - Cumulative run count (counter)",
    ]
    for m in metrics_items:
        story.append(Paragraph(
            f"\u2022  {m}", s["BulletItem"],
        ))

    story.append(Paragraph(
        "4.3 Idempotent GPU Cleanup", s["SubSection"],
    ))
    story.append(Paragraph(
        "The cleanup module executes automatically after every diagnostic "
        "run. It performs four reset operations: CUDA context and memory "
        "cache clearing, application clock reset to default, power limit "
        "restoration, and pending ECC page retirement detection. Each "
        "operation is independently error-handled; a failure in one "
        "does not block the others. This ensures diagnostic tests never "
        "leave GPUs in states that could impact subsequent workloads.",
        s["BodyText2"],
    ))

    # ─── Section 5: Execution Modes ─────────────────────────
    story.append(Paragraph(
        "5. Execution Modes", s["SectionHead"],
    ))

    mode_table = make_table(
        ["Mode", "Duration", "Use Case", "CLI Flag"],
        [
            ["Pre-flight", "~60s",
             "Run before every training job.\n"
             "Validates drivers, health, telemetry.",
             "--mode preflight"],
            ["Standard", "2-10 min",
             "Routine validation with\n"
             "configurable run level.",
             "--level <quick|med|long|ext>"],
            ["Burn-in", "1-24 hours",
             "New hardware acceptance.\n"
             "Sustained stress at 100% TDP.",
             "--mode burnin\n--duration 86400"],
        ],
        col_widths=[W * 0.14, W * 0.12, W * 0.42, W * 0.32],
    )
    story.append(mode_table)
    story.append(Paragraph(
        "Table 7. Execution modes.",
        s["Caption"],
    ))
    story.append(Paragraph(
        "The distinction between pre-flight and burn-in reflects "
        "production GPU fleet operations. Pre-flight checks run in "
        "Slurm prologue scripts before every job, catching driver "
        "crashes and thermal issues in seconds. Burn-in runs on newly "
        "racked servers, exercising every subsystem under sustained "
        "thermal and electrical stress to surface infant mortality "
        "failures before the node enters the production pool.",
        s["BodyText2"],
    ))

    story.append(PageBreak())

    # ─── Section 6: Failure Code Registry ───────────────────
    story.append(Paragraph(
        "6. Diagnostic Failure Code Registry", s["SectionHead"],
    ))
    story.append(Paragraph(
        "Every test failure produces a unique diagnostic code enabling "
        "automated triage and historical analysis.",
        s["BodyText2"],
    ))

    code_table = make_table(
        ["Code Range", "Category", "Examples"],
        [
            ["DIAG-001 - 010", "Deployment",
             "Driver not loaded, GPU count mismatch,\n"
             "model mismatch, ECC misconfigured"],
            ["DIAG-100 - 199", "GPU Health",
             "Temperature critical, clocks stuck"],
            ["DIAG-200 - 299", "PCIe Validation",
             "Link gen degraded, width degraded,\n"
             "replay counter non-zero"],
            ["DIAG-300 - 399", "Memory Test",
             "VRAM allocation failure,\n"
             "pattern integrity mismatch"],
            ["DIAG-400 - 499", "PCIe Bandwidth",
             "H2D/D2H below minimum threshold"],
            ["DIAG-500 - 599", "Memory Bandwidth",
             "STREAM copy/triad below spec"],
            ["DIAG-600 - 699", "Compute Stress",
             "GEMM errors under sustained load"],
            ["DIAG-700 - 799", "SM Stress",
             "FP32/FP16 throughput below target"],
            ["DIAG-800 - 899", "Power Test",
             "Failed to reach target TDP,\n"
             "power delivery instability"],
            ["DIAG-900 - 999", "Telemetry / Interconnect",
             "Critical XID, clock throttle,\n"
             "DBE detected, NVLink degraded,\n"
             "NCCL below threshold, topology fault"],
        ],
        col_widths=[W * 0.22, W * 0.22, W * 0.56],
    )
    story.append(code_table)
    story.append(Paragraph(
        "Table 8. Diagnostic failure code registry.",
        s["Caption"],
    ))

    # ─── Section 7: Configuration ───────────────────────────
    story.append(Paragraph(
        "7. Configuration Architecture", s["SectionHead"],
    ))
    story.append(Paragraph(
        "The framework uses a two-tier configuration system. A master "
        "configuration file (test_config.yaml) defines run levels, "
        "database connections, and monitoring parameters. Per-GPU "
        "profiles define hardware specifications and pass/fail "
        "thresholds. Deploying to a new GPU SKU requires creating "
        "a single YAML profile.",
        s["BodyText2"],
    ))

    profile_table = make_table(
        ["Profile", "VRAM", "PCIe", "TDP",
         "ECC", "NVLink"],
        [
            ["RTX 5070 Ti", "16 GiB", "Gen4 x16",
             "300W", "No", "No"],
            ["A100 80GB PCIe", "80 GiB", "Gen4 x16",
             "300W", "Yes", "No"],
            ["H100 SXM 80GB", "80 GiB", "Gen5 x16",
             "700W", "Yes", "18 links"],
        ],
        col_widths=[
            W * 0.22, W * 0.13, W * 0.17,
            W * 0.12, W * 0.12, W * 0.24,
        ],
    )
    story.append(profile_table)
    story.append(Paragraph(
        "Table 9. Included GPU profiles.",
        s["Caption"],
    ))

    # ─── Section 8: Containerized Deployment ────────────────
    story.append(Paragraph(
        "8. Containerized Deployment", s["SectionHead"],
    ))
    story.append(Paragraph(
        "The framework ships as a fully containerized stack via Docker "
        "Compose, providing reproducible execution across any environment "
        "with NVIDIA Container Toolkit installed. This eliminates "
        "dependency conflicts and ensures consistent behavior across "
        "development, CI, and production environments.",
        s["BodyText2"],
    ))

    story.append(Paragraph(
        "8.1 Multi-Stage Docker Build", s["SubSection"],
    ))
    story.append(Paragraph(
        "The Dockerfile uses a two-stage build pattern on the "
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04 base image. The builder "
        "stage installs Python dependencies into an isolated prefix; the "
        "runtime stage copies only the compiled packages, reducing the "
        "final image size by excluding build tools and headers. The "
        "container runs as a non-root user (gpudiag) for security.",
        s["BodyText2"],
    ))

    docker_features = [
        "<b>GPU passthrough.</b> Uses NVIDIA_VISIBLE_DEVICES=all and "
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility for full GPU access.",
        "<b>Health check.</b> Built-in Docker HEALTHCHECK probes the "
        "/health endpoint every 30 seconds, enabling orchestrator "
        "liveness detection.",
        "<b>Configurable entry point.</b> ENTRYPOINT is the CLI; CMD "
        "defaults to medium-level diagnostics with metrics server. "
        "Override via: docker run --gpus all gpu-diag diag --level long.",
        "<b>Volume mounts.</b> Reports persist to ./reports on the host; "
        "configuration is mounted read-only from ./config.",
    ]
    for item in docker_features:
        story.append(Paragraph(
            f"\u2022  {item}", s["BulletItem"],
        ))

    story.append(Paragraph(
        "8.2 Three-Service Compose Stack", s["SubSection"],
    ))
    story.append(Paragraph(
        "Docker Compose orchestrates the diagnostic service alongside "
        "its observability infrastructure. A single command deploys the "
        "complete monitoring pipeline.",
        s["BodyText2"],
    ))

    compose_table = make_table(
        ["Service", "Image", "Port", "Purpose"],
        [
            ["gpu-diag", "gpu-diag:latest\n(custom build)",
             "9835", "Diagnostic runner +\nPrometheus exporter"],
            ["prometheus", "prom/prometheus:v2.51.0",
             "9090", "Metrics scraping +\nalert evaluation"],
            ["grafana", "grafana/grafana:10.4.0",
             "3000", "Dashboard visualization\n+ alerting UI"],
        ],
        col_widths=[W * 0.16, W * 0.28, W * 0.12, W * 0.44],
    )
    story.append(compose_table)
    story.append(Paragraph(
        "Table 10. Docker Compose service architecture.",
        s["Caption"],
    ))

    story.append(Paragraph(
        "The gpu-diag service requires the NVIDIA runtime; Prometheus "
        "and Grafana run on standard containers. Prometheus is configured "
        "with a 10-second scrape interval and 30-day retention. Grafana "
        "auto-provisions the Prometheus datasource and a pre-built GPU "
        "diagnostics dashboard on first boot via file-based provisioning.",
        s["BodyText2"],
    ))

    story.append(PageBreak())

    # ─── Section 9: Grafana Dashboard ─────────────────────
    story.append(Paragraph(
        "9. Grafana Observability Dashboard", s["SectionHead"],
    ))
    story.append(Paragraph(
        "The pre-built Grafana dashboard provides six monitoring panels "
        "organized into four sections, designed for both real-time "
        "operator use during burn-in and historical trend analysis.",
        s["BodyText2"],
    ))

    grafana_table = make_table(
        ["Panel", "Visualization", "Metrics"],
        [
            ["Overall Verdict", "Stat (PASS/FAIL)",
             "min(gpu_diagnostic_status)"],
            ["Temperature", "Time series + thresholds",
             "gpu_temperature_celsius\nper GPU, 75C/85C thresholds"],
            ["Power Draw", "Time series + thresholds",
             "gpu_power_draw_watts\nper GPU, 250W/290W thresholds"],
            ["Test Results", "Stat grid (color-coded)",
             "gpu_diagnostic_status\nper test, PASS=green FAIL=red"],
            ["Test Duration", "Bar chart",
             "gpu_diagnostic_duration_seconds\nper test"],
            ["VRAM Usage", "Time series",
             "gpu_memory_used_mib per GPU"],
            ["Graphics Clock", "Time series",
             "gpu_clock_graphics_mhz per GPU"],
        ],
        col_widths=[W * 0.22, W * 0.28, W * 0.50],
    )
    story.append(grafana_table)
    story.append(Paragraph(
        "Table 11. Grafana dashboard panels.",
        s["Caption"],
    ))

    story.append(Paragraph(
        "9.1 Prometheus Alerting Rules", s["SubSection"],
    ))
    story.append(Paragraph(
        "Six alerting rules are pre-configured for critical GPU events. "
        "These rules evaluate on the Prometheus server and can route to "
        "PagerDuty, Slack, or email via Alertmanager.",
        s["BodyText2"],
    ))

    alert_table = make_table(
        ["Alert", "Condition", "Severity"],
        [
            ["GPUTemperatureCritical",
             "> 85C for 2 minutes", "Critical"],
            ["GPUTemperatureWarning",
             "> 75C for 5 minutes", "Warning"],
            ["GPUDiagnosticFailed",
             "Any test status = 0", "Critical"],
            ["GPUPowerExcessive",
             "> 290W for 2 minutes", "Warning"],
            ["GPUECCDoublebitError",
             "DBE count > 0", "Critical"],
            ["GPUECCSinglebitRising",
             "SBE rate > 0.1/hr for 10min", "Warning"],
        ],
        col_widths=[W * 0.35, W * 0.40, W * 0.25],
    )
    story.append(alert_table)
    story.append(Paragraph(
        "Table 12. Pre-configured Prometheus alerting rules.",
        s["Caption"],
    ))

    # ─── Section 10: Testing Strategy ─────────────────────
    story.append(Paragraph(
        "10. Testing and CI/CD", s["SectionHead"],
    ))
    story.append(Paragraph(
        "The framework maintains 153 test assertions across 12 test "
        "files, achieving full CI compatibility without GPU hardware. "
        "All NVML and CUDA operations are comprehensively mocked, "
        "enabling the complete test suite to run in GitHub Actions on "
        "standard Ubuntu runners in under 3 seconds.",
        s["BodyText2"],
    ))
    story.append(Paragraph(
        "The CI pipeline runs two parallel jobs: static analysis via "
        "ruff (import ordering, unused variables, line length) and "
        "test execution on Python 3.10 and 3.12 matrices. Test results "
        "are uploaded as JUnit XML artifacts for GitHub Actions "
        "integration.",
        s["BodyText2"],
    ))

    # ─── Section 11: Deployment Quick Reference ────────────
    story.append(Paragraph(
        "11. Deployment Quick Reference", s["SectionHead"],
    ))

    deploy_table = make_table(
        ["Method", "Command", "Use Case"],
        [
            ["Local Python",
             "python -m src.main diag --level medium",
             "Development, single-server"],
            ["Docker (single)",
             "docker run --gpus all gpu-diag\n"
             "diag --level long",
             "Isolated execution"],
            ["Docker Compose",
             "docker compose up -d",
             "Full stack with monitoring"],
            ["CI Pipeline",
             "pytest tests/ -v --tb=short",
             "GitHub Actions (no GPU)"],
            ["Burn-in",
             "docker run --gpus all gpu-diag\n"
             "diag --mode burnin --duration 86400",
             "New hardware acceptance"],
        ],
        col_widths=[W * 0.20, W * 0.45, W * 0.35],
    )
    story.append(deploy_table)
    story.append(Paragraph(
        "Table 13. Deployment methods.",
        s["Caption"],
    ))

    story.append(PageBreak())

    # ─── Section 12: Maturity Matrix ──────────────────────
    story.append(Paragraph(
        "12. Maturity Assessment", s["SectionHead"],
    ))
    story.append(Paragraph(
        "The following matrix maps the framework's current capabilities "
        "against the industry standard for production GPU fleet tooling.",
        s["BodyText2"],
    ))

    maturity_table = make_table(
        ["Capability", "Industry Standard", "Current State"],
        [
            ["Diagnostics",
             "DCGM-level (XID, ECC, throttle)",
             "Implemented: 16 modules"],
            ["Networking",
             "NCCL/NVLink bandwidth tests",
             "Implemented: P2P + NCCL + topo"],
            ["Output",
             "JSON / Prometheus structured data",
             "Implemented: JSON + JUnit + Prom"],
            ["Environment",
             "Dockerized / Kubernetes-ready",
             "Implemented: Docker Compose stack"],
            ["Visualization",
             "Grafana dashboards + alerting",
             "Implemented: 7 panels + 6 alerts"],
            ["CI/CD",
             "Automated lint + test pipeline",
             "Implemented: GitHub Actions"],
            ["Cleanup",
             "Idempotent GPU state reset",
             "Implemented: 4-stage cleanup"],
        ],
        col_widths=[W * 0.22, W * 0.39, W * 0.39],
    )
    story.append(maturity_table)
    story.append(Paragraph(
        "Table 14. Industry maturity assessment.",
        s["Caption"],
    ))

    # ─── Section 13: Future Work ──────────────────────────
    story.append(Paragraph(
        "13. Future Work", s["SectionHead"],
    ))
    future_items = [
        "<b>Multi-process NCCL benchmarks.</b> Integrate nccl-tests "
        "with torchrun for true distributed AllReduce measurement "
        "across multi-node GPU clusters.",
        "<b>Kubernetes-native deployment.</b> Helm chart for running "
        "diagnostics as a DaemonSet with results exported to the "
        "cluster's Prometheus/Grafana stack.",
        "<b>Historical trend analysis.</b> PostgreSQL persistence with "
        "SQLAlchemy for tracking error rate trends, enabling predictive "
        "maintenance alerts before hardware failure.",
        "<b>Memtest (Level 4).</b> Bit-pattern walking memory test "
        "for exhaustive VRAM integrity validation during burn-in.",
        "<b>Alertmanager integration.</b> Route Prometheus alerts to "
        "PagerDuty, Slack, and email for on-call GPU fleet engineers.",
    ]
    for item in future_items:
        story.append(Paragraph(
            f"\u2022  {item}", s["BulletItem"],
        ))

    story.append(Spacer(1, 24))
    story.append(HRFlowable(
        width="100%", thickness=0.5, color=MED_GRAY,
        spaceAfter=12,
    ))
    story.append(Paragraph(
        "Repository: "
        "github.com/omnipotence-eth/gpu-server-test-suite",
        ParagraphStyle(
            "repo", parent=s["BodyText2"],
            alignment=TA_CENTER,
            fontSize=9,
            textColor=ACCENT_BLUE,
        ),
    ))

    # Build
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print("PDF generated: docs/GPU_Server_Diagnostic_Framework.pdf")


if __name__ == "__main__":
    build_document()
