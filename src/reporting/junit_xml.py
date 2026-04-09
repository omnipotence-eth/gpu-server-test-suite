"""JUnit XML report generation for CI/CD integration.

Produces JUnit XML output compatible with Jenkins, GitLab CI,
GitHub Actions, and Slurm job schedulers. Enables GPU diagnostic
results to be parsed by standard CI tooling.

JUnit XML maps test results as:
  - PASS → <testcase> (no child elements)
  - FAIL → <testcase><failure>
  - ERROR → <testcase><error>
  - SKIP → <testcase><skipped>
  - WARN → <testcase> with <system-out> warning

Output conforms to JUnit XML schema:
  https://llg.cubic.org/docs/junit/
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

from src.reporting.models import DiagnosticRun, TestResult, TestStatus


def _result_to_testcase(result: TestResult) -> ET.Element:
    """Convert a TestResult to a JUnit <testcase> XML element."""
    # Split test_name into classname.name (e.g., "health.temperature")
    parts = result.test_name.rsplit(".", 1)
    if len(parts) == 2:
        classname, name = parts
    else:
        classname = "gpu_diagnostics"
        name = result.test_name

    tc = ET.Element(
        "testcase",
        {
            "classname": classname,
            "name": name,
            "time": f"{result.duration_seconds:.3f}",
        },
    )

    if result.status == TestStatus.FAIL:
        failure = ET.SubElement(
            tc,
            "failure",
            {
                "message": result.message,
                "type": result.failure_code or "AssertionError",
            },
        )
        failure.text = _format_details(result)

    elif result.status == TestStatus.ERROR:
        error = ET.SubElement(
            tc,
            "error",
            {
                "message": result.message,
                "type": result.failure_code or "RuntimeError",
            },
        )
        error.text = _format_details(result)

    elif result.status == TestStatus.SKIP:
        ET.SubElement(
            tc,
            "skipped",
            {
                "message": result.message,
            },
        )

    elif result.status == TestStatus.WARN:
        # JUnit has no WARN — use system-out for visibility
        sysout = ET.SubElement(tc, "system-out")
        sysout.text = f"WARNING: {result.message}"

    # Add GPU UUID as property if present
    if result.gpu_uuid:
        props = ET.SubElement(tc, "properties")
        ET.SubElement(
            props,
            "property",
            {
                "name": "gpu_uuid",
                "value": result.gpu_uuid,
            },
        )

    return tc


def _format_details(result: TestResult) -> str:
    """Format TestResult details as readable text for XML body."""
    lines = [
        f"Test: {result.test_name}",
        f"Status: {result.status.value}",
        f"Message: {result.message}",
    ]
    if result.failure_code:
        lines.append(f"Failure Code: {result.failure_code}")
    if result.gpu_uuid:
        lines.append(f"GPU UUID: {result.gpu_uuid}")
    if result.details:
        lines.append("Details:")
        for key, value in result.details.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def results_to_junit_xml(
    results: list[TestResult],
    suite_name: str = "gpu_diagnostics",
    run_id: str = "",
) -> str:
    """Convert a list of TestResults to JUnit XML string.

    Args:
        results: List of TestResult objects.
        suite_name: Name for the <testsuite> element.
        run_id: Optional run ID for the suite.

    Returns:
        Pretty-printed JUnit XML string.
    """
    # Compute summary counts
    tests = len(results)
    failures = sum(1 for r in results if r.status == TestStatus.FAIL)
    errors = sum(1 for r in results if r.status == TestStatus.ERROR)
    skipped = sum(1 for r in results if r.status == TestStatus.SKIP)
    total_time = sum(r.duration_seconds for r in results)

    # Build XML
    suite = ET.Element(
        "testsuite",
        {
            "name": suite_name,
            "tests": str(tests),
            "failures": str(failures),
            "errors": str(errors),
            "skipped": str(skipped),
            "time": f"{total_time:.3f}",
        },
    )

    if run_id:
        suite.set("id", run_id)

    for result in results:
        suite.append(_result_to_testcase(result))

    # Pretty print
    rough = ET.tostring(suite, encoding="unicode")
    dom = minidom.parseString(rough)
    return dom.toprettyxml(indent="  ", encoding=None)


def diagnostic_run_to_junit_xml(run: DiagnosticRun) -> str:
    """Convert a DiagnosticRun to JUnit XML string.

    Maps the DiagnosticRun metadata as suite-level properties.
    """
    xml = results_to_junit_xml(
        results=run.results,
        suite_name=f"gpu_diagnostics.{run.run_level}",
        run_id=run.run_id,
    )
    return xml


def write_junit_xml(
    results: list[TestResult],
    output_path: str,
    suite_name: str = "gpu_diagnostics",
) -> str:
    """Write JUnit XML report to a file.

    Args:
        results: List of TestResult objects.
        output_path: File path for the XML output.
        suite_name: Name for the <testsuite>.

    Returns:
        The output file path.
    """
    xml_str = results_to_junit_xml(results, suite_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return output_path
