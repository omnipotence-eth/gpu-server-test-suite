"""Generate a demo SVG of the diagnostic output for the repository.

Runs a real medium-level diagnostic against the local GPU, captures
the Rich terminal output, and exports it as an SVG to docs/demo.svg.

Usage:
    conda activate mlenv
    python scripts/generate_demo.py
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

# Patch the global console in main BEFORE click imports or runs anything.
# This intercepts all console.print() calls made during the diagnostic run.
import src.main as _main_module

recording = Console(record=True, width=110, force_terminal=True, force_jupyter=False)
_main_module.console = recording

from click.testing import CliRunner

from src.main import cli

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "demo.svg"
)

if __name__ == "__main__":
    print("Running medium-level diagnostic for demo capture...")
    runner = CliRunner()
    result = runner.invoke(cli, ["diag", "--level", "medium"])

    if result.exit_code != 0 and result.exception:
        import traceback
        traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    recording.save_svg(OUTPUT_PATH, title="GPU Server Diagnostic Suite — RTX 5070 Ti")

    # Rich's SVG export hardcodes macOS-style traffic light buttons regardless
    # of platform. Strip them — this tool runs on Linux/Windows servers.
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        svg = f.read()
    svg = svg.replace('<circle cx="0" cy="0" r="7" fill="#ff5f57"/>', "")
    svg = svg.replace('<circle cx="22" cy="0" r="7" fill="#febc2e"/>', "")
    svg = svg.replace('<circle cx="44" cy="0" r="7" fill="#28c840"/>', "")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"Saved: {OUTPUT_PATH}")
