"""Reproduce every result table and figure of the paper, in order.

    python src/run_all.py            # full pipeline
    python src/run_all.py --figures  # figures only (needs results/)

The pipeline needs the LEMIS extract as a DuckDB database; see docs/DATA.md.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS: list[tuple[str, list[str]]] = [
    ("whole_animals.py", []),
    ("metric_robustness.py", []),
    ("analysis.py", ["rolling", "bootstrap", "sweep", "lambda", "composition",
                     "concentration", "alt_score", "shipment", "purpose_p"]),
    ("upgrade.py", ["delete_top_ci", "shadow_cost"]),
    ("revisions.py", []),
    ("descriptives.py", []),
    ("sankey.py", []),
    ("figdata.py", []),
    ("split_probe.py", []),
    ("case_audit.py", []),
    ("figures.py", []),
]

FIGURES = [s for s in STEPS if s[0] == "figures.py"]


def run(step: tuple[str, list[str]]) -> None:
    name, args = step
    print(f"\n=== {name} {' '.join(args)}", flush=True)
    res = subprocess.run([sys.executable, str(HERE / name), *args])
    if res.returncode != 0:
        sys.exit(f"FAILED: {name}")


def main() -> None:
    steps = FIGURES if "--figures" in sys.argv else STEPS
    for step in steps:
        run(step)
    print("\nALL DONE: results/ and figures/ are up to date")


if __name__ == "__main__":
    main()
