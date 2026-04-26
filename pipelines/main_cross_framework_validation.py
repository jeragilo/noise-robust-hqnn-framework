"""
Main Pipeline: Cross-Framework Validation

This pipeline evaluates whether quantum results are consistent across:

- Qiskit
- Cirq
- PennyLane

It supports the thesis claim that results are not framework-specific.
"""

import json
from pathlib import Path

from framework.benchmark_runner import run_cross_framework_validation
from framework.reporting import save_csv, save_json, plot_accuracy_comparison


RESULTS_DIR = Path("results")
FRAMEWORK_DIR = RESULTS_DIR / "framework"


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def run_pipeline():
    # ------------------------------------------------------------
    # Load Demo 07 results
    # ------------------------------------------------------------
    demo07_path = RESULTS_DIR / "demo07" / "results_demo07.json"

    if not demo07_path.exists():
        print("ERROR: Demo 07 results not found.")
        print("Make sure demo07 has been run.")
        return

    demo07 = load_json(demo07_path)

    # Expected structure:
    # {
    #   "qiskit": {"Z0":..., "Z1":..., "ZZ":..., "XX":...},
    #   "cirq": {...},
    #   "pennylane": {...}
    # }

    rows = run_cross_framework_validation(demo07)

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------
    save_csv(
        rows,
        str(FRAMEWORK_DIR / "cross_framework_summary.csv"),
    )

    # ------------------------------------------------------------
    # Prepare plot (ZZ comparison for simplicity)
    # ------------------------------------------------------------
    frameworks = ["qiskit", "cirq", "pennylane"]
    zz_values = [demo07[fw]["ZZ"] for fw in frameworks]

    plot_accuracy_comparison(
        frameworks,
        zz_values,
        str(FRAMEWORK_DIR / "cross_framework_ZZ_comparison.png"),
        title="Cross-Framework ZZ Expectation Comparison",
        ylabel="Expectation Value",
    )

    # ------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------
    save_json(
        {
            "description": "Cross-framework validation results using Demo 07 outputs",
            "rows": rows,
        },
        str(FRAMEWORK_DIR / "cross_framework_summary.json"),
    )

    print("Cross-framework validation complete.")
    print("Saved outputs to results/framework/")


if __name__ == "__main__":
    run_pipeline()
