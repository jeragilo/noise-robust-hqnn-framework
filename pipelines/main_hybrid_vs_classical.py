"""
Main Pipeline: Hybrid vs Classical Benchmark

This pipeline highlights one of the core framework contributions of the thesis:
a unified comparison between hybrid quantum models and classical baselines
using standardized reporting.

This version reads REAL outputs from Demo 09 and Demo 11.
"""

import json
from pathlib import Path

from framework.reporting import save_csv, save_json, plot_accuracy_comparison


RESULTS_DIR = Path("results")
FRAMEWORK_DIR = RESULTS_DIR / "framework"


def load_json(path: Path):
    """Load a JSON result file."""
    with open(path, "r") as f:
        return json.load(f)


def run_pipeline():
    rows = []

    # ------------------------------------------------------------
    # Demo 09 — Medical Risk Classification
    # ------------------------------------------------------------
    demo09_path = RESULTS_DIR / "demo09" / "results_demo09_medical.json"
    demo09 = load_json(demo09_path)

    medical_results = {
        "HQNN": float(demo09["hqnn_final_accuracy"]),
        "Logistic Regression": float(demo09["classical_accuracy"]),
    }

    for model, acc in medical_results.items():
        rows.append(
            {
                "source_demo": "Demo 09",
                "dataset": "Synthetic Medical Risk",
                "task": "Medical risk classification",
                "model": model,
                "accuracy": acc,
                "auc": "",
                "notes": "Real output from Demo 09 JSON",
            }
        )

    plot_accuracy_comparison(
        list(medical_results.keys()),
        list(medical_results.values()),
        str(FRAMEWORK_DIR / "demo09_medical_hybrid_vs_classical.png"),
        title="Demo 09 — Medical Risk: Hybrid vs Classical",
    )

    # ------------------------------------------------------------
    # Demo 11 — Cybersecurity Anomaly Detection
    # ------------------------------------------------------------
    demo11_path = RESULTS_DIR / "demo11" / "results_demo11_cyber.json"
    demo11 = load_json(demo11_path)

    cyber_results = {
        "QSVM": float(demo11["qsvm_accuracy"]),
        "HQNN": float(demo11["hqnn_accuracy"]),
        "Logistic Regression": float(demo11["classical_accuracy"]),
    }

    for model, acc in cyber_results.items():
        rows.append(
            {
                "source_demo": "Demo 11",
                "dataset": "Synthetic Cybersecurity Traffic",
                "task": "Cybersecurity anomaly detection",
                "model": model,
                "accuracy": acc,
                "auc": float(demo11["qsvm_auc"]) if model == "QSVM" else "",
                "notes": "Real output from Demo 11 JSON",
            }
        )

    plot_accuracy_comparison(
        list(cyber_results.keys()),
        list(cyber_results.values()),
        str(FRAMEWORK_DIR / "demo11_cyber_hybrid_vs_classical.png"),
        title="Demo 11 — Cybersecurity: Quantum vs Hybrid vs Classical",
    )

    # ------------------------------------------------------------
    # Save combined framework-level summary
    # ------------------------------------------------------------
    save_csv(
        rows,
        str(FRAMEWORK_DIR / "hybrid_vs_classical_real_summary.csv"),
    )

    save_json(
        {
            "description": (
                "Framework-level hybrid-versus-classical comparison using real "
                "outputs from Demo 09 and Demo 11."
            ),
            "rows": rows,
        },
        str(FRAMEWORK_DIR / "hybrid_vs_classical_real_summary.json"),
    )

    print("Hybrid vs classical REAL results pipeline complete.")
    print("Saved outputs to results/framework/")


if __name__ == "__main__":
    run_pipeline()
