"""
Main Pipeline: Framework Capabilities Report

This script generates a machine-readable and human-readable report explaining
what the HQNN framework provides beyond default libraries such as Qiskit,
PennyLane, Cirq, scikit-learn, and Qiskit Machine Learning.
"""

from pathlib import Path
from framework.reporting import save_json

FRAMEWORK_DIR = Path("results/framework")


def run_pipeline():
    FRAMEWORK_DIR.mkdir(parents=True, exist_ok=True)

    capabilities = {
        "framework_name": "Noise-Robust HQNN Benchmarking Framework",
        "purpose": (
            "Provide a reusable benchmarking layer for evaluating hybrid quantum "
            "machine learning models under NISQ-era constraints."
        ),
        "uses_existing_libraries": [
            "Qiskit",
            "Qiskit Aer",
            "Qiskit Machine Learning",
            "Cirq",
            "PennyLane",
            "scikit-learn",
        ],
        "what_default_libraries_provide": [
            "Quantum circuits",
            "Quantum simulators",
            "Quantum kernels",
            "Classical machine learning models",
            "Basic plotting and numerical tools",
        ],
        "what_this_framework_adds": [
            {
                "feature": "Standardized dataset layer",
                "code_location": "framework/datasets.py",
                "description": (
                    "Provides synthetic, Iris, and Wisconsin Diagnostic Breast Cancer "
                    "dataset loaders with consistent quantum-compatible preprocessing."
                ),
            },
            {
                "feature": "Noise-analysis toolbox",
                "code_location": "framework/noise_channels.py",
                "description": (
                    "Centralizes depolarizing, bit-flip, phase-flip, and amplitude-damping "
                    "noise models with reusable sweep support."
                ),
            },
            {
                "feature": "Robustness metrics",
                "code_location": "framework/robustness_metrics.py",
                "description": (
                    "Adds accuracy drop, relative robustness score, degradation slope, "
                    "training instability, and cross-framework deviation metrics."
                ),
            },
            {
                "feature": "Standardized reporting",
                "code_location": "framework/reporting.py",
                "description": (
                    "Standardizes JSON, CSV, accuracy plots, noise curves, heatmaps, "
                    "and training-curve outputs."
                ),
            },
            {
                "feature": "Benchmark orchestration",
                "code_location": "framework/benchmark_runner.py",
                "description": (
                    "Aggregates hybrid-vs-classical, noise-robustness, and cross-framework "
                    "validation outputs into thesis-ready benchmark rows."
                ),
            },
            {
                "feature": "Main contribution pipelines",
                "code_location": "pipelines/",
                "description": (
                    "Provides main pipeline files for hybrid-vs-classical comparison, "
                    "noise robustness, cross-framework validation, and final benchmark summary."
                ),
            },
            {
                "feature": "Cybersecurity anomaly benchmark integration",
                "code_location": "demos/industry/demo11_cyber_anomaly_qiskit.py",
                "description": (
                    "Combines quantum kernel classification, HQNN training, and classical "
                    "logistic regression in one domain-inspired cybersecurity benchmark."
                ),
            },
        ],
        "nonstandard_framework_features": [
            "accuracy_drop",
            "robustness_score",
            "degradation_slope",
            "training_instability",
            "cross_framework_deviation",
            "final_benchmark_summary",
        ],
        "main_pipelines": [
            "python -m pipelines.main_hybrid_vs_classical",
            "python -m pipelines.main_noise_robustness",
            "python -m pipelines.main_cross_framework_validation",
            "python -m pipelines.main_full_benchmark_summary",
            "python -m pipelines.main_framework_capabilities_report",
        ],
    }

    save_json(
        capabilities,
        str(FRAMEWORK_DIR / "framework_capabilities_report.json"),
    )

    markdown_path = FRAMEWORK_DIR / "framework_capabilities_report.md"

    with open(markdown_path, "w") as f:
        f.write("# Framework Capabilities Report\n\n")
        f.write("## Purpose\n\n")
        f.write(capabilities["purpose"] + "\n\n")

        f.write("## Existing Libraries Used\n\n")
        for item in capabilities["uses_existing_libraries"]:
            f.write(f"- {item}\n")

        f.write("\n## What Default Libraries Provide\n\n")
        for item in capabilities["what_default_libraries_provide"]:
            f.write(f"- {item}\n")

        f.write("\n## What This Framework Adds\n\n")
        for item in capabilities["what_this_framework_adds"]:
            f.write(f"### {item['feature']}\n")
            f.write(f"- Code: `{item['code_location']}`\n")
            f.write(f"- Description: {item['description']}\n\n")

        f.write("## Nonstandard Framework Features\n\n")
        for item in capabilities["nonstandard_framework_features"]:
            f.write(f"- `{item}`\n")

        f.write("\n## Main Pipelines\n\n")
        for item in capabilities["main_pipelines"]:
            f.write(f"```bash\n{item}\n```\n\n")

    print("Framework capabilities report complete.")
    print("Saved:")
    print(FRAMEWORK_DIR / "framework_capabilities_report.json")
    print(FRAMEWORK_DIR / "framework_capabilities_report.md")


if __name__ == "__main__":
    run_pipeline()
