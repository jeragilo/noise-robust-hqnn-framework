"""
Main Pipeline: Noise Robustness Benchmark

This pipeline converts real Demo 13 cross-noise results into
framework-level robustness summaries, curves, and heatmaps.

It highlights the framework's noise-analysis capability using real
outputs from the thesis demonstrations.
"""

import json
from pathlib import Path

from framework.reporting import save_csv, save_json, plot_noise_curves, plot_heatmap
from framework.robustness_metrics import degradation_slope, cross_framework_deviation


RESULTS_DIR = Path("results")
FRAMEWORK_DIR = RESULTS_DIR / "framework"


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def run_pipeline():
    demo13_path = RESULTS_DIR / "demo13" / "noise_matrix_demo13.json"

    if not demo13_path.exists():
        print("ERROR: Demo 13 results not found.")
        print("Run Demo 13 first.")
        return

    demo13 = load_json(demo13_path)

    noise_levels = demo13["noise_levels"]
    matrix = demo13["matrix"]

    rows = []

    for framework_name, values in matrix.items():
        rows.append(
            {
                "source_demo": "Demo 13",
                "framework": framework_name,
                "noise_levels": str(noise_levels),
                "values": str(values),
                "mean_value": sum(values) / len(values),
                "min_value": min(values),
                "max_value": max(values),
                "degradation_slope": degradation_slope(noise_levels, values),
                "interpretation": "Expectation-value drift under depolarizing noise benchmark.",
            }
        )

    # Cross-framework deviation at each noise level
    deviation_rows = []

    for i, level in enumerate(noise_levels):
        values_at_level = [matrix[fw][i] for fw in matrix.keys()]
        deviation_rows.append(
            {
                "source_demo": "Demo 13",
                "noise_level": level,
                "cross_framework_deviation": cross_framework_deviation(values_at_level),
            }
        )

    save_csv(rows, str(FRAMEWORK_DIR / "noise_robustness_real_summary.csv"))
    save_csv(deviation_rows, str(FRAMEWORK_DIR / "noise_cross_framework_deviation.csv"))

    save_json(
        {
            "description": "Framework-level noise robustness summary using real Demo 13 outputs.",
            "noise_levels": noise_levels,
            "matrix": matrix,
            "summary_rows": rows,
            "cross_framework_deviation": deviation_rows,
        },
        str(FRAMEWORK_DIR / "noise_robustness_real_summary.json"),
    )

    plot_noise_curves(
        noise_levels,
        matrix,
        str(FRAMEWORK_DIR / "noise_robustness_real_curves.png"),
        title="Demo 13 — Real Cross-Framework Noise Drift",
        ylabel="Expectation Value <ZZ>",
    )

    heatmap_matrix = [matrix[fw] for fw in matrix.keys()]

    plot_heatmap(
        heatmap_matrix,
        [str(level) for level in noise_levels],
        list(matrix.keys()),
        str(FRAMEWORK_DIR / "noise_robustness_real_heatmap.png"),
        title="Demo 13 — Real Noise Robustness Heatmap",
        colorbar_label="Expectation Value <ZZ>",
    )

    print("Noise robustness REAL results pipeline complete.")
    print("Saved outputs to results/framework/")


if __name__ == "__main__":
    run_pipeline()
