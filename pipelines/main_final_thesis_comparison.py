"""
Pipeline: Final Thesis Comparison

Combines the major algorithmic improvements into one thesis-ready comparison:
1. Fixed parity HQNN baseline
2. Learned readout HQNN
3. Multi-observable HQNN
4. Architecture search
5. Best-architecture noise sweep
6. Repeated-trial validation
7. Statistical validation ranking
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


FRAMEWORK_DIR = Path("results") / "framework"
OUTPUT_DIR = FRAMEWORK_DIR / "final_thesis_comparison"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def add_row(
    rows: List[Dict[str, Any]],
    experiment: str,
    method: str,
    clean_accuracy: float,
    noisy_accuracy: float,
    accuracy_drop: float,
    robustness_score: float,
    main_contribution: str,
) -> None:
    rows.append(
        {
            "experiment": experiment,
            "method": method,
            "clean_accuracy": float(clean_accuracy),
            "noisy_accuracy": float(noisy_accuracy),
            "accuracy_drop": float(accuracy_drop),
            "robustness_score": float(robustness_score),
            "main_contribution": main_contribution,
        }
    )


def plot_final_comparison(rows: List[Dict[str, Any]], output_path: Path) -> None:
    labels = [row["method"] for row in rows]
    clean = [row["clean_accuracy"] for row in rows]
    noisy = [row["noisy_accuracy"] for row in rows]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, clean, width, label="Clean Accuracy")
    plt.bar(x + width / 2, noisy, width, label="Noisy Accuracy")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Final Thesis Comparison: HQNN Algorithmic Improvements")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_noisy_only(rows: List[Dict[str, Any]], output_path: Path) -> None:
    ranked = sorted(rows, key=lambda r: r["noisy_accuracy"], reverse=True)

    labels = [row["method"] for row in ranked]
    noisy = [row["noisy_accuracy"] for row in ranked]

    plt.figure(figsize=(14, 6))
    plt.bar(labels, noisy)
    plt.ylim(0, 1)
    plt.ylabel("Noisy Accuracy")
    plt.title("Final Thesis Ranking by Noisy Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_pipeline() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    learned = load_json(
        FRAMEWORK_DIR / "learned_readout_hqnn" / "learned_readout_summary.json"
    )

    multi = load_json(
        FRAMEWORK_DIR / "multi_observable_hqnn" / "multi_observable_summary.json"
    )

    arch = load_json(
        FRAMEWORK_DIR / "architecture_search_hqnn" / "architecture_search_summary.json"
    )

    sweep = load_json(
        FRAMEWORK_DIR / "best_architecture_noise_sweep" / "best_architecture_noise_sweep_summary.json"
    )

    repeated = load_json(
        FRAMEWORK_DIR / "best_architecture_repeated_trials" / "repeated_trials_summary.json"
    )

    statistical = load_json(
        FRAMEWORK_DIR / "statistical_validation" / "statistical_validation_summary.json"
    )

    rows: List[Dict[str, Any]] = []

    add_row(
        rows,
        "Learned Readout HQNN",
        "Fixed Parity Readout",
        learned["parity_readout"]["clean_accuracy"],
        learned["parity_readout"]["noisy_accuracy"],
        learned["parity_readout"]["accuracy_drop"],
        learned["parity_readout"]["robustness_score"],
        "Baseline HQNN readout using one fixed parity observable.",
    )

    add_row(
        rows,
        "Learned Readout HQNN",
        "Learned Logistic Readout",
        learned["learned_readout"]["clean_accuracy"],
        learned["learned_readout"]["noisy_accuracy"],
        learned["learned_readout"]["accuracy_drop"],
        learned["learned_readout"]["robustness_score"],
        "Uses quantum measurement distributions as features for a learned classical readout.",
    )

    add_row(
        rows,
        "Multi-Observable HQNN",
        "Multi-Observable LR",
        multi["multi_observable_logistic_readout"]["clean_accuracy"],
        multi["multi_observable_logistic_readout"]["noisy_accuracy"],
        multi["multi_observable_logistic_readout"]["accuracy_drop"],
        multi["multi_observable_logistic_readout"]["robustness_score"],
        "Uses probability, Z expectation, ZZ correlation, parity, and distribution statistics.",
    )

    add_row(
        rows,
        "Multi-Observable HQNN",
        "Multi-Observable RF",
        multi["multi_observable_random_forest_readout"]["clean_accuracy"],
        multi["multi_observable_random_forest_readout"]["noisy_accuracy"],
        multi["multi_observable_random_forest_readout"]["accuracy_drop"],
        multi["multi_observable_random_forest_readout"]["robustness_score"],
        "Improves HQNN readout by using richer quantum features and a nonlinear classifier.",
    )

    best_arch = "linear"
    best_arch_result = arch["results"][best_arch]["random_forest"]

    add_row(
        rows,
        "Architecture Search HQNN",
        "Best Architecture: Linear + RF",
        best_arch_result["clean_accuracy"],
        best_arch_result["noisy_accuracy"],
        best_arch_result["accuracy_drop"],
        best_arch_result["robustness_score"],
        "Shows that entanglement architecture affects HQNN clean and noisy performance.",
    )

    add_row(
        rows,
        "Best Architecture Noise Sweep",
        "Linear + Multi-Observable RF Sweep",
        sweep["clean_accuracy"],
        sweep["worst_noisy_accuracy"],
        sweep["max_accuracy_drop"],
        sweep["mean_robustness_score"],
        "Validates the optimized HQNN configuration across increasing noise levels.",
    )

    repeated_agg = repeated["aggregate_results"]

    add_row(
        rows,
        "Repeated-Trial Validation",
        "Repeated Trials Mean",
        repeated_agg["mean_clean_accuracy"],
        repeated_agg["mean_noisy_accuracy"],
        repeated_agg["mean_accuracy_drop"],
        repeated_agg["mean_robustness_score"],
        "Tests whether the optimized HQNN result remains strong across randomized trials.",
    )

    stat_best = statistical["best_method"]

    add_row(
        rows,
        "Statistical Validation",
        "Statistical Best Method",
        stat_best["mean_clean_accuracy"],
        stat_best["mean_noisy_accuracy"],
        stat_best["mean_accuracy_drop"],
        stat_best["mean_robustness_score"],
        "Ranks all thesis methods by noisy accuracy, robustness, and stability.",
    )

    best_overall = max(rows, key=lambda row: row["noisy_accuracy"])
    baseline = rows[0]
    best_gain = best_overall["noisy_accuracy"] - baseline["noisy_accuracy"]

    output = {
        "description": (
            "Final thesis comparison combining fixed parity HQNN, learned readout, "
            "multi-observable readout, architecture search, noise sweep, repeated trials, "
            "and statistical validation."
        ),
        "rows": rows,
        "best_overall_by_noisy_accuracy": best_overall,
        "baseline_noisy_accuracy": baseline["noisy_accuracy"],
        "best_noisy_accuracy": best_overall["noisy_accuracy"],
        "best_noisy_accuracy_gain_over_baseline": float(best_gain),
        "thesis_contribution_summary": (
            "The final comparison shows that HQNN performance improves substantially "
            "when the framework moves beyond fixed parity readout and adds learned "
            "readout, multi-observable quantum feature extraction, architecture search, "
            "noise-sweep validation, repeated trials, and statistical validation."
        ),
    }

    json_path = OUTPUT_DIR / "final_thesis_comparison.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    plot_final_comparison(
        rows,
        OUTPUT_DIR / "final_thesis_comparison.png",
    )

    plot_noisy_only(
        rows,
        OUTPUT_DIR / "final_thesis_noisy_accuracy_ranking.png",
    )

    print("\nFinal thesis comparison pipeline complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {OUTPUT_DIR / 'final_thesis_comparison.png'}")
    print(f"Saved: {OUTPUT_DIR / 'final_thesis_noisy_accuracy_ranking.png'}")

    print("\nBest overall by noisy accuracy:")
    print(best_overall)

    print("\nGain over fixed parity baseline:")
    print(best_gain)


if __name__ == "__main__":
    run_pipeline()
