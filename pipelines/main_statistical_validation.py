"""
Pipeline: Statistical Validation Summary

Combines all major framework outputs into one statistical validation report.
This supports the thesis claim that the framework is not just a collection of
demos, but a reusable HQNN robustness and optimization evaluation system.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results") / "framework"
OUTPUT_DIR = RESULTS_DIR / "statistical_validation"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def confidence_interval_95(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "lower_95": mean - margin,
        "upper_95": mean + margin,
        "margin": margin,
    }


def summarize_method(method: str, clean_values: List[float], noisy_values: List[float]) -> Dict[str, Any]:
    clean_arr = np.array(clean_values, dtype=float)
    noisy_arr = np.array(noisy_values, dtype=float)

    noisy_std = float(np.std(noisy_arr, ddof=1)) if len(noisy_arr) > 1 else 0.0
    noisy_mean = float(np.mean(noisy_arr))
    clean_mean = float(np.mean(clean_arr))

    robustness = noisy_mean / clean_mean if clean_mean != 0 else 0.0
    coefficient_variation = noisy_std / noisy_mean if noisy_mean != 0 else 0.0
    stability_score = max(0.0, 1.0 - coefficient_variation)

    ci = confidence_interval_95(noisy_values)

    return {
        "method": method,
        "num_observations": len(noisy_values),
        "mean_clean_accuracy": clean_mean,
        "mean_noisy_accuracy": noisy_mean,
        "std_noisy_accuracy": noisy_std,
        "min_noisy_accuracy": float(np.min(noisy_arr)),
        "max_noisy_accuracy": float(np.max(noisy_arr)),
        "mean_accuracy_drop": clean_mean - noisy_mean,
        "mean_robustness_score": robustness,
        "coefficient_of_variation": coefficient_variation,
        "stability_score": stability_score,
        "confidence_interval_95": ci,
    }


def collect_methods() -> List[Dict[str, Any]]:
    methods = []

    learned = load_json(
        RESULTS_DIR / "learned_readout_hqnn" / "learned_readout_summary.json"
    )

    multi = load_json(
        RESULTS_DIR / "multi_observable_hqnn" / "multi_observable_summary.json"
    )

    arch = load_json(
        RESULTS_DIR / "architecture_search_hqnn" / "architecture_search_summary.json"
    )

    sweep = load_json(
        RESULTS_DIR / "best_architecture_noise_sweep" / "best_architecture_noise_sweep_summary.json"
    )

    repeated = load_json(
        RESULTS_DIR / "best_architecture_repeated_trials" / "repeated_trials_summary.json"
    )

    methods.append(
        summarize_method(
            "Fixed Parity Readout",
            [learned["parity_readout"]["clean_accuracy"]],
            [learned["parity_readout"]["noisy_accuracy"]],
        )
    )

    methods.append(
        summarize_method(
            "Learned Logistic Readout",
            [learned["learned_readout"]["clean_accuracy"]],
            [learned["learned_readout"]["noisy_accuracy"]],
        )
    )

    methods.append(
        summarize_method(
            "Multi-Observable Logistic Readout",
            [multi["multi_observable_logistic_readout"]["clean_accuracy"]],
            [multi["multi_observable_logistic_readout"]["noisy_accuracy"]],
        )
    )

    methods.append(
        summarize_method(
            "Multi-Observable Random Forest",
            [multi["multi_observable_random_forest_readout"]["clean_accuracy"]],
            [multi["multi_observable_random_forest_readout"]["noisy_accuracy"]],
        )
    )

    best_arch = "linear"
    methods.append(
        summarize_method(
            "Best Architecture: Linear + RF",
            [arch["results"][best_arch]["random_forest"]["clean_accuracy"]],
            [arch["results"][best_arch]["random_forest"]["noisy_accuracy"]],
        )
    )

    sweep_clean = sweep["clean_accuracy"]
    sweep_noisy_values = [row["accuracy"] for row in sweep["accuracy_by_noise"]]

    methods.append(
        summarize_method(
            "Linear + Multi-Observable RF Noise Sweep",
            [sweep_clean for _ in sweep_noisy_values],
            sweep_noisy_values,
        )
    )

    repeated_clean = [row["clean_accuracy"] for row in repeated["trials"]]
    repeated_noisy = [row["noisy_accuracy"] for row in repeated["trials"]]

    methods.append(
        summarize_method(
            "Repeated Trials: Linear + Multi-Observable RF",
            repeated_clean,
            repeated_noisy,
        )
    )

    return methods


def rank_methods(methods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        methods,
        key=lambda row: (
            row["mean_noisy_accuracy"],
            row["stability_score"],
            row["mean_robustness_score"],
        ),
        reverse=True,
    )

    for i, row in enumerate(ranked, start=1):
        row["rank"] = i

    return ranked


def plot_noisy_accuracy_ranking(ranked: List[Dict[str, Any]], output_path: Path) -> None:
    labels = [row["method"] for row in ranked]
    values = [row["mean_noisy_accuracy"] for row in ranked]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Mean Noisy Accuracy")
    plt.title("Statistical Validation: Mean Noisy Accuracy Ranking")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_stability_scores(ranked: List[Dict[str, Any]], output_path: Path) -> None:
    labels = [row["method"] for row in ranked]
    values = [row["stability_score"] for row in ranked]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.ylim(0, 1.05)
    plt.ylabel("Stability Score")
    plt.title("Statistical Validation: Stability Score by Method")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confidence_intervals(ranked: List[Dict[str, Any]], output_path: Path) -> None:
    labels = [row["method"] for row in ranked]
    means = [row["confidence_interval_95"]["mean"] for row in ranked]
    errors = [row["confidence_interval_95"]["margin"] for row in ranked]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, means, yerr=errors, capsize=5)
    plt.ylim(0, 1)
    plt.ylabel("Mean Noisy Accuracy")
    plt.title("Statistical Validation: 95% Confidence Intervals")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_pipeline() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    methods = collect_methods()
    ranked = rank_methods(methods)

    output = {
        "description": (
            "Statistical validation report combining learned readout, "
            "multi-observable readout, architecture search, noise sweep, "
            "and repeated-trial validation."
        ),
        "ranked_methods": ranked,
        "best_method": ranked[0],
        "thesis_contribution": (
            "This statistical validation supports the thesis claim that HQNN "
            "performance improves when the framework uses learned readouts, "
            "multi-observable quantum features, architecture search, and "
            "robustness validation rather than relying only on fixed parity readout."
        ),
    }

    json_path = OUTPUT_DIR / "statistical_validation_summary.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    plot_noisy_accuracy_ranking(
        ranked,
        OUTPUT_DIR / "statistical_noisy_accuracy_ranking.png",
    )

    plot_stability_scores(
        ranked,
        OUTPUT_DIR / "statistical_stability_scores.png",
    )

    plot_confidence_intervals(
        ranked,
        OUTPUT_DIR / "statistical_confidence_intervals.png",
    )

    print("\nStatistical validation pipeline complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {OUTPUT_DIR / 'statistical_noisy_accuracy_ranking.png'}")
    print(f"Saved: {OUTPUT_DIR / 'statistical_stability_scores.png'}")
    print(f"Saved: {OUTPUT_DIR / 'statistical_confidence_intervals.png'}")

    print("\nBest method:")
    print(ranked[0])

    print("\nRanked methods:")
    for row in ranked:
        print(
            f"#{row['rank']} | {row['method']} | "
            f"mean noisy acc={row['mean_noisy_accuracy']:.4f} | "
            f"stability={row['stability_score']:.4f}"
        )


if __name__ == "__main__":
    run_pipeline()
