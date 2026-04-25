"""
Benchmark runner for the Noise-Robust HQNN framework.

This module connects datasets, models, noise channels, metrics, and reporting.
It provides high-level functions used by pipeline scripts to highlight the
main thesis contributions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from framework.robustness_metrics import (
    accuracy_drop,
    robustness_score,
    degradation_slope,
    cross_framework_deviation,
    training_instability,
)


def run_hybrid_vs_classical_benchmark(
    dataset_name: str,
    model_results: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Convert hybrid/classical model results into standardized benchmark rows.

    Args:
        dataset_name:
            Name of the dataset used.

        model_results:
            Dictionary mapping model names to accuracy values.

            Example:
                {
                    "HQNN": 0.54,
                    "QSVM": 0.84,
                    "Logistic Regression": 0.97
                }

    Returns:
        List of rows suitable for CSV reporting.
    """
    rows = []

    for model_name, accuracy in model_results.items():
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "accuracy": float(accuracy),
            }
        )

    return rows


def run_noise_robustness_benchmark(
    *,
    model_name: str,
    noise_type: str,
    noiseless_accuracy: float,
    noise_levels: Sequence[float],
    noisy_accuracies: Sequence[float],
) -> Dict[str, Any]:
    """
    Summarize model robustness under a specific noise channel.

    This function produces thesis-ready metrics:
    - maximum accuracy drop
    - mean robustness score
    - degradation slope
    - worst noisy accuracy
    """
    if len(noise_levels) != len(noisy_accuracies):
        raise ValueError("noise_levels and noisy_accuracies must have the same length.")

    drops = [
        accuracy_drop(noiseless_accuracy, acc)
        for acc in noisy_accuracies
    ]

    scores = [
        robustness_score(acc, noiseless_accuracy)
        for acc in noisy_accuracies
    ]

    return {
        "model": model_name,
        "noise_type": noise_type,
        "noiseless_accuracy": float(noiseless_accuracy),
        "mean_noisy_accuracy": float(np.mean(noisy_accuracies)),
        "worst_noisy_accuracy": float(np.min(noisy_accuracies)),
        "best_noisy_accuracy": float(np.max(noisy_accuracies)),
        "max_accuracy_drop": float(np.max(drops)),
        "mean_robustness_score": float(np.mean(scores)),
        "degradation_slope": degradation_slope(noise_levels, noisy_accuracies),
    }


def run_cross_framework_validation(
    expectation_results: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """
    Summarize cross-framework expectation value consistency.

    Args:
        expectation_results:
            Example:
                {
                    "qiskit": {"ZZ": 0.70, "XX": 0.50},
                    "cirq": {"ZZ": 0.71, "XX": 0.49},
                    "pennylane": {"ZZ": 0.70, "XX": 0.51}
                }

    Returns:
        Rows with observable-level cross-framework deviation.
    """
    frameworks = list(expectation_results.keys())

    if not frameworks:
        return []

    observables = list(expectation_results[frameworks[0]].keys())
    rows = []

    for obs in observables:
        values = [expectation_results[fw][obs] for fw in frameworks]

        row = {
            "observable": obs,
            "cross_framework_deviation": cross_framework_deviation(values),
        }

        for fw, value in zip(frameworks, values):
            row[fw] = float(value)

        rows.append(row)

    return rows


def summarize_training_runs(
    model_name: str,
    final_accuracies: Sequence[float],
    final_losses: Sequence[float] | None = None,
) -> Dict[str, Any]:
    """
    Summarize training sensitivity across repeated runs.

    This supports claims about initialization sensitivity and optimizer instability.
    """
    summary = {
        "model": model_name,
        "mean_final_accuracy": float(np.mean(final_accuracies)),
        "std_final_accuracy": training_instability(final_accuracies),
        "min_final_accuracy": float(np.min(final_accuracies)),
        "max_final_accuracy": float(np.max(final_accuracies)),
    }

    if final_losses is not None:
        summary.update(
            {
                "mean_final_loss": float(np.mean(final_losses)),
                "std_final_loss": training_instability(final_losses),
                "min_final_loss": float(np.min(final_losses)),
                "max_final_loss": float(np.max(final_losses)),
            }
        )

    return summary
