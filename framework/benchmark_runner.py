"""
Benchmark runner for the Noise-Robust HQNN framework.

This module connects datasets, models, noise channels, metrics, and reporting.
It provides high-level functions used by pipeline scripts to highlight the
main thesis contributions.

Main purpose:
1. Standardize hybrid/classical comparison.
2. Standardize noise robustness evaluation.
3. Standardize cross-framework validation.
4. Standardize training instability summaries.
5. Compare standard, noise-aware, dual-loss, and curriculum training modes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

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
    Convert hybrid, quantum, and classical model results into standardized rows.

    Example:
        {
            "HQNN": 0.54,
            "QSVM": 0.84,
            "Logistic Regression": 0.97,
            "Random Forest": 0.96,
            "XGBoost": 0.98
        }
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

    Metrics:
    - mean noisy accuracy
    - worst noisy accuracy
    - best noisy accuracy
    - maximum accuracy drop
    - mean robustness score
    - degradation slope
    """
    if len(noise_levels) != len(noisy_accuracies):
        raise ValueError("noise_levels and noisy_accuracies must have the same length.")

    if len(noise_levels) == 0:
        raise ValueError("noise_levels cannot be empty.")

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
        "degradation_slope": float(degradation_slope(noise_levels, noisy_accuracies)),
    }


def run_cross_framework_validation(
    expectation_results: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """
    Summarize cross-framework expectation-value consistency.

    Example:
        {
            "qiskit": {"ZZ": 0.70, "XX": 0.50},
            "cirq": {"ZZ": 0.71, "XX": 0.49},
            "pennylane": {"ZZ": 0.70, "XX": 0.51}
        }
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
            "cross_framework_deviation": float(cross_framework_deviation(values)),
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
    if len(final_accuracies) == 0:
        raise ValueError("final_accuracies cannot be empty.")

    summary = {
        "model": model_name,
        "mean_final_accuracy": float(np.mean(final_accuracies)),
        "std_final_accuracy": float(training_instability(final_accuracies)),
        "min_final_accuracy": float(np.min(final_accuracies)),
        "max_final_accuracy": float(np.max(final_accuracies)),
    }

    if final_losses is not None:
        if len(final_losses) == 0:
            raise ValueError("final_losses cannot be empty when provided.")

        summary.update(
            {
                "mean_final_loss": float(np.mean(final_losses)),
                "std_final_loss": float(training_instability(final_losses)),
                "min_final_loss": float(np.min(final_losses)),
                "max_final_loss": float(np.max(final_losses)),
            }
        )

    return summary


def compare_training_modes(
    *,
    dataset_name: str,
    model_name: str,
    training_results: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """
    Compare standard, noise-aware, dual-loss, and curriculum training modes.

    This function is important for the thesis contribution because it makes the
    noise-aware training pipeline visible as a reusable benchmarking layer.

    Expected input:
        {
            "standard": {
                "clean_accuracy": 0.58,
                "noisy_accuracy": 0.46,
                "robustness_score": 0.79
            },
            "noise_aware": {
                "clean_accuracy": 0.62,
                "noisy_accuracy": 0.55,
                "robustness_score": 0.88
            }
        }
    """
    rows: List[Dict[str, Any]] = []

    for training_mode, metrics in training_results.items():
        clean_accuracy = float(metrics.get("clean_accuracy", metrics.get("accuracy", 0.0)))
        noisy_accuracy = float(metrics.get("noisy_accuracy", clean_accuracy))

        drop = float(metrics.get(
            "accuracy_drop",
            accuracy_drop(clean_accuracy, noisy_accuracy),
        ))

        robust = float(metrics.get(
            "robustness_score",
            robustness_score(noisy_accuracy, clean_accuracy),
        ))

        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "training_mode": training_mode,
                "clean_accuracy": clean_accuracy,
                "noisy_accuracy": noisy_accuracy,
                "accuracy_drop": drop,
                "robustness_score": robust,
            }
        )

    return rows


def compare_architectures(
    *,
    dataset_name: str,
    training_mode: str,
    architecture_results: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """
    Compare HQNN architecture variants.

    Example architectures:
    - no_entanglement
    - linear_entanglement
    - ring_entanglement
    - full_entanglement
    - shallow
    - deep

    This supports the architecture contribution of the thesis.
    """
    rows: List[Dict[str, Any]] = []

    for architecture_name, metrics in architecture_results.items():
        clean_accuracy = float(metrics.get("clean_accuracy", metrics.get("accuracy", 0.0)))
        noisy_accuracy = float(metrics.get("noisy_accuracy", clean_accuracy))

        rows.append(
            {
                "dataset": dataset_name,
                "training_mode": training_mode,
                "architecture": architecture_name,
                "clean_accuracy": clean_accuracy,
                "noisy_accuracy": noisy_accuracy,
                "accuracy_drop": float(
                    metrics.get(
                        "accuracy_drop",
                        accuracy_drop(clean_accuracy, noisy_accuracy),
                    )
                ),
                "robustness_score": float(
                    metrics.get(
                        "robustness_score",
                        robustness_score(noisy_accuracy, clean_accuracy),
                    )
                ),
            }
        )

    return rows


def summarize_noise_aware_contribution(
    *,
    baseline_clean_accuracy: float,
    baseline_noisy_accuracy: float,
    improved_clean_accuracy: float,
    improved_noisy_accuracy: float,
) -> Dict[str, Any]:
    """
    Summarize the improvement produced by the proposed noise-aware method.

    This gives one thesis-ready statement:
    how much the improved method improves noisy accuracy and robustness.
    """
    baseline_drop = accuracy_drop(baseline_clean_accuracy, baseline_noisy_accuracy)
    improved_drop = accuracy_drop(improved_clean_accuracy, improved_noisy_accuracy)

    baseline_robustness = robustness_score(baseline_noisy_accuracy, baseline_clean_accuracy)
    improved_robustness = robustness_score(improved_noisy_accuracy, improved_clean_accuracy)

    return {
        "baseline_clean_accuracy": float(baseline_clean_accuracy),
        "baseline_noisy_accuracy": float(baseline_noisy_accuracy),
        "improved_clean_accuracy": float(improved_clean_accuracy),
        "improved_noisy_accuracy": float(improved_noisy_accuracy),
        "noisy_accuracy_gain": float(improved_noisy_accuracy - baseline_noisy_accuracy),
        "baseline_accuracy_drop": float(baseline_drop),
        "improved_accuracy_drop": float(improved_drop),
        "accuracy_drop_reduction": float(baseline_drop - improved_drop),
        "baseline_robustness_score": float(baseline_robustness),
        "improved_robustness_score": float(improved_robustness),
        "robustness_score_gain": float(improved_robustness - baseline_robustness),
    }
