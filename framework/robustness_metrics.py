"""
Robustness metrics for the Noise-Robust HQNN framework.

This module provides quantitative measures used to support
thesis claims with numerical evidence.

These metrics are NOT provided as a unified system in
Qiskit Machine Learning or PennyLane — this is part of your framework contribution.
"""

from __future__ import annotations
from typing import Sequence, Dict
import numpy as np


def accuracy_drop(noiseless_acc: float, noisy_acc: float) -> float:
    """
    Absolute accuracy loss due to noise.

    Example:
        0.90 → 0.70 = 0.20 drop
    """
    return float(noiseless_acc - noisy_acc)


def robustness_score(noisy_acc: float, noiseless_acc: float) -> float:
    """
    Relative robustness (normalized performance).

    1.0 = no degradation
    0.0 = complete failure
    """
    if noiseless_acc == 0:
        return 0.0
    return float(noisy_acc / noiseless_acc)


def degradation_slope(noise_levels: Sequence[float], accuracies: Sequence[float]) -> float:
    """
    Linear slope of accuracy vs noise.

    Negative slope = performance drops with noise
    """
    noise_levels = np.asarray(noise_levels)
    accuracies = np.asarray(accuracies)

    if len(noise_levels) < 2:
        return 0.0

    slope, _ = np.polyfit(noise_levels, accuracies, 1)
    return float(slope)


def training_instability(values: Sequence[float]) -> float:
    """
    Standard deviation across runs or epochs.

    Measures sensitivity to:
    - initialization
    - noise
    - optimizer randomness
    """
    values = np.asarray(values)
    return float(np.std(values))


def cross_framework_deviation(values: Sequence[float]) -> float:
    """
    Maximum difference across frameworks.

    Example:
        Qiskit vs Cirq vs PennyLane
    """
    values = np.asarray(values)
    return float(np.max(values) - np.min(values))


def summarize_noise_results(
    noiseless_acc: float,
    noise_levels: Sequence[float],
    noisy_accuracies: Sequence[float],
) -> Dict[str, float]:
    """
    Creates a summary of noise robustness.

    This is VERY important for your thesis tables.
    """
    noisy_accuracies = np.asarray(noisy_accuracies)

    drops = [accuracy_drop(noiseless_acc, acc) for acc in noisy_accuracies]
    scores = [robustness_score(acc, noiseless_acc) for acc in noisy_accuracies]

    return {
        "noiseless_accuracy": float(noiseless_acc),
        "mean_noisy_accuracy": float(np.mean(noisy_accuracies)),
        "worst_noisy_accuracy": float(np.min(noisy_accuracies)),
        "best_noisy_accuracy": float(np.max(noisy_accuracies)),
        "max_accuracy_drop": float(np.max(drops)),
        "mean_robustness_score": float(np.mean(scores)),
        "degradation_slope": degradation_slope(noise_levels, noisy_accuracies),
    }
