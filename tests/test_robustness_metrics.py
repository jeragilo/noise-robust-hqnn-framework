import numpy as np
import pytest

from framework.robustness_metrics import (
    accuracy_drop,
    cross_framework_deviation,
    degradation_slope,
    robustness_score,
    summarize_noise_results,
    training_instability,
)


def test_accuracy_drop():
    assert accuracy_drop(0.90, 0.70) == pytest.approx(0.20)


def test_robustness_score():
    assert robustness_score(0.72, 0.90) == pytest.approx(0.80)
    assert robustness_score(0.0, 0.0) == 0.0


def test_degradation_slope():
    levels = [0.0, 0.1, 0.2]
    accuracies = [0.9, 0.8, 0.7]
    assert degradation_slope(levels, accuracies) == pytest.approx(-1.0)


def test_degradation_slope_with_single_point():
    assert degradation_slope([0.0], [0.9]) == 0.0


def test_training_instability():
    values = [0.8, 0.9, 1.0]
    assert training_instability(values) == pytest.approx(float(np.std(values)))


def test_cross_framework_deviation():
    assert cross_framework_deviation([0.82, 0.85, 0.81]) == pytest.approx(0.04)


def test_summarize_noise_results():
    summary = summarize_noise_results(
        noiseless_acc=0.90,
        noise_levels=[0.05, 0.10, 0.20],
        noisy_accuracies=[0.88, 0.84, 0.75],
    )

    assert summary["noiseless_accuracy"] == pytest.approx(0.90)
    assert summary["mean_noisy_accuracy"] == pytest.approx(np.mean([0.88, 0.84, 0.75]))
    assert summary["worst_noisy_accuracy"] == pytest.approx(0.75)
    assert summary["best_noisy_accuracy"] == pytest.approx(0.88)
    assert summary["max_accuracy_drop"] == pytest.approx(0.15)
    assert summary["mean_robustness_score"] == pytest.approx(np.mean([0.88 / 0.90, 0.84 / 0.90, 0.75 / 0.90]))
    assert summary["degradation_slope"] < 0
