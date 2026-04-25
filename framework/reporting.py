"""
Reporting utilities for the Noise-Robust HQNN framework.

This module standardizes how experiments save JSON, CSV, and plots.
The goal is to make every demo and benchmark pipeline produce consistent,
thesis-ready output files.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save dictionary data to a JSON file."""
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Save a list of dictionaries to a CSV file."""
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)

    if not rows:
        raise ValueError("Cannot save empty CSV rows.")

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy_comparison(
    labels: Sequence[str],
    accuracies: Sequence[float],
    path: str,
    *,
    title: str = "Accuracy Comparison",
    ylabel: str = "Accuracy",
) -> None:
    """Create a bar chart comparing model accuracies."""
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, accuracies)
    plt.ylim(0, 1)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_noise_curves(
    noise_levels: Sequence[float],
    series: Dict[str, Sequence[float]],
    path: str,
    *,
    title: str = "Noise Robustness Curves",
    ylabel: str = "Metric",
) -> None:
    """Plot metric values across noise levels for one or more noise types."""
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)

    plt.figure(figsize=(8, 5))

    for label, values in series.items():
        plt.plot(noise_levels, values, marker="o", label=label)

    plt.xlabel("Noise Level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_heatmap(
    matrix: Sequence[Sequence[float]],
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    path: str,
    *,
    title: str = "Heatmap",
    colorbar_label: str = "Value",
) -> None:
    """Create a heatmap for robustness or framework comparison results."""
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)

    data = np.asarray(matrix, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.imshow(data, aspect="auto")
    plt.colorbar(label=colorbar_label)
    plt.xticks(range(len(x_labels)), x_labels, rotation=30, ha="right")
    plt.yticks(range(len(y_labels)), y_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_training_curve(
    values: Sequence[float],
    path: str,
    *,
    title: str = "Training Curve",
    xlabel: str = "Epoch",
    ylabel: str = "Value",
) -> None:
    """Plot a single training curve, such as loss or accuracy over epochs."""
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(values)), values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
