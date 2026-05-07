"""
Pipeline: Best-Architecture Noise Sweep

Contribution:
This pipeline takes the strongest architecture/readout found so far and tests
whether it remains robust as depolarizing noise increases.

It supports the thesis claim that the framework can identify HQNN
configurations that preserve useful classification performance under noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from framework.noise_channels import create_noise_model
from framework.robustness_metrics import (
    accuracy_drop,
    robustness_score,
    degradation_slope,
)


RESULTS_DIR = Path("results") / "framework" / "best_architecture_noise_sweep"


def build_feature_map(num_qubits: int, x: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(float(x[i]), i)
        qc.rz(float(x[i]) ** 2, i)
    return qc


def build_variational_layer(
    num_qubits: int,
    weights: np.ndarray,
    architecture: str,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    for i in range(num_qubits):
        qc.rx(float(weights[i]), i)
        qc.ry(float(weights[num_qubits + i]), i)
        qc.rz(float(weights[2 * num_qubits + i]), i)

    if architecture == "linear":
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)
    elif architecture == "ring":
        for i in range(num_qubits):
            qc.cz(i, (i + 1) % num_qubits)
    elif architecture == "full":
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qc.cz(i, j)
    elif architecture == "none":
        pass
    else:
        raise ValueError("architecture must be one of: none, linear, ring, full")

    return qc


def build_hqnn_circuit(
    num_qubits: int,
    x: np.ndarray,
    weights: np.ndarray,
    architecture: str,
) -> QuantumCircuit:
    x_pad = np.zeros(num_qubits)
    x_pad[: len(x)] = x

    qc = build_feature_map(num_qubits, x_pad)
    qc = qc.compose(build_variational_layer(num_qubits, weights, architecture))
    qc.measure_all()
    return qc


def all_bitstrings(num_qubits: int) -> List[str]:
    return [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]


def bitstring_probabilities(counts: Dict[str, int], num_qubits: int) -> np.ndarray:
    shots = sum(counts.values())
    return np.array(
        [counts.get(bitstring, 0) / shots for bitstring in all_bitstrings(num_qubits)],
        dtype=float,
    )


def parity_expectation(counts: Dict[str, int]) -> float:
    shots = sum(counts.values())
    exp = 0.0

    for bitstring, count in counts.items():
        parity = bitstring.count("1") % 2
        sign = 1 if parity == 0 else -1
        exp += sign * count / shots

    return float(exp)


def z_expectations(counts: Dict[str, int], num_qubits: int) -> np.ndarray:
    shots = sum(counts.values())
    values = []

    for q in range(num_qubits):
        exp = 0.0
        for bitstring, count in counts.items():
            bit = int(bitstring[::-1][q])
            sign = 1 if bit == 0 else -1
            exp += sign * count / shots
        values.append(exp)

    return np.array(values, dtype=float)


def zz_correlations(counts: Dict[str, int], num_qubits: int) -> np.ndarray:
    shots = sum(counts.values())
    values = []

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            exp = 0.0
            for bitstring, count in counts.items():
                bi = int(bitstring[::-1][i])
                bj = int(bitstring[::-1][j])
                zi = 1 if bi == 0 else -1
                zj = 1 if bj == 0 else -1
                exp += zi * zj * count / shots
            values.append(exp)

    return np.array(values, dtype=float)


def probability_statistics(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps))
    max_prob = np.max(probs)
    min_prob = np.min(probs)
    variance = np.var(probs)

    return np.array([entropy, max_prob, min_prob, variance], dtype=float)


def counts_to_multi_observable_features(
    counts: Dict[str, int],
    num_qubits: int,
) -> np.ndarray:
    probs = bitstring_probabilities(counts, num_qubits)
    z_vals = z_expectations(counts, num_qubits)
    zz_vals = zz_correlations(counts, num_qubits)
    parity = np.array([parity_expectation(counts)], dtype=float)
    stats = probability_statistics(probs)

    return np.concatenate([probs, z_vals, zz_vals, parity, stats])


def extract_features(
    simulator: AerSimulator,
    X: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    shots: int,
) -> np.ndarray:
    features = []

    for x in X:
        qc = build_hqnn_circuit(num_qubits, x, weights, architecture)
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        features.append(counts_to_multi_observable_features(counts, num_qubits))

    return np.vstack(features)


def make_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        class_sep=2.6,
        flip_y=0.005,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.clip(X_train, -np.pi, np.pi)
    X_test = np.clip(X_test, -np.pi, np.pi)

    return X_train, X_test, y_train, y_test


def plot_noise_sweep(
    noise_levels: List[float],
    accuracies: List[float],
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(noise_levels, accuracies, marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Depolarizing Noise Probability")
    plt.ylabel("Noisy Accuracy")
    plt.title("Best HQNN Architecture Noise Sweep")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_pipeline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    X_train, X_test, y_train, y_test = make_dataset()

    num_qubits = X_train.shape[1]
    num_params = 3 * num_qubits

    architecture = "linear"
    noise_type = "depolarizing"
    shots = 4096

    noise_levels = [0.00, 0.01, 0.03, 0.05, 0.07, 0.10]

    weights = np.random.uniform(-np.pi, np.pi, num_params)

    clean_sim = AerSimulator()

    Z_train_clean = extract_features(
        clean_sim,
        X_train,
        weights,
        num_qubits,
        architecture,
        shots,
    )

    readout = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
    )
    readout.fit(Z_train_clean, y_train)

    rows = []
    accuracies = []

    for noise in noise_levels:
        if noise == 0.0:
            sim = AerSimulator()
        else:
            sim = AerSimulator(noise_model=create_noise_model(noise_type, noise))

        Z_test = extract_features(
            sim,
            X_test,
            weights,
            num_qubits,
            architecture,
            shots,
        )

        preds = readout.predict(Z_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(float(acc))

        rows.append(
            {
                "noise_level": float(noise),
                "accuracy": float(acc),
            }
        )

        print(f"noise={noise:.2f} | accuracy={acc:.4f}")

    clean_accuracy = accuracies[0]
    worst_accuracy = min(accuracies)
    best_accuracy = max(accuracies)

    summary = {
        "description": (
            "Best-architecture noise sweep using the linear HQNN architecture, "
            "multi-observable feature extraction, and Random Forest learned readout."
        ),
        "architecture": architecture,
        "readout": "RandomForestClassifier",
        "noise_type": noise_type,
        "shots": shots,
        "num_qubits": num_qubits,
        "noise_levels": noise_levels,
        "accuracy_by_noise": rows,
        "clean_accuracy": float(clean_accuracy),
        "worst_noisy_accuracy": float(worst_accuracy),
        "best_accuracy": float(best_accuracy),
        "max_accuracy_drop": float(accuracy_drop(clean_accuracy, worst_accuracy)),
        "mean_robustness_score": float(
            np.mean([robustness_score(acc, clean_accuracy) for acc in accuracies])
        ),
        "degradation_slope": float(degradation_slope(noise_levels, accuracies)),
        "thesis_contribution": (
            "This experiment tests whether the optimized HQNN configuration "
            "continues to perform well as noise increases. It turns the framework "
            "from a single-result benchmark into a reusable robustness-analysis "
            "pipeline."
        ),
    }

    json_path = RESULTS_DIR / "best_architecture_noise_sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_path = RESULTS_DIR / "best_architecture_noise_sweep.png"
    plot_noise_sweep(noise_levels, accuracies, plot_path)

    print("\nBest-architecture noise sweep complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_pipeline()
