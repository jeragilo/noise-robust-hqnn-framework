"""
Pipeline: Multi-Channel Flagship HQNN Validation

This pipeline validates the flagship HQNN configuration across multiple
NISQ-relevant noise channels.

Flagship configuration:
- Stability-regularized training
- Multi-observable quantum features
- Linear entanglement architecture
- Learned classical readout

Noise channels tested:
- depolarizing
- bit_flip
- phase_flip
- amplitude_damping
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from framework.noise_channels import create_noise_model
from framework.robustness_metrics import accuracy_drop, robustness_score


RESULTS_DIR = (
    Path("results")
    / "framework"
    / "multichannel_flagship_validation"
)


def build_feature_map(num_qubits: int, x: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    for i in range(num_qubits):
        qc.ry(float(x[i]), i)
        qc.rz(float(x[i]) ** 2, i)

    return qc


def build_variational_layer(
    num_qubits: int,
    weights: np.ndarray,
    architecture: str = "linear",
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
        raise ValueError("Invalid architecture.")

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


def bitstring_probabilities(
    counts: Dict[str, int],
    num_qubits: int,
) -> np.ndarray:
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


def z_expectations(
    counts: Dict[str, int],
    num_qubits: int,
) -> np.ndarray:
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


def zz_correlations(
    counts: Dict[str, int],
    num_qubits: int,
) -> np.ndarray:
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

    return np.array(
        [entropy, max_prob, min_prob, variance],
        dtype=float,
    )


def counts_to_multi_observable_features(
    counts: Dict[str, int],
    num_qubits: int,
) -> np.ndarray:
    probs = bitstring_probabilities(counts, num_qubits)
    z_vals = z_expectations(counts, num_qubits)
    zz_vals = zz_correlations(counts, num_qubits)

    parity = np.array([parity_expectation(counts)], dtype=float)
    stats = probability_statistics(probs)

    return np.concatenate(
        [
            probs,
            z_vals,
            zz_vals,
            parity,
            stats,
        ]
    )


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
        qc = build_hqnn_circuit(
            num_qubits,
            x,
            weights,
            architecture,
        )

        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()

        features.append(
            counts_to_multi_observable_features(
                counts,
                num_qubits,
            )
        )

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


def evaluate_noise_channel(
    noise_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    weights: np.ndarray,
    architecture: str,
    eval_noise: float,
    shots: int,
) -> Dict[str, object]:
    num_qubits = X_train.shape[1]

    clean_sim = AerSimulator()

    noisy_sim = AerSimulator(
        noise_model=create_noise_model(
            noise_type,
            eval_noise,
        )
    )

    Z_train_clean = extract_features(
        clean_sim,
        X_train,
        weights,
        num_qubits,
        architecture,
        shots,
    )

    Z_test_clean = extract_features(
        clean_sim,
        X_test,
        weights,
        num_qubits,
        architecture,
        shots,
    )

    Z_test_noisy = extract_features(
        noisy_sim,
        X_test,
        weights,
        num_qubits,
        architecture,
        shots,
    )

    lr = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=7,
        min_samples_leaf=2,
        random_state=42,
    )

    lr.fit(Z_train_clean, y_train)
    rf.fit(Z_train_clean, y_train)

    lr_clean_pred = lr.predict(Z_test_clean)
    lr_noisy_pred = lr.predict(Z_test_noisy)

    rf_clean_pred = rf.predict(Z_test_clean)
    rf_noisy_pred = rf.predict(Z_test_noisy)

    lr_clean_acc = accuracy_score(y_test, lr_clean_pred)
    lr_noisy_acc = accuracy_score(y_test, lr_noisy_pred)

    rf_clean_acc = accuracy_score(y_test, rf_clean_pred)
    rf_noisy_acc = accuracy_score(y_test, rf_noisy_pred)

    return {
        "noise_type": noise_type,
        "logistic_regression": {
            "clean_accuracy": float(lr_clean_acc),
            "noisy_accuracy": float(lr_noisy_acc),
            "accuracy_drop": float(
                accuracy_drop(lr_clean_acc, lr_noisy_acc)
            ),
            "robustness_score": float(
                robustness_score(lr_noisy_acc, lr_clean_acc)
            ),
        },
        "random_forest": {
            "clean_accuracy": float(rf_clean_acc),
            "noisy_accuracy": float(rf_noisy_acc),
            "accuracy_drop": float(
                accuracy_drop(rf_clean_acc, rf_noisy_acc)
            ),
            "robustness_score": float(
                robustness_score(rf_noisy_acc, rf_clean_acc)
            ),
        },
    }


def plot_results(
    rows: List[Dict[str, object]],
    output_path: Path,
) -> None:
    labels = []
    values = []

    for row in rows:
        labels.append(f"{row['noise_type']}\nLR")
        values.append(
            row["logistic_regression"]["noisy_accuracy"]
        )

        labels.append(f"{row['noise_type']}\nRF")
        values.append(
            row["random_forest"]["noisy_accuracy"]
        )

    plt.figure(figsize=(12, 5))

    plt.bar(labels, values)

    plt.ylim(0, 1)

    plt.ylabel("Noisy Accuracy")

    plt.title(
        "Flagship HQNN Validation Across Multiple Noise Channels"
    )

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()


def run_pipeline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(46)

    X_train, X_test, y_train, y_test = make_dataset()

    architecture = "linear"

    eval_noise = 0.05

    shots = 4096

    num_qubits = X_train.shape[1]
    num_params = 3 * num_qubits

    weights = np.random.uniform(
        -np.pi,
        np.pi,
        num_params,
    )

    noise_types = [
        "depolarizing",
        "bit_flip",
        "phase_flip",
        "amplitude_damping",
    ]

    rows = []

    for noise_type in noise_types:
        print(f"\n=== Testing noise channel: {noise_type} ===")

        result = evaluate_noise_channel(
            noise_type=noise_type,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            weights=weights,
            architecture=architecture,
            eval_noise=eval_noise,
            shots=shots,
        )

        rows.append(result)

        print(result)

    best_overall = max(
        rows,
        key=lambda row: row["random_forest"]["noisy_accuracy"],
    )

    summary = {
        "description": (
            "Multi-channel validation of the flagship HQNN configuration "
            "across multiple NISQ-relevant noise channels."
        ),
        "architecture": architecture,
        "eval_noise": eval_noise,
        "shots": shots,
        "results": rows,
        "best_overall_noise_channel": best_overall,
        "thesis_contribution": (
            "This experiment tests whether the flagship HQNN "
            "configuration generalizes across multiple noise channels "
            "instead of only depolarizing noise."
        ),
    }

    json_path = RESULTS_DIR / "multichannel_flagship_summary.json"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_path = RESULTS_DIR / "multichannel_flagship_accuracy.png"

    plot_results(rows, plot_path)

    print("\nMulti-channel flagship validation complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")

    print("\nBest overall:")
    print(best_overall)


if __name__ == "__main__":
    run_pipeline()
