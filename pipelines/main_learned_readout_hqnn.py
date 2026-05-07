"""
Pipeline: Learned Readout HQNN

Contribution:
Instead of using a fixed parity threshold, this pipeline extracts quantum
measurement features from the HQNN circuit and trains a classical Logistic
Regression readout on top of those features.

This tests whether a hybrid learned readout improves HQNN classification
performance under clean and noisy evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from framework.noise_channels import create_noise_model
from framework.robustness_metrics import accuracy_drop, robustness_score


RESULTS_DIR = Path("results") / "framework" / "learned_readout_hqnn"


def build_feature_map(num_qubits: int, x: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(float(x[i]), i)
    return qc


def build_variational_layer(
    num_qubits: int,
    weights: np.ndarray,
    architecture: str = "ring",
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


def counts_to_features(counts: Dict[str, int], num_qubits: int) -> np.ndarray:
    shots = sum(counts.values())
    probs = []

    for i in range(2**num_qubits):
        bitstring = format(i, f"0{num_qubits}b")
        probs.append(counts.get(bitstring, 0) / shots)

    return np.array(probs, dtype=float)


def extract_quantum_features(
    simulator: AerSimulator,
    X: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    shots: int = 1024,
) -> np.ndarray:
    features = []

    for x in X:
        qc = build_hqnn_circuit(num_qubits, x, weights, architecture)
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        features.append(counts_to_features(counts, num_qubits))

    return np.vstack(features)


def parity_predict(
    simulator: AerSimulator,
    X: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    shots: int = 1024,
) -> np.ndarray:
    preds = []

    for x in X:
        qc = build_hqnn_circuit(num_qubits, x, weights, architecture)
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()

        total = sum(counts.values())
        exp = 0.0

        for bitstring, count in counts.items():
            parity = bitstring.count("1") % 2
            sign = 1 if parity == 0 else -1
            exp += sign * count / total

        p1 = (1 - exp) / 2
        preds.append(1 if p1 >= 0.5 else 0)

    return np.array(preds)


def make_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        class_sep=2.2,
        flip_y=0.01,
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


def run_pipeline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    X_train, X_test, y_train, y_test = make_dataset()

    num_qubits = X_train.shape[1]
    num_params = 3 * num_qubits
    architecture = "ring"
    noise_type = "depolarizing"
    eval_noise = 0.05
    shots = 2048

    weights = np.random.uniform(-np.pi, np.pi, num_params)

    clean_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=create_noise_model(noise_type, eval_noise))

    y_pred_parity_clean = parity_predict(
        clean_sim, X_test, weights, num_qubits, architecture, shots
    )
    y_pred_parity_noisy = parity_predict(
        noisy_sim, X_test, weights, num_qubits, architecture, shots
    )

    parity_clean_acc = accuracy_score(y_test, y_pred_parity_clean)
    parity_noisy_acc = accuracy_score(y_test, y_pred_parity_noisy)

    Z_train_clean = extract_quantum_features(
        clean_sim, X_train, weights, num_qubits, architecture, shots
    )
    Z_test_clean = extract_quantum_features(
        clean_sim, X_test, weights, num_qubits, architecture, shots
    )
    Z_test_noisy = extract_quantum_features(
        noisy_sim, X_test, weights, num_qubits, architecture, shots
    )

    readout = LogisticRegression(max_iter=1000, random_state=42)
    readout.fit(Z_train_clean, y_train)

    learned_clean_pred = readout.predict(Z_test_clean)
    learned_noisy_pred = readout.predict(Z_test_noisy)

    learned_clean_acc = accuracy_score(y_test, learned_clean_pred)
    learned_noisy_acc = accuracy_score(y_test, learned_noisy_pred)

    summary = {
        "description": (
            "Learned-readout HQNN experiment. Quantum measurement distributions "
            "are used as features for a classical Logistic Regression readout."
        ),
        "architecture": architecture,
        "noise_type": noise_type,
        "eval_noise": eval_noise,
        "shots": shots,
        "parity_readout": {
            "clean_accuracy": float(parity_clean_acc),
            "noisy_accuracy": float(parity_noisy_acc),
            "accuracy_drop": float(accuracy_drop(parity_clean_acc, parity_noisy_acc)),
            "robustness_score": float(robustness_score(parity_noisy_acc, parity_clean_acc)),
        },
        "learned_readout": {
            "clean_accuracy": float(learned_clean_acc),
            "noisy_accuracy": float(learned_noisy_acc),
            "accuracy_drop": float(accuracy_drop(learned_clean_acc, learned_noisy_acc)),
            "robustness_score": float(robustness_score(learned_noisy_acc, learned_clean_acc)),
        },
        "improvement": {
            "clean_accuracy_gain": float(learned_clean_acc - parity_clean_acc),
            "noisy_accuracy_gain": float(learned_noisy_acc - parity_noisy_acc),
        },
    }

    json_path = RESULTS_DIR / "learned_readout_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    labels = ["Parity Clean", "Parity Noisy", "Learned Clean", "Learned Noisy"]
    values = [
        parity_clean_acc,
        parity_noisy_acc,
        learned_clean_acc,
        learned_noisy_acc,
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("HQNN Fixed Parity Readout vs Learned Classical Readout")
    plt.grid(axis="y")
    plt.tight_layout()

    plot_path = RESULTS_DIR / "learned_readout_accuracy.png"
    plt.savefig(plot_path)
    plt.close()

    print("\nLearned-readout HQNN pipeline complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    run_pipeline()
