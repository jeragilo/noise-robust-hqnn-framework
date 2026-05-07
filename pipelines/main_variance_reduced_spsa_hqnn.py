"""
Pipeline: Variance-Reduced Noise-Aware SPSA for HQNN Training

Contribution:
This pipeline improves HQNN optimization under NISQ-style noise by reducing
gradient variance during SPSA optimization.

Instead of using one noisy SPSA gradient estimate per update, this method
averages multiple noisy gradient realizations:

k = 1  -> standard SPSA
k = 3  -> variance-reduced SPSA
k = 5  -> stronger variance reduction

The goal is to stabilize noisy HQNN training and improve robustness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from framework.noise_channels import create_noise_model
from framework.robustness_metrics import (
    accuracy_drop,
    robustness_score,
    training_instability,
)

RESULTS_DIR = (
    Path("results")
    / "framework"
    / "variance_reduced_spsa_hqnn"
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
    qc = qc.compose(
        build_variational_layer(
            num_qubits,
            weights,
            architecture,
        )
    )

    qc.measure_all()

    return qc


def parity_expectation(counts: Dict[str, int]) -> float:
    shots = sum(counts.values())

    exp = 0.0

    for bitstring, count in counts.items():
        parity = bitstring.count("1") % 2
        sign = 1 if parity == 0 else -1
        exp += sign * count / shots

    return float(exp)


def predict_probs(
    simulator: AerSimulator,
    X: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    shots: int,
) -> np.ndarray:
    probs = []

    for x in X:
        qc = build_hqnn_circuit(
            num_qubits,
            x,
            weights,
            architecture,
        )

        result = simulator.run(qc, shots=shots).result()

        counts = result.get_counts()

        exp = parity_expectation(counts)

        probs.append((1 - exp) / 2)

    return np.array(probs, dtype=float)


def binary_cross_entropy(
    probs: np.ndarray,
    y: np.ndarray,
) -> float:
    eps = 1e-10

    probs = np.clip(probs, eps, 1 - eps)

    return float(
        -np.mean(
            y * np.log(probs)
            + (1 - y) * np.log(1 - probs)
        )
    )


def evaluate_accuracy(
    probs: np.ndarray,
    y: np.ndarray,
) -> float:
    preds = (probs >= 0.5).astype(int)

    return float(accuracy_score(y, preds))


def compute_loss(
    simulator: AerSimulator,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    shots: int,
) -> float:
    probs = predict_probs(
        simulator,
        X,
        weights,
        num_qubits,
        architecture,
        shots,
    )

    return binary_cross_entropy(probs, y)


def variance_reduced_spsa_step(
    simulator: AerSimulator,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    learning_rate: float,
    perturbation: float,
    gradient_samples: int,
    shots: int,
) -> np.ndarray:
    dim = len(weights)

    gradients = []

    for _ in range(gradient_samples):
        delta = 2 * np.random.randint(0, 2, dim) - 1

        w_plus = weights + perturbation * delta
        w_minus = weights - perturbation * delta

        loss_plus = compute_loss(
            simulator,
            X,
            y,
            w_plus,
            num_qubits,
            architecture,
            shots,
        )

        loss_minus = compute_loss(
            simulator,
            X,
            y,
            w_minus,
            num_qubits,
            architecture,
            shots,
        )

        grad_hat = (
            (loss_plus - loss_minus)
            / (2 * perturbation * delta)
        )

        gradients.append(grad_hat)

    avg_gradient = np.mean(gradients, axis=0)

    return weights - learning_rate * avg_gradient


def make_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=320,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        class_sep=2.4,
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


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
    gradient_samples: int,
    seed: int,
    architecture: str,
    noise_type: str,
    train_noise: float,
    eval_noise: float,
    epochs: int,
    shots: int,
) -> Dict[str, object]:
    np.random.seed(seed)

    num_qubits = X_train.shape[1]
    num_params = 3 * num_qubits

    weights = np.random.uniform(
        -np.pi,
        np.pi,
        num_params,
    )

    noisy_train_simulator = AerSimulator(
        noise_model=create_noise_model(
            noise_type,
            train_noise,
        )
    )

    noisy_eval_simulator = AerSimulator(
        noise_model=create_noise_model(
            noise_type,
            eval_noise,
        )
    )

    clean_eval_simulator = AerSimulator()

    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        learning_rate = 0.18 / np.sqrt(epoch + 1)
        perturbation = 0.12 / np.sqrt(epoch + 1)

        weights = variance_reduced_spsa_step(
            simulator=noisy_train_simulator,
            X=X_train,
            y=y_train,
            weights=weights,
            num_qubits=num_qubits,
            architecture=architecture,
            learning_rate=learning_rate,
            perturbation=perturbation,
            gradient_samples=gradient_samples,
            shots=shots,
        )

        clean_probs = predict_probs(
            clean_eval_simulator,
            X_test,
            weights,
            num_qubits,
            architecture,
            shots,
        )

        noisy_probs = predict_probs(
            noisy_eval_simulator,
            X_test,
            weights,
            num_qubits,
            architecture,
            shots,
        )

        clean_acc = evaluate_accuracy(
            clean_probs,
            y_test,
        )

        noisy_acc = evaluate_accuracy(
            noisy_probs,
            y_test,
        )

        noisy_loss = binary_cross_entropy(
            noisy_probs,
            y_test,
        )

        row = {
            "epoch": float(epoch),
            "clean_accuracy": float(clean_acc),
            "noisy_accuracy": float(noisy_acc),
            "noisy_loss": float(noisy_loss),
        }

        history.append(row)

        print(
            f"{label:30s} | "
            f"epoch={epoch:02d} | "
            f"clean_acc={clean_acc:.3f} | "
            f"noisy_acc={noisy_acc:.3f}"
        )

    final = history[-1]

    noisy_acc_series = [
        h["noisy_accuracy"]
        for h in history
    ]

    noisy_loss_series = [
        h["noisy_loss"]
        for h in history
    ]

    return {
        "label": label,
        "gradient_samples": int(gradient_samples),
        "clean_accuracy": float(final["clean_accuracy"]),
        "noisy_accuracy": float(final["noisy_accuracy"]),
        "accuracy_drop": float(
            accuracy_drop(
                final["clean_accuracy"],
                final["noisy_accuracy"],
            )
        ),
        "robustness_score": float(
            robustness_score(
                final["noisy_accuracy"],
                final["clean_accuracy"],
            )
        ),
        "training_instability": float(
            training_instability(noisy_acc_series)
        ),
        "loss_instability": float(
            training_instability(noisy_loss_series)
        ),
        "history": history,
    }


def plot_accuracy(
    results: Dict[str, Dict[str, object]],
    output_path: Path,
) -> None:
    labels = list(results.keys())

    clean = [
        results[m]["clean_accuracy"]
        for m in labels
    ]

    noisy = [
        results[m]["noisy_accuracy"]
        for m in labels
    ]

    x = np.arange(len(labels))

    width = 0.35

    plt.figure(figsize=(11, 5))

    plt.bar(
        x - width / 2,
        clean,
        width,
        label="Clean Accuracy",
    )

    plt.bar(
        x + width / 2,
        noisy,
        width,
        label="Noisy Accuracy",
    )

    plt.xticks(
        x,
        labels,
        rotation=15,
    )

    plt.ylim(0, 1)

    plt.ylabel("Accuracy")

    plt.title(
        "Variance-Reduced SPSA HQNN Accuracy"
    )

    plt.grid(axis="y")

    plt.legend()

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()


def plot_instability(
    results: Dict[str, Dict[str, object]],
    output_path: Path,
) -> None:
    labels = list(results.keys())

    values = [
        results[m]["training_instability"]
        for m in labels
    ]

    plt.figure(figsize=(10, 5))

    plt.bar(labels, values)

    plt.ylabel("Training Instability")

    plt.title(
        "Variance-Reduced SPSA: Accuracy Instability"
    )

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()


def run_pipeline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = make_dataset()

    architecture = "linear"

    noise_type = "depolarizing"

    train_noise = 0.05
    eval_noise = 0.05

    epochs = 25

    shots = 1024

    configs = [
        ("standard_spsa", 1),
        ("variance_reduced_spsa_k3", 3),
        ("variance_reduced_spsa_k5", 5),
    ]

    results: Dict[str, Dict[str, object]] = {}

    for i, (label, k) in enumerate(configs):
        print(f"\n=== Training: {label} ===")

        results[label] = train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            label=label,
            gradient_samples=k,
            seed=42 + i,
            architecture=architecture,
            noise_type=noise_type,
            train_noise=train_noise,
            eval_noise=eval_noise,
            epochs=epochs,
            shots=shots,
        )

    best_method = max(
        results.values(),
        key=lambda row: row["noisy_accuracy"],
    )

    summary = {
        "description": (
            "Variance-reduced noise-aware SPSA optimization "
            "for HQNN training."
        ),
        "architecture": architecture,
        "noise_type": noise_type,
        "train_noise": train_noise,
        "eval_noise": eval_noise,
        "epochs": epochs,
        "shots": shots,
        "results": results,
        "best_method": best_method,
        "thesis_contribution": (
            "This experiment reduces noisy SPSA gradient "
            "variance by averaging multiple noisy gradient "
            "realizations per update step."
        ),
    }

    json_path = (
        RESULTS_DIR
        / "variance_reduced_spsa_summary.json"
    )

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    accuracy_plot = (
        RESULTS_DIR
        / "variance_reduced_spsa_accuracy.png"
    )

    instability_plot = (
        RESULTS_DIR
        / "variance_reduced_spsa_instability.png"
    )

    plot_accuracy(results, accuracy_plot)

    plot_instability(results, instability_plot)

    print("\nVariance-reduced SPSA pipeline complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {accuracy_plot}")
    print(f"Saved: {instability_plot}")

    print("\nBest method:")
    print(best_method)


if __name__ == "__main__":
    run_pipeline()

