"""
Main Pipeline: Training Mode Comparison

This pipeline demonstrates the main thesis contribution:

A reusable noise-aware HQNN training framework that compares:
1. Standard training
2. Noise-aware training
3. Curriculum noise training

The goal is to show that the way noise is inserted into the training process
changes HQNN learning behavior, robustness, and final noisy accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from framework.noise_channels import (
    create_noise_model,
    make_training_noise_schedule,
)
from framework.benchmark_runner import (
    compare_training_modes,
    summarize_noise_aware_contribution,
)
from framework.robustness_metrics import accuracy_drop, robustness_score


RESULTS_DIR = Path("results") / "framework" / "training_mode_comparison"


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

    if architecture == "none":
        return qc

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

    else:
        raise ValueError("architecture must be one of: none, linear, ring, full")

    return qc


def build_hqnn_circuit(
    num_qubits: int,
    x: np.ndarray,
    weights: np.ndarray,
    architecture: str = "ring",
) -> QuantumCircuit:
    x_pad = np.zeros(num_qubits)
    x_pad[: len(x)] = x

    qc = build_feature_map(num_qubits, x_pad)
    qc = qc.compose(build_variational_layer(num_qubits, weights, architecture))
    qc.measure_all()
    return qc


def parity_expval(counts: Dict[str, int]) -> float:
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
    shots: int = 1024,
) -> np.ndarray:
    probs = []

    for x in X:
        qc = build_hqnn_circuit(num_qubits, x, weights, architecture)
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        exp = parity_expval(counts)
        probs.append((1 - exp) / 2)

    return np.array(probs)


def binary_cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-10
    return float(
        -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
    )


def evaluate_loss(
    simulator: AerSimulator,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
) -> float:
    probs = predict_probs(simulator, X, weights, num_qubits, architecture)
    return binary_cross_entropy(probs, y)


def evaluate_accuracy(
    simulator: AerSimulator,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
) -> float:
    probs = predict_probs(simulator, X, weights, num_qubits, architecture)
    preds = (probs >= 0.5).astype(int)
    return float(accuracy_score(y, preds))


def spsa_step(
    simulator: AerSimulator,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    alpha: float,
    c: float,
) -> np.ndarray:
    dim = len(weights)
    delta = 2 * np.random.randint(0, 2, dim) - 1

    w_plus = weights + c * delta
    w_minus = weights - c * delta

    loss_plus = evaluate_loss(simulator, X, y, w_plus, num_qubits, architecture)
    loss_minus = evaluate_loss(simulator, X, y, w_minus, num_qubits, architecture)

    grad_hat = (loss_plus - loss_minus) / (2 * c * delta)

    return weights - alpha * grad_hat


def train_hqnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    training_mode: str,
    noise_type: str,
    max_train_noise: float,
    eval_noise: float,
    architecture: str = "ring",
    epochs: int = 20,
    seed: int = 42,
) -> Dict[str, object]:
    np.random.seed(seed)

    num_qubits = X_train.shape[1]
    num_params = 3 * num_qubits

    weights = np.random.uniform(-np.pi, np.pi, num_params)

    noise_schedule = make_training_noise_schedule(
        mode=training_mode,
        epochs=epochs,
        max_noise=max_train_noise,
    )

    history: List[Dict[str, float]] = []

    for epoch, train_noise in enumerate(noise_schedule):
        train_noise_model = create_noise_model(noise_type, train_noise)
        train_simulator = AerSimulator(noise_model=train_noise_model)

        alpha = 0.18 / np.sqrt(epoch + 1)
        c = 0.12 / np.sqrt(epoch + 1)

        weights = spsa_step(
            train_simulator,
            X_train,
            y_train,
            weights,
            num_qubits,
            architecture,
            alpha=alpha,
            c=c,
        )

        clean_simulator = AerSimulator()
        noisy_eval_simulator = AerSimulator(
            noise_model=create_noise_model(noise_type, eval_noise)
        )

        clean_acc = evaluate_accuracy(
            clean_simulator,
            X_test,
            y_test,
            weights,
            num_qubits,
            architecture,
        )

        noisy_acc = evaluate_accuracy(
            noisy_eval_simulator,
            X_test,
            y_test,
            weights,
            num_qubits,
            architecture,
        )

        train_loss = evaluate_loss(
            train_simulator,
            X_train,
            y_train,
            weights,
            num_qubits,
            architecture,
        )

        history.append(
            {
                "epoch": float(epoch),
                "training_noise": float(train_noise),
                "loss": float(train_loss),
                "clean_accuracy": float(clean_acc),
                "noisy_accuracy": float(noisy_acc),
            }
        )

        print(
            f"{training_mode:12s} | epoch={epoch:02d} | "
            f"train_noise={train_noise:.3f} | "
            f"clean_acc={clean_acc:.3f} | noisy_acc={noisy_acc:.3f}"
        )

    final = history[-1]

    return {
        "training_mode": training_mode,
        "architecture": architecture,
        "noise_type": noise_type,
        "max_train_noise": float(max_train_noise),
        "eval_noise": float(eval_noise),
        "clean_accuracy": float(final["clean_accuracy"]),
        "noisy_accuracy": float(final["noisy_accuracy"]),
        "accuracy_drop": float(
            accuracy_drop(final["clean_accuracy"], final["noisy_accuracy"])
        ),
        "robustness_score": float(
            robustness_score(final["noisy_accuracy"], final["clean_accuracy"])
        ),
        "history": history,
    }


def make_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=240,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        class_sep=1.8,
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


def plot_training_curves(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    plt.figure(figsize=(9, 5))

    for mode, result in results.items():
        history = result["history"]
        epochs = [h["epoch"] for h in history]
        noisy_acc = [h["noisy_accuracy"] for h in history]
        plt.plot(epochs, noisy_acc, marker="o", label=mode)

    plt.xlabel("Epoch")
    plt.ylabel("Noisy Test Accuracy")
    plt.title("Training Mode Comparison: Noisy Accuracy During Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_final_comparison(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    labels = list(results.keys())
    clean = [results[m]["clean_accuracy"] for m in labels]
    noisy = [results[m]["noisy_accuracy"] for m in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, clean, width, label="Clean Accuracy")
    plt.bar(x + width / 2, noisy, width, label="Noisy Accuracy")
    plt.xticks(x, labels)
    plt.ylabel("Accuracy")
    plt.title("Final Accuracy by Training Mode")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_pipeline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = make_dataset()

    noise_type = "depolarizing"
    max_train_noise = 0.05
    eval_noise = 0.05
    architecture = "linear"
    epochs = 50

    training_modes = ["standard", "noise_aware", "curriculum"]

    results: Dict[str, Dict[str, object]] = {}

    for i, mode in enumerate(training_modes):
        results[mode] = train_hqnn(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            training_mode=mode,
            noise_type=noise_type,
            max_train_noise=max_train_noise,
            eval_noise=eval_noise,
            architecture=architecture,
            epochs=epochs,
            seed=42 + i,
        )

    training_rows = compare_training_modes(
        dataset_name="Synthetic classification",
        model_name="HQNN",
        training_results={
            mode: {
                "clean_accuracy": results[mode]["clean_accuracy"],
                "noisy_accuracy": results[mode]["noisy_accuracy"],
                "accuracy_drop": results[mode]["accuracy_drop"],
                "robustness_score": results[mode]["robustness_score"],
            }
            for mode in results
        },
    )

    improvement_summary = summarize_noise_aware_contribution(
        baseline_clean_accuracy=results["standard"]["clean_accuracy"],
        baseline_noisy_accuracy=results["standard"]["noisy_accuracy"],
        improved_clean_accuracy=results["noise_aware"]["clean_accuracy"],
        improved_noisy_accuracy=results["noise_aware"]["noisy_accuracy"],
    )

    output = {
        "description": (
            "Training mode comparison showing how embedding noise into the "
            "training process changes HQNN robustness and noisy accuracy."
        ),
        "dataset": "Synthetic classification",
        "noise_type": noise_type,
        "max_train_noise": max_train_noise,
        "eval_noise": eval_noise,
        "architecture": architecture,
        "epochs": epochs,
        "results": results,
        "training_rows": training_rows,
        "noise_aware_improvement_summary": improvement_summary,
    }

    json_path = RESULTS_DIR / "training_mode_comparison.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    plot_training_curves(
        results,
        RESULTS_DIR / "training_mode_noisy_accuracy_curves.png",
    )

    plot_final_comparison(
        results,
        RESULTS_DIR / "training_mode_final_accuracy_comparison.png",
    )

    print("\nTraining mode comparison pipeline complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {RESULTS_DIR / 'training_mode_noisy_accuracy_curves.png'}")
    print(f"Saved: {RESULTS_DIR / 'training_mode_final_accuracy_comparison.png'}")
    print("\nTraining rows:")
    for row in training_rows:
        print(row)

    print("\nNoise-aware improvement summary:")
    print(improvement_summary)


if __name__ == "__main__":
    run_pipeline()
