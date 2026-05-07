"""
Pipeline: Dual-Loss Stability-Regularized Noise-Aware HQNN

Contribution:
This pipeline implements a true noise-aware optimization objective for HQNNs.

Instead of treating noise only as an evaluation condition, this method embeds
noise directly into training through:

1. Clean loss
2. Noisy loss
3. Stability penalty between clean and noisy predictions

The goal is to compare standard HQNN training against a robustness-aware
optimization method.
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
from framework.robustness_metrics import accuracy_drop, robustness_score


RESULTS_DIR = Path("results") / "framework" / "dual_loss_noise_aware_hqnn"


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
        qc = build_hqnn_circuit(num_qubits, x, weights, architecture)
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()

        exp = parity_expectation(counts)
        p1 = (1 - exp) / 2
        probs.append(p1)

    return np.array(probs, dtype=float)


def binary_cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-10
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def evaluate_accuracy_from_probs(probs: np.ndarray, y: np.ndarray) -> float:
    preds = (probs >= 0.5).astype(int)
    return float(accuracy_score(y, preds))


def compute_objective(
    clean_simulator: AerSimulator,
    noisy_simulator: AerSimulator,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    mode: str,
    alpha_clean: float,
    beta_noisy: float,
    lambda_stability: float,
    shots: int,
) -> float:
    clean_probs = predict_probs(
        clean_simulator, X, weights, num_qubits, architecture, shots
    )
    noisy_probs = predict_probs(
        noisy_simulator, X, weights, num_qubits, architecture, shots
    )

    clean_loss = binary_cross_entropy(clean_probs, y)
    noisy_loss = binary_cross_entropy(noisy_probs, y)
    stability_loss = float(np.mean(np.abs(clean_probs - noisy_probs)))

    if mode == "standard":
        return clean_loss

    if mode == "noise_aware":
        return noisy_loss

    if mode == "dual_loss":
        return alpha_clean * clean_loss + beta_noisy * noisy_loss

    if mode == "stability_regularized":
        return (
            alpha_clean * clean_loss
            + beta_noisy * noisy_loss
            + lambda_stability * stability_loss
        )

    raise ValueError(
        "mode must be one of: standard, noise_aware, dual_loss, stability_regularized"
    )


def spsa_step(
    clean_simulator: AerSimulator,
    noisy_simulator: AerSimulator,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    mode: str,
    alpha_clean: float,
    beta_noisy: float,
    lambda_stability: float,
    learning_rate: float,
    perturbation: float,
    shots: int,
) -> np.ndarray:
    dim = len(weights)
    delta = 2 * np.random.randint(0, 2, dim) - 1

    w_plus = weights + perturbation * delta
    w_minus = weights - perturbation * delta

    loss_plus = compute_objective(
        clean_simulator,
        noisy_simulator,
        X,
        y,
        w_plus,
        num_qubits,
        architecture,
        mode,
        alpha_clean,
        beta_noisy,
        lambda_stability,
        shots,
    )

    loss_minus = compute_objective(
        clean_simulator,
        noisy_simulator,
        X,
        y,
        w_minus,
        num_qubits,
        architecture,
        mode,
        alpha_clean,
        beta_noisy,
        lambda_stability,
        shots,
    )

    grad_hat = (loss_plus - loss_minus) / (2 * perturbation * delta)

    return weights - learning_rate * grad_hat


def make_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=260,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        class_sep=2.0,
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


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mode: str,
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

    weights = np.random.uniform(-np.pi, np.pi, num_params)

    clean_simulator = AerSimulator()
    noisy_train_simulator = AerSimulator(
        noise_model=create_noise_model(noise_type, train_noise)
    )
    noisy_eval_simulator = AerSimulator(
        noise_model=create_noise_model(noise_type, eval_noise)
    )

    alpha_clean = 0.50
    beta_noisy = 0.50
    lambda_stability = 0.75

    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        learning_rate = 0.20 / np.sqrt(epoch + 1)
        perturbation = 0.12 / np.sqrt(epoch + 1)

        weights = spsa_step(
            clean_simulator=clean_simulator,
            noisy_simulator=noisy_train_simulator,
            X=X_train,
            y=y_train,
            weights=weights,
            num_qubits=num_qubits,
            architecture=architecture,
            mode=mode,
            alpha_clean=alpha_clean,
            beta_noisy=beta_noisy,
            lambda_stability=lambda_stability,
            learning_rate=learning_rate,
            perturbation=perturbation,
            shots=shots,
        )

        clean_probs = predict_probs(
            clean_simulator,
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

        clean_acc = evaluate_accuracy_from_probs(clean_probs, y_test)
        noisy_acc = evaluate_accuracy_from_probs(noisy_probs, y_test)

        clean_loss = binary_cross_entropy(clean_probs, y_test)
        noisy_loss = binary_cross_entropy(noisy_probs, y_test)
        stability_loss = float(np.mean(np.abs(clean_probs - noisy_probs)))

        total_loss = (
            alpha_clean * clean_loss
            + beta_noisy * noisy_loss
            + lambda_stability * stability_loss
        )

        row = {
            "epoch": float(epoch),
            "clean_accuracy": float(clean_acc),
            "noisy_accuracy": float(noisy_acc),
            "clean_loss": float(clean_loss),
            "noisy_loss": float(noisy_loss),
            "stability_loss": float(stability_loss),
            "total_stability_regularized_loss": float(total_loss),
        }

        history.append(row)

        print(
            f"{mode:24s} | epoch={epoch:02d} | "
            f"clean_acc={clean_acc:.3f} | noisy_acc={noisy_acc:.3f} | "
            f"stability={stability_loss:.4f}"
        )

    final = history[-1]

    return {
        "mode": mode,
        "architecture": architecture,
        "noise_type": noise_type,
        "train_noise": float(train_noise),
        "eval_noise": float(eval_noise),
        "clean_accuracy": float(final["clean_accuracy"]),
        "noisy_accuracy": float(final["noisy_accuracy"]),
        "accuracy_drop": float(
            accuracy_drop(final["clean_accuracy"], final["noisy_accuracy"])
        ),
        "robustness_score": float(
            robustness_score(final["noisy_accuracy"], final["clean_accuracy"])
        ),
        "final_clean_loss": float(final["clean_loss"]),
        "final_noisy_loss": float(final["noisy_loss"]),
        "final_stability_loss": float(final["stability_loss"]),
        "history": history,
    }


def plot_final_accuracy(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    labels = list(results.keys())
    clean = [results[m]["clean_accuracy"] for m in labels]
    noisy = [results[m]["noisy_accuracy"] for m in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, clean, width, label="Clean Accuracy")
    plt.bar(x + width / 2, noisy, width, label="Noisy Accuracy")
    plt.xticks(x, labels, rotation=20)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Dual-Loss Noise-Aware HQNN: Final Accuracy Comparison")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_training_curves(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    plt.figure(figsize=(10, 5))

    for mode, result in results.items():
        history = result["history"]
        epochs = [h["epoch"] for h in history]
        noisy_acc = [h["noisy_accuracy"] for h in history]
        plt.plot(epochs, noisy_acc, marker="o", label=mode)

    plt.xlabel("Epoch")
    plt.ylabel("Noisy Test Accuracy")
    plt.title("Dual-Loss Noise-Aware HQNN: Noisy Accuracy During Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_stability(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    labels = list(results.keys())
    stability = [results[m]["final_stability_loss"] for m in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, stability)
    plt.xticks(rotation=20)
    plt.ylabel("Final Stability Loss")
    plt.title("Clean/Noisy Prediction Disagreement by Training Method")
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

    modes = [
        "standard",
        "noise_aware",
        "dual_loss",
        "stability_regularized",
    ]

    results: Dict[str, Dict[str, object]] = {}

    for i, mode in enumerate(modes):
        print(f"\n=== Training mode: {mode} ===")
        results[mode] = train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            mode=mode,
            seed=42 + i,
            architecture=architecture,
            noise_type=noise_type,
            train_noise=train_noise,
            eval_noise=eval_noise,
            epochs=epochs,
            shots=shots,
        )

    best_mode = max(results.values(), key=lambda row: row["noisy_accuracy"])

    summary = {
        "description": (
            "Dual-loss stability-regularized noise-aware HQNN optimization. "
            "This experiment embeds noise awareness directly into the training "
            "objective using clean loss, noisy loss, and a stability penalty."
        ),
        "architecture": architecture,
        "noise_type": noise_type,
        "train_noise": train_noise,
        "eval_noise": eval_noise,
        "epochs": epochs,
        "shots": shots,
        "results": results,
        "best_mode_by_noisy_accuracy": best_mode,
        "thesis_contribution": (
            "This pipeline moves the framework beyond post-hoc noise evaluation "
            "by incorporating noise directly into the optimization objective. "
            "Dual-loss and stability-regularized training test whether HQNN "
            "parameters can be optimized to preserve performance under noisy "
            "NISQ-style evaluation."
        ),
    }

    json_path = RESULTS_DIR / "dual_loss_noise_aware_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    final_plot = RESULTS_DIR / "dual_loss_final_accuracy.png"
    curve_plot = RESULTS_DIR / "dual_loss_training_curves.png"
    stability_plot = RESULTS_DIR / "dual_loss_stability_loss.png"

    plot_final_accuracy(results, final_plot)
    plot_training_curves(results, curve_plot)
    plot_stability(results, stability_plot)

    print("\nDual-loss noise-aware HQNN pipeline complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {final_plot}")
    print(f"Saved: {curve_plot}")
    print(f"Saved: {stability_plot}")

    print("\nBest mode by noisy accuracy:")
    print(best_mode)


if __name__ == "__main__":
    run_pipeline()
