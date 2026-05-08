"""
Pipeline: Noise Curriculum Learning for HQNNs

Contribution:
This pipeline trains HQNN parameters under a progressively increasing noise
schedule instead of using a fixed noise level from the beginning.

It compares:
1. Standard clean training
2. Fixed-noise training
3. Linear noise curriculum
4. Adaptive stability-guided noise curriculum

Core idea:
The HQNN first learns under easier low-noise conditions and is gradually exposed
to stronger NISQ-style noise. This tests whether staged noise exposure improves
robustness compared with fixed-noise training.
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


RESULTS_DIR = Path("results") / "framework" / "noise_curriculum_hqnn"


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


def evaluate_accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    preds = (probs >= 0.5).astype(int)
    return float(accuracy_score(y, preds))


def make_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=320,
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


def build_simulator(noise_type: str, noise_level: float) -> AerSimulator:
    if noise_level <= 0:
        return AerSimulator()
    return AerSimulator(noise_model=create_noise_model(noise_type, noise_level))


def curriculum_noise_level(
    mode: str,
    epoch: int,
    epochs: int,
    max_noise: float,
    previous_noise: float,
    stability_loss: float | None,
) -> float:
    if mode == "clean_training":
        return 0.0

    if mode == "fixed_noise":
        return max_noise

    if mode == "linear_curriculum":
        return max_noise * (epoch / max(1, epochs - 1))

    if mode == "adaptive_curriculum":
        if epoch == 0 or stability_loss is None:
            return 0.0

        if stability_loss < 0.030:
            return min(max_noise, previous_noise + 0.01)

        if stability_loss > 0.060:
            return max(0.0, previous_noise - 0.01)

        return previous_noise

    raise ValueError(
        "mode must be one of: clean_training, fixed_noise, "
        "linear_curriculum, adaptive_curriculum"
    )


def objective(
    clean_simulator: AerSimulator,
    train_simulator: AerSimulator,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    shots: int,
    alpha_clean: float,
    beta_noisy: float,
    lambda_stability: float,
) -> Tuple[float, float, float, float]:
    clean_probs = predict_probs(
        clean_simulator, X, weights, num_qubits, architecture, shots
    )
    noisy_probs = predict_probs(
        train_simulator, X, weights, num_qubits, architecture, shots
    )

    clean_loss = binary_cross_entropy(clean_probs, y)
    noisy_loss = binary_cross_entropy(noisy_probs, y)
    stability_loss = float(np.mean(np.abs(clean_probs - noisy_probs)))

    total_loss = (
        alpha_clean * clean_loss
        + beta_noisy * noisy_loss
        + lambda_stability * stability_loss
    )

    return total_loss, clean_loss, noisy_loss, stability_loss


def spsa_step(
    clean_simulator: AerSimulator,
    train_simulator: AerSimulator,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    shots: int,
    learning_rate: float,
    perturbation: float,
    alpha_clean: float,
    beta_noisy: float,
    lambda_stability: float,
    gradient_samples: int,
) -> np.ndarray:
    dim = len(weights)
    grad_total = np.zeros(dim)

    for _ in range(gradient_samples):
        delta = 2 * np.random.randint(0, 2, dim) - 1

        w_plus = weights + perturbation * delta
        w_minus = weights - perturbation * delta

        loss_plus, _, _, _ = objective(
            clean_simulator,
            train_simulator,
            X,
            y,
            w_plus,
            num_qubits,
            architecture,
            shots,
            alpha_clean,
            beta_noisy,
            lambda_stability,
        )

        loss_minus, _, _, _ = objective(
            clean_simulator,
            train_simulator,
            X,
            y,
            w_minus,
            num_qubits,
            architecture,
            shots,
            alpha_clean,
            beta_noisy,
            lambda_stability,
        )

        grad_total += (loss_plus - loss_minus) / (2 * perturbation * delta)

    grad_hat = grad_total / gradient_samples
    return weights - learning_rate * grad_hat


def train_with_mode(
    mode: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, object]:
    np.random.seed(seed)

    num_qubits = X_train.shape[1]
    num_params = 3 * num_qubits

    architecture = "linear"
    noise_type = "depolarizing"
    max_noise = 0.10
    eval_noise = 0.10
    epochs = 25
    shots = 1024
    gradient_samples = 3

    alpha_clean = 0.45
    beta_noisy = 0.55
    lambda_stability = 0.75

    weights = np.random.uniform(-np.pi, np.pi, num_params)

    clean_simulator = AerSimulator()
    eval_simulator = build_simulator(noise_type, eval_noise)

    history: List[Dict[str, float]] = []

    current_noise = 0.0
    previous_stability = None

    for epoch in range(epochs):
        current_noise = curriculum_noise_level(
            mode=mode,
            epoch=epoch,
            epochs=epochs,
            max_noise=max_noise,
            previous_noise=current_noise,
            stability_loss=previous_stability,
        )

        train_simulator = build_simulator(noise_type, current_noise)

        learning_rate = 0.18 / np.sqrt(epoch + 1)
        perturbation = 0.12 / np.sqrt(epoch + 1)

        weights = spsa_step(
            clean_simulator=clean_simulator,
            train_simulator=train_simulator,
            X=X_train,
            y=y_train,
            weights=weights,
            num_qubits=num_qubits,
            architecture=architecture,
            shots=shots,
            learning_rate=learning_rate,
            perturbation=perturbation,
            alpha_clean=alpha_clean,
            beta_noisy=beta_noisy,
            lambda_stability=lambda_stability,
            gradient_samples=gradient_samples,
        )

        train_total_loss, train_clean_loss, train_noisy_loss, train_stability_loss = objective(
            clean_simulator,
            train_simulator,
            X_train,
            y_train,
            weights,
            num_qubits,
            architecture,
            shots,
            alpha_clean,
            beta_noisy,
            lambda_stability,
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
            eval_simulator,
            X_test,
            weights,
            num_qubits,
            architecture,
            shots,
        )

        clean_acc = evaluate_accuracy(clean_probs, y_test)
        noisy_acc = evaluate_accuracy(noisy_probs, y_test)
        eval_stability = float(np.mean(np.abs(clean_probs - noisy_probs)))

        previous_stability = train_stability_loss

        row = {
            "epoch": float(epoch),
            "train_noise": float(current_noise),
            "clean_accuracy": float(clean_acc),
            "noisy_accuracy": float(noisy_acc),
            "train_total_loss": float(train_total_loss),
            "train_clean_loss": float(train_clean_loss),
            "train_noisy_loss": float(train_noisy_loss),
            "train_stability_loss": float(train_stability_loss),
            "eval_stability_loss": float(eval_stability),
        }

        history.append(row)

        print(
            f"{mode:22s} | epoch={epoch:02d} | "
            f"train_noise={current_noise:.3f} | "
            f"clean={clean_acc:.3f} | noisy={noisy_acc:.3f} | "
            f"eval_stability={eval_stability:.4f}"
        )

    final = history[-1]

    noisy_acc_history = np.array([row["noisy_accuracy"] for row in history])

    return {
        "mode": mode,
        "architecture": architecture,
        "noise_type": noise_type,
        "max_train_noise": float(max_noise),
        "eval_noise": float(eval_noise),
        "epochs": epochs,
        "shots": shots,
        "gradient_samples": gradient_samples,
        "clean_accuracy": float(final["clean_accuracy"]),
        "noisy_accuracy": float(final["noisy_accuracy"]),
        "accuracy_drop": float(
            accuracy_drop(final["clean_accuracy"], final["noisy_accuracy"])
        ),
        "robustness_score": float(
            robustness_score(final["noisy_accuracy"], final["clean_accuracy"])
        ),
        "final_train_noise": float(final["train_noise"]),
        "final_eval_stability_loss": float(final["eval_stability_loss"]),
        "mean_noisy_accuracy_last_5": float(np.mean(noisy_acc_history[-5:])),
        "max_noisy_accuracy": float(np.max(noisy_acc_history)),
        "training_instability": float(np.std(noisy_acc_history, ddof=0)),
        "history": history,
    }


def plot_final_accuracy(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    labels = list(results.keys())
    clean = [results[label]["clean_accuracy"] for label in labels]
    noisy = [results[label]["noisy_accuracy"] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, clean, width, label="Clean Accuracy")
    plt.bar(x + width / 2, noisy, width, label="Noisy Accuracy")
    plt.xticks(x, labels, rotation=20)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Noise Curriculum HQNN: Final Accuracy")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_noisy_accuracy_curves(
    results: Dict[str, Dict[str, object]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))

    for mode, result in results.items():
        history = result["history"]
        epochs = [row["epoch"] for row in history]
        noisy = [row["noisy_accuracy"] for row in history]
        plt.plot(epochs, noisy, marker="o", label=mode)

    plt.xlabel("Epoch")
    plt.ylabel("Noisy Accuracy at Evaluation Noise = 0.10")
    plt.title("Noise Curriculum HQNN: Noisy Accuracy During Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_noise_schedule(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    plt.figure(figsize=(10, 5))

    for mode, result in results.items():
        history = result["history"]
        epochs = [row["epoch"] for row in history]
        train_noise = [row["train_noise"] for row in history]
        plt.plot(epochs, train_noise, marker="o", label=mode)

    plt.xlabel("Epoch")
    plt.ylabel("Training Noise Level")
    plt.title("HQNN Noise Curriculum Schedules")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_pipeline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = make_dataset()

    modes = [
        "clean_training",
        "fixed_noise",
        "linear_curriculum",
        "adaptive_curriculum",
    ]

    results = {}

    for idx, mode in enumerate(modes):
        print(f"\n=== Training mode: {mode} ===")
        results[mode] = train_with_mode(
            mode=mode,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            seed=42 + idx,
        )

    best_by_final_noisy = max(
        results.values(),
        key=lambda row: row["noisy_accuracy"],
    )

    best_by_last5_noisy = max(
        results.values(),
        key=lambda row: row["mean_noisy_accuracy_last_5"],
    )

    summary = {
        "description": (
            "Noise Curriculum Learning for HQNNs. This experiment compares clean "
            "training, fixed-noise training, linear noise curriculum, and adaptive "
            "stability-guided curriculum training under high-noise evaluation."
        ),
        "results": results,
        "best_by_final_noisy_accuracy": best_by_final_noisy,
        "best_by_last5_mean_noisy_accuracy": best_by_last5_noisy,
        "thesis_contribution": (
            "This pipeline embeds curriculum learning into HQNN optimization by "
            "progressively exposing the model to stronger noise conditions. It "
            "tests whether staged noise exposure improves high-noise robustness "
            "compared with clean-only or fixed-noise training."
        ),
    }

    json_path = RESULTS_DIR / "noise_curriculum_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    final_plot = RESULTS_DIR / "noise_curriculum_final_accuracy.png"
    curve_plot = RESULTS_DIR / "noise_curriculum_noisy_accuracy_curves.png"
    schedule_plot = RESULTS_DIR / "noise_curriculum_schedules.png"

    plot_final_accuracy(results, final_plot)
    plot_noisy_accuracy_curves(results, curve_plot)
    plot_noise_schedule(results, schedule_plot)

    print("\nNoise curriculum HQNN pipeline complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {final_plot}")
    print(f"Saved: {curve_plot}")
    print(f"Saved: {schedule_plot}")

    print("\nBest by final noisy accuracy:")
    print(best_by_final_noisy)

    print("\nBest by last-5 mean noisy accuracy:")
    print(best_by_last5_noisy)


if __name__ == "__main__":
    run_pipeline()


