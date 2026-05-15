"""
Pipeline: Framework Ablation Study for Noise-Robust HQNN

Purpose:
Isolate the contribution of the two strongest thesis mechanisms:

1. Noise-aware quantum feature generation
   - standard objective vs dual-loss + stability objective

2. Noise-aware classical readout
   - standard readout training vs noise-augmented readout training

This directly supports the thesis ablation table requested by Dr. Hart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from framework.noise_channels import create_noise_model
from framework.robustness_metrics import accuracy_drop, robustness_score


RESULTS_DIR = Path("results") / "framework" / "framework_ablation_study"


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
    x: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
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


def parity_expectation(counts: Dict[str, int]) -> float:
    shots = sum(counts.values())
    exp = 0.0

    for bitstring, count in counts.items():
        parity = bitstring.count("1") % 2
        sign = 1 if parity == 0 else -1
        exp += sign * count / shots

    return float(exp)


def measurement_entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(p * np.log2(p)))


def counts_to_features(counts: Dict[str, int], num_qubits: int) -> np.ndarray:
    probs = bitstring_probabilities(counts, num_qubits)
    z_vals = z_expectations(counts, num_qubits)
    zz_vals = zz_correlations(counts, num_qubits)
    parity = np.array([parity_expectation(counts)], dtype=float)
    entropy = np.array([measurement_entropy(probs)], dtype=float)

    return np.concatenate([probs, z_vals, zz_vals, parity, entropy])


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
        qc = build_hqnn_circuit(x, weights, num_qubits, architecture)
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        features.append(counts_to_features(counts, num_qubits))

    return np.vstack(features)


def binary_cross_entropy_from_features(
    features: np.ndarray,
    y: np.ndarray,
) -> float:
    # Lightweight proxy objective: use parity feature as probability-like score.
    # Parity feature is second-to-last because entropy is last.
    parity = features[:, -2]
    probs = (1.0 - parity) / 2.0
    eps = 1e-10
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def compute_objective(
    clean_simulator: AerSimulator,
    noisy_simulator: AerSimulator,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    objective_mode: str,
    shots: int,
) -> float:
    clean_features = extract_features(
        clean_simulator, X, weights, num_qubits, architecture, shots
    )
    noisy_features = extract_features(
        noisy_simulator, X, weights, num_qubits, architecture, shots
    )

    clean_loss = binary_cross_entropy_from_features(clean_features, y)
    noisy_loss = binary_cross_entropy_from_features(noisy_features, y)
    stability_loss = float(np.mean(np.abs(clean_features - noisy_features)))

    if objective_mode == "standard_loss":
        return clean_loss

    if objective_mode == "dual_loss_stability":
        return 0.50 * clean_loss + 0.50 * noisy_loss + 0.75 * stability_loss

    raise ValueError("objective_mode must be standard_loss or dual_loss_stability")


def spsa_train_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_qubits: int,
    architecture: str,
    noise_type: str,
    train_noise: float,
    objective_mode: str,
    epochs: int,
    shots: int,
    seed: int,
) -> np.ndarray:
    np.random.seed(seed)

    num_params = 3 * num_qubits
    weights = np.random.uniform(-np.pi, np.pi, num_params)

    clean_simulator = AerSimulator()
    noisy_simulator = AerSimulator(
        noise_model=create_noise_model(noise_type, train_noise)
    )

    for epoch in range(epochs):
        dim = len(weights)
        delta = 2 * np.random.randint(0, 2, dim) - 1

        learning_rate = 0.20 / np.sqrt(epoch + 1)
        perturbation = 0.12 / np.sqrt(epoch + 1)

        w_plus = weights + perturbation * delta
        w_minus = weights - perturbation * delta

        loss_plus = compute_objective(
            clean_simulator,
            noisy_simulator,
            X_train,
            y_train,
            w_plus,
            num_qubits,
            architecture,
            objective_mode,
            shots,
        )

        loss_minus = compute_objective(
            clean_simulator,
            noisy_simulator,
            X_train,
            y_train,
            w_minus,
            num_qubits,
            architecture,
            objective_mode,
            shots,
        )

        grad_hat = (loss_plus - loss_minus) / (2 * perturbation * delta)
        weights = weights - learning_rate * grad_hat

        print(
            f"{objective_mode:22s} | epoch={epoch:02d} | "
            f"loss_plus={loss_plus:.4f} | loss_minus={loss_minus:.4f}"
        )

    return weights


def make_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=500,
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


def get_readout_models() -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "svm_rbf": SVC(kernel="rbf", gamma="scale", C=2.0, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }


def train_and_evaluate_readouts(
    X_train_features_clean: np.ndarray,
    X_test_features_clean: np.ndarray,
    X_train_features_noisy: np.ndarray,
    X_test_features_noisy: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    readout_training_mode: str,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    if readout_training_mode == "standard_readout":
        X_readout_train = X_train_features_clean
        y_readout_train = y_train

    elif readout_training_mode == "noise_augmented_readout":
        X_readout_train = np.vstack(
            [X_train_features_clean, X_train_features_noisy]
        )
        y_readout_train = np.concatenate([y_train, y_train])

    else:
        raise ValueError(
            "readout_training_mode must be standard_readout or noise_augmented_readout"
        )

    for model_name, model in get_readout_models().items():
        model.fit(X_readout_train, y_readout_train)

        clean_preds = model.predict(X_test_features_clean)
        noisy_preds = model.predict(X_test_features_noisy)

        clean_acc = float(accuracy_score(y_test, clean_preds))
        noisy_acc = float(accuracy_score(y_test, noisy_preds))

        results[model_name] = {
            "clean_accuracy": clean_acc,
            "noisy_accuracy": noisy_acc,
            "accuracy_drop": float(accuracy_drop(clean_acc, noisy_acc)),
            "robustness_score": float(robustness_score(noisy_acc, clean_acc)),
        }

    return results


def run_configuration(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    objective_mode: str,
    readout_training_mode: str,
    architecture: str,
    noise_type: str,
    train_noise: float,
    eval_noise: float,
    epochs: int,
    shots: int,
    seed: int,
) -> Dict[str, object]:
    num_qubits = X_train.shape[1]

    print("\n" + "=" * 80)
    print(f"Objective: {objective_mode}")
    print(f"Readout:   {readout_training_mode}")
    print("=" * 80)

    weights = spsa_train_weights(
        X_train=X_train,
        y_train=y_train,
        num_qubits=num_qubits,
        architecture=architecture,
        noise_type=noise_type,
        train_noise=train_noise,
        objective_mode=objective_mode,
        epochs=epochs,
        shots=shots,
        seed=seed,
    )

    clean_simulator = AerSimulator()
    noisy_eval_simulator = AerSimulator(
        noise_model=create_noise_model(noise_type, eval_noise)
    )

    X_train_features_clean = extract_features(
        clean_simulator, X_train, weights, num_qubits, architecture, shots
    )
    X_test_features_clean = extract_features(
        clean_simulator, X_test, weights, num_qubits, architecture, shots
    )
    X_train_features_noisy = extract_features(
        noisy_eval_simulator, X_train, weights, num_qubits, architecture, shots
    )
    X_test_features_noisy = extract_features(
        noisy_eval_simulator, X_test, weights, num_qubits, architecture, shots
    )

    readout_results = train_and_evaluate_readouts(
        X_train_features_clean=X_train_features_clean,
        X_test_features_clean=X_test_features_clean,
        X_train_features_noisy=X_train_features_noisy,
        X_test_features_noisy=X_test_features_noisy,
        y_train=y_train,
        y_test=y_test,
        readout_training_mode=readout_training_mode,
    )

    best_model_name, best_metrics = max(
        readout_results.items(), key=lambda item: item[1]["noisy_accuracy"]
    )

    print("\nReadout results:")
    for model_name, metrics in readout_results.items():
        print(
            f"{model_name:20s} | clean={metrics['clean_accuracy']:.4f} | "
            f"noisy={metrics['noisy_accuracy']:.4f} | "
            f"drop={metrics['accuracy_drop']:.4f}"
        )

    print("\nBest readout:")
    print(best_model_name, best_metrics)

    return {
        "objective_mode": objective_mode,
        "readout_training_mode": readout_training_mode,
        "architecture": architecture,
        "noise_type": noise_type,
        "train_noise": train_noise,
        "eval_noise": eval_noise,
        "epochs": epochs,
        "shots": shots,
        "feature_dimension": int(X_train_features_clean.shape[1]),
        "readout_results": readout_results,
        "best_model": best_model_name,
        "best_metrics": best_metrics,
    }


def plot_ablation_best(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    labels = []
    clean = []
    noisy = []

    for label, result in results.items():
        labels.append(label.replace(" + ", "\n+\n"))
        clean.append(result["best_metrics"]["clean_accuracy"])
        noisy.append(result["best_metrics"]["noisy_accuracy"])

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, clean, width, label="Clean Accuracy")
    plt.bar(x + width / 2, noisy, width, label="Noisy Accuracy")
    plt.xticks(x, labels, rotation=0)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Framework Ablation Study: Dual-Loss vs Noise-Augmented Readout")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_ablation_drop(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    labels = []
    drops = []

    for label, result in results.items():
        labels.append(label.replace(" + ", "\n+\n"))
        drops.append(result["best_metrics"]["accuracy_drop"])

    plt.figure(figsize=(11, 5))
    plt.bar(labels, drops)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Accuracy Drop")
    plt.title("Framework Ablation Study: Accuracy Drop by Mechanism")
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
    epochs = 12
    shots = 1024

    configs = {
        "standard_loss + standard_readout": {
            "objective_mode": "standard_loss",
            "readout_training_mode": "standard_readout",
            "seed": 101,
        },
        "standard_loss + noise_augmented_readout": {
            "objective_mode": "standard_loss",
            "readout_training_mode": "noise_augmented_readout",
            "seed": 102,
        },
        "dual_loss_stability + standard_readout": {
            "objective_mode": "dual_loss_stability",
            "readout_training_mode": "standard_readout",
            "seed": 103,
        },
        "dual_loss_stability + noise_augmented_readout": {
            "objective_mode": "dual_loss_stability",
            "readout_training_mode": "noise_augmented_readout",
            "seed": 104,
        },
    }

    results: Dict[str, Dict[str, object]] = {}

    for label, config in configs.items():
        results[label] = run_configuration(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            objective_mode=config["objective_mode"],
            readout_training_mode=config["readout_training_mode"],
            architecture=architecture,
            noise_type=noise_type,
            train_noise=train_noise,
            eval_noise=eval_noise,
            epochs=epochs,
            shots=shots,
            seed=config["seed"],
        )

    best_label, best_result = max(
        results.items(),
        key=lambda item: item[1]["best_metrics"]["noisy_accuracy"],
    )

    summary = {
        "description": (
            "Cross-ablation study isolating the contributions of noise-aware "
            "quantum feature generation and noise-augmented classical readout."
        ),
        "architecture": architecture,
        "noise_type": noise_type,
        "train_noise": train_noise,
        "eval_noise": eval_noise,
        "epochs": epochs,
        "shots": shots,
        "results": results,
        "best_configuration": {
            "label": best_label,
            **best_result,
        },
        "thesis_contribution": (
            "This experiment directly tests whether robustness gains come from "
            "noise-aware quantum training, noise-augmented readout training, or "
            "their combination."
        ),
    }

    json_path = RESULTS_DIR / "framework_ablation_summary.json"
    plot_path = RESULTS_DIR / "framework_ablation_accuracy.png"
    drop_path = RESULTS_DIR / "framework_ablation_accuracy_drop.png"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_ablation_best(results, plot_path)
    plot_ablation_drop(results, drop_path)

    print("\nFramework ablation study complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {drop_path}")

    print("\nAblation summary:")
    for label, result in results.items():
        metrics = result["best_metrics"]
        print(
            f"{label:45s} | best={result['best_model']:20s} | "
            f"clean={metrics['clean_accuracy']:.4f} | "
            f"noisy={metrics['noisy_accuracy']:.4f} | "
            f"drop={metrics['accuracy_drop']:.4f}"
        )

    print("\nBest configuration:")
    print(best_label)
    print(best_result["best_metrics"])


if __name__ == "__main__":
    run_pipeline()


