"""
Pipeline: Dual-Loss Multi-Observable Noise-Aware HQNN

Flagship contribution:
This pipeline combines two major thesis improvements:

1. Dual-loss stability-aware HQNN optimization
2. Multi-observable quantum feature extraction with learned classical readout

The purpose is to test whether training the quantum circuit with a noise-aware
objective improves the quality of the quantum features used by learned readouts.
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


RESULTS_DIR = Path("results") / "framework" / "dual_loss_multi_observable_hqnn"


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


def parity_probabilities_from_counts(counts: Dict[str, int]) -> float:
    exp = parity_expectation(counts)
    return float((1 - exp) / 2)


def predict_parity_probs(
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
        probs.append(parity_probabilities_from_counts(counts))

    return np.array(probs, dtype=float)


def extract_multi_features(
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


def binary_cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-10
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def dual_loss_objective(
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
    clean_probs = predict_parity_probs(
        clean_simulator, X, weights, num_qubits, architecture, shots
    )
    noisy_probs = predict_parity_probs(
        noisy_simulator, X, weights, num_qubits, architecture, shots
    )

    clean_loss = binary_cross_entropy(clean_probs, y)
    noisy_loss = binary_cross_entropy(noisy_probs, y)
    stability_loss = float(np.mean(np.abs(clean_probs - noisy_probs)))

    if mode == "random_untrained":
        return clean_loss

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

    raise ValueError("Unknown training mode.")


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

    loss_plus = dual_loss_objective(
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

    loss_minus = dual_loss_objective(
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


def train_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mode: str,
    seed: int,
    architecture: str,
    noise_type: str,
    train_noise: float,
    epochs: int,
    shots: int,
) -> np.ndarray:
    np.random.seed(seed)

    num_qubits = X_train.shape[1]
    num_params = 3 * num_qubits

    weights = np.random.uniform(-np.pi, np.pi, num_params)

    if mode == "random_untrained":
        return weights

    clean_simulator = AerSimulator()
    noisy_train_simulator = AerSimulator(
        noise_model=create_noise_model(noise_type, train_noise)
    )

    alpha_clean = 0.50
    beta_noisy = 0.50
    lambda_stability = 0.75

    for epoch in range(epochs):
        learning_rate = 0.18 / np.sqrt(epoch + 1)
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

        if epoch % 5 == 0 or epoch == epochs - 1:
            obj = dual_loss_objective(
                clean_simulator,
                noisy_train_simulator,
                X_train,
                y_train,
                weights,
                num_qubits,
                architecture,
                mode,
                alpha_clean,
                beta_noisy,
                lambda_stability,
                shots,
            )
            print(f"{mode:24s} | epoch={epoch:02d} | objective={obj:.4f}")

    return weights


def evaluate_trained_feature_extractor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    weights: np.ndarray,
    architecture: str,
    noise_type: str,
    eval_noise: float,
    shots: int,
) -> Dict[str, Dict[str, float]]:
    num_qubits = X_train.shape[1]

    clean_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=create_noise_model(noise_type, eval_noise))

    clean_parity_probs = predict_parity_probs(
        clean_sim, X_test, weights, num_qubits, architecture, shots
    )
    noisy_parity_probs = predict_parity_probs(
        noisy_sim, X_test, weights, num_qubits, architecture, shots
    )

    parity_clean_pred = (clean_parity_probs >= 0.5).astype(int)
    parity_noisy_pred = (noisy_parity_probs >= 0.5).astype(int)

    parity_clean_acc = accuracy_score(y_test, parity_clean_pred)
    parity_noisy_acc = accuracy_score(y_test, parity_noisy_pred)

    Z_train_clean = extract_multi_features(
        clean_sim, X_train, weights, num_qubits, architecture, shots
    )
    Z_test_clean = extract_multi_features(
        clean_sim, X_test, weights, num_qubits, architecture, shots
    )
    Z_test_noisy = extract_multi_features(
        noisy_sim, X_test, weights, num_qubits, architecture, shots
    )

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(Z_train_clean, y_train)

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=7,
        min_samples_leaf=2,
        random_state=42,
    )
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
        "parity_readout": {
            "clean_accuracy": float(parity_clean_acc),
            "noisy_accuracy": float(parity_noisy_acc),
            "accuracy_drop": float(accuracy_drop(parity_clean_acc, parity_noisy_acc)),
            "robustness_score": float(robustness_score(parity_noisy_acc, parity_clean_acc)),
        },
        "multi_observable_logistic_regression": {
            "clean_accuracy": float(lr_clean_acc),
            "noisy_accuracy": float(lr_noisy_acc),
            "accuracy_drop": float(accuracy_drop(lr_clean_acc, lr_noisy_acc)),
            "robustness_score": float(robustness_score(lr_noisy_acc, lr_clean_acc)),
        },
        "multi_observable_random_forest": {
            "clean_accuracy": float(rf_clean_acc),
            "noisy_accuracy": float(rf_noisy_acc),
            "accuracy_drop": float(accuracy_drop(rf_clean_acc, rf_noisy_acc)),
            "robustness_score": float(robustness_score(rf_noisy_acc, rf_clean_acc)),
        },
        "feature_dimensions": {
            "multi_observable_features": int(Z_train_clean.shape[1]),
        },
    }


def plot_noisy_accuracy(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    labels = []
    values = []

    for mode, row in results.items():
        labels.append(f"{mode}\nParity")
        values.append(row["evaluation"]["parity_readout"]["noisy_accuracy"])

        labels.append(f"{mode}\nMulti-LR")
        values.append(
            row["evaluation"]["multi_observable_logistic_regression"]["noisy_accuracy"]
        )

        labels.append(f"{mode}\nMulti-RF")
        values.append(row["evaluation"]["multi_observable_random_forest"]["noisy_accuracy"])

    plt.figure(figsize=(15, 6))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Noisy Accuracy")
    plt.title("Dual-Loss Training + Multi-Observable Learned Readout")
    plt.grid(axis="y")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_best_by_mode(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    labels = []
    values = []

    for mode, row in results.items():
        evals = row["evaluation"]
        best = max(
            [
                evals["parity_readout"]["noisy_accuracy"],
                evals["multi_observable_logistic_regression"]["noisy_accuracy"],
                evals["multi_observable_random_forest"]["noisy_accuracy"],
            ]
        )
        labels.append(mode)
        values.append(best)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Best Noisy Accuracy")
    plt.title("Best Readout Accuracy by Training Objective")
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
    train_shots = 512
    eval_shots = 4096

    modes = [
        "random_untrained",
        "standard",
        "noise_aware",
        "dual_loss",
        "stability_regularized",
    ]

    results: Dict[str, Dict[str, object]] = {}

    for i, mode in enumerate(modes):
        print(f"\n=== Training feature extractor: {mode} ===")

        weights = train_weights(
            X_train=X_train,
            y_train=y_train,
            mode=mode,
            seed=42 + i,
            architecture=architecture,
            noise_type=noise_type,
            train_noise=train_noise,
            epochs=epochs,
            shots=train_shots,
        )

        evaluation = evaluate_trained_feature_extractor(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            weights=weights,
            architecture=architecture,
            noise_type=noise_type,
            eval_noise=eval_noise,
            shots=eval_shots,
        )

        results[mode] = {
            "training_mode": mode,
            "architecture": architecture,
            "evaluation": evaluation,
        }

        print(evaluation)

    best_records = []

    for mode, row in results.items():
        for readout_name, metrics in row["evaluation"].items():
            if readout_name == "feature_dimensions":
                continue
            best_records.append(
                {
                    "training_mode": mode,
                    "readout": readout_name,
                    "clean_accuracy": metrics["clean_accuracy"],
                    "noisy_accuracy": metrics["noisy_accuracy"],
                    "accuracy_drop": metrics["accuracy_drop"],
                    "robustness_score": metrics["robustness_score"],
                }
            )

    best_overall = max(best_records, key=lambda row: row["noisy_accuracy"])

    baseline = results["random_untrained"]["evaluation"]["parity_readout"]["noisy_accuracy"]
    best_gain = best_overall["noisy_accuracy"] - baseline

    summary = {
        "description": (
            "Dual-loss multi-observable HQNN experiment. This pipeline combines "
            "noise-aware optimization with multi-observable quantum feature "
            "extraction and learned classical readouts."
        ),
        "architecture": architecture,
        "noise_type": noise_type,
        "train_noise": train_noise,
        "eval_noise": eval_noise,
        "epochs": epochs,
        "train_shots": train_shots,
        "eval_shots": eval_shots,
        "results": results,
        "best_overall_by_noisy_accuracy": best_overall,
        "baseline_random_parity_noisy_accuracy": float(baseline),
        "best_noisy_accuracy_gain_over_random_parity": float(best_gain),
        "thesis_contribution": (
            "This experiment tests whether noise-aware training objectives improve "
            "the quantum feature representations used by learned classical readouts. "
            "It combines dual-loss optimization, stability regularization, "
            "multi-observable feature extraction, and hybrid readout learning."
        ),
    }

    json_path = RESULTS_DIR / "dual_loss_multi_observable_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    noisy_plot = RESULTS_DIR / "dual_loss_multi_observable_noisy_accuracy.png"
    best_plot = RESULTS_DIR / "dual_loss_multi_observable_best_by_mode.png"

    plot_noisy_accuracy(results, noisy_plot)
    plot_best_by_mode(results, best_plot)

    print("\nDual-loss multi-observable HQNN pipeline complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {noisy_plot}")
    print(f"Saved: {best_plot}")

    print("\nBest overall:")
    print(best_overall)
    print("\nGain over random parity baseline:")
    print(best_gain)


if __name__ == "__main__":
    run_pipeline()
