"""
Pipeline: Classical-Challenging Dataset Benchmark for HQNN Feature Fusion

Contribution:
This pipeline tests whether quantum multi-observable features become more useful
on nonlinear datasets where simple classical linear models struggle.

It compares:
1. Classical-only features
2. Quantum-only multi-observable features
3. Quantum-classical feature fusion
4. Noise-augmented quantum-classical feature fusion

Datasets:
- moons
- circles
- XOR-style synthetic classification

Purpose:
Evaluate whether HQNN feature extraction provides complementary value on
nonlinear decision boundaries instead of only easy linearly separable datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_circles, make_moons
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from framework.noise_channels import create_noise_model
from framework.robustness_metrics import accuracy_drop, robustness_score


RESULTS_DIR = Path("results") / "framework" / "classical_challenging_benchmark"


def make_xor_dataset(n_samples: int = 500, noise: float = 0.20, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, size=(n_samples, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    X += rng.normal(0, noise, size=X.shape)
    return X, y


def make_dataset(dataset_name: str):
    if dataset_name == "moons":
        X, y = make_moons(n_samples=500, noise=0.22, random_state=42)
    elif dataset_name == "circles":
        X, y = make_circles(n_samples=500, noise=0.12, factor=0.45, random_state=42)
    elif dataset_name == "xor":
        X, y = make_xor_dataset(n_samples=500, noise=0.20, seed=42)
    else:
        raise ValueError("dataset_name must be one of: moons, circles, xor")

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


def build_feature_map(num_qubits: int, x: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    x_pad = np.zeros(num_qubits)
    x_pad[: len(x)] = x

    for i in range(num_qubits):
        qc.ry(float(x_pad[i]), i)
        qc.rz(float(x_pad[i]) ** 2, i)

    # Feature interaction encoding
    if num_qubits >= 2:
        qc.cz(0, 1)
        qc.rz(float(x_pad[0] * x_pad[1]), 1)

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
    qc = build_feature_map(num_qubits, x)
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


def counts_to_multi_observable_features(counts: Dict[str, int], num_qubits: int):
    probs = bitstring_probabilities(counts, num_qubits)
    z_vals = z_expectations(counts, num_qubits)
    zz_vals = zz_correlations(counts, num_qubits)
    parity = np.array([parity_expectation(counts)], dtype=float)
    stats = probability_statistics(probs)
    return np.concatenate([probs, z_vals, zz_vals, parity, stats])


def build_simulator(noise_type: str | None, noise_level: float):
    if noise_type is None:
        return AerSimulator()
    return AerSimulator(noise_model=create_noise_model(noise_type, noise_level))


def extract_quantum_features(
    simulator,
    X,
    weights,
    num_qubits,
    architecture,
    shots,
):
    features = []

    for x in X:
        qc = build_hqnn_circuit(num_qubits, x, weights, architecture)
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        features.append(counts_to_multi_observable_features(counts, num_qubits))

    return np.vstack(features)


def model_factory(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=3000, random_state=42)

    if model_name == "svm_rbf":
        return SVC(kernel="rbf", C=3.0, gamma="scale", random_state=42)

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
        )

    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def prepare_features(X_classical, Z_quantum, mode: str):
    if mode == "classical_only":
        return X_classical
    if mode == "quantum_only":
        return Z_quantum
    if mode == "fusion":
        return np.hstack([X_classical, Z_quantum])
    raise ValueError("mode must be classical_only, quantum_only, or fusion")


def evaluate_model(model, Z_train, y_train, Z_test_clean, Z_test_noisy, y_test):
    model.fit(Z_train, y_train)

    clean_pred = model.predict(Z_test_clean)
    noisy_pred = model.predict(Z_test_noisy)

    clean_acc = accuracy_score(y_test, clean_pred)
    noisy_acc = accuracy_score(y_test, noisy_pred)

    return {
        "clean_accuracy": float(clean_acc),
        "noisy_accuracy": float(noisy_acc),
        "accuracy_drop": float(accuracy_drop(clean_acc, noisy_acc)),
        "robustness_score": float(robustness_score(noisy_acc, clean_acc)),
    }


def augment_noise_training(
    X_train,
    y_train,
    weights,
    num_qubits,
    architecture,
    shots,
    noise_level,
    mode,
):
    noise_channels = [None, "depolarizing", "bit_flip", "phase_flip", "amplitude_damping"]

    blocks = []
    labels = []

    for noise_type in noise_channels:
        sim = build_simulator(noise_type, noise_level)
        Z_quantum = extract_quantum_features(
            sim, X_train, weights, num_qubits, architecture, shots
        )
        blocks.append(prepare_features(X_train, Z_quantum, mode))
        labels.append(y_train)

    return np.vstack(blocks), np.concatenate(labels)


def run_single_dataset(dataset_name: str):
    print(f"\n==============================")
    print(f"Dataset: {dataset_name}")
    print(f"==============================")

    X_train, X_test, y_train, y_test = make_dataset(dataset_name)

    num_qubits = 2
    num_params = 3 * num_qubits
    architecture = "linear"
    noise_type = "depolarizing"
    noise_level = 0.05
    shots = 4096

    rng = np.random.default_rng(42)
    weights = rng.uniform(-np.pi, np.pi, num_params)

    clean_sim = build_simulator(None, noise_level)
    noisy_sim = build_simulator(noise_type, noise_level)

    Z_train_quantum_clean = extract_quantum_features(
        clean_sim, X_train, weights, num_qubits, architecture, shots
    )
    Z_test_quantum_clean = extract_quantum_features(
        clean_sim, X_test, weights, num_qubits, architecture, shots
    )
    Z_test_quantum_noisy = extract_quantum_features(
        noisy_sim, X_test, weights, num_qubits, architecture, shots
    )

    modes = {
        "classical_only": {
            "mode": "classical_only",
            "noise_augmented": False,
        },
        "quantum_only": {
            "mode": "quantum_only",
            "noise_augmented": False,
        },
        "fusion_clean_train": {
            "mode": "fusion",
            "noise_augmented": False,
        },
        "fusion_noise_augmented": {
            "mode": "fusion",
            "noise_augmented": True,
        },
    }

    models = [
        "logistic_regression",
        "svm_rbf",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
    ]

    dataset_results = {}

    for experiment_name, cfg in modes.items():
        print(f"\n--- Mode: {experiment_name} ---")

        mode = cfg["mode"]

        if cfg["noise_augmented"]:
            Z_train, y_train_used = augment_noise_training(
                X_train,
                y_train,
                weights,
                num_qubits,
                architecture,
                shots,
                noise_level,
                mode,
            )
        else:
            Z_train = prepare_features(X_train, Z_train_quantum_clean, mode)
            y_train_used = y_train

        Z_test_clean = prepare_features(X_test, Z_test_quantum_clean, mode)
        Z_test_noisy = prepare_features(X_test, Z_test_quantum_noisy, mode)

        dataset_results[experiment_name] = {}

        for model_name in models:
            result = evaluate_model(
                model_factory(model_name),
                Z_train,
                y_train_used,
                Z_test_clean,
                Z_test_noisy,
                y_test,
            )

            dataset_results[experiment_name][model_name] = result

            print(
                f"{model_name:20s} | clean={result['clean_accuracy']:.4f} | "
                f"noisy={result['noisy_accuracy']:.4f} | "
                f"drop={result['accuracy_drop']:.4f}"
            )

    flat = []
    for experiment_name, experiment_results in dataset_results.items():
        for model_name, result in experiment_results.items():
            flat.append(
                {
                    "dataset": dataset_name,
                    "experiment": experiment_name,
                    "model": model_name,
                    **result,
                }
            )

    best_overall = max(flat, key=lambda row: row["noisy_accuracy"])
    best_classical = max(
        [row for row in flat if row["experiment"] == "classical_only"],
        key=lambda row: row["noisy_accuracy"],
    )
    best_fusion = max(
        [row for row in flat if "fusion" in row["experiment"]],
        key=lambda row: row["noisy_accuracy"],
    )

    return {
        "dataset_name": dataset_name,
        "results": dataset_results,
        "best_overall": best_overall,
        "best_classical": best_classical,
        "best_fusion": best_fusion,
        "fusion_gain_over_best_classical": float(
            best_fusion["noisy_accuracy"] - best_classical["noisy_accuracy"]
        ),
    }


def plot_dataset_best(summary, output_path):
    datasets = []
    classical = []
    fusion = []
    quantum = []

    for dataset_name, result in summary["datasets"].items():
        datasets.append(dataset_name)
        classical.append(result["best_classical"]["noisy_accuracy"])
        fusion.append(result["best_fusion"]["noisy_accuracy"])

        quantum_candidates = []
        for model_name, row in result["results"]["quantum_only"].items():
            quantum_candidates.append(row["noisy_accuracy"])
        quantum.append(max(quantum_candidates))

    x = np.arange(len(datasets))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, classical, width, label="Best Classical")
    plt.bar(x, quantum, width, label="Best Quantum-Only")
    plt.bar(x + width, fusion, width, label="Best Fusion")
    plt.xticks(x, datasets)
    plt.ylim(0, 1)
    plt.ylabel("Noisy Accuracy")
    plt.title("Classical-Challenging Dataset Benchmark")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = ["moons", "circles", "xor"]

    results = {}

    for dataset_name in datasets:
        results[dataset_name] = run_single_dataset(dataset_name)

    summary = {
        "description": (
            "Classical-challenging nonlinear benchmark comparing classical-only, "
            "quantum-only, quantum-classical fusion, and noise-augmented fusion "
            "on moons, circles, and XOR-style datasets."
        ),
        "datasets": results,
        "thesis_contribution": (
            "This experiment tests whether quantum-derived multi-observable "
            "features provide more value on nonlinear decision boundaries where "
            "simple classical linear models struggle. It provides a fairer test "
            "of hybrid feature fusion than highly separable synthetic datasets."
        ),
    }

    json_path = RESULTS_DIR / "classical_challenging_benchmark_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_path = RESULTS_DIR / "classical_challenging_best_accuracy.png"
    plot_dataset_best(summary, plot_path)

    print("\nClassical-challenging benchmark complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")

    print("\nDataset summary:")
    for dataset_name, result in results.items():
        print(f"\n{dataset_name}")
        print("Best classical:", result["best_classical"])
        print("Best fusion:", result["best_fusion"])
        print("Fusion gain:", result["fusion_gain_over_best_classical"])


if __name__ == "__main__":
    run_pipeline()
