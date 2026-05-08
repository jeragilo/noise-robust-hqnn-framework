"""
Pipeline: Quantum-Classical Feature Fusion HQNN

Contribution:
This pipeline separates and formalizes the strongest result from the
noise-augmented readout experiment.

It compares:

1. Classical-only learning
2. Quantum-only multi-observable learning
3. Quantum-classical feature fusion
4. Noise-augmented quantum-classical feature fusion

Goal:
Test whether quantum-derived multi-observable features complement classical
features and improve clean/noisy classification performance under NISQ-style
noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from framework.noise_channels import create_noise_model
from framework.robustness_metrics import accuracy_drop, robustness_score


RESULTS_DIR = Path("results") / "framework" / "quantum_classical_feature_fusion"


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


def extract_quantum_features(
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


def build_simulator(noise_type: str | None, noise_level: float) -> AerSimulator:
    if noise_type is None:
        return AerSimulator()
    return AerSimulator(noise_model=create_noise_model(noise_type, noise_level))


def make_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        class_sep=2.8,
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


def model_factory(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=3000, random_state=42)

    if model_name == "svm_rbf":
        return SVC(kernel="rbf", C=3.0, gamma="scale", random_state=42)

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=9,
            min_samples_leaf=2,
            random_state=42,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
        )

    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def augment_training_set(
    X_classical: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    shots: int,
    noise_channels: List[str | None],
    noise_level: float,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_blocks = []
    label_blocks = []

    for noise_type in noise_channels:
        simulator = build_simulator(noise_type, noise_level)

        Z_quantum = extract_quantum_features(
            simulator=simulator,
            X=X_classical,
            weights=weights,
            num_qubits=num_qubits,
            architecture=architecture,
            shots=shots,
        )

        if mode == "classical_only":
            features = X_classical

        elif mode == "quantum_only":
            features = Z_quantum

        elif mode == "fusion":
            features = np.hstack([X_classical, Z_quantum])

        else:
            raise ValueError("mode must be one of: classical_only, quantum_only, fusion")

        feature_blocks.append(features)
        label_blocks.append(y)

    return np.vstack(feature_blocks), np.concatenate(label_blocks)


def prepare_test_features(
    X_classical: np.ndarray,
    Z_quantum: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "classical_only":
        return X_classical

    if mode == "quantum_only":
        return Z_quantum

    if mode == "fusion":
        return np.hstack([X_classical, Z_quantum])

    raise ValueError("mode must be one of: classical_only, quantum_only, fusion")


def evaluate_model_across_noise(
    model,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    test_features_clean: np.ndarray,
    test_features_by_noise: Dict[str, np.ndarray],
    y_test: np.ndarray,
) -> Dict[str, object]:
    model.fit(Z_train, y_train)

    clean_pred = model.predict(test_features_clean)
    clean_acc = accuracy_score(y_test, clean_pred)

    noise_results = {}

    for noise_type, test_features in test_features_by_noise.items():
        noisy_pred = model.predict(test_features)
        noisy_acc = accuracy_score(y_test, noisy_pred)

        noise_results[noise_type] = {
            "noisy_accuracy": float(noisy_acc),
            "accuracy_drop": float(accuracy_drop(clean_acc, noisy_acc)),
            "robustness_score": float(robustness_score(noisy_acc, clean_acc)),
        }

    noisy_values = [row["noisy_accuracy"] for row in noise_results.values()]

    return {
        "clean_accuracy": float(clean_acc),
        "mean_noisy_accuracy": float(np.mean(noisy_values)),
        "min_noisy_accuracy": float(np.min(noisy_values)),
        "max_noisy_accuracy": float(np.max(noisy_values)),
        "std_noisy_accuracy": float(np.std(noisy_values, ddof=0)),
        "noise_results": noise_results,
    }


def plot_summary(summary: Dict[str, object], output_path: Path, metric: str) -> None:
    labels = []
    values = []

    for mode_name, mode_results in summary["results"].items():
        for model_name, model_results in mode_results.items():
            labels.append(f"{mode_name}\n{model_name}")
            values.append(model_results[metric])

    plt.figure(figsize=(15, 6))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Quantum-Classical Feature Fusion: {metric.replace('_', ' ').title()}")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y")
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
    noise_level = 0.05
    shots = 4096

    weights = np.random.uniform(-np.pi, np.pi, num_params)

    eval_noise_channels = [
        "depolarizing",
        "bit_flip",
        "phase_flip",
        "amplitude_damping",
    ]

    clean_sim = AerSimulator()

    Z_test_quantum_clean = extract_quantum_features(
        simulator=clean_sim,
        X=X_test,
        weights=weights,
        num_qubits=num_qubits,
        architecture=architecture,
        shots=shots,
    )

    Z_test_quantum_by_noise = {}

    for noise_type in eval_noise_channels:
        noisy_sim = build_simulator(noise_type, noise_level)

        Z_test_quantum_by_noise[noise_type] = extract_quantum_features(
            simulator=noisy_sim,
            X=X_test,
            weights=weights,
            num_qubits=num_qubits,
            architecture=architecture,
            shots=shots,
        )

    experiment_modes = {
        "classical_only": {
            "mode": "classical_only",
            "training_noise_channels": [None],
        },
        "quantum_only_clean_train": {
            "mode": "quantum_only",
            "training_noise_channels": [None],
        },
        "fusion_clean_train": {
            "mode": "fusion",
            "training_noise_channels": [None],
        },
        "fusion_noise_augmented": {
            "mode": "fusion",
            "training_noise_channels": [
                None,
                "depolarizing",
                "bit_flip",
                "phase_flip",
                "amplitude_damping",
            ],
        },
    }

    model_names = [
        "logistic_regression",
        "svm_rbf",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
    ]

    results: Dict[str, Dict[str, object]] = {}

    for experiment_name, experiment_config in experiment_modes.items():
        print(f"\n=== Experiment mode: {experiment_name} ===")

        mode = experiment_config["mode"]
        training_noise_channels = experiment_config["training_noise_channels"]

        Z_train, y_train_augmented = augment_training_set(
            X_classical=X_train,
            y=y_train,
            weights=weights,
            num_qubits=num_qubits,
            architecture=architecture,
            shots=shots,
            noise_channels=training_noise_channels,
            noise_level=noise_level,
            mode=mode,
        )

        test_features_clean = prepare_test_features(
            X_classical=X_test,
            Z_quantum=Z_test_quantum_clean,
            mode=mode,
        )

        test_features_by_noise = {
            noise_type: prepare_test_features(
                X_classical=X_test,
                Z_quantum=Z_quantum,
                mode=mode,
            )
            for noise_type, Z_quantum in Z_test_quantum_by_noise.items()
        }

        results[experiment_name] = {}

        for model_name in model_names:
            model = model_factory(model_name)

            model_results = evaluate_model_across_noise(
                model=model,
                Z_train=Z_train,
                y_train=y_train_augmented,
                test_features_clean=test_features_clean,
                test_features_by_noise=test_features_by_noise,
                y_test=y_test,
            )

            results[experiment_name][model_name] = model_results

            print(
                f"{experiment_name:28s} | {model_name:20s} | "
                f"clean={model_results['clean_accuracy']:.4f} | "
                f"mean_noisy={model_results['mean_noisy_accuracy']:.4f} | "
                f"min_noisy={model_results['min_noisy_accuracy']:.4f}"
            )

    flat_results = []

    for experiment_name, experiment_results in results.items():
        for model_name, model_results in experiment_results.items():
            flat_results.append(
                {
                    "experiment": experiment_name,
                    "model": model_name,
                    "clean_accuracy": model_results["clean_accuracy"],
                    "mean_noisy_accuracy": model_results["mean_noisy_accuracy"],
                    "min_noisy_accuracy": model_results["min_noisy_accuracy"],
                    "max_noisy_accuracy": model_results["max_noisy_accuracy"],
                    "std_noisy_accuracy": model_results["std_noisy_accuracy"],
                }
            )

    best_by_mean_noisy = max(flat_results, key=lambda row: row["mean_noisy_accuracy"])
    best_by_worst_case = max(flat_results, key=lambda row: row["min_noisy_accuracy"])

    classical_best = max(
        [row for row in flat_results if row["experiment"] == "classical_only"],
        key=lambda row: row["mean_noisy_accuracy"],
    )

    fusion_best = max(
        [row for row in flat_results if "fusion" in row["experiment"]],
        key=lambda row: row["mean_noisy_accuracy"],
    )

    summary = {
        "description": (
            "Dedicated quantum-classical feature fusion experiment. This pipeline "
            "compares classical-only learning, quantum-only multi-observable "
            "learning, clean-trained feature fusion, and noise-augmented feature "
            "fusion under multiple NISQ-style noise channels."
        ),
        "architecture": architecture,
        "noise_level": noise_level,
        "shots": shots,
        "num_qubits": num_qubits,
        "feature_dimensions": {
            "classical_features": int(X_train.shape[1]),
            "quantum_multi_observable_features": int(Z_test_quantum_clean.shape[1]),
            "fusion_features": int(X_train.shape[1] + Z_test_quantum_clean.shape[1]),
        },
        "eval_noise_channels": eval_noise_channels,
        "experiment_modes": experiment_modes,
        "results": results,
        "best_by_mean_noisy_accuracy": best_by_mean_noisy,
        "best_by_worst_case_noisy_accuracy": best_by_worst_case,
        "best_classical_only": classical_best,
        "best_fusion": fusion_best,
        "fusion_gain_over_best_classical_mean_noisy": float(
            fusion_best["mean_noisy_accuracy"] - classical_best["mean_noisy_accuracy"]
        ),
        "fusion_gain_over_best_classical_worst_case": float(
            fusion_best["min_noisy_accuracy"] - classical_best["min_noisy_accuracy"]
        ),
        "thesis_contribution": (
            "This experiment tests whether quantum-derived multi-observable "
            "features complement classical features when both are combined in a "
            "noise-aware hybrid readout. The contribution is not framed as pure "
            "quantum advantage, but as evidence that quantum feature extraction "
            "can participate in a stronger hybrid quantum-classical learning "
            "pipeline under NISQ noise."
        ),
    }

    json_path = RESULTS_DIR / "quantum_classical_feature_fusion_summary.json"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    mean_plot = RESULTS_DIR / "feature_fusion_mean_noisy_accuracy.png"
    worst_plot = RESULTS_DIR / "feature_fusion_worst_case_accuracy.png"
    clean_plot = RESULTS_DIR / "feature_fusion_clean_accuracy.png"

    plot_summary(summary, mean_plot, "mean_noisy_accuracy")
    plot_summary(summary, worst_plot, "min_noisy_accuracy")
    plot_summary(summary, clean_plot, "clean_accuracy")

    print("\nQuantum-classical feature fusion pipeline complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {mean_plot}")
    print(f"Saved: {worst_plot}")
    print(f"Saved: {clean_plot}")

    print("\nBest by mean noisy accuracy:")
    print(best_by_mean_noisy)

    print("\nBest by worst-case noisy accuracy:")
    print(best_by_worst_case)

    print("\nBest classical-only:")
    print(classical_best)

    print("\nBest fusion:")
    print(fusion_best)

    print("\nFusion gain over best classical mean noisy:")
    print(summary["fusion_gain_over_best_classical_mean_noisy"])


if __name__ == "__main__":
    run_pipeline()
