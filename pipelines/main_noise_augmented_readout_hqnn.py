"""
Pipeline: Noise-Augmented Readout HQNN

Contribution:
This pipeline improves the learned-readout HQNN by training the classical
readout on quantum features generated under multiple noise conditions.

Instead of:
    train readout on clean quantum features only
    evaluate on noisy quantum features

this pipeline compares:
    1. Clean-trained readout
    2. Single-noise augmented readout
    3. Multi-channel noise-augmented readout
    4. Multi-channel noise-augmented + classical feature fusion readout

Core idea:
The quantum circuit is treated as a feature extractor, and the classical
readout is trained to recognize how those quantum features deform under
NISQ-style noise.
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

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from framework.noise_channels import create_noise_model
from framework.robustness_metrics import accuracy_drop, robustness_score


RESULTS_DIR = Path("results") / "framework" / "noise_augmented_readout_hqnn"


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


def extract_multi_observable_features(
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


def build_simulator(noise_type: str | None, noise_level: float) -> AerSimulator:
    if noise_type is None:
        return AerSimulator()

    return AerSimulator(noise_model=create_noise_model(noise_type, noise_level))


def augment_features_across_noise_channels(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    shots: int,
    noise_channels: List[str | None],
    noise_level: float,
    include_classical_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_blocks = []
    label_blocks = []

    for noise_type in noise_channels:
        simulator = build_simulator(noise_type, noise_level)

        Z = extract_multi_observable_features(
            simulator=simulator,
            X=X,
            weights=weights,
            num_qubits=num_qubits,
            architecture=architecture,
            shots=shots,
        )

        if include_classical_features:
            Z = np.hstack([X, Z])

        feature_blocks.append(Z)
        label_blocks.append(y)

    return np.vstack(feature_blocks), np.concatenate(label_blocks)


def evaluate_model(
    model,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test_clean: np.ndarray,
    Z_test_by_noise: Dict[str, np.ndarray],
    y_test: np.ndarray,
) -> Dict[str, object]:
    model.fit(Z_train, y_train)

    clean_pred = model.predict(Z_test_clean)
    clean_acc = accuracy_score(y_test, clean_pred)

    noise_results = {}

    for noise_type, Z_test_noisy in Z_test_by_noise.items():
        noisy_pred = model.predict(Z_test_noisy)
        noisy_acc = accuracy_score(y_test, noisy_pred)

        noise_results[noise_type] = {
            "noisy_accuracy": float(noisy_acc),
            "accuracy_drop": float(accuracy_drop(clean_acc, noisy_acc)),
            "robustness_score": float(robustness_score(noisy_acc, clean_acc)),
        }

    noisy_values = [
        row["noisy_accuracy"]
        for row in noise_results.values()
    ]

    return {
        "clean_accuracy": float(clean_acc),
        "noise_results": noise_results,
        "mean_noisy_accuracy": float(np.mean(noisy_values)),
        "min_noisy_accuracy": float(np.min(noisy_values)),
        "max_noisy_accuracy": float(np.max(noisy_values)),
        "std_noisy_accuracy": float(np.std(noisy_values, ddof=0)),
    }


def model_factory(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=42)

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
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

    raise ValueError("Unknown model name.")


def plot_mean_noisy_accuracy(summary: Dict[str, object], output_path: Path) -> None:
    labels = []
    values = []

    for strategy_name, strategy_results in summary["results"].items():
        for model_name, model_results in strategy_results.items():
            labels.append(f"{strategy_name}\n{model_name}")
            values.append(model_results["mean_noisy_accuracy"])

    plt.figure(figsize=(14, 6))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Mean Noisy Accuracy")
    plt.title("Noise-Augmented HQNN Readout Training")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_min_noisy_accuracy(summary: Dict[str, object], output_path: Path) -> None:
    labels = []
    values = []

    for strategy_name, strategy_results in summary["results"].items():
        for model_name, model_results in strategy_results.items():
            labels.append(f"{strategy_name}\n{model_name}")
            values.append(model_results["min_noisy_accuracy"])

    plt.figure(figsize=(14, 6))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Worst-Case Noisy Accuracy")
    plt.title("Worst-Case Accuracy Across Noise Channels")
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

    training_strategies = {
        "clean_only": {
            "noise_channels": [None],
            "include_classical_features": False,
        },
        "single_noise_depolarizing": {
            "noise_channels": [None, "depolarizing"],
            "include_classical_features": False,
        },
        "multi_noise_augmented": {
            "noise_channels": [
                None,
                "depolarizing",
                "bit_flip",
                "phase_flip",
                "amplitude_damping",
            ],
            "include_classical_features": False,
        },
        "multi_noise_feature_fusion": {
            "noise_channels": [
                None,
                "depolarizing",
                "bit_flip",
                "phase_flip",
                "amplitude_damping",
            ],
            "include_classical_features": True,
        },
    }

    models = [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ]

    clean_sim = AerSimulator()

    Z_test_clean_base = extract_multi_observable_features(
        simulator=clean_sim,
        X=X_test,
        weights=weights,
        num_qubits=num_qubits,
        architecture=architecture,
        shots=shots,
    )

    Z_test_by_noise_base = {}

    for noise_type in eval_noise_channels:
        noisy_sim = build_simulator(noise_type, noise_level)

        Z_test_by_noise_base[noise_type] = extract_multi_observable_features(
            simulator=noisy_sim,
            X=X_test,
            weights=weights,
            num_qubits=num_qubits,
            architecture=architecture,
            shots=shots,
        )

    results: Dict[str, Dict[str, object]] = {}

    for strategy_name, strategy_config in training_strategies.items():
        print(f"\n=== Training strategy: {strategy_name} ===")

        include_classical = bool(strategy_config["include_classical_features"])

        Z_train, y_train_aug = augment_features_across_noise_channels(
            X=X_train,
            y=y_train,
            weights=weights,
            num_qubits=num_qubits,
            architecture=architecture,
            shots=shots,
            noise_channels=strategy_config["noise_channels"],
            noise_level=noise_level,
            include_classical_features=include_classical,
        )

        if include_classical:
            Z_test_clean = np.hstack([X_test, Z_test_clean_base])
            Z_test_by_noise = {
                noise_type: np.hstack([X_test, Z_test])
                for noise_type, Z_test in Z_test_by_noise_base.items()
            }
        else:
            Z_test_clean = Z_test_clean_base
            Z_test_by_noise = Z_test_by_noise_base

        results[strategy_name] = {}

        for model_name in models:
            model = model_factory(model_name)

            model_results = evaluate_model(
                model=model,
                Z_train=Z_train,
                y_train=y_train_aug,
                Z_test_clean=Z_test_clean,
                Z_test_by_noise=Z_test_by_noise,
                y_test=y_test,
            )

            results[strategy_name][model_name] = model_results

            print(
                f"{strategy_name:28s} | {model_name:20s} | "
                f"clean={model_results['clean_accuracy']:.4f} | "
                f"mean_noisy={model_results['mean_noisy_accuracy']:.4f} | "
                f"min_noisy={model_results['min_noisy_accuracy']:.4f}"
            )

    flat_results = []

    for strategy_name, strategy_results in results.items():
        for model_name, model_results in strategy_results.items():
            flat_results.append(
                {
                    "strategy": strategy_name,
                    "model": model_name,
                    "clean_accuracy": model_results["clean_accuracy"],
                    "mean_noisy_accuracy": model_results["mean_noisy_accuracy"],
                    "min_noisy_accuracy": model_results["min_noisy_accuracy"],
                    "max_noisy_accuracy": model_results["max_noisy_accuracy"],
                    "std_noisy_accuracy": model_results["std_noisy_accuracy"],
                }
            )

    best_by_mean = max(flat_results, key=lambda row: row["mean_noisy_accuracy"])
    best_by_worst_case = max(flat_results, key=lambda row: row["min_noisy_accuracy"])

    summary = {
        "description": (
            "Noise-augmented readout HQNN experiment. This pipeline trains "
            "classical readouts on quantum multi-observable features generated "
            "under clean and noisy conditions, then evaluates generalization "
            "across multiple NISQ-style noise channels."
        ),
        "architecture": architecture,
        "noise_level": noise_level,
        "shots": shots,
        "num_qubits": num_qubits,
        "feature_dimensions": {
            "multi_observable_features": int(Z_test_clean_base.shape[1]),
            "feature_fusion_features": int(X_test.shape[1] + Z_test_clean_base.shape[1]),
        },
        "eval_noise_channels": eval_noise_channels,
        "training_strategies": training_strategies,
        "results": results,
        "best_by_mean_noisy_accuracy": best_by_mean,
        "best_by_worst_case_noisy_accuracy": best_by_worst_case,
        "thesis_contribution": (
            "This experiment embeds noise awareness into the learned readout "
            "stage by augmenting the readout training set with quantum features "
            "generated under multiple NISQ noise channels. It tests whether the "
            "classical decoder can learn noise-deformed quantum feature geometry "
            "rather than being trained only on clean circuit outputs."
        ),
    }

    json_path = RESULTS_DIR / "noise_augmented_readout_summary.json"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    mean_plot_path = RESULTS_DIR / "noise_augmented_mean_noisy_accuracy.png"
    worst_plot_path = RESULTS_DIR / "noise_augmented_worst_case_accuracy.png"

    plot_mean_noisy_accuracy(summary, mean_plot_path)
    plot_min_noisy_accuracy(summary, worst_plot_path)

    print("\nNoise-augmented readout HQNN pipeline complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {mean_plot_path}")
    print(f"Saved: {worst_plot_path}")

    print("\nBest by mean noisy accuracy:")
    print(best_by_mean)

    print("\nBest by worst-case noisy accuracy:")
    print(best_by_worst_case)


if __name__ == "__main__":
    run_pipeline()
