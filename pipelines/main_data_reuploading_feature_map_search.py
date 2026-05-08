"""
Pipeline: Data-Reuploading Quantum Feature Map Search

Contribution:
This pipeline tests whether stronger quantum feature maps improve HQNN-style
feature extraction under noise.

It compares multiple quantum feature-map designs:

1. angle
2. angle_square
3. interaction
4. reuploading_2
5. reuploading_3
6. interaction_reuploading

Each feature map is evaluated with:
- quantum-only multi-observable readout
- quantum-classical fusion readout
- clean and noisy evaluation
- multiple classical decoders

Purpose:
Determine whether data reuploading and interaction-aware encodings create more
useful quantum-derived representations than a simple one-pass angle encoding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from framework.noise_channels import create_noise_model
from framework.robustness_metrics import accuracy_drop, robustness_score


RESULTS_DIR = Path("results") / "framework" / "data_reuploading_feature_map_search"


# ---------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------


def make_xor_dataset(n_samples: int = 500, noise: float = 0.20, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, size=(n_samples, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    X += rng.normal(0, noise, size=X.shape)
    return X, y


def make_dataset(dataset_name: str):
    if dataset_name == "synthetic_4d":
        X, y = make_classification(
            n_samples=500,
            n_features=4,
            n_informative=4,
            n_redundant=0,
            n_repeated=0,
            class_sep=2.4,
            flip_y=0.01,
            random_state=42,
        )

    elif dataset_name == "moons":
        X, y = make_moons(n_samples=500, noise=0.22, random_state=42)

    elif dataset_name == "circles":
        X, y = make_circles(
            n_samples=500,
            noise=0.12,
            factor=0.45,
            random_state=42,
        )

    elif dataset_name == "xor":
        X, y = make_xor_dataset(n_samples=500, noise=0.20, seed=42)

    else:
        raise ValueError(
            "dataset_name must be one of: synthetic_4d, moons, circles, xor"
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


# ---------------------------------------------------------------------
# Quantum circuit construction
# ---------------------------------------------------------------------


def pad_features(x: np.ndarray, num_qubits: int) -> np.ndarray:
    x_pad = np.zeros(num_qubits)
    x_pad[: min(len(x), num_qubits)] = x[: min(len(x), num_qubits)]
    return x_pad


def add_angle_layer(qc: QuantumCircuit, x_pad: np.ndarray, num_qubits: int) -> None:
    for i in range(num_qubits):
        qc.ry(float(x_pad[i]), i)


def add_angle_square_layer(qc: QuantumCircuit, x_pad: np.ndarray, num_qubits: int) -> None:
    for i in range(num_qubits):
        qc.ry(float(x_pad[i]), i)
        qc.rz(float(x_pad[i]) ** 2, i)


def add_interaction_layer(qc: QuantumCircuit, x_pad: np.ndarray, num_qubits: int) -> None:
    for i in range(num_qubits):
        qc.ry(float(x_pad[i]), i)
        qc.rz(float(x_pad[i]) ** 2, i)

    for i in range(num_qubits - 1):
        qc.cz(i, i + 1)
        qc.rz(float(x_pad[i] * x_pad[i + 1]), i + 1)

    if num_qubits > 2:
        qc.cz(num_qubits - 1, 0)
        qc.rz(float(x_pad[-1] * x_pad[0]), 0)


def add_reuploading_block(
    qc: QuantumCircuit,
    x_pad: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    layer_index: int,
    interaction: bool,
) -> None:
    offset = layer_index * 3 * num_qubits

    for i in range(num_qubits):
        qc.ry(float(x_pad[i]), i)
        qc.rz(float(x_pad[i]) ** 2, i)

        qc.rx(float(weights[offset + i]), i)
        qc.ry(float(weights[offset + num_qubits + i]), i)
        qc.rz(float(weights[offset + 2 * num_qubits + i]), i)

    if interaction:
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)
            qc.rz(float(x_pad[i] * x_pad[i + 1]), i + 1)

        if num_qubits > 2:
            qc.cz(num_qubits - 1, 0)
            qc.rz(float(x_pad[-1] * x_pad[0]), 0)
    else:
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)


def build_feature_map(
    num_qubits: int,
    x: np.ndarray,
    weights: np.ndarray,
    feature_map_type: str,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    x_pad = pad_features(x, num_qubits)

    if feature_map_type == "angle":
        add_angle_layer(qc, x_pad, num_qubits)

    elif feature_map_type == "angle_square":
        add_angle_square_layer(qc, x_pad, num_qubits)

    elif feature_map_type == "interaction":
        add_interaction_layer(qc, x_pad, num_qubits)

    elif feature_map_type == "reuploading_2":
        for layer in range(2):
            add_reuploading_block(
                qc,
                x_pad,
                weights,
                num_qubits,
                layer_index=layer,
                interaction=False,
            )

    elif feature_map_type == "reuploading_3":
        for layer in range(3):
            add_reuploading_block(
                qc,
                x_pad,
                weights,
                num_qubits,
                layer_index=layer,
                interaction=False,
            )

    elif feature_map_type == "interaction_reuploading":
        for layer in range(2):
            add_reuploading_block(
                qc,
                x_pad,
                weights,
                num_qubits,
                layer_index=layer,
                interaction=True,
            )

    else:
        raise ValueError(
            "feature_map_type must be one of: angle, angle_square, interaction, "
            "reuploading_2, reuploading_3, interaction_reuploading"
        )

    return qc


def build_variational_tail(
    num_qubits: int,
    weights: np.ndarray,
    architecture: str,
    tail_offset: int,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    for i in range(num_qubits):
        qc.rx(float(weights[tail_offset + i]), i)
        qc.ry(float(weights[tail_offset + num_qubits + i]), i)
        qc.rz(float(weights[tail_offset + 2 * num_qubits + i]), i)

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


def parameter_count(num_qubits: int, feature_map_type: str) -> int:
    if feature_map_type in {"angle", "angle_square", "interaction"}:
        reupload_params = 0
    elif feature_map_type == "reuploading_2":
        reupload_params = 2 * 3 * num_qubits
    elif feature_map_type == "reuploading_3":
        reupload_params = 3 * 3 * num_qubits
    elif feature_map_type == "interaction_reuploading":
        reupload_params = 2 * 3 * num_qubits
    else:
        raise ValueError(f"Unknown feature map type: {feature_map_type}")

    tail_params = 3 * num_qubits
    return reupload_params + tail_params


def tail_offset(num_qubits: int, feature_map_type: str) -> int:
    return parameter_count(num_qubits, feature_map_type) - 3 * num_qubits


def build_hqnn_circuit(
    num_qubits: int,
    x: np.ndarray,
    weights: np.ndarray,
    architecture: str,
    feature_map_type: str,
) -> QuantumCircuit:
    qc = build_feature_map(num_qubits, x, weights, feature_map_type)
    qc = qc.compose(
        build_variational_tail(
            num_qubits=num_qubits,
            weights=weights,
            architecture=architecture,
            tail_offset=tail_offset(num_qubits, feature_map_type),
        )
    )
    qc.measure_all()
    return qc


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------


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


def build_simulator(noise_type: str | None, noise_level: float) -> AerSimulator:
    if noise_type is None:
        return AerSimulator()
    return AerSimulator(noise_model=create_noise_model(noise_type, noise_level))


def extract_quantum_features(
    simulator: AerSimulator,
    X: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    feature_map_type: str,
    shots: int,
) -> np.ndarray:
    features = []

    for x in X:
        qc = build_hqnn_circuit(
            num_qubits=num_qubits,
            x=x,
            weights=weights,
            architecture=architecture,
            feature_map_type=feature_map_type,
        )
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        features.append(counts_to_multi_observable_features(counts, num_qubits))

    return np.vstack(features)


# ---------------------------------------------------------------------
# Classical readouts
# ---------------------------------------------------------------------


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


def prepare_features(
    X_classical: np.ndarray,
    Z_quantum: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "quantum_only":
        return Z_quantum

    if mode == "fusion":
        return np.hstack([X_classical, Z_quantum])

    raise ValueError("mode must be quantum_only or fusion")


def evaluate_model(
    model,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test_clean: np.ndarray,
    Z_test_noisy: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
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


# ---------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------


def run_single_configuration(
    dataset_name: str,
    feature_map_type: str,
    readout_mode: str,
    model_name: str,
    seed: int = 42,
) -> Dict[str, object]:
    X_train, X_test, y_train, y_test = make_dataset(dataset_name)

    num_qubits = X_train.shape[1]
    num_params = parameter_count(num_qubits, feature_map_type)

    architecture = "linear"
    noise_type = "depolarizing"
    noise_level = 0.05
    shots = 4096

    rng = np.random.default_rng(seed)
    weights = rng.uniform(-np.pi, np.pi, num_params)

    clean_sim = build_simulator(None, noise_level)
    noisy_sim = build_simulator(noise_type, noise_level)

    Z_train_quantum = extract_quantum_features(
        simulator=clean_sim,
        X=X_train,
        weights=weights,
        num_qubits=num_qubits,
        architecture=architecture,
        feature_map_type=feature_map_type,
        shots=shots,
    )

    Z_test_quantum_clean = extract_quantum_features(
        simulator=clean_sim,
        X=X_test,
        weights=weights,
        num_qubits=num_qubits,
        architecture=architecture,
        feature_map_type=feature_map_type,
        shots=shots,
    )

    Z_test_quantum_noisy = extract_quantum_features(
        simulator=noisy_sim,
        X=X_test,
        weights=weights,
        num_qubits=num_qubits,
        architecture=architecture,
        feature_map_type=feature_map_type,
        shots=shots,
    )

    Z_train = prepare_features(X_train, Z_train_quantum, readout_mode)
    Z_test_clean = prepare_features(X_test, Z_test_quantum_clean, readout_mode)
    Z_test_noisy = prepare_features(X_test, Z_test_quantum_noisy, readout_mode)

    result = evaluate_model(
        model=model_factory(model_name),
        Z_train=Z_train,
        y_train=y_train,
        Z_test_clean=Z_test_clean,
        Z_test_noisy=Z_test_noisy,
        y_test=y_test,
    )

    result.update(
        {
            "dataset": dataset_name,
            "feature_map": feature_map_type,
            "readout_mode": readout_mode,
            "model": model_name,
            "num_qubits": num_qubits,
            "num_parameters": num_params,
            "feature_dimension": int(Z_train.shape[1]),
            "noise_type": noise_type,
            "noise_level": noise_level,
            "shots": shots,
        }
    )

    return result


def plot_best_by_feature_map(summary: Dict[str, object], output_path: Path) -> None:
    feature_maps = summary["feature_maps"]

    labels = []
    values = []

    for feature_map in feature_maps:
        candidates = [
            row for row in summary["flat_results"]
            if row["feature_map"] == feature_map
        ]
        best = max(candidates, key=lambda row: row["noisy_accuracy"])
        labels.append(feature_map)
        values.append(best["noisy_accuracy"])

    plt.figure(figsize=(11, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Best Noisy Accuracy")
    plt.title("Data-Reuploading Feature Map Search: Best Noisy Accuracy")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_best_by_dataset(summary: Dict[str, object], output_path: Path) -> None:
    datasets = summary["datasets"]

    labels = []
    values = []

    for dataset in datasets:
        candidates = [
            row for row in summary["flat_results"]
            if row["dataset"] == dataset
        ]
        best = max(candidates, key=lambda row: row["noisy_accuracy"])
        labels.append(dataset)
        values.append(best["noisy_accuracy"])

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Best Noisy Accuracy")
    plt.title("Best Data-Reuploading HQNN Result by Dataset")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_pipeline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [
        "synthetic_4d",
        "moons",
        "circles",
        "xor",
    ]

    feature_maps = [
        "angle",
        "angle_square",
        "interaction",
        "reuploading_2",
        "reuploading_3",
        "interaction_reuploading",
    ]

    readout_modes = [
        "quantum_only",
        "fusion",
    ]

    model_names = [
        "logistic_regression",
        "svm_rbf",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
    ]

    flat_results = []

    total = len(datasets) * len(feature_maps) * len(readout_modes) * len(model_names)
    counter = 0

    for dataset_name in datasets:
        for feature_map_type in feature_maps:
            for readout_mode in readout_modes:
                for model_name in model_names:
                    counter += 1
                    print(
                        f"[{counter}/{total}] dataset={dataset_name} | "
                        f"feature_map={feature_map_type} | "
                        f"mode={readout_mode} | model={model_name}"
                    )

                    result = run_single_configuration(
                        dataset_name=dataset_name,
                        feature_map_type=feature_map_type,
                        readout_mode=readout_mode,
                        model_name=model_name,
                        seed=42,
                    )

                    flat_results.append(result)

                    print(
                        f"    clean={result['clean_accuracy']:.4f} | "
                        f"noisy={result['noisy_accuracy']:.4f} | "
                        f"drop={result['accuracy_drop']:.4f}"
                    )

    best_overall = max(flat_results, key=lambda row: row["noisy_accuracy"])

    best_quantum_only = max(
        [row for row in flat_results if row["readout_mode"] == "quantum_only"],
        key=lambda row: row["noisy_accuracy"],
    )

    best_fusion = max(
        [row for row in flat_results if row["readout_mode"] == "fusion"],
        key=lambda row: row["noisy_accuracy"],
    )

    best_by_dataset = {}
    for dataset_name in datasets:
        candidates = [row for row in flat_results if row["dataset"] == dataset_name]
        best_by_dataset[dataset_name] = max(
            candidates,
            key=lambda row: row["noisy_accuracy"],
        )

    best_by_feature_map = {}
    for feature_map_type in feature_maps:
        candidates = [
            row for row in flat_results if row["feature_map"] == feature_map_type
        ]
        best_by_feature_map[feature_map_type] = max(
            candidates,
            key=lambda row: row["noisy_accuracy"],
        )

    summary = {
        "description": (
            "Data-reuploading quantum feature-map search for HQNN-style "
            "multi-observable feature extraction. This experiment compares "
            "simple angle encoding, interaction-aware encoding, and repeated "
            "data reuploading under noisy evaluation."
        ),
        "datasets": datasets,
        "feature_maps": feature_maps,
        "readout_modes": readout_modes,
        "models": model_names,
        "flat_results": flat_results,
        "best_overall": best_overall,
        "best_quantum_only": best_quantum_only,
        "best_fusion": best_fusion,
        "best_by_dataset": best_by_dataset,
        "best_by_feature_map": best_by_feature_map,
        "thesis_contribution": (
            "This pipeline tests whether quantum-side feature-map design affects "
            "the usefulness and robustness of HQNN-derived representations. "
            "Data reuploading and interaction-aware encodings are evaluated as "
            "algorithmic mechanisms for increasing the expressive power of the "
            "quantum feature extractor before learned classical readout."
        ),
    }

    json_path = RESULTS_DIR / "data_reuploading_feature_map_search_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    feature_map_plot = RESULTS_DIR / "feature_map_search_best_by_feature_map.png"
    dataset_plot = RESULTS_DIR / "feature_map_search_best_by_dataset.png"

    plot_best_by_feature_map(summary, feature_map_plot)
    plot_best_by_dataset(summary, dataset_plot)

    print("\nData-reuploading feature map search complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {feature_map_plot}")
    print(f"Saved: {dataset_plot}")

    print("\nBest overall:")
    print(best_overall)

    print("\nBest quantum-only:")
    print(best_quantum_only)

    print("\nBest fusion:")
    print(best_fusion)

    print("\nBest by feature map:")
    for feature_map, row in best_by_feature_map.items():
        print(feature_map, row)


if __name__ == "__main__":
    run_pipeline()


