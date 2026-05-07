"""
Pipeline: Best Architecture Repeated Trials

Purpose:
Validates that the best HQNN configuration is not a one-run accident.

Configuration:
- Linear HQNN architecture
- Multi-observable quantum feature extraction
- Random Forest learned readout
- Depolarizing noise evaluation
- Multiple random seeds

Outputs:
- results/framework/best_architecture_repeated_trials/repeated_trials_summary.json
- results/framework/best_architecture_repeated_trials/repeated_trials_accuracy.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from framework.noise_channels import create_noise_model
from framework.robustness_metrics import accuracy_drop, robustness_score


RESULTS_DIR = Path("results") / "framework" / "best_architecture_repeated_trials"


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
        qc = build_hqnn_circuit(num_qubits, x, weights, architecture)
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        features.append(counts_to_multi_observable_features(counts, num_qubits))

    return np.vstack(features)


def make_dataset(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        class_sep=2.6,
        flip_y=0.005,
        random_state=seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.clip(X_train, -np.pi, np.pi)
    X_test = np.clip(X_test, -np.pi, np.pi)

    return X_train, X_test, y_train, y_test


def run_single_trial(seed: int) -> Dict[str, float]:
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = make_dataset(seed)

    num_qubits = X_train.shape[1]
    num_params = 3 * num_qubits

    architecture = "linear"
    noise_type = "depolarizing"
    eval_noise = 0.05
    shots = 4096

    weights = np.random.uniform(-np.pi, np.pi, num_params)

    clean_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=create_noise_model(noise_type, eval_noise))

    Z_train_clean = extract_features(
        clean_sim, X_train, weights, num_qubits, architecture, shots
    )
    Z_test_clean = extract_features(
        clean_sim, X_test, weights, num_qubits, architecture, shots
    )
    Z_test_noisy = extract_features(
        noisy_sim, X_test, weights, num_qubits, architecture, shots
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        random_state=seed,
    )

    model.fit(Z_train_clean, y_train)

    clean_pred = model.predict(Z_test_clean)
    noisy_pred = model.predict(Z_test_noisy)

    clean_acc = accuracy_score(y_test, clean_pred)
    noisy_acc = accuracy_score(y_test, noisy_pred)

    return {
        "seed": seed,
        "clean_accuracy": float(clean_acc),
        "noisy_accuracy": float(noisy_acc),
        "accuracy_drop": float(accuracy_drop(clean_acc, noisy_acc)),
        "robustness_score": float(robustness_score(noisy_acc, clean_acc)),
    }


def plot_trials(trials: List[Dict[str, float]], output_path: Path) -> None:
    seeds = [str(t["seed"]) for t in trials]
    clean = [t["clean_accuracy"] for t in trials]
    noisy = [t["noisy_accuracy"] for t in trials]

    x = np.arange(len(seeds))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, clean, width, label="Clean Accuracy")
    plt.bar(x + width / 2, noisy, width, label="Noisy Accuracy")
    plt.xticks(x, seeds)
    plt.ylim(0, 1)
    plt.xlabel("Random Seed")
    plt.ylabel("Accuracy")
    plt.title("Repeated Trials: Linear HQNN + Multi-Observable RF Readout")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_pipeline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [42, 43, 44, 45, 46]

    trials = []

    for seed in seeds:
        print(f"\nRunning trial seed={seed}")
        result = run_single_trial(seed)
        print(result)
        trials.append(result)

    clean_values = np.array([t["clean_accuracy"] for t in trials])
    noisy_values = np.array([t["noisy_accuracy"] for t in trials])
    drop_values = np.array([t["accuracy_drop"] for t in trials])
    robustness_values = np.array([t["robustness_score"] for t in trials])

    summary = {
        "description": (
            "Repeated-trial validation of the optimized HQNN configuration: "
            "linear architecture, multi-observable quantum features, and Random Forest readout."
        ),
        "architecture": "linear",
        "readout": "RandomForestClassifier",
        "noise_type": "depolarizing",
        "eval_noise": 0.05,
        "shots": 4096,
        "seeds": seeds,
        "trials": trials,
        "aggregate_results": {
            "mean_clean_accuracy": float(np.mean(clean_values)),
            "std_clean_accuracy": float(np.std(clean_values)),
            "mean_noisy_accuracy": float(np.mean(noisy_values)),
            "std_noisy_accuracy": float(np.std(noisy_values)),
            "min_noisy_accuracy": float(np.min(noisy_values)),
            "max_noisy_accuracy": float(np.max(noisy_values)),
            "mean_accuracy_drop": float(np.mean(drop_values)),
            "mean_robustness_score": float(np.mean(robustness_values)),
        },
        "thesis_contribution": (
            "This experiment checks reproducibility. It supports the claim that the "
            "optimized HQNN improvement is not merely a single lucky run, but remains "
            "strong across repeated randomized trials."
        ),
    }

    json_path = RESULTS_DIR / "repeated_trials_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_path = RESULTS_DIR / "repeated_trials_accuracy.png"
    plot_trials(trials, plot_path)

    print("\nBest-architecture repeated trials complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")
    print("\nAggregate Results:")
    print(json.dumps(summary["aggregate_results"], indent=2))


if __name__ == "__main__":
    run_pipeline()
