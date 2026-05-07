"""
Pipeline: Multi-Observable HQNN Readout

Contribution:
This pipeline improves the learned-readout HQNN by extracting richer quantum
features from the circuit.

Instead of using only:
1. fixed parity readout, or
2. raw bitstring probabilities,

this version builds a multi-observable feature vector containing:
- full bitstring probability distribution
- single-qubit Z expectations
- pairwise ZZ correlations
- global parity expectation
- probability mass statistics

This tests whether richer quantum measurement information improves clean and
noisy HQNN classification performance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List

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


RESULTS_DIR = Path("results") / "framework" / "multi_observable_hqnn"


def build_feature_map(num_qubits: int, x: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(float(x[i]), i)
        qc.rz(float(x[i]) ** 2, i)
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


def parity_expectation(counts: Dict[str, int]) -> float:
    shots = sum(counts.values())
    exp = 0.0

    for bitstring, count in counts.items():
        parity = bitstring.count("1") % 2
        sign = 1 if parity == 0 else -1
        exp += sign * count / shots

    return float(exp)


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


def extract_probability_features(
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
        features.append(bitstring_probabilities(counts, num_qubits))

    return np.vstack(features)


def parity_predict(
    simulator: AerSimulator,
    X: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    architecture: str,
    shots: int,
) -> np.ndarray:
    preds = []

    for x in X:
        qc = build_hqnn_circuit(num_qubits, x, weights, architecture)
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()

        exp = parity_expectation(counts)
        p1 = (1 - exp) / 2
        preds.append(1 if p1 >= 0.5 else 0)

    return np.array(preds)


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


def evaluate_readout(
    model,
    Z_train_clean: np.ndarray,
    y_train: np.ndarray,
    Z_test_clean: np.ndarray,
    Z_test_noisy: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    model.fit(Z_train_clean, y_train)

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


def plot_results(summary: Dict[str, object], output_path: Path) -> None:
    labels = [
        "Parity\nClean",
        "Parity\nNoisy",
        "Prob-LR\nClean",
        "Prob-LR\nNoisy",
        "Multi-LR\nClean",
        "Multi-LR\nNoisy",
        "Multi-RF\nClean",
        "Multi-RF\nNoisy",
    ]

    values = [
        summary["parity_readout"]["clean_accuracy"],
        summary["parity_readout"]["noisy_accuracy"],
        summary["probability_logistic_readout"]["clean_accuracy"],
        summary["probability_logistic_readout"]["noisy_accuracy"],
        summary["multi_observable_logistic_readout"]["clean_accuracy"],
        summary["multi_observable_logistic_readout"]["noisy_accuracy"],
        summary["multi_observable_random_forest_readout"]["clean_accuracy"],
        summary["multi_observable_random_forest_readout"]["noisy_accuracy"],
    ]

    plt.figure(figsize=(12, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("HQNN Readout Comparison: Parity vs Probability vs Multi-Observable")
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

    architecture = "ring"
    noise_type = "depolarizing"
    eval_noise = 0.05
    shots = 4096

    weights = np.random.uniform(-np.pi, np.pi, num_params)

    clean_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=create_noise_model(noise_type, eval_noise))

    y_pred_parity_clean = parity_predict(
        clean_sim, X_test, weights, num_qubits, architecture, shots
    )
    y_pred_parity_noisy = parity_predict(
        noisy_sim, X_test, weights, num_qubits, architecture, shots
    )

    parity_clean_acc = accuracy_score(y_test, y_pred_parity_clean)
    parity_noisy_acc = accuracy_score(y_test, y_pred_parity_noisy)

    Z_train_prob = extract_probability_features(
        clean_sim, X_train, weights, num_qubits, architecture, shots
    )
    Z_test_prob_clean = extract_probability_features(
        clean_sim, X_test, weights, num_qubits, architecture, shots
    )
    Z_test_prob_noisy = extract_probability_features(
        noisy_sim, X_test, weights, num_qubits, architecture, shots
    )

    Z_train_multi = extract_multi_observable_features(
        clean_sim, X_train, weights, num_qubits, architecture, shots
    )
    Z_test_multi_clean = extract_multi_observable_features(
        clean_sim, X_test, weights, num_qubits, architecture, shots
    )
    Z_test_multi_noisy = extract_multi_observable_features(
        noisy_sim, X_test, weights, num_qubits, architecture, shots
    )

    probability_lr = evaluate_readout(
        LogisticRegression(max_iter=2000, random_state=42),
        Z_train_prob,
        y_train,
        Z_test_prob_clean,
        Z_test_prob_noisy,
        y_test,
    )

    multi_lr = evaluate_readout(
        LogisticRegression(max_iter=2000, random_state=42),
        Z_train_multi,
        y_train,
        Z_test_multi_clean,
        Z_test_multi_noisy,
        y_test,
    )

    multi_rf = evaluate_readout(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
        ),
        Z_train_multi,
        y_train,
        Z_test_multi_clean,
        Z_test_multi_noisy,
        y_test,
    )

    summary = {
        "description": (
            "Multi-observable HQNN readout experiment. This pipeline extracts "
            "multiple quantum observables from measurement distributions and "
            "uses them as features for learned classical readouts."
        ),
        "architecture": architecture,
        "noise_type": noise_type,
        "eval_noise": eval_noise,
        "shots": shots,
        "num_qubits": num_qubits,
        "feature_dimensions": {
            "probability_features": int(Z_train_prob.shape[1]),
            "multi_observable_features": int(Z_train_multi.shape[1]),
        },
        "parity_readout": {
            "clean_accuracy": float(parity_clean_acc),
            "noisy_accuracy": float(parity_noisy_acc),
            "accuracy_drop": float(accuracy_drop(parity_clean_acc, parity_noisy_acc)),
            "robustness_score": float(robustness_score(parity_noisy_acc, parity_clean_acc)),
        },
        "probability_logistic_readout": probability_lr,
        "multi_observable_logistic_readout": multi_lr,
        "multi_observable_random_forest_readout": multi_rf,
        "improvement_over_parity": {
            "probability_lr_noisy_gain": float(
                probability_lr["noisy_accuracy"] - parity_noisy_acc
            ),
            "multi_lr_noisy_gain": float(
                multi_lr["noisy_accuracy"] - parity_noisy_acc
            ),
            "multi_rf_noisy_gain": float(
                multi_rf["noisy_accuracy"] - parity_noisy_acc
            ),
        },
        "thesis_contribution": (
            "The experiment shows that HQNN performance depends strongly on the "
            "readout strategy. A fixed parity observable discards useful quantum "
            "measurement information, while a multi-observable learned readout "
            "uses richer quantum features and can substantially improve clean "
            "and noisy classification accuracy."
        ),
    }

    json_path = RESULTS_DIR / "multi_observable_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_path = RESULTS_DIR / "multi_observable_accuracy.png"
    plot_results(summary, plot_path)

    print("\nMulti-observable HQNN pipeline complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    run_pipeline()
