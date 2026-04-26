#!/usr/bin/env python
"""
Demo 05 — Noise-Robust HQNN (Qiskit)

This demo compares:
1. Noiseless HQNN
2. Noisy HQNN (depolarizing error model)
3. HQNN + ZNE (if mthree installed)
4. Classical baseline (MLP)

Outputs:
- results_demo05.json
- accuracy_demo05.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Optional error mitigation
try:
    from mthree.zne import zne
    ZNE_AVAILABLE = True
except:
    ZNE_AVAILABLE = False


# ============================================================
# HQNN Circuit Components
# ============================================================

def build_feature_map(num_qubits, x):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(float(x[i]), i)
    return qc


def build_variational_layer(num_qubits, weights):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(weights[i], i)
        qc.rz(weights[num_qubits + i], i)
    for i in range(num_qubits):
        qc.cz(i, (i + 1) % num_qubits)
    return qc


def build_hqnn_circuit(num_qubits, x, weights):
    fm = build_feature_map(num_qubits, x)
    var = build_variational_layer(num_qubits, weights)
    qc = fm.compose(var)
    qc.measure_all()
    return qc


# ============================================================
# Parity + Prediction Logic
# ============================================================

def parity_expval(counts):
    shots = sum(counts.values())
    exp = 0
    for bitstring, count in counts.items():
        parity = bitstring.count("1") % 2
        sign = 1 if parity == 0 else -1
        exp += sign * count / shots
    return exp


def predict_probs(sim, num_qubits, weights, X):
    probs = []
    for x in X:
        x_pad = np.zeros(num_qubits)
        x_pad[:len(x)] = x

        qc = build_hqnn_circuit(num_qubits, x_pad, weights)
        result = sim.run(qc, shots=1024).result()
        counts = result.get_counts()

        exp = parity_expval(counts)
        p1 = (1 - exp) / 2
        probs.append(p1)

    return np.array(probs)


# ============================================================
# Noise Model
# ============================================================

def make_noise_model(p=0.01):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p, 1),
        ["rx", "ry", "rz", "u1", "u2", "u3"],
    )
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p, 2),
        ["cx", "cz"],
    )
    return nm


def zne_predict(qc):
    """Perform Zero Noise Extrapolation on an unmeasured circuit."""
    if not ZNE_AVAILABLE:
        return None
    return zne.simulation(qc, observable="parity", shots=1024)


# ============================================================
# Main Demo Execution
# ============================================================

def run_demo(output_dir, noise_p=0.05):
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. Dataset (FIXED)
    # --------------------------------------------------------
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        class_sep=1.2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    num_qubits = 4
    num_params = 2 * num_qubits
    weights = np.random.uniform(-np.pi, np.pi, num_params)

    sim_noiseless = AerSimulator()
    sim_noisy = AerSimulator(noise_model=make_noise_model(noise_p))

    # --------------------------------------------------------
    # 2. Noiseless HQNN
    # --------------------------------------------------------
    test_nf = predict_probs(sim_noiseless, num_qubits, weights, X_test)
    acc_nf = accuracy_score(y_test, test_nf >= 0.5)

    # --------------------------------------------------------
    # 3. Noisy HQNN
    # --------------------------------------------------------
    test_n = predict_probs(sim_noisy, num_qubits, weights, X_test)
    acc_n = accuracy_score(y_test, test_n >= 0.5)

    # --------------------------------------------------------
    # 4. HQNN + ZNE (fixed)
    # --------------------------------------------------------
    if ZNE_AVAILABLE:
        zne_list = []
        for x in X_test:
            x_pad = np.zeros(num_qubits)
            x_pad[:len(x)] = x

            # Build *unmeasured* circuit
            fm = build_feature_map(num_qubits, x_pad)
            var = build_variational_layer(num_qubits, weights)
            qc = fm.compose(var)  # unmeasured

            exp = zne_predict(qc)
            p1 = (1 - exp) / 2
            zne_list.append(p1)

        acc_zne = accuracy_score(y_test, np.array(zne_list) >= 0.5)
    else:
        acc_zne = None

    # --------------------------------------------------------
    # 5. Classical baseline
    # --------------------------------------------------------
    clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=300, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_cl = clf.predict(X_test)
    acc_cl = accuracy_score(y_test, y_pred_cl)

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------
    summary = {
        "accuracy_noiseless": float(acc_nf),
        "accuracy_noisy": float(acc_n),
        "accuracy_zne": float(acc_zne) if acc_zne is not None else "Not available",
        "accuracy_classical": float(acc_cl),
        "noise_probability": float(noise_p)
    }

    json_path = os.path.join(output_dir, "results_demo05.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    labels = ["Noiseless HQNN", "Noisy HQNN", "Classical Baseline"]
    accs = [acc_nf, acc_n, acc_cl]

    if acc_zne is not None:
        labels.append("HQNN + ZNE")
        accs.append(acc_zne)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, accs, color=["green", "red", "blue", "purple"][:len(labels)])
    plt.ylabel("Accuracy")
    plt.title("Demo 05 — HQNN Noise Robustness")
    plt.xticks(rotation=20)
    plt.tight_layout()

    png_path = os.path.join(output_dir, "accuracy_demo05.png")
    plt.savefig(png_path)
    plt.close()

    print("\n===== DEMO 05 SUMMARY =====")
    print(summary)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved plot: {png_path}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/demo05")
    parser.add_argument("--noise_p", type=float, default=0.05)
    args = parser.parse_args()
    run_demo(args.output_dir, args.noise_p)
