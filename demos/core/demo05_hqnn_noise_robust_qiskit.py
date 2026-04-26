#!/usr/bin/env python
"""
Demo 05 — Noise-Robust HQNN (Qiskit)

This demo compares:
1. Noiseless HQNN
2. Noisy HQNN using the framework noise toolbox
3. HQNN + ZNE if mthree is installed
4. Classical baseline using MLP

Framework integration:
- Uses framework.noise_channels.create_depolarizing_noise
- Uses framework.robustness_metrics.accuracy_drop
- Uses framework.robustness_metrics.robustness_score
- Uses framework.reporting.save_json
- Uses framework.reporting.plot_accuracy_comparison

Outputs:
- results/demo05/results_demo05.json
- results/demo05/accuracy_demo05.png
"""

import argparse
import os

import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from framework.noise_channels import create_depolarizing_noise
from framework.reporting import save_json, plot_accuracy_comparison
from framework.robustness_metrics import accuracy_drop, robustness_score


# Optional error mitigation
try:
    from mthree.zne import zne
    ZNE_AVAILABLE = True
except Exception:
    ZNE_AVAILABLE = False


# ============================================================
# HQNN Circuit Components
# ============================================================

def build_feature_map(num_qubits, x):
    """Encode classical features into qubits using RY rotations."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(float(x[i]), i)
    return qc


def build_variational_layer(num_qubits, weights):
    """RX/RZ variational layer with CZ ring entanglement."""
    qc = QuantumCircuit(num_qubits)

    for i in range(num_qubits):
        qc.rx(float(weights[i]), i)
        qc.rz(float(weights[num_qubits + i]), i)

    for i in range(num_qubits):
        qc.cz(i, (i + 1) % num_qubits)

    return qc


def build_hqnn_circuit(num_qubits, x, weights):
    """Build full measured HQNN circuit."""
    fm = build_feature_map(num_qubits, x)
    var = build_variational_layer(num_qubits, weights)
    qc = fm.compose(var)
    qc.measure_all()
    return qc


def build_unmeasured_hqnn_circuit(num_qubits, x, weights):
    """Build unmeasured HQNN circuit for optional ZNE."""
    fm = build_feature_map(num_qubits, x)
    var = build_variational_layer(num_qubits, weights)
    return fm.compose(var)


# ============================================================
# Parity + Prediction Logic
# ============================================================

def parity_expval(counts):
    """Convert measured bitstring counts into a parity expectation value."""
    shots = sum(counts.values())
    exp = 0.0

    for bitstring, count in counts.items():
        parity = bitstring.count("1") % 2
        sign = 1 if parity == 0 else -1
        exp += sign * count / shots

    return exp


def predict_probs(sim, num_qubits, weights, X, shots=1024):
    """
    Predict P(y=1) using parity expectation.

    P(y=1) = (1 - expectation) / 2
    """
    probs = []

    for x in X:
        x_pad = np.zeros(num_qubits)
        x_pad[:len(x)] = x

        qc = build_hqnn_circuit(num_qubits, x_pad, weights)
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts()

        exp = parity_expval(counts)
        p1 = (1 - exp) / 2
        probs.append(p1)

    return np.array(probs)


def zne_predict(qc):
    """Perform Zero Noise Extrapolation on an unmeasured circuit if available."""
    if not ZNE_AVAILABLE:
        return None
    return zne.simulation(qc, observable="parity", shots=1024)


# ============================================================
# Main Demo Execution
# ============================================================

def run_demo(output_dir, noise_p=0.05):
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        class_sep=1.2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --------------------------------------------------------
    # 2. HQNN setup
    # --------------------------------------------------------
    num_qubits = 4
    num_params = 2 * num_qubits
    weights = np.random.uniform(-np.pi, np.pi, num_params)

    sim_noiseless = AerSimulator()

    # Framework integration:
    # noise model comes from framework/noise_channels.py
    sim_noisy = AerSimulator(
        noise_model=create_depolarizing_noise(noise_p)
    )

    # --------------------------------------------------------
    # 3. Noiseless HQNN
    # --------------------------------------------------------
    test_nf = predict_probs(sim_noiseless, num_qubits, weights, X_test)
    acc_nf = accuracy_score(y_test, test_nf >= 0.5)

    # --------------------------------------------------------
    # 4. Noisy HQNN
    # --------------------------------------------------------
    test_n = predict_probs(sim_noisy, num_qubits, weights, X_test)
    acc_n = accuracy_score(y_test, test_n >= 0.5)

    # --------------------------------------------------------
    # 5. HQNN + ZNE
    # --------------------------------------------------------
    if ZNE_AVAILABLE:
        zne_list = []

        for x in X_test:
            x_pad = np.zeros(num_qubits)
            x_pad[:len(x)] = x

            qc = build_unmeasured_hqnn_circuit(num_qubits, x_pad, weights)
            exp = zne_predict(qc)

            if exp is None:
                continue

            p1 = (1 - exp) / 2
            zne_list.append(p1)

        acc_zne = (
            accuracy_score(y_test, np.array(zne_list) >= 0.5)
            if len(zne_list) == len(y_test)
            else None
        )
    else:
        acc_zne = None

    # --------------------------------------------------------
    # 6. Classical baseline
    # --------------------------------------------------------
    clf = MLPClassifier(
        hidden_layer_sizes=(8,),
        max_iter=300,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred_cl = clf.predict(X_test)
    acc_cl = accuracy_score(y_test, y_pred_cl)

    # --------------------------------------------------------
    # 7. Framework robustness metrics
    # --------------------------------------------------------
    drop_value = accuracy_drop(float(acc_nf), float(acc_n))
    robustness_value = robustness_score(float(acc_n), float(acc_nf))

    summary = {
        "demo": "Demo 05 — Noise-Robust HQNN",
        "dataset": "Synthetic binary classification",
        "framework_role": (
            "Demonstrates the framework noise-analysis and robustness-metric layer "
            "using a depolarizing noise model."
        ),
        "accuracy_noiseless": float(acc_nf),
        "accuracy_noisy": float(acc_n),
        "accuracy_zne": float(acc_zne) if acc_zne is not None else "Not available",
        "accuracy_classical": float(acc_cl),
        "noise_probability": float(noise_p),
        "accuracy_drop_noisy_vs_noiseless": float(drop_value),
        "robustness_score_noisy_vs_noiseless": float(robustness_value),
        "framework_components_used": [
            "framework.noise_channels.create_depolarizing_noise",
            "framework.robustness_metrics.accuracy_drop",
            "framework.robustness_metrics.robustness_score",
            "framework.reporting.save_json",
            "framework.reporting.plot_accuracy_comparison",
        ],
        "interpretation": (
            "This demo quantifies the difference between noiseless and noisy HQNN "
            "performance and compares the HQNN result with a classical MLP baseline."
        ),
    }

    # --------------------------------------------------------
    # 8. Save JSON using framework reporting
    # --------------------------------------------------------
    json_path = os.path.join(output_dir, "results_demo05.json")
    save_json(summary, json_path)

    # --------------------------------------------------------
    # 9. Plot using framework reporting
    # --------------------------------------------------------
    labels = ["Noiseless HQNN", "Noisy HQNN", "Classical Baseline"]
    accs = [float(acc_nf), float(acc_n), float(acc_cl)]

    if acc_zne is not None:
        labels.append("HQNN + ZNE")
        accs.append(float(acc_zne))

    png_path = os.path.join(output_dir, "accuracy_demo05.png")
    plot_accuracy_comparison(
        labels,
        accs,
        png_path,
        title="Demo 05 — HQNN Noise Robustness",
        ylabel="Accuracy",
    )

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
