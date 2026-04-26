#!/usr/bin/env python
"""
Demo 11 — Cybersecurity Anomaly Detection (QSVM + HQNN)

This demo compares:
1. Quantum Support Vector Machine (QSVM)
2. Hybrid Quantum Neural Network (HQNN)
3. Classical Logistic Regression baseline

Framework integration:
- Uses framework.reporting.save_json
- Uses framework.reporting.plot_accuracy_comparison
- Adds framework-level comparison metrics to the JSON output

Outputs:
- results/demo11/results_demo11_cyber.json
- results/demo11/cyber_roc_demo11.png
- results/demo11/cyber_accuracy_demo11.png
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from framework.reporting import save_json, plot_accuracy_comparison


# ============================================================
# Synthetic Cyber Dataset
# ============================================================

def generate_cyber_data():
    """
    Synthetic network-traffic anomaly dataset.

    Features are interpreted as:
    - packet interval variance
    - byte entropy
    - port randomness
    - flag sequence irregularity
    """
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.8,
        random_state=11,
    )
    return X, y


# ============================================================
# Quantum Kernel Feature Map
# ============================================================

def feature_map():
    """
    Parameterized 2-qubit feature map for 4 classical features.

    Required by the Qiskit Machine Learning kernel API:
    number of parameters must match the feature dimension.
    """
    x = ParameterVector("x", 4)
    qc = QuantumCircuit(2)

    qc.ry(x[0], 0)
    qc.ry(x[1], 1)
    qc.ry(x[2], 0)
    qc.ry(x[3], 1)
    qc.cx(0, 1)

    return qc


# ============================================================
# HQNN Components
# ============================================================

def build_feature_map_hqnn(num_qubits, x):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(float(x[i]), i)
    return qc


def build_var_layer_hqnn(num_qubits, w):
    qc = QuantumCircuit(num_qubits)

    for i in range(num_qubits):
        qc.rx(float(w[i]), i)
        qc.rz(float(w[num_qubits + i]), i)

    for i in range(num_qubits):
        qc.cz(i, (i + 1) % num_qubits)

    return qc


def build_hqnn_circuit(num_qubits, x, w):
    fm = build_feature_map_hqnn(num_qubits, x)
    var = build_var_layer_hqnn(num_qubits, w)
    qc = fm.compose(var)
    qc.measure_all()
    return qc


def parity_expval(counts):
    shots = sum(counts.values())
    exp = 0.0

    for bits, c in counts.items():
        parity = bits.count("1") % 2
        sign = 1 if parity == 0 else -1
        exp += sign * c / shots

    return exp


def predict_prob_hqnn(sim, num_qubits, w, x):
    x_pad = np.zeros(num_qubits)
    x_pad[:len(x)] = x

    qc = build_hqnn_circuit(num_qubits, x_pad, w)
    result = sim.run(qc, shots=1024).result()
    counts = result.get_counts()

    exp = parity_expval(counts)
    return (1 - exp) / 2


# ============================================================
# SPSA for HQNN training
# ============================================================

def loss_fn(sim, num_qubits, w, X, y):
    eps = 1e-10
    preds = np.array([predict_prob_hqnn(sim, num_qubits, w, x) for x in X])
    return float(-np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps)))


def spsa_step(sim, num_qubits, w, X, y, alpha=0.15, c=0.15):
    dim = len(w)
    delta = 2 * np.random.randint(0, 2, dim) - 1

    wplus = w + c * delta
    wminus = w - c * delta

    loss_p = loss_fn(sim, num_qubits, wplus, X, y)
    loss_m = loss_fn(sim, num_qubits, wminus, X, y)

    ghat = (loss_p - loss_m) / (2 * c * delta)
    return w - alpha * ghat


# ============================================================
# Main Demo
# ============================================================

def run_demo(output_dir, epochs=10):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------
    X, y = generate_cyber_data()

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
    # 2. Quantum Kernel SVM
    # --------------------------------------------------------
    fm = feature_map()
    kernel = FidelityQuantumKernel(feature_map=fm)

    print("[QSVM] Computing quantum kernel...")
    K_train = kernel.evaluate(X_train, X_train)
    K_test = kernel.evaluate(X_test, X_train)

    qsvm = SVC(kernel="precomputed")
    qsvm.fit(K_train, y_train)
    y_pred_qsvm = qsvm.predict(K_test)

    qsvm_acc = accuracy_score(y_test, y_pred_qsvm)
    scores = qsvm.decision_function(K_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    qsvm_auc = auc(fpr, tpr)

    # --------------------------------------------------------
    # 3. HQNN Training
    # --------------------------------------------------------
    num_qubits = 4
    w = np.random.uniform(-np.pi, np.pi, 2 * num_qubits)
    sim = AerSimulator()

    hqnn_loss_history = []

    for ep in range(epochs):
        w = spsa_step(sim, num_qubits, w, X_train, y_train)
        current_loss = loss_fn(sim, num_qubits, w, X_train, y_train)
        hqnn_loss_history.append(float(current_loss))
        print(f"[HQNN][Epoch {ep}] Loss={current_loss:.4f}")

    preds_hqnn = np.array([predict_prob_hqnn(sim, num_qubits, w, x) for x in X_test])
    y_pred_hqnn = (preds_hqnn >= 0.5).astype(int)
    hqnn_acc = accuracy_score(y_test, y_pred_hqnn)

    # --------------------------------------------------------
    # 4. Classical baseline
    # --------------------------------------------------------
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    y_pred_cl = clf.predict(X_test)
    cl_acc = accuracy_score(y_test, y_pred_cl)

    # --------------------------------------------------------
    # 5. Framework-level comparison metrics
    # --------------------------------------------------------
    hybrid_vs_classical_gap = float(cl_acc - hqnn_acc)
    qsvm_vs_hqnn_gap = float(qsvm_acc - hqnn_acc)
    classical_vs_qsvm_gap = float(cl_acc - qsvm_acc)

    summary = {
        "demo": "Demo 11 — Cybersecurity Anomaly Detection",
        "dataset": "Synthetic cybersecurity traffic dataset",
        "framework_role": (
            "Demonstrates the framework's domain-inspired benchmarking capability "
            "by comparing QSVM, HQNN, and a classical baseline on the same task."
        ),
        "qsvm_accuracy": float(qsvm_acc),
        "qsvm_auc": float(qsvm_auc),
        "hqnn_accuracy": float(hqnn_acc),
        "classical_accuracy": float(cl_acc),
        "hybrid_vs_classical_accuracy_gap": hybrid_vs_classical_gap,
        "qsvm_vs_hqnn_accuracy_gap": qsvm_vs_hqnn_gap,
        "classical_vs_qsvm_accuracy_gap": classical_vs_qsvm_gap,
        "epochs": epochs,
        "hqnn_loss_history": hqnn_loss_history,
        "framework_components_used": [
            "framework.reporting.save_json",
            "framework.reporting.plot_accuracy_comparison",
        ],
        "interpretation": (
            "The cybersecurity benchmark compares a quantum kernel method, a hybrid "
            "quantum neural model, and a classical baseline under the same dataset split. "
            "This supports the framework-level contribution of standardized hybrid-versus-classical "
            "comparison in a domain-inspired task."
        ),
    }

    # --------------------------------------------------------
    # 6. Save JSON using framework reporting
    # --------------------------------------------------------
    json_path = os.path.join(output_dir, "results_demo11_cyber.json")
    save_json(summary, json_path)

    # --------------------------------------------------------
    # 7. Save ROC plot for QSVM
    # --------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"QSVM (AUC={qsvm_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Demo 11 — QSVM ROC Curve (Cyber Anomaly Detection)")
    plt.grid(True)
    plt.legend()

    roc_path = os.path.join(output_dir, "cyber_roc_demo11.png")
    plt.savefig(roc_path)
    plt.close()

    # --------------------------------------------------------
    # 8. Accuracy comparison using framework reporting
    # --------------------------------------------------------
    labels = ["QSVM", "HQNN", "Classical"]
    accs = [float(qsvm_acc), float(hqnn_acc), float(cl_acc)]

    acc_path = os.path.join(output_dir, "cyber_accuracy_demo11.png")
    plot_accuracy_comparison(
        labels,
        accs,
        acc_path,
        title="Demo 11 — Cybersecurity Anomaly Detection Accuracy",
        ylabel="Accuracy",
    )

    print("\n===== DEMO 11 SUMMARY =====")
    print(summary)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved ROC Plot: {roc_path}")
    print(f"Saved Accuracy Plot: {acc_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/demo11")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run_demo(args.output_dir, epochs=args.epochs)
