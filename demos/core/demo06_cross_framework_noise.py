#!/usr/bin/env python
"""
Demo 06 — Cross-Framework Noise Benchmark
(Qiskit Aer vs Cirq vs PennyLane)

This demo:
- Builds the same 2-qubit parity circuit
- Computes ZZ expectation value in:
      1. Qiskit (noiseless)
      2. Qiskit (noisy)
      3. Cirq (noiseless)
      4. Cirq (noisy)
      5. PennyLane (noiseless)
      6. PennyLane (noisy)
- (Optional) Zero Noise Extrapolation (ZNE) in Qiskit (if mthree installed)
- Saves comparison JSON + PNG
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import cirq
import pennylane as qml

# Optional ZNE
try:
    from mthree.zne import zne
    ZNE_AVAILABLE = True
except:
    ZNE_AVAILABLE = False


# ------------------------------------------------------------
# Build same test circuit (Qiskit/Cirq/PL)
# ------------------------------------------------------------

def build_qiskit_circuit(theta):
    qc = QuantumCircuit(2)
    qc.h(0); qc.h(1)
    qc.cx(0,1)
    qc.ry(theta, 0)
    qc.cx(0,1)
    qc.measure_all()
    return qc


def build_cirq_circuit(theta):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.H(q1),
        cirq.CNOT(q0, q1),
        cirq.ry(theta)(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key="m")
    )
    return circuit, (q0, q1)


def build_pl_noiseless(theta):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.H(0); qml.H(1)
        qml.CNOT(wires=[0,1])
        qml.RY(theta, wires=0)
        qml.CNOT(wires=[0,1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return circuit


def build_pl_noisy(theta, p):
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.H(0); qml.H(1)
        qml.CNOT(wires=[0,1])

        # Depolarizing noise — FIXED SYNTAX
        qml.DepolarizingChannel(p, wires=0)
        qml.DepolarizingChannel(p, wires=1)

        qml.RY(theta, wires=0)
        qml.CNOT(wires=[0,1])

        qml.DepolarizingChannel(p, wires=0)
        qml.DepolarizingChannel(p, wires=1)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return circuit


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def parity_expval(counts):
    shots = sum(counts.values())
    exp = 0
    for b, c in counts.items():
        parity = b.count("1") % 2
        sign = 1 if parity == 0 else -1
        exp += sign * c/shots
    return exp


# ------------------------------------------------------------
# Backends
# ------------------------------------------------------------

def run_qiskit(theta, noisy=False, p=0.05):
    qc = build_qiskit_circuit(theta)

    if noisy:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(p,1), ["h","ry"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p,2), ["cx"])
        sim = AerSimulator(noise_model=nm)
    else:
        sim = AerSimulator()

    res = sim.run(qc, shots=2000).result()
    return parity_expval(res.get_counts())


def run_cirq(theta, noisy=False, p=0.05):
    circuit, (q0,q1) = build_cirq_circuit(theta)

    if noisy:
        noisy_circuit = circuit.with_noise(cirq.depolarize(p))
        sim = cirq.DensityMatrixSimulator()
        result = sim.run(noisy_circuit, repetitions=2000)
    else:
        sim = cirq.Simulator()
        result = sim.run(circuit, repetitions=2000)

    bits = result.measurements["m"]
    shots = len(bits)
    exp = 0

    for b in bits:
        parity = int(b[0]) ^ int(b[1])
        sign = 1 if parity == 0 else -1
        exp += sign/shots

    return exp


def run_pl(theta, noisy=False, p=0.05):
    if noisy:
        circ = build_pl_noisy(theta, p)
    else:
        circ = build_pl_noiseless(theta)

    return float(circ())


def run_zne(theta, p=0.05):
    if not ZNE_AVAILABLE:
        return None

    # Build circuit WITHOUT measurements
    qc = QuantumCircuit(2)
    qc.h(0); qc.h(1)
    qc.cx(0,1)
    qc.ry(theta, 0)
    qc.cx(0,1)

    exp = zne.simulation(qc, observable="parity", shots=2000)
    return exp


# ------------------------------------------------------------
# DEMO EXECUTION
# ------------------------------------------------------------

def run_demo(output_dir="results/demo06", theta=pi/4, noise_p=0.05):
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Running Demo 06: Cross-Framework Noise Benchmark ===\n")

    results = {
        "qiskit_noiseless":      run_qiskit(theta, noisy=False),
        "qiskit_noisy":          run_qiskit(theta, noisy=True, p=noise_p),
        "cirq_noiseless":        run_cirq(theta, noisy=False),
        "cirq_noisy":            run_cirq(theta, noisy=True, p=noise_p),
        "pennylane_noiseless":   run_pl(theta, noisy=False),
        "pennylane_noisy":       run_pl(theta, noisy=True, p=noise_p),
        "zne_qiskit":            run_zne(theta, p=noise_p)
    }

    print(results)

    # Save JSON
    json_path = os.path.join(output_dir, "results_demo06.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    labels = list(results.keys())
    values = [0 if v is None else float(v) for v in results.values()]

    plt.figure(figsize=(12,5))
    plt.bar(labels, values, color="teal")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("ZZ Expectation Value")
    plt.title("Demo 06 — Cross-Framework Noisy vs Noiseless Comparison")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "parity_demo06.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"\nSaved JSON to {json_path}")
    print(f"Saved plot to {plot_path}")


# CLI
if __name__ == "__main__":
    run_demo()
