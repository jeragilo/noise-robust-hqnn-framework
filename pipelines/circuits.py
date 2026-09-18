"""
Canonical quantum circuits for the HQNN readout publication experiments.

The central readout ablation holds the quantum computation fixed and changes
only the measurement/readout representation.

Canonical four-qubit circuit:
1. RY(x_i) RZ(x_i^2) feature encoding
2. Trainable RX-RY-RZ rotations
3. Ring CZ entanglement

Measurement basis is selected separately so the same underlying quantum
circuit can support:
- Z-basis counts for R0, R0L, R1, R2, R4, R5, R6, R7
- X/Y/Z measurements for R3 (Zeng AMM)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit import QuantumCircuit


MeasurementBasis = Literal["X", "Y", "Z"]


def pad_features(
    x: np.ndarray,
    num_qubits: int,
) -> np.ndarray:
    """
    Pad an input vector with zeros to match the number of qubits.
    """

    x = np.asarray(x, dtype=float)

    if len(x) > num_qubits:
        raise ValueError(
            f"Input has {len(x)} features but circuit has only "
            f"{num_qubits} qubits."
        )

    x_pad = np.zeros(num_qubits, dtype=float)
    x_pad[: len(x)] = x

    return x_pad


def build_feature_map(
    x: np.ndarray,
    num_qubits: int,
) -> QuantumCircuit:
    """
    Canonical feature map.

    For each qubit i:
        RY(x_i)
        RZ(x_i^2)
    """

    x_pad = pad_features(x, num_qubits)

    qc = QuantumCircuit(num_qubits)

    for i in range(num_qubits):
        qc.ry(float(x_pad[i]), i)
        qc.rz(float(x_pad[i]) ** 2, i)

    return qc


def build_variational_layer(
    weights: np.ndarray,
    num_qubits: int,
) -> QuantumCircuit:
    """
    Canonical variational layer.

    Each qubit receives RX-RY-RZ trainable rotations followed by
    ring CZ entanglement.
    """

    weights = np.asarray(weights, dtype=float)

    expected_parameters = 3 * num_qubits

    if len(weights) != expected_parameters:
        raise ValueError(
            f"Expected {expected_parameters} variational parameters "
            f"for {num_qubits} qubits, got {len(weights)}."
        )

    qc = QuantumCircuit(num_qubits)

    for i in range(num_qubits):
        qc.rx(float(weights[i]), i)
        qc.ry(float(weights[num_qubits + i]), i)
        qc.rz(float(weights[2 * num_qubits + i]), i)

    if num_qubits > 1:
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

        # Close the ring only when this does not duplicate the same
        # two-qubit edge used above.
        if num_qubits > 2:
            qc.cz(num_qubits - 1, 0)

    return qc


def build_unmeasured_circuit(
    x: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
) -> QuantumCircuit:
    """
    Build the canonical HQNN quantum circuit without measurement.
    """

    qc = build_feature_map(
        x=x,
        num_qubits=num_qubits,
    )

    qc.compose(
        build_variational_layer(
            weights=weights,
            num_qubits=num_qubits,
        ),
        inplace=True,
    )

    return qc


def add_measurement_basis(
    circuit: QuantumCircuit,
    basis: MeasurementBasis,
) -> QuantumCircuit:
    """
    Add basis rotations and measure all qubits.

    Z basis:
        direct computational-basis measurement

    X basis:
        H before computational-basis measurement

    Y basis:
        S-dagger then H before computational-basis measurement
    """

    basis = basis.upper()

    if basis not in {"X", "Y", "Z"}:
        raise ValueError(
            f"Unsupported measurement basis: {basis}"
        )

    qc = circuit.copy()

    if basis == "X":
        for qubit in range(qc.num_qubits):
            qc.h(qubit)

    elif basis == "Y":
        for qubit in range(qc.num_qubits):
            qc.sdg(qubit)
            qc.h(qubit)

    qc.measure_all()

    return qc


def build_measurement_circuit(
    x: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    basis: MeasurementBasis = "Z",
) -> QuantumCircuit:
    """
    Build one canonical HQNN circuit and append the requested
    measurement basis.
    """

    qc = build_unmeasured_circuit(
        x=x,
        weights=weights,
        num_qubits=num_qubits,
    )

    return add_measurement_basis(
        circuit=qc,
        basis=basis,
    )


def initialize_weights(
    num_qubits: int,
    seed: int,
) -> np.ndarray:
    """
    Reproducibly initialize the canonical circuit parameters.

    All readout representations within a matched experimental trial
    must use exactly these same weights.
    """

    rng = np.random.default_rng(seed)

    return rng.uniform(
        low=-np.pi,
        high=np.pi,
        size=3 * num_qubits,
    )


def circuit_metadata(
    num_qubits: int,
) -> dict:
    """
    Metadata stored with publication results for reproducibility.
    """

    return {
        "num_qubits": num_qubits,
        "feature_map": "RY(x_i) + RZ(x_i^2)",
        "variational_rotations": "RX-RY-RZ",
        "entanglement": "ring_CZ",
        "num_trainable_quantum_parameters": 3 * num_qubits,
    }
