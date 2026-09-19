"""
Publication experiment: HQNN readout representations.

Defines the standardized readout representations used in the
thesis-derived publication experiments.

R0  : parity
R0L : learned parity
R1  : single-qubit Z
R2  : all-qubit Z
R3  : all-qubit XYZ / Zeng AMM
R4  : full computational-basis probabilities
R5  : Z + ZZ correlations
R6  : probabilities + Z + ZZ
R7  : full engineered representation

Important:
R0, R0L, R1, R2, R4, R5, R6, and R7 can be derived from the
same computational-basis measurement counts.

R3 requires additional X- and Y-basis measurements.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List

import numpy as np


class Representation(str, Enum):
    PARITY = "R0_parity"
    LEARNED_PARITY = "R0L_learned_parity"
    SINGLE_Z = "R1_single_z"
    ALL_Z = "R2_all_z"
    ZENG_AMM = "R3_zeng_xyz"
    PROBABILITIES = "R4_probabilities"
    Z_ZZ = "R5_z_zz"
    PROB_Z_ZZ = "R6_prob_z_zz"
    FULL = "R7_full"


def all_bitstrings(num_qubits: int) -> List[str]:
    return [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]


def bitstring_probabilities(
    counts: Dict[str, int],
    num_qubits: int,
) -> np.ndarray:
    shots = sum(counts.values())

    if shots == 0:
        raise ValueError("Measurement counts contain zero shots.")

    return np.array(
        [
            counts.get(bitstring, 0) / shots
            for bitstring in all_bitstrings(num_qubits)
        ],
        dtype=float,
    )


def z_expectations(
    counts: Dict[str, int],
    num_qubits: int,
) -> np.ndarray:
    shots = sum(counts.values())

    if shots == 0:
        raise ValueError("Measurement counts contain zero shots.")

    values = []

    for q in range(num_qubits):
        expectation = 0.0

        for bitstring, count in counts.items():
            bit = int(bitstring[::-1][q])
            sign = 1.0 if bit == 0 else -1.0
            expectation += sign * count / shots

        values.append(expectation)

    return np.asarray(values, dtype=float)


def zz_correlations(
    counts: Dict[str, int],
    num_qubits: int,
) -> np.ndarray:
    shots = sum(counts.values())

    if shots == 0:
        raise ValueError("Measurement counts contain zero shots.")

    values = []

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            expectation = 0.0

            for bitstring, count in counts.items():
                bit_i = int(bitstring[::-1][i])
                bit_j = int(bitstring[::-1][j])

                z_i = 1.0 if bit_i == 0 else -1.0
                z_j = 1.0 if bit_j == 0 else -1.0

                expectation += z_i * z_j * count / shots

            values.append(expectation)

    return np.asarray(values, dtype=float)


def parity_expectation(counts: Dict[str, int]) -> float:
    shots = sum(counts.values())

    if shots == 0:
        raise ValueError("Measurement counts contain zero shots.")

    expectation = 0.0

    for bitstring, count in counts.items():
        parity = bitstring.count("1") % 2
        sign = 1.0 if parity == 0 else -1.0
        expectation += sign * count / shots

    return float(expectation)


def probability_statistics(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12

    entropy = -np.sum(probs * np.log(probs + eps))
    max_probability = np.max(probs)
    min_probability = np.min(probs)
    variance = np.var(probs)

    return np.array(
        [
            entropy,
            max_probability,
            min_probability,
            variance,
        ],
        dtype=float,
    )


def zeng_amm_features(
    x_counts: Dict[str, int],
    y_counts: Dict[str, int],
    z_counts: Dict[str, int],
    num_qubits: int,
) -> np.ndarray:
    """
    Construct the all-qubit X/Y/Z measurement representation.

    Ordering:
        X0, Y0, Z0, X1, Y1, Z1, ...
    """

    x_values = z_expectations(x_counts, num_qubits)
    y_values = z_expectations(y_counts, num_qubits)
    z_values = z_expectations(z_counts, num_qubits)

    features = []

    for q in range(num_qubits):
        features.extend(
            [
                x_values[q],
                y_values[q],
                z_values[q],
            ]
        )

    return np.asarray(features, dtype=float)


def representation_dimension(
    representation: Representation,
    num_qubits: int,
) -> int:
    num_probabilities = 2**num_qubits
    num_zz = num_qubits * (num_qubits - 1) // 2

    dimensions = {
        Representation.PARITY: 1,
        Representation.LEARNED_PARITY: 1,
        Representation.SINGLE_Z: 1,
        Representation.ALL_Z: num_qubits,
        Representation.ZENG_AMM: 3 * num_qubits,
        Representation.PROBABILITIES: num_probabilities,
        Representation.Z_ZZ: num_qubits + num_zz,
        Representation.PROB_Z_ZZ: (
            num_probabilities + num_qubits + num_zz
        ),
        Representation.FULL: (
            num_probabilities
            + num_qubits
            + num_zz
            + 1
            + 4
        ),
    }

    return dimensions[representation]


def extract_z_basis_representation(
    counts: Dict[str, int],
    num_qubits: int,
    representation: Representation,
) -> np.ndarray:
    """
    Derive a representation from one common set of Z-basis counts.

    This guarantees that all supported representations are based on
    exactly the same finite-shot quantum measurement sample.
    """

    probabilities = bitstring_probabilities(counts, num_qubits)
    z_values = z_expectations(counts, num_qubits)
    zz_values = zz_correlations(counts, num_qubits)
    parity = np.array([parity_expectation(counts)], dtype=float)
    statistics = probability_statistics(probabilities)

    if representation in {
        Representation.PARITY,
        Representation.LEARNED_PARITY,
    }:
        features = parity

    elif representation == Representation.SINGLE_Z:
        features = z_values[:1]

    elif representation == Representation.ALL_Z:
        features = z_values

    elif representation == Representation.PROBABILITIES:
        features = probabilities

    elif representation == Representation.Z_ZZ:
        features = np.concatenate(
            [
                z_values,
                zz_values,
            ]
        )

    elif representation == Representation.PROB_Z_ZZ:
        features = np.concatenate(
            [
                probabilities,
                z_values,
                zz_values,
            ]
        )

    elif representation == Representation.FULL:
        features = np.concatenate(
            [
                probabilities,
                z_values,
                zz_values,
                parity,
                statistics,
            ]
        )

    elif representation == Representation.ZENG_AMM:
        raise ValueError(
            "R3 Zeng AMM requires X-, Y-, and Z-basis counts. "
            "Use zeng_amm_features()."
        )

    else:
        raise ValueError(
            f"Unsupported representation: {representation}"
        )

    expected_dimension = representation_dimension(
        representation,
        num_qubits,
    )

    if len(features) != expected_dimension:
        raise RuntimeError(
            f"{representation.value}: expected dimension "
            f"{expected_dimension}, got {len(features)}."
        )

    return np.asarray(features, dtype=float)


def fixed_parity_prediction(counts: Dict[str, int]) -> int:
    """
    Original fixed parity classifier from the thesis experiments.
    """

    expectation = parity_expectation(counts)
    probability_class_one = (1.0 - expectation) / 2.0

    return int(probability_class_one >= 0.5)
