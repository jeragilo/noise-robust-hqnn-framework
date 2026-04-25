"""
Noise channel utilities for the Noise-Robust HQNN framework.

This module provides reusable quantum noise models and noise sweep helpers.
It is designed to support robustness experiments across multiple demos.

This is one of the framework components that extends beyond simply using
Qiskit or PennyLane examples: it standardizes how noise channels are created,
named, swept, and reported across experiments.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple, Any

import numpy as np

try:
    from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, amplitude_damping_error
    QISKIT_AER_AVAILABLE = True
except Exception:
    QISKIT_AER_AVAILABLE = False


DEFAULT_NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]


def _require_qiskit_aer() -> None:
    """Raise a clear error if qiskit-aer is unavailable."""
    if not QISKIT_AER_AVAILABLE:
        raise ImportError(
            "qiskit-aer is required for Qiskit noise models. "
            "Install it with: pip install qiskit-aer"
        )


def create_depolarizing_noise(p: float) -> "NoiseModel":
    """
    Create a Qiskit depolarizing noise model.

    Depolarizing noise randomly corrupts the quantum state and is often
    one of the most disruptive NISQ noise channels.
    """
    _require_qiskit_aer()

    noise_model = NoiseModel()

    if p <= 0:
        return noise_model

    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p, 1),
        ["rx", "ry", "rz", "x", "y", "z", "h", "sx", "u1", "u2", "u3"],
    )

    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p, 2),
        ["cx", "cz"],
    )

    return noise_model


def create_bit_flip_noise(p: float) -> "NoiseModel":
    """
    Create a Qiskit bit-flip noise model.

    Bit-flip noise applies an X error with probability p.
    """
    _require_qiskit_aer()

    noise_model = NoiseModel()

    if p <= 0:
        return noise_model

    bit_flip = pauli_error([("X", p), ("I", 1 - p)])

    noise_model.add_all_qubit_quantum_error(
        bit_flip,
        ["rx", "ry", "rz", "x", "y", "z", "h", "sx", "u1", "u2", "u3"],
    )

    return noise_model


def create_phase_flip_noise(p: float) -> "NoiseModel":
    """
    Create a Qiskit phase-flip noise model.

    Phase-flip noise applies a Z error with probability p.
    """
    _require_qiskit_aer()

    noise_model = NoiseModel()

    if p <= 0:
        return noise_model

    phase_flip = pauli_error([("Z", p), ("I", 1 - p)])

    noise_model.add_all_qubit_quantum_error(
        phase_flip,
        ["rx", "ry", "rz", "x", "y", "z", "h", "sx", "u1", "u2", "u3"],
    )

    return noise_model


def create_amplitude_damping_noise(p: float) -> "NoiseModel":
    """
    Create a Qiskit amplitude damping noise model.

    Amplitude damping models energy loss from |1> toward |0>.
    """
    _require_qiskit_aer()

    noise_model = NoiseModel()

    if p <= 0:
        return noise_model

    amp_damp = amplitude_damping_error(p)

    noise_model.add_all_qubit_quantum_error(
        amp_damp,
        ["rx", "ry", "rz", "x", "y", "z", "h", "sx", "u1", "u2", "u3"],
    )

    return noise_model


def get_noise_factory(noise_type: str) -> Callable[[float], "NoiseModel"]:
    """
    Return the noise model factory for a given noise type.

    Supported:
        depolarizing
        bit_flip
        phase_flip
        amplitude_damping
    """
    factories = {
        "depolarizing": create_depolarizing_noise,
        "bit_flip": create_bit_flip_noise,
        "phase_flip": create_phase_flip_noise,
        "amplitude_damping": create_amplitude_damping_noise,
    }

    if noise_type not in factories:
        raise ValueError(
            f"Unsupported noise_type={noise_type}. "
            f"Choose from {list(factories.keys())}."
        )

    return factories[noise_type]


def available_noise_types() -> List[str]:
    """Return supported noise channel names."""
    return ["depolarizing", "bit_flip", "phase_flip", "amplitude_damping"]


def run_noise_sweep(
    evaluate_fn: Callable[[Any], float],
    noise_type: str,
    noise_levels: Iterable[float] = DEFAULT_NOISE_LEVELS,
) -> Dict[str, Any]:
    """
    Run an evaluation function across noise levels.

    Args:
        evaluate_fn:
            A function that accepts a Qiskit NoiseModel and returns a metric,
            usually accuracy or expectation value.

        noise_type:
            One of: depolarizing, bit_flip, phase_flip, amplitude_damping.

        noise_levels:
            Noise probabilities to evaluate.

    Returns:
        Dictionary containing the noise type, levels, and metric values.

    Why this matters:
        This turns noise testing into a reusable framework operation instead of
        custom one-off code inside each demo.
    """
    factory = get_noise_factory(noise_type)

    levels = list(noise_levels)
    values = []

    for p in levels:
        noise_model = factory(float(p))
        metric_value = evaluate_fn(noise_model)
        values.append(float(metric_value))

    return {
        "noise_type": noise_type,
        "noise_levels": levels,
        "values": values,
    }
