"""
Noise channel utilities for the Noise-Robust HQNN framework.

This module standardizes how noise is created, swept, and reported across
training-time and evaluation-time experiments.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

try:
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        pauli_error,
        amplitude_damping_error,
    )
    QISKIT_AER_AVAILABLE = True
except Exception:
    QISKIT_AER_AVAILABLE = False


DEFAULT_NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]


def _require_qiskit_aer() -> None:
    if not QISKIT_AER_AVAILABLE:
        raise ImportError(
            "qiskit-aer is required. Install with: pip install qiskit-aer"
        )


def create_depolarizing_noise(p: float) -> "NoiseModel":
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
    return ["depolarizing", "bit_flip", "phase_flip", "amplitude_damping"]


def create_noise_model(noise_type: str, p: float) -> "NoiseModel":
    factory = get_noise_factory(noise_type)
    return factory(float(p))


def run_noise_sweep(
    evaluate_fn: Callable[[Any], float],
    noise_type: str,
    noise_levels: Iterable[float] = DEFAULT_NOISE_LEVELS,
) -> Dict[str, Any]:
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


def run_multi_noise_sweep(
    evaluate_fn: Callable[[Any], float],
    noise_types: Iterable[str] | None = None,
    noise_levels: Iterable[float] = DEFAULT_NOISE_LEVELS,
) -> Dict[str, Any]:
    if noise_types is None:
        noise_types = available_noise_types()

    results = {}

    for noise_type in noise_types:
        results[noise_type] = run_noise_sweep(
            evaluate_fn=evaluate_fn,
            noise_type=noise_type,
            noise_levels=noise_levels,
        )

    return {
        "noise_types": list(noise_types),
        "noise_levels": list(noise_levels),
        "results": results,
    }


def make_training_noise_schedule(
    mode: str,
    epochs: int,
    max_noise: float = 0.10,
) -> List[float]:
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    if mode == "standard":
        return [0.0 for _ in range(epochs)]

    if mode == "noise_aware":
        return [float(max_noise) for _ in range(epochs)]

    if mode == "curriculum":
        return [float(max_noise) * (i / max(1, epochs - 1)) for i in range(epochs)]

    raise ValueError(
        "Unsupported schedule mode. Choose: standard, noise_aware, curriculum."
    )
