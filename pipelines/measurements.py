"""
Measurement execution and caching for publication experiments.

The central design principle is:

    execute quantum circuit once
        ↓
    save raw counts
        ↓
    derive multiple readout representations from identical measurements

Z-basis counts support R0, R0L, R1, R2, R4, R5, R6, and R7.

R3 (Zeng AMM) additionally requires X- and Y-basis measurements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from pipelines.publication.circuits import build_measurement_circuit


Counts = Dict[str, int]


def normalize_counts(counts: Dict[str, int]) -> Counts:
    """
    Normalize Qiskit count keys.

    Removes spaces that may appear when multiple classical registers
    are present and converts all values to Python integers.
    """

    normalized: Counts = {}

    for bitstring, count in counts.items():
        key = str(bitstring).replace(" ", "")
        normalized[key] = normalized.get(key, 0) + int(count)

    return normalized


def run_one_circuit(
    backend,
    circuit,
    shots: int,
    seed_simulator: Optional[int] = None,
) -> Counts:
    """
    Execute one circuit and return normalized measurement counts.

    seed_simulator is used by Aer when supported.
    """

    compiled = transpile(circuit, backend)

    run_kwargs = {"shots": shots}

    if seed_simulator is not None and isinstance(backend, AerSimulator):
        run_kwargs["seed_simulator"] = seed_simulator

    result = backend.run(
        compiled,
        **run_kwargs,
    ).result()

    return normalize_counts(result.get_counts())


def measure_sample(
    backend,
    x: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    shots: int,
    include_xyz: bool = False,
    seed_simulator: Optional[int] = None,
) -> Dict[str, Counts]:
    """
    Measure one sample.

    Always executes the Z-basis circuit.

    When include_xyz=True, also executes X- and Y-basis circuits for
    the Zeng all-qubit multi-observable baseline.
    """

    bases = ["Z"]

    if include_xyz:
        bases = ["X", "Y", "Z"]

    measurements: Dict[str, Counts] = {}

    for basis_index, basis in enumerate(bases):
        circuit = build_measurement_circuit(
            x=x,
            weights=weights,
            num_qubits=num_qubits,
            basis=basis,
        )

        basis_seed = None

        if seed_simulator is not None:
            basis_seed = seed_simulator + basis_index

        measurements[basis] = run_one_circuit(
            backend=backend,
            circuit=circuit,
            shots=shots,
            seed_simulator=basis_seed,
        )

    return measurements


def measure_dataset(
    backend,
    X: np.ndarray,
    weights: np.ndarray,
    num_qubits: int,
    shots: int,
    include_xyz: bool = False,
    seed_simulator: Optional[int] = None,
) -> List[Dict[str, Counts]]:
    """
    Execute the canonical circuit for every sample in a dataset.

    Returns raw measurement records only. Feature representations
    are intentionally derived later.
    """

    records: List[Dict[str, Counts]] = []

    for sample_index, x in enumerate(X):
        sample_seed = None

        if seed_simulator is not None:
            # Give every sample a deterministic but distinct simulator seed.
            sample_seed = seed_simulator + 1000 * sample_index

        records.append(
            measure_sample(
                backend=backend,
                x=x,
                weights=weights,
                num_qubits=num_qubits,
                shots=shots,
                include_xyz=include_xyz,
                seed_simulator=sample_seed,
            )
        )

    return records


def counts_total(counts: Counts) -> int:
    return int(sum(counts.values()))


def validate_measurement_records(
    records: Iterable[Dict[str, Counts]],
    shots: int,
    require_xyz: bool = False,
) -> None:
    """
    Validate that cached measurement records are structurally complete.
    """

    required_bases = {"Z"}

    if require_xyz:
        required_bases = {"X", "Y", "Z"}

    for sample_index, record in enumerate(records):
        missing = required_bases.difference(record.keys())

        if missing:
            raise ValueError(
                f"Sample {sample_index} is missing measurement bases: "
                f"{sorted(missing)}"
            )

        for basis in required_bases:
            observed_shots = counts_total(record[basis])

            if observed_shots != shots:
                raise ValueError(
                    f"Sample {sample_index}, basis {basis}: expected "
                    f"{shots} shots, observed {observed_shots}."
                )


def save_measurement_records(
    records: List[Dict[str, Counts]],
    output_path: Path,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save raw counts so readout experiments can be reproduced without
    rerunning quantum circuits.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "metadata": metadata or {},
        "records": records,
    }

    with open(output_path, "w") as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def load_measurement_records(
    input_path: Path,
) -> Dict[str, object]:
    """
    Load previously cached quantum measurement records.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Measurement cache not found: {input_path}"
        )

    with open(input_path, "r") as file:
        payload = json.load(file)

    if "records" not in payload:
        raise ValueError(
            f"Invalid measurement cache: {input_path}"
        )

    return payload


def make_aer_backend(
    noise_model=None,
) -> AerSimulator:
    """
    Construct an Aer backend for ideal or noisy simulation.
    """

    if noise_model is None:
        return AerSimulator()

    return AerSimulator(
        noise_model=noise_model,
    )
