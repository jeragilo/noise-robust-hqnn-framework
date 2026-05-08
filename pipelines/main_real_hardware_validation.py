"""
Pipeline: Real Hardware Validation for Noise-Robust HQNN Framework

Runs a small HQNN-style feature extraction experiment on:
1. Qiskit Aer simulator
2. IBM Quantum hardware through SamplerV2, if configured

This validates whether the same multi-observable readout logic used in the
framework can operate on real NISQ hardware measurement counts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


RESULTS_DIR = Path("results") / "framework" / "real_hardware_validation"


def build_hardware_validation_circuit(
    x: np.ndarray,
    weights: np.ndarray,
    num_qubits: int = 2,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, num_qubits)

    x_pad = np.zeros(num_qubits)
    x_pad[: len(x)] = x

    for i in range(num_qubits):
        qc.ry(float(x_pad[i]), i)
        qc.rz(float(x_pad[i]) ** 2, i)

    for i in range(num_qubits):
        qc.rx(float(weights[i]), i)
        qc.ry(float(weights[num_qubits + i]), i)
        qc.rz(float(weights[2 * num_qubits + i]), i)

    qc.cz(0, 1)
    qc.measure(range(num_qubits), range(num_qubits))

    return qc


def all_bitstrings(num_qubits: int) -> List[str]:
    return [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]


def bitstring_probabilities(counts: Dict[str, int], num_qubits: int) -> np.ndarray:
    shots = sum(counts.values())
    return np.array(
        [counts.get(bitstring, 0) / shots for bitstring in all_bitstrings(num_qubits)],
        dtype=float,
    )


def parity_expectation(counts: Dict[str, int]) -> float:
    shots = sum(counts.values())
    exp = 0.0

    for bitstring, count in counts.items():
        parity = bitstring.count("1") % 2
        sign = 1 if parity == 0 else -1
        exp += sign * count / shots

    return float(exp)


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


def zz_correlation(counts: Dict[str, int]) -> float:
    shots = sum(counts.values())
    exp = 0.0

    for bitstring, count in counts.items():
        b0 = int(bitstring[::-1][0])
        b1 = int(bitstring[::-1][1])

        z0 = 1 if b0 == 0 else -1
        z1 = 1 if b1 == 0 else -1

        exp += z0 * z1 * count / shots

    return float(exp)


def counts_to_features(counts: Dict[str, int], num_qubits: int = 2) -> np.ndarray:
    probs = bitstring_probabilities(counts, num_qubits)
    z_vals = z_expectations(counts, num_qubits)
    zz = np.array([zz_correlation(counts)], dtype=float)
    parity = np.array([parity_expectation(counts)], dtype=float)

    return np.concatenate([probs, z_vals, zz, parity])


def parity_predict_from_counts(counts: Dict[str, int]) -> int:
    exp = parity_expectation(counts)
    p1 = (1 - exp) / 2
    return int(p1 >= 0.5)


def make_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=40,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        class_sep=2.0,
        flip_y=0.02,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.clip(X_train, -np.pi, np.pi)
    X_test = np.clip(X_test, -np.pi, np.pi)

    return X_train, X_test, y_train, y_test


def is_ibm_backend(backend) -> bool:
    return "qiskit_ibm_runtime" in type(backend).__module__


def extract_sampler_counts(pub_result) -> Dict[str, int]:
    data = pub_result.data

    for register_name in ["c", "meas", "cr", "creg"]:
        if hasattr(data, register_name):
            register = getattr(data, register_name)
            if hasattr(register, "get_counts"):
                return dict(register.get_counts())

    available_fields = [
        attr for attr in dir(data)
        if not attr.startswith("_")
    ]

    raise RuntimeError(
        "Could not find classical measurement register in SamplerV2 result. "
        f"Available data fields: {available_fields}"
    )


def run_circuits_on_backend(
    backend,
    circuits: List[QuantumCircuit],
    shots: int,
    use_transpile: bool = True,
) -> List[Dict[str, int]]:
    if use_transpile:
        circuits = transpile(circuits, backend)

    if is_ibm_backend(backend):
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        sampler = Sampler(mode=backend)
        job = sampler.run(circuits, shots=shots)
        result = job.result()

        counts_list = []

        for i in range(len(circuits)):
            counts_list.append(extract_sampler_counts(result[i]))

        return counts_list

    job = backend.run(circuits, shots=shots)
    result = job.result()

    return [result.get_counts(i) for i in range(len(circuits))]


def try_get_ibm_backend() -> Tuple[Optional[object], Dict[str, object]]:
    metadata: Dict[str, object] = {
        "ibm_runtime_available": False,
        "ibm_backend_available": False,
        "backend_name": None,
        "note": None,
    }

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:
        metadata["note"] = (
            "qiskit-ibm-runtime is not installed or could not be imported. "
            f"Import error: {repr(exc)}"
        )
        return None, metadata

    metadata["ibm_runtime_available"] = True

    try:
        service = QiskitRuntimeService()
        backend = service.least_busy(operational=True, simulator=False)

        metadata["ibm_backend_available"] = True
        metadata["backend_name"] = backend.name
        metadata["note"] = "IBM backend selected successfully."

        try:
            metadata["num_qubits"] = backend.num_qubits
        except Exception:
            metadata["num_qubits"] = None

        return backend, metadata

    except Exception as exc:
        metadata["note"] = (
            "IBM Runtime is installed, but no configured account/backend was available. "
            f"Runtime error: {repr(exc)}"
        )
        return None, metadata


def evaluate_counts(
    train_counts: List[Dict[str, int]],
    test_counts: List[Dict[str, int]],
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_qubits: int,
) -> Dict[str, object]:
    parity_preds = np.array(
        [parity_predict_from_counts(counts) for counts in test_counts],
        dtype=int,
    )

    parity_accuracy = accuracy_score(y_test, parity_preds)

    Z_train = np.vstack(
        [counts_to_features(counts, num_qubits) for counts in train_counts]
    )

    Z_test = np.vstack(
        [counts_to_features(counts, num_qubits) for counts in test_counts]
    )

    readout = LogisticRegression(max_iter=1000, random_state=42)
    readout.fit(Z_train, y_train)

    learned_preds = readout.predict(Z_test)
    learned_accuracy = accuracy_score(y_test, learned_preds)

    return {
        "parity_accuracy": float(parity_accuracy),
        "multi_observable_logistic_accuracy": float(learned_accuracy),
        "feature_dimension": int(Z_train.shape[1]),
    }


def plot_results(summary: Dict[str, object], output_path: Path) -> None:
    labels = ["Aer\nParity", "Aer\nMulti-Obs LR"]
    values = [
        summary["aer_simulator"]["parity_accuracy"],
        summary["aer_simulator"]["multi_observable_logistic_accuracy"],
    ]

    if summary.get("ibm_hardware") is not None:
        labels.extend(["IBM\nParity", "IBM\nMulti-Obs LR"])
        values.extend(
            [
                summary["ibm_hardware"]["parity_accuracy"],
                summary["ibm_hardware"]["multi_observable_logistic_accuracy"],
            ]
        )

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Real Hardware Validation: Parity vs Multi-Observable Readout")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_pipeline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    X_train, X_test, y_train, y_test = make_dataset()

    num_qubits = 2
    num_params = 3 * num_qubits
    shots = 1024

    weights = np.random.uniform(-np.pi, np.pi, num_params)

    train_circuits = [
        build_hardware_validation_circuit(x, weights, num_qubits)
        for x in X_train
    ]

    test_circuits = [
        build_hardware_validation_circuit(x, weights, num_qubits)
        for x in X_test
    ]

    aer_backend = AerSimulator()

    print("\nRunning local Aer simulator validation...")

    aer_train_counts = run_circuits_on_backend(
        aer_backend,
        train_circuits,
        shots=shots,
        use_transpile=True,
    )

    aer_test_counts = run_circuits_on_backend(
        aer_backend,
        test_circuits,
        shots=shots,
        use_transpile=True,
    )

    aer_results = evaluate_counts(
        aer_train_counts,
        aer_test_counts,
        y_train,
        y_test,
        num_qubits,
    )

    print("Aer results:")
    print(aer_results)

    ibm_backend, ibm_metadata = try_get_ibm_backend()

    ibm_results = None

    if ibm_backend is not None:
        print("\nRunning IBM Quantum hardware validation...")
        print(f"Selected backend: {ibm_metadata.get('backend_name')}")

        ibm_train_counts = run_circuits_on_backend(
            ibm_backend,
            train_circuits,
            shots=shots,
            use_transpile=True,
        )

        ibm_test_counts = run_circuits_on_backend(
            ibm_backend,
            test_circuits,
            shots=shots,
            use_transpile=True,
        )

        ibm_results = evaluate_counts(
            ibm_train_counts,
            ibm_test_counts,
            y_train,
            y_test,
            num_qubits,
        )

        print("IBM hardware results:")
        print(ibm_results)

    else:
        print("\nIBM hardware was skipped.")
        print(ibm_metadata["note"])

    summary = {
        "description": (
            "Small-scale real hardware validation for the noise-robust HQNN "
            "framework. This compares fixed parity readout against "
            "multi-observable learned readout on local Aer simulation and, "
            "when configured, an IBM Quantum backend."
        ),
        "num_qubits": num_qubits,
        "shots": shots,
        "dataset_size": {
            "train": int(len(X_train)),
            "test": int(len(X_test)),
        },
        "aer_simulator": aer_results,
        "ibm_metadata": ibm_metadata,
        "ibm_hardware": ibm_results,
        "thesis_contribution": (
            "This hardware-validation substudy tests whether the same "
            "multi-observable readout logic used in the framework can be applied "
            "to real NISQ backend measurement counts."
        ),
    }

    json_path = RESULTS_DIR / "real_hardware_validation_summary.json"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_path = RESULTS_DIR / "real_hardware_validation_accuracy.png"

    plot_results(summary, plot_path)

    print("\nReal hardware validation pipeline complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    run_pipeline()


