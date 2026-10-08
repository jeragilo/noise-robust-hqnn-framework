
"""
Beyond Parity — IBM Quantum hardware validation.

Preflight only. No QPU jobs are submitted.
"""

from pathlib import Path
import json
import hashlib

from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)
from qiskit_ibm_runtime import QiskitRuntimeService

from pipelines.publication.circuits import (
    build_measurement_circuit,
    initialize_weights,
)
from pipelines.publication.run_iris_validation import (
    make_iris_dataset,
)

BACKEND_NAME = "ibm_marrakesh"
INSTANCE = "hqnn-thesis-instance"

SEEDS = (101, 202, 303, 404, 505)
BASES = ("X", "Y", "Z")

NUM_QUBITS = 4
SHOTS = 1024
OPTIMIZATION_LEVEL = 3
SEED_TRANSPILER = 42

OUTPUT_DIR = Path(
    "results/publication/ibm_hardware_validation"
)

# Hard execution lock.
EXECUTE_HARDWARE = False


def main():
    if EXECUTE_HARDWARE:
        raise RuntimeError(
            "Hardware execution is not implemented "
            "in this preflight script."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    service = QiskitRuntimeService(
        instance=INSTANCE,
    )
    backend = service.backend(BACKEND_NAME)

    if not backend.status().operational:
        raise RuntimeError(
            "Selected IBM backend is not operational."
        )

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILER,
    )

    manifest = []
    max_depth = 0
    max_cz = 0

    for seed in SEEDS:
        dataset = make_iris_dataset(seed)
        weights = initialize_weights(
            NUM_QUBITS,
            seed,
        )

        splits = {
            "train": dataset.X_train,
            "validation": dataset.X_validation,
            "test": dataset.X_test,
        }

        for split_name, X in splits.items():
            for sample_index, x in enumerate(X):
                for basis in BASES:
                    qc = build_measurement_circuit(
                        x=x,
                        weights=weights,
                        num_qubits=NUM_QUBITS,
                        basis=basis,
                    )

                    tqc = pm.run(qc)
                    ops = dict(tqc.count_ops())

                    if tqc.num_clbits != NUM_QUBITS:
                        raise RuntimeError(
                            "Unexpected classical-bit count."
                        )

                    depth = tqc.depth()
                    cz_count = ops.get("cz", 0)

                    max_depth = max(max_depth, depth)
                    max_cz = max(max_cz, cz_count)

                    manifest.append({
                        "seed": seed,
                        "split": split_name,
                        "sample_index": sample_index,
                        "basis": basis,
                        "shots": SHOTS,
                        "depth": depth,
                        "cz_count": cz_count,
                        "operations": ops,
                    })

    if len(manifest) != 1500:
        raise RuntimeError(
            f"Expected 1500 circuits, "
            f"found {len(manifest)}."
        )

    protocol = {
        "backend": BACKEND_NAME,
        "instance": INSTANCE,
        "seeds": SEEDS,
        "bases": BASES,
        "shots_per_circuit": SHOTS,
        "num_qubits": NUM_QUBITS,
        "optimization_level": OPTIMIZATION_LEVEL,
        "seed_transpiler": SEED_TRANSPILER,
        "total_circuits": len(manifest),
        "total_shots": len(manifest) * SHOTS,
        "max_transpiled_depth": max_depth,
        "max_cz_count": max_cz,
        "hardware_execution_enabled": False,
    }

    payload = {
        "protocol": protocol,
        "manifest": manifest,
    }

    output_path = OUTPUT_DIR / "preflight_manifest.json"

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    manifest_hash = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()

    print("=" * 65)
    print("BEYOND PARITY | IBM HARDWARE PREFLIGHT")
    print("=" * 65)
    print("Backend:", BACKEND_NAME)
    print("Circuits:", len(manifest))
    print("Shots:", len(manifest) * SHOTS)
    print("Maximum depth:", max_depth)
    print("Maximum CZ count:", max_cz)
    print("Manifest SHA256:", manifest_hash)
    print("Manifest:", output_path)
    print("Hardware execution: DISABLED")
    print("=" * 65)


if __name__ == "__main__":
    main()

