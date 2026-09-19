"""
Publication Experiment 1:
Controlled HQNN Readout Representation Ablation

Purpose
-------
Evaluate how quantum-to-classical readout representation affects
classification performance under increasing NISQ noise while holding
the underlying quantum circuit fixed.

Smoke-test protocol
-------------------
Dataset: synthetic binary classification
Qubits: 4
Architecture: canonical ring-CZ circuit
Seeds: 3
Noise: depolarizing
Noise levels: 0.00, 0.05, 0.10
Shots: 1024

Representations
---------------
R0  : fixed parity
R0L : learned parity
R1  : single Z
R2  : all Z
R3  : Zeng all-qubit XYZ AMM
R4  : full computational-basis probabilities
R5  : Z + ZZ
R6  : probabilities + Z + ZZ
R7  : full engineered representation

This is a development smoke test, not a final publication run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from framework.noise_channels import create_noise_model

from pipelines.publication.circuits import (
    circuit_metadata,
    initialize_weights,
)

from pipelines.publication.experiment import (
    evaluate_all_representations,
    make_synthetic_dataset,
)

from pipelines.publication.measurements import (
    make_aer_backend,
    measure_dataset,
    save_measurement_records,
    validate_measurement_records,
)


RESULTS_DIR = (
    Path("results")
    / "publication"
    / "readout_ablation_smoke_test"
)

CACHE_DIR = RESULTS_DIR / "measurement_cache"


SEEDS = [101, 202, 303]

NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
]

NOISE_TYPE = "depolarizing"

NUM_QUBITS = 4

SHOTS = 1024


def make_backend(
    noise_level: float,
):
    """
    Construct ideal or noisy Aer backend.
    """

    if noise_level == 0.0:
        return make_aer_backend()

    noise_model = create_noise_model(
        NOISE_TYPE,
        noise_level,
    )

    return make_aer_backend(
        noise_model=noise_model,
    )


def measure_split(
    backend,
    X: np.ndarray,
    weights: np.ndarray,
    seed: int,
    noise_level: float,
    split_name: str,
) -> List[Dict[str, Dict[str, int]]]:
    """
    Execute and cache one dataset split.

    XYZ measurements are included so all representations,
    including R3, can be evaluated from this experiment.
    """

    simulator_seed = (
        seed * 100000
        + int(round(noise_level * 10000))
    )

    if split_name == "validation":
        simulator_seed += 10000

    elif split_name == "test":
        simulator_seed += 20000

    records = measure_dataset(
        backend=backend,
        X=X,
        weights=weights,
        num_qubits=NUM_QUBITS,
        shots=SHOTS,
        include_xyz=True,
        seed_simulator=simulator_seed,
    )

    validate_measurement_records(
        records=records,
        shots=SHOTS,
        require_xyz=True,
    )

    cache_path = (
        CACHE_DIR
        / f"seed_{seed}"
        / f"noise_{noise_level:.2f}"
        / f"{split_name}.json"
    )

    save_measurement_records(
        records=records,
        output_path=cache_path,
        metadata={
            "seed": seed,
            "noise_type": NOISE_TYPE,
            "noise_level": noise_level,
            "shots": SHOTS,
            "num_qubits": NUM_QUBITS,
            "split": split_name,
            **circuit_metadata(NUM_QUBITS),
        },
    )

    return records


def extract_test_metrics(
    representation_results: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    """
    Reduce the full decoder-selection output to the principal
    metrics used by the smoke-test summary.
    """

    reduced: Dict[str, Dict[str, object]] = {}

    for representation, result in representation_results.items():

        if representation == "R0_parity":
            reduced[representation] = {
                "decoder": "fixed_threshold",
                "accuracy": result["test"]["accuracy"],
                "macro_f1": result["test"]["macro_f1"],
            }

        else:
            reduced[representation] = {
                "decoder": result["selected_decoder"],
                "feature_dimension": result[
                    "feature_dimension"
                ],
                "validation_accuracy": result[
                    "selected_validation_accuracy"
                ],
                "accuracy": result["test"]["accuracy"],
                "macro_f1": result["test"]["macro_f1"],
            }

    return reduced


def summarize_across_seeds(
    runs: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    Compute mean and sample SD across matched seeds for each
    representation and noise level.
    """

    rows = []

    representations = sorted(
        {
            representation
            for run in runs
            for representation in run["results"].keys()
        }
    )

    for noise_level in NOISE_LEVELS:
        noise_runs = [
            run
            for run in runs
            if run["noise_level"] == noise_level
        ]

        for representation in representations:
            accuracies = np.array(
                [
                    run["results"][representation]["accuracy"]
                    for run in noise_runs
                ],
                dtype=float,
            )

            macro_f1 = np.array(
                [
                    run["results"][representation]["macro_f1"]
                    for run in noise_runs
                ],
                dtype=float,
            )

            rows.append(
                {
                    "noise_level": noise_level,
                    "representation": representation,
                    "n": int(len(accuracies)),
                    "mean_accuracy": float(
                        np.mean(accuracies)
                    ),
                    "std_accuracy": float(
                        np.std(
                            accuracies,
                            ddof=1,
                        )
                        if len(accuracies) > 1
                        else 0.0
                    ),
                    "mean_macro_f1": float(
                        np.mean(macro_f1)
                    ),
                    "std_macro_f1": float(
                        np.std(
                            macro_f1,
                            ddof=1,
                        )
                        if len(macro_f1) > 1
                        else 0.0
                    ),
                }
            )

    return rows


def plot_noise_curves(
    summary_rows: List[Dict[str, object]],
    output_path: Path,
) -> None:
    """
    Plot mean test accuracy against depolarizing noise.
    """

    representations = sorted(
        {
            row["representation"]
            for row in summary_rows
        }
    )

    plt.figure(
        figsize=(12, 7)
    )

    for representation in representations:
        rows = [
            row
            for row in summary_rows
            if row["representation"] == representation
        ]

        rows = sorted(
            rows,
            key=lambda row: row["noise_level"],
        )

        x = [
            row["noise_level"]
            for row in rows
        ]

        y = [
            row["mean_accuracy"]
            for row in rows
        ]

        yerr = [
            row["std_accuracy"]
            for row in rows
        ]

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=representation,
        )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.xlabel(
        "Depolarizing Noise Probability"
    )

    plt.ylabel(
        "Mean Test Accuracy"
    )

    plt.title(
        "Controlled HQNN Readout Ablation"
    )

    plt.grid(
        axis="y"
    )

    plt.legend(
        fontsize=8,
        ncol=2,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def run_pipeline() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_runs: List[Dict[str, object]] = []

    for seed in SEEDS:

        print("\n" + "=" * 80)
        print(f"PUBLICATION SMOKE TEST | SEED {seed}")
        print("=" * 80)

        dataset = make_synthetic_dataset(
            seed=seed,
            n_samples=500,
            n_features=NUM_QUBITS,
        )

        weights = initialize_weights(
            num_qubits=NUM_QUBITS,
            seed=seed,
        )

        for noise_level in NOISE_LEVELS:

            print(
                f"\nSeed={seed} | "
                f"Noise={noise_level:.2f}"
            )

            backend = make_backend(
                noise_level=noise_level,
            )

            train_records = measure_split(
                backend=backend,
                X=dataset.X_train,
                weights=weights,
                seed=seed,
                noise_level=noise_level,
                split_name="train",
            )

            validation_records = measure_split(
                backend=backend,
                X=dataset.X_validation,
                weights=weights,
                seed=seed,
                noise_level=noise_level,
                split_name="validation",
            )

            test_records = measure_split(
                backend=backend,
                X=dataset.X_test,
                weights=weights,
                seed=seed,
                noise_level=noise_level,
                split_name="test",
            )

            representation_results = (
                evaluate_all_representations(
                    train_records=train_records,
                    validation_records=validation_records,
                    test_records=test_records,
                    y_train=dataset.y_train,
                    y_validation=dataset.y_validation,
                    y_test=dataset.y_test,
                    num_qubits=NUM_QUBITS,
                )
            )

            reduced_results = extract_test_metrics(
                representation_results
            )

            run_result = {
                "seed": seed,
                "noise_type": NOISE_TYPE,
                "noise_level": noise_level,
                "shots": SHOTS,
                "results": reduced_results,
            }

            all_runs.append(
                run_result
            )

            print("\nTest results:")

            for representation, result in reduced_results.items():
                print(
                    f"{representation:24s} | "
                    f"acc={result['accuracy']:.4f} | "
                    f"F1={result['macro_f1']:.4f} | "
                    f"decoder={result['decoder']}"
                )

    summary_rows = summarize_across_seeds(
        all_runs
    )

    output = {
        "description": (
            "Development smoke test for the controlled HQNN "
            "readout-representation publication experiment."
        ),
        "warning": (
            "These are development results based on only three seeds "
            "and must not be used as final publication statistics."
        ),
        "protocol": {
            "seeds": SEEDS,
            "noise_type": NOISE_TYPE,
            "noise_levels": NOISE_LEVELS,
            "shots": SHOTS,
            "num_qubits": NUM_QUBITS,
            **circuit_metadata(NUM_QUBITS),
        },
        "runs": all_runs,
        "summary": summary_rows,
    }

    json_path = (
        RESULTS_DIR
        / "readout_ablation_smoke_test.json"
    )

    with open(
        json_path,
        "w",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    plot_path = (
        RESULTS_DIR
        / "readout_ablation_noise_curves.png"
    )

    plot_noise_curves(
        summary_rows=summary_rows,
        output_path=plot_path,
    )

    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETE")
    print("=" * 80)

    print(
        f"Saved results: {json_path}"
    )

    print(
        f"Saved figure:  {plot_path}"
    )

    print(
        "\nIMPORTANT: These are development results only."
    )


if __name__ == "__main__":
    run_pipeline()
