"""
Main Publication Study:
Measurement Representation and Decoder Capacity in Noise-Robust HQNNs

This is the publication-scale follow-up to the development smoke test.

Research questions
------------------
RQ1:
How much classification information is lost by fixed parity readout?

RQ2:
Which quantum measurement representations provide the strongest
classification performance under increasing NISQ noise?

RQ3:
Does the relative value of a quantum measurement representation depend
on the capacity of the classical decoder?

RQ4:
How much additional predictive performance is obtained from multi-basis
XYZ measurement relative to representations derived from Z-basis
measurements?

Experimental design
-------------------
- 20 matched random seeds
- 5 depolarizing-noise levels
- R0 through R7 readout representations
- fixed Logistic Regression and Random Forest decoders
- matched quantum circuit / dataset configuration within each seed
- measurement caching for resumability
- long-form CSV output
- JSON metadata and summary output

IMPORTANT:
This script does NOT select the best decoder independently for each
representation. Decoder type is fixed as an experimental factor.
"""

from __future__ import annotations

import csv
import json
import time
from scipy import stats
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from pipelines.publication.circuits import initialize_weights

from pipelines.publication.experiment import (
    make_synthetic_dataset,
    records_to_feature_matrix,
)

from pipelines.publication.measurements import (
    load_measurement_records,
    make_aer_backend,
    save_measurement_records,
)

from pipelines.publication.representations import (
    Representation,
    fixed_parity_prediction,
)


# =============================================================================
# PATHS
# =============================================================================

RESULTS_DIR = (
    Path("results")
    / "publication"
    / "main_readout_study"
)

CACHE_DIR = RESULTS_DIR / "measurement_cache"

LONG_FORM_PATH = (
    RESULTS_DIR
    / "main_readout_study_long_form.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "main_readout_study_summary.json"
)

METADATA_PATH = (
    RESULTS_DIR
    / "main_readout_study_metadata.json"
)


# =============================================================================
# LOCKED EXPERIMENTAL PROTOCOL
# =============================================================================

SEEDS = [
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
    909,
    1010,
    1111,
    1212,
    1313,
    1414,
    1515,
    1616,
    1717,
    1818,
    1919,
    2020,
]

NOISE_LEVELS = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.20,
]

NOISE_TYPE = "depolarizing"

NUM_QUBITS = 4

NUM_SAMPLES = 500

SHOTS = 1024

ARCHITECTURE = "ring"


# =============================================================================
# REPRESENTATIONS
# =============================================================================

LEARNED_REPRESENTATIONS = [
    Representation.LEARNED_PARITY,
    Representation.SINGLE_Z,
    Representation.ALL_Z,
    Representation.ZENG_AMM,
    Representation.PROBABILITIES,
    Representation.Z_ZZ,
    Representation.PROB_Z_ZZ,
    Representation.FULL,
]


# =============================================================================
# DECODERS
# =============================================================================

def decoder_templates() -> Dict[str, object]:
    """
    Fixed decoder configurations.

    Hyperparameters are locked before the main experiment.
    No per-representation decoder selection is performed.
    """

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }


# =============================================================================
# METRICS
# =============================================================================

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
            )
        ),
    }


# =============================================================================
# CACHE PATHS
# =============================================================================

def cache_condition_dir(
    seed: int,
    noise_level: float,
) -> Path:

    return (
        CACHE_DIR
        / f"seed_{seed}"
        / f"noise_{noise_level:.2f}"
    )


def cache_split_path(
    seed: int,
    noise_level: float,
    split_name: str,
) -> Path:

    return (
        cache_condition_dir(
            seed,
            noise_level,
        )
        / f"{split_name}.json"
    )


def cache_complete(
    seed: int,
    noise_level: float,
) -> bool:

    required = [
        cache_split_path(
            seed,
            noise_level,
            "train",
        ),
        cache_split_path(
            seed,
            noise_level,
            "validation",
        ),
        cache_split_path(
            seed,
            noise_level,
            "test",
        ),
    ]

    return all(
        path.exists()
        for path in required
    )


# =============================================================================
# MEASUREMENT GENERATION
# =============================================================================

def generate_measurements_for_condition(
    seed: int,
    noise_level: float,
) -> None:
    """
    Generate and cache quantum measurement records for one matched
    seed/noise condition.

    The main study deliberately reuses make_backend() and measure_split()
    from the validated publication smoke-test pipeline. This keeps circuit
    construction, X/Y/Z measurement, validation, and measurement semantics
    consistent between the development and publication-scale experiments.
    """

    if cache_complete(
        seed,
        noise_level,
    ):
        print(
            f"CACHE HIT | seed={seed} | "
            f"noise={noise_level:.2f}"
        )
        return

    print(
        f"CACHE MISS | seed={seed} | "
        f"noise={noise_level:.2f}"
    )

    dataset = make_synthetic_dataset(
        seed=seed,
        n_samples=NUM_SAMPLES,
        n_features=NUM_QUBITS,
    )

    weights = initialize_weights(
        num_qubits=NUM_QUBITS,
        seed=seed,
    )

    from pipelines.publication.run_readout_ablation import (
        make_backend,
        measure_split,
    )

    backend = make_backend(
        noise_level=noise_level,
    )

    split_data = {
        "train": dataset.X_train,
        "validation": dataset.X_validation,
        "test": dataset.X_test,
    }

    for split_name, X_split in split_data.items():

        output_path = cache_split_path(
            seed,
            noise_level,
            split_name,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = measure_split(
            backend=backend,
            X=X_split,
            weights=weights,
            seed=seed,
            noise_level=noise_level,
            split_name=split_name,
        )

        save_measurement_records(
            records=records,
            output_path=output_path,
            metadata={
                "seed": seed,
                "noise_type": NOISE_TYPE,
                "noise_level": noise_level,
                "shots": SHOTS,
                "num_qubits": NUM_QUBITS,
                "architecture": ARCHITECTURE,
                "split": split_name,
                "study": "main_readout_study",
            },
        )

# =============================================================================
# CACHE LOADING
# =============================================================================

def load_split_records(
    seed: int,
    noise_level: float,
    split_name: str,
) -> List[Dict[str, Any]]:

    path = cache_split_path(
        seed,
        noise_level,
        split_name,
    )

    payload = load_measurement_records(
        path
    )

    return payload["records"]


# =============================================================================
# FIXED PARITY BASELINE
# =============================================================================

def evaluate_fixed_parity(
    test_records: List[Dict[str, Any]],
    y_test: np.ndarray,
) -> Dict[str, float]:

    predictions = np.asarray(
        [
            fixed_parity_prediction(
                record["Z"]
            )
            for record in test_records
        ],
        dtype=int,
    )

    return classification_metrics(
        y_test,
        predictions,
    )


# =============================================================================
# LEARNED REPRESENTATION EVALUATION
# =============================================================================

def evaluate_representation(
    representation: Representation,
    decoder_template,
    train_records: List[Dict[str, Any]],
    validation_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:

    X_train = records_to_feature_matrix(
        train_records,
        NUM_QUBITS,
        representation,
    )

    X_validation = records_to_feature_matrix(
        validation_records,
        NUM_QUBITS,
        representation,
    )

    X_test = records_to_feature_matrix(
        test_records,
        NUM_QUBITS,
        representation,
    )

    # No model selection is performed in the main study.
    # Validation data can therefore be folded into the final training set.

    X_fit = np.vstack(
        [
            X_train,
            X_validation,
        ]
    )

    y_fit = np.concatenate(
        [
            y_train,
            y_validation,
        ]
    )

    model = clone(
        decoder_template
    )

    model.fit(
        X_fit,
        y_fit,
    )

    predictions = model.predict(
        X_test
    )

    return {
        "feature_dimension": int(
            X_fit.shape[1]
        ),
        **classification_metrics(
            y_test,
            predictions,
        ),
    }


# =============================================================================
# REPRESENTATION RESOURCE METADATA
# =============================================================================

def measurement_basis_count(
    representation_name: str,
) -> int:
    """
    Approximate number of measurement bases required by the representation.

    XYZ requires three measurement bases.
    The remaining representations in this experiment are extracted from
    computational-basis measurement records.
    """

    if representation_name == Representation.ZENG_AMM.value:
        return 3

    return 1


# =============================================================================
# ONE CONDITION
# =============================================================================

def evaluate_condition(
    seed: int,
    noise_level: float,
) -> List[Dict[str, Any]]:

    dataset = make_synthetic_dataset(
        seed=seed,
        n_samples=NUM_SAMPLES,
        n_features=NUM_QUBITS,
    )

    train_records = load_split_records(
        seed,
        noise_level,
        "train",
    )

    validation_records = load_split_records(
        seed,
        noise_level,
        "validation",
    )

    test_records = load_split_records(
        seed,
        noise_level,
        "test",
    )

    rows: List[Dict[str, Any]] = []

    parity_result = evaluate_fixed_parity(
        test_records,
        dataset.y_test,
    )

    rows.append(
        {
            "seed": seed,
            "noise_type": NOISE_TYPE,
            "noise_level": noise_level,
            "shots": SHOTS,
            "num_qubits": NUM_QUBITS,
            "architecture": ARCHITECTURE,
            "representation": "R0_parity",
            "decoder": "fixed_threshold",
            "measurement_bases": 1,
            "feature_dimension": 1,
            **parity_result,
        }
    )

    for decoder_name, decoder_template in (
        decoder_templates().items()
    ):

        for representation in LEARNED_REPRESENTATIONS:

            result = evaluate_representation(
                representation=representation,
                decoder_template=decoder_template,
                train_records=train_records,
                validation_records=validation_records,
                test_records=test_records,
                y_train=dataset.y_train,
                y_validation=dataset.y_validation,
                y_test=dataset.y_test,
            )

            rows.append(
                {
                    "seed": seed,
                    "noise_type": NOISE_TYPE,
                    "noise_level": noise_level,
                    "shots": SHOTS,
                    "num_qubits": NUM_QUBITS,
                    "architecture": ARCHITECTURE,
                    "representation": representation.value,
                    "decoder": decoder_name,
                    "measurement_bases": (
                        measurement_basis_count(
                            representation.value
                        )
                    ),
                    "feature_dimension": result[
                        "feature_dimension"
                    ],
                    "accuracy": result[
                        "accuracy"
                    ],
                    "macro_f1": result[
                        "macro_f1"
                    ],
                }
            )

    return rows


# =============================================================================
# LONG-FORM OUTPUT
# =============================================================================

CSV_FIELDS = [
    "seed",
    "noise_type",
    "noise_level",
    "shots",
    "num_qubits",
    "architecture",
    "representation",
    "decoder",
    "measurement_bases",
    "feature_dimension",
    "accuracy",
    "macro_f1",
]


def write_long_form(
    rows: List[Dict[str, Any]],
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        LONG_FORM_PATH,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# =============================================================================
# SUMMARY
# =============================================================================

def summarize_results(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    groups: Dict[
        tuple,
        List[Dict[str, Any]],
    ] = {}

    for row in rows:

        key = (
            row["noise_level"],
            row["representation"],
            row["decoder"],
        )

        groups.setdefault(
            key,
            [],
        ).append(row)

    summary = []

    for (
        noise_level,
        representation,
        decoder,
    ), group_rows in groups.items():

        accuracy_values = np.asarray(
            [
                row["accuracy"]
                for row in group_rows
            ],
            dtype=float,
        )

        f1_values = np.asarray(
            [
                row["macro_f1"]
                for row in group_rows
            ],
            dtype=float,
        )

        n = len(
            accuracy_values
        )

        accuracy_mean = float(
            np.mean(
                accuracy_values
            )
        )

        accuracy_std = float(
            np.std(
                accuracy_values,
                ddof=1,
            )
        ) if n > 1 else 0.0

        accuracy_se = (
            accuracy_std
            / np.sqrt(n)
            if n > 1
            else 0.0
        )

            1.96 * accuracy_se
)if n > 1:
    critical_t = float(
        stats.t.ppf(
            0.975,
            df=n - 1,
        )
    )

    ci_margin = (
        critical_t
        * accuracy_se
    )
else:
    ci_margin = 0.0

        summary.append(
            {
                "noise_level": noise_level,
                "representation": representation,
                "decoder": decoder,
                "n": n,
                "mean_accuracy": accuracy_mean,
                "std_accuracy": accuracy_std,
                "ci95_accuracy_lower": (
                    accuracy_mean
                    - ci_margin
                ),
                "ci95_accuracy_upper": (
                    accuracy_mean
                    + ci_margin
                ),
                "mean_macro_f1": float(
                    np.mean(
                        f1_values
                    )
                ),
                "measurement_bases": group_rows[
                    0
                ]["measurement_bases"],
                "feature_dimension": group_rows[
                    0
                ]["feature_dimension"],
            }
        )

    summary.sort(
        key=lambda row: (
            row["noise_level"],
            row["decoder"],
            row["representation"],
        )
    )

    return summary


# =============================================================================
# ROBUSTNESS METRICS
# =============================================================================

def add_robustness_metrics(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    clean_lookup = {}

    for row in rows:

        if np.isclose(
            row["noise_level"],
            0.0,
        ):

            key = (
                row["seed"],
                row["representation"],
                row["decoder"],
            )

            clean_lookup[
                key
            ] = row["accuracy"]

    enriched_rows = []

    for row in rows:

        new_row = dict(
            row
        )

        key = (
            row["seed"],
            row["representation"],
            row["decoder"],
        )

        clean_accuracy = clean_lookup.get(
            key
        )

        if clean_accuracy is None:

            new_row[
                "accuracy_drop_from_clean"
            ] = None

            new_row[
                "accuracy_retention"
            ] = None

        else:

            new_row[
                "accuracy_drop_from_clean"
            ] = float(
                clean_accuracy
                - row["accuracy"]
            )

            new_row[
                "accuracy_retention"
            ] = (
                float(
                    row["accuracy"]
                    / clean_accuracy
                )
                if clean_accuracy != 0
                else None
            )

        enriched_rows.append(
            new_row
        )

    return enriched_rows


# =============================================================================
# PROGRESS / CHECKPOINT OUTPUT
# =============================================================================

def print_condition_summary(
    seed: int,
    noise_level: float,
    rows: List[Dict[str, Any]],
) -> None:

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"COMPLETE | seed={seed} | "
        f"noise={noise_level:.2f}"
    )

    print(
        "-" * 80
    )

    for row in rows:

        print(
		            f"{row['representation']:24s} | "
            f"{row['decoder']:20s} | "
            f"acc={row['accuracy']:.4f} | "
            f"F1={row['macro_f1']:.4f}"
        )


# =============================================================================
# METADATA
# =============================================================================

def write_metadata() -> None:

    metadata = {
        "study_name": (
            "Main Publication Study: Measurement Representation "
            "and Decoder Capacity in Noise-Robust HQNNs"
        ),
        "study_type": "publication_main_experiment",
        "status": "main_experiment",
        "experimental_protocol": {
            "num_seeds": len(SEEDS),
            "seeds": SEEDS,
            "noise_type": NOISE_TYPE,
            "noise_levels": NOISE_LEVELS,
            "num_qubits": NUM_QUBITS,
            "num_samples": NUM_SAMPLES,
            "shots": SHOTS,
            "architecture": ARCHITECTURE,
        },
        "representations": [
            "R0_parity",
            *[
                representation.value
                for representation in LEARNED_REPRESENTATIONS
            ],
        ],
        "decoders": [
            "fixed_threshold",
            *list(decoder_templates().keys()),
        ],
        "research_questions": {
            "RQ1": (
                "How much classification information is lost "
                "by fixed parity readout?"
            ),
            "RQ2": (
                "Which quantum measurement representations "
                "provide the strongest classification performance "
                "under increasing NISQ noise?"
            ),
            "RQ3": (
                "Does the relative value of a quantum measurement "
                "representation depend on classical decoder capacity?"
            ),
            "RQ4": (
                "How much additional predictive performance is "
                "obtained from multi-basis XYZ measurement relative "
                "to representations derived from Z-basis measurements?"
            ),
        },
        "methodological_controls": {
            "matched_seeds": True,
            "fixed_decoders": True,
            "per_representation_decoder_selection": False,
            "measurement_caching": True,
            "resumable": True,
            "clean_noisy_pairing": True,
        },
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METADATA_PATH,
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


# =============================================================================
# CHECKPOINT STORAGE
# =============================================================================

CHECKPOINT_PATH = (
    RESULTS_DIR
    / "main_readout_study_checkpoint.json"
)


def condition_key(
    seed: int,
    noise_level: float,
) -> str:

    return (
        f"seed_{seed}"
        f"__noise_{noise_level:.2f}"
    )


def load_checkpoint() -> Dict[str, Any]:

    if not CHECKPOINT_PATH.exists():

        return {
            "completed_conditions": {},
        }

    with open(
        CHECKPOINT_PATH,
        "r",
    ) as file:

        return json.load(file)


def save_checkpoint(
    checkpoint: Dict[str, Any],
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        CHECKPOINT_PATH.with_suffix(
            ".tmp"
        )
    )

    with open(
        temporary_path,
        "w",
    ) as file:

        json.dump(
            checkpoint,
            file,
            indent=2,
        )

    temporary_path.replace(
        CHECKPOINT_PATH
    )


def checkpoint_rows(
    checkpoint: Dict[str, Any],
) -> List[Dict[str, Any]]:

    rows = []

    for condition in checkpoint[
        "completed_conditions"
    ].values():

        rows.extend(
            condition["rows"]
        )

    return rows


# =============================================================================
# FINAL SUMMARY PRINTING
# =============================================================================

def print_final_summary(
    summary: List[Dict[str, Any]],
) -> None:

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MAIN READOUT STUDY SUMMARY"
    )

    print(
        "=" * 80
    )

    for noise_level in NOISE_LEVELS:

        print(
            f"\nNOISE = {noise_level:.2f}"
        )

        parity_rows = [
            row
            for row in summary
            if (
                np.isclose(
                    row["noise_level"],
                    noise_level,
                )
                and row["representation"]
                == "R0_parity"
            )
        ]

        if parity_rows:

            row = parity_rows[0]

            print(
                "\nfixed_threshold:"
            )

            print(
                f"{row['representation']:24s} | "
                f"mean={row['mean_accuracy']:.4f} | "
                f"sd={row['std_accuracy']:.4f} | "
                f"95% CI=["
                f"{row['ci95_accuracy_lower']:.4f}, "
                f"{row['ci95_accuracy_upper']:.4f}]"
            )

        for decoder_name in (
            decoder_templates().keys()
        ):

            print(
                f"\n{decoder_name}:"
            )

            decoder_rows = [
                row
                for row in summary
                if (
                    np.isclose(
                        row["noise_level"],
                        noise_level,
                    )
                    and row["decoder"]
                    == decoder_name
                )
            ]

            decoder_rows = sorted(
                decoder_rows,
                key=lambda row: (
                    row["mean_accuracy"]
                ),
                reverse=True,
            )

            for row in decoder_rows:

                print(
                    f"{row['representation']:24s} | "
                    f"mean={row['mean_accuracy']:.4f} | "
                    f"sd={row['std_accuracy']:.4f} | "
                    f"95% CI=["
                    f"{row['ci95_accuracy_lower']:.4f}, "
                    f"{row['ci95_accuracy_upper']:.4f}] | "
                    f"bases={row['measurement_bases']}"
                )


# =============================================================================
# RUN PIPELINE
# =============================================================================

def run_pipeline() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_metadata()

    checkpoint = load_checkpoint()

    total_conditions = (
        len(SEEDS)
        * len(NOISE_LEVELS)
    )

    condition_counter = 0

    study_start = time.time()

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MAIN PUBLICATION READOUT STUDY"
    )

    print(
        "=" * 80
    )

    print(
        f"Seeds: {len(SEEDS)}"
    )

    print(
        f"Noise levels: {NOISE_LEVELS}"
    )

    print(
        f"Total seed/noise conditions: "
        f"{total_conditions}"
    )

    print(
        f"Shots: {SHOTS}"
    )

    print(
        f"Qubits: {NUM_QUBITS}"
    )

    print(
        f"Architecture: {ARCHITECTURE}"
    )

    print(
        "\nThis experiment is resumable."
    )

    print(
        "Completed conditions will not be "
        "re-evaluated after interruption."
    )

    for seed in SEEDS:

        for noise_level in NOISE_LEVELS:

            condition_counter += 1

            key = condition_key(
                seed,
                noise_level,
            )

            print(
                "\n"
                + "=" * 80
            )

            print(
                f"CONDITION "
                f"{condition_counter}/"
                f"{total_conditions}"
            )

            print(
                f"seed={seed} | "
                f"noise={noise_level:.2f}"
            )

            print(
                "=" * 80
            )

            if key in checkpoint[
                "completed_conditions"
            ]:

                print(
                    "CHECKPOINT HIT | "
                    "condition already complete"
                )

                continue

            condition_start = time.time()

            # -------------------------------------------------------------
            # Step 1:
            # Generate quantum measurements if they do not already exist.
            # -------------------------------------------------------------

            generate_measurements_for_condition(
                seed=seed,
                noise_level=noise_level,
            )

            # -------------------------------------------------------------
            # Step 2:
            # Evaluate all representations using fixed decoders.
            # -------------------------------------------------------------

            rows = evaluate_condition(
                seed=seed,
                noise_level=noise_level,
            )

            print_condition_summary(
                seed,
                noise_level,
                rows,
            )

            elapsed = (
                time.time()
                - condition_start
            )

            checkpoint[
                "completed_conditions"
            ][key] = {
                "seed": seed,
                "noise_level": noise_level,
                "elapsed_seconds": elapsed,
                "rows": rows,
            }

            save_checkpoint(
                checkpoint
            )

            # -------------------------------------------------------------
            # Continuously update long-form results.
            # This means partial results survive an interrupted run.
            # -------------------------------------------------------------

            current_rows = checkpoint_rows(
                checkpoint
            )

            current_rows = (
                add_robustness_metrics(
                    current_rows
                )
            )

            write_long_form(
                current_rows
            )

            print(
                f"\nCheckpoint saved | "
                f"elapsed={elapsed / 60:.2f} min"
            )

    # =========================================================================
    # FINALIZE
    # =========================================================================

    all_rows = checkpoint_rows(
        checkpoint
    )

    all_rows = add_robustness_metrics(
        all_rows
    )

    # The base CSV writer intentionally uses the original core fields.
    # Write the enriched publication table separately so robustness
    # quantities are preserved as well.

    enriched_csv_path = (
        RESULTS_DIR
        / "main_readout_study_enriched.csv"
    )

    enriched_fields = [
        *CSV_FIELDS,
        "accuracy_drop_from_clean",
        "accuracy_retention",
    ]

    with open(
        enriched_csv_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=enriched_fields,
        )

        writer.writeheader()

        for row in all_rows:
            writer.writerow(row)

    # Keep the ordinary long-form table as well.

    base_rows = [
        {
            field: row[field]
            for field in CSV_FIELDS
        }
        for row in all_rows
    ]

    write_long_form(
        base_rows
    )

    summary = summarize_results(
        all_rows
    )

    final_output = {
        "description": (
            "Publication-scale controlled study of "
            "measurement representation and decoder "
            "capacity in noise-robust HQNNs."
        ),
        "experimental_protocol": {
            "seeds": SEEDS,
            "num_seeds": len(SEEDS),
            "noise_type": NOISE_TYPE,
            "noise_levels": NOISE_LEVELS,
            "num_qubits": NUM_QUBITS,
            "num_samples": NUM_SAMPLES,
            "shots": SHOTS,
            "architecture": ARCHITECTURE,
        },
        "summary": summary,
        "output_files": {
            "long_form_csv": str(
                LONG_FORM_PATH
            ),
            "enriched_csv": str(
                enriched_csv_path
            ),
            "metadata": str(
                METADATA_PATH
            ),
            "checkpoint": str(
                CHECKPOINT_PATH
            ),
        },
    }

    with open(
        SUMMARY_PATH,
        "w",
    ) as file:

        json.dump(
            final_output,
            file,
            indent=2,
        )

    study_elapsed = (
        time.time()
        - study_start
    )

    print_final_summary(
        summary
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MAIN READOUT STUDY COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Completed conditions: "
        f"{len(checkpoint['completed_conditions'])}"
        f"/{total_conditions}"
    )

    print(
        f"Total elapsed time: "
        f"{study_elapsed / 60:.2f} minutes"
    )

    print(
        f"\nSaved summary:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"\nSaved long-form results:\n"
        f"{LONG_FORM_PATH}"
    )

    print(
        f"\nSaved enriched results:\n"
        f"{enriched_csv_path}"
    )

    print(
        f"\nSaved metadata:\n"
        f"{METADATA_PATH}"
    )

    print(
        f"\nCheckpoint:\n"
        f"{CHECKPOINT_PATH}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not modify the locked experimental "
        "protocol after inspecting these results."
    )


if __name__ == "__main__":
    run_pipeline()
