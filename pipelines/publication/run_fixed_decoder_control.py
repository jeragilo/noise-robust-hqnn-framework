"""
Publication Experiment 2:
Fixed-Decoder Control for HQNN Readout Representations

Reuses cached quantum measurements from the publication smoke test.
No quantum circuits are rerun.

Purpose:
Determine whether representation rankings persist when every learned
representation uses exactly the same classical decoder.

Controls:
A. Logistic Regression
B. Random Forest

R0 fixed parity remains unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from pipelines.publication.experiment import (
    make_synthetic_dataset,
    records_to_feature_matrix,
)

from pipelines.publication.measurements import (
    load_measurement_records,
)

from pipelines.publication.representations import (
    Representation,
    fixed_parity_prediction,
)


SOURCE_DIR = (
    Path("results")
    / "publication"
    / "readout_ablation_smoke_test"
)

CACHE_DIR = SOURCE_DIR / "measurement_cache"

OUTPUT_DIR = (
    Path("results")
    / "publication"
    / "fixed_decoder_control"
)

SEEDS = [101, 202, 303]

NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
]

NUM_QUBITS = 4


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


def load_split(
    seed: int,
    noise_level: float,
    split_name: str,
) -> List[Dict[str, Dict[str, int]]]:

    path = (
        CACHE_DIR
        / f"seed_{seed}"
        / f"noise_{noise_level:.2f}"
        / f"{split_name}.json"
    )

    payload = load_measurement_records(path)

    return payload["records"]


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
            )
        ),
    }


def fixed_parity_predictions(
    records: List[Dict[str, Dict[str, int]]],
) -> np.ndarray:

    return np.asarray(
        [
            fixed_parity_prediction(record["Z"])
            for record in records
        ],
        dtype=int,
    )


def decoder_templates() -> Dict[str, object]:

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
        ),
    }


def evaluate_learned_representation(
    representation: Representation,
    decoder_template,
    train_records,
    validation_records,
    test_records,
    y_train,
    y_validation,
    y_test,
) -> Dict[str, object]:

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

    # No model selection occurs here.
    # The decoder type is fixed before evaluation.
    #
    # Train + validation are combined because no hyperparameters
    # are being selected using the validation set.
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

    model = clone(decoder_template)

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
        **metrics(
            y_test,
            predictions,
        ),
    }


def run_one_condition(
    seed: int,
    noise_level: float,
) -> Dict[str, object]:

    dataset = make_synthetic_dataset(
        seed=seed,
        n_samples=500,
        n_features=NUM_QUBITS,
    )

    train_records = load_split(
        seed,
        noise_level,
        "train",
    )

    validation_records = load_split(
        seed,
        noise_level,
        "validation",
    )

    test_records = load_split(
        seed,
        noise_level,
        "test",
    )

    parity_predictions = fixed_parity_predictions(
        test_records
    )

    result: Dict[str, object] = {
        "seed": seed,
        "noise_level": noise_level,
        "R0_parity": {
            "decoder": "fixed_threshold",
            **metrics(
                dataset.y_test,
                parity_predictions,
            ),
        },
        "decoders": {},
    }

    for decoder_name, decoder_template in (
        decoder_templates().items()
    ):

        decoder_results = {}

        for representation in LEARNED_REPRESENTATIONS:

            decoder_results[
                representation.value
            ] = evaluate_learned_representation(
                representation=representation,
                decoder_template=decoder_template,
                train_records=train_records,
                validation_records=validation_records,
                test_records=test_records,
                y_train=dataset.y_train,
                y_validation=dataset.y_validation,
                y_test=dataset.y_test,
            )

        result["decoders"][
            decoder_name
        ] = decoder_results

    return result


def summarize(
    runs: List[Dict[str, object]],
) -> List[Dict[str, object]]:

    rows = []

    representation_names = [
        representation.value
        for representation in LEARNED_REPRESENTATIONS
    ]

    for noise_level in NOISE_LEVELS:

        matching_runs = [
            run
            for run in runs
            if run["noise_level"] == noise_level
        ]

        parity_values = np.asarray(
            [
                run["R0_parity"]["accuracy"]
                for run in matching_runs
            ],
            dtype=float,
        )

        rows.append(
            {
                "noise_level": noise_level,
                "decoder": "fixed_threshold",
                "representation": "R0_parity",
                "n": len(parity_values),
                "mean_accuracy": float(
                    np.mean(parity_values)
                ),
                "std_accuracy": float(
                    np.std(
                        parity_values,
                        ddof=1,
                    )
                ),
            }
        )

        for decoder_name in decoder_templates():

            for representation in representation_names:

                values = np.asarray(
                    [
                        run["decoders"][
                            decoder_name
                        ][representation]["accuracy"]
                        for run in matching_runs
                    ],
                    dtype=float,
                )

                rows.append(
                    {
                        "noise_level": noise_level,
                        "decoder": decoder_name,
                        "representation": representation,
                        "n": len(values),
                        "mean_accuracy": float(
                            np.mean(values)
                        ),
                        "std_accuracy": float(
                            np.std(
                                values,
                                ddof=1,
                            )
                        ),
                    }
                )

    return rows


def print_summary(
    summary_rows: List[Dict[str, object]],
) -> None:

    for noise_level in NOISE_LEVELS:

        print("\n" + "=" * 80)
        print(
            f"FIXED DECODER CONTROL | "
            f"NOISE={noise_level:.2f}"
        )
        print("=" * 80)

        parity_row = next(
            row
            for row in summary_rows
            if (
                row["noise_level"] == noise_level
                and row["representation"] == "R0_parity"
            )
        )

        print(
            f"R0_parity | "
            f"mean={parity_row['mean_accuracy']:.4f} | "
            f"sd={parity_row['std_accuracy']:.4f}"
        )

        for decoder_name in decoder_templates():

            print(
                f"\n{decoder_name}:"
            )

            decoder_rows = [
                row
                for row in summary_rows
                if (
                    row["noise_level"] == noise_level
                    and row["decoder"] == decoder_name
                )
            ]

            decoder_rows = sorted(
                decoder_rows,
                key=lambda row: row["mean_accuracy"],
                reverse=True,
            )

            for row in decoder_rows:

                print(
                    f"{row['representation']:24s} | "
                    f"mean={row['mean_accuracy']:.4f} | "
                    f"sd={row['std_accuracy']:.4f}"
                )


def run_pipeline() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = []

    for seed in SEEDS:

        for noise_level in NOISE_LEVELS:

            print(
                f"Evaluating cached measurements | "
                f"seed={seed} | "
                f"noise={noise_level:.2f}"
            )

            runs.append(
                run_one_condition(
                    seed=seed,
                    noise_level=noise_level,
                )
            )

    summary_rows = summarize(
        runs
    )

    output = {
        "description": (
            "Fixed-decoder control using cached quantum "
            "measurements from the publication smoke test."
        ),
        "note": (
            "No quantum circuits were rerun. Logistic Regression "
            "and Random Forest are applied consistently across "
            "all learned representations."
        ),
        "seeds": SEEDS,
        "noise_levels": NOISE_LEVELS,
        "runs": runs,
        "summary": summary_rows,
    }

    output_path = (
        OUTPUT_DIR
        / "fixed_decoder_control.json"
    )

    with open(
        output_path,
        "w",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print_summary(
        summary_rows
    )

    print("\nFixed-decoder control complete.")
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    run_pipeline()
