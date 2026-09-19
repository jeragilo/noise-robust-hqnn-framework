"""
Core experiment utilities for the HQNN readout publication study.

Responsibilities:
1. Create reproducible train/validation/test splits.
2. Convert cached quantum measurement records into R0-R7 feature matrices.
3. Train learned classical readouts without using the test set for selection.
4. Evaluate fixed parity and learned representations consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipelines.publication.representations import (
    Representation,
    extract_z_basis_representation,
    fixed_parity_prediction,
    zeng_amm_features,
)


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray


def make_synthetic_dataset(
    seed: int,
    n_samples: int = 500,
    n_features: int = 4,
) -> DatasetSplit:
    """
    Publication development dataset.

    Split:
        60% training
        20% validation
        20% test
    """

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        class_sep=2.2,
        flip_y=0.01,
        random_state=seed,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=seed,
        stratify=y,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=seed + 1,
        stratify=y_temp,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    X_train = np.clip(X_train, -np.pi, np.pi)
    X_validation = np.clip(X_validation, -np.pi, np.pi)
    X_test = np.clip(X_test, -np.pi, np.pi)

    return DatasetSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def record_to_features(
    record: Dict[str, Dict[str, int]],
    num_qubits: int,
    representation: Representation,
) -> np.ndarray:
    """
    Convert one cached measurement record into one feature vector.
    """

    if representation == Representation.ZENG_AMM:
        required = {"X", "Y", "Z"}
        missing = required.difference(record.keys())

        if missing:
            raise ValueError(
                "Zeng AMM requires X/Y/Z measurements. "
                f"Missing: {sorted(missing)}"
            )

        return zeng_amm_features(
            x_counts=record["X"],
            y_counts=record["Y"],
            z_counts=record["Z"],
            num_qubits=num_qubits,
        )

    if "Z" not in record:
        raise ValueError(
            "Z-basis counts are required for this representation."
        )

    return extract_z_basis_representation(
        counts=record["Z"],
        num_qubits=num_qubits,
        representation=representation,
    )


def records_to_feature_matrix(
    records: List[Dict[str, Dict[str, int]]],
    num_qubits: int,
    representation: Representation,
) -> np.ndarray:
    """
    Convert cached measurement records into a 2D feature matrix.
    """

    features = [
        record_to_features(
            record=record,
            num_qubits=num_qubits,
            representation=representation,
        )
        for record in records
    ]

    return np.vstack(features)


def fixed_parity_predictions(
    records: List[Dict[str, Dict[str, int]]],
) -> np.ndarray:
    """
    Apply the original fixed parity threshold to cached Z-basis counts.
    """

    predictions = []

    for record in records:
        if "Z" not in record:
            raise ValueError(
                "Fixed parity requires Z-basis measurement counts."
            )

        predictions.append(
            fixed_parity_prediction(record["Z"])
        )

    return np.asarray(predictions, dtype=int)


def get_decoder_candidates() -> Dict[str, object]:
    """
    Candidate classical readouts.

    Hyperparameter search is intentionally small for the first
    publication experiment. Selection is performed only on validation data.
    """

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=42,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }


def classification_metrics(
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


def evaluate_fixed_parity(
    validation_records: List[Dict[str, Dict[str, int]]],
    test_records: List[Dict[str, Dict[str, int]]],
    y_validation: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    """
    Evaluate R0 using the original fixed threshold.
    """

    validation_predictions = fixed_parity_predictions(
        validation_records
    )

    test_predictions = fixed_parity_predictions(
        test_records
    )

    return {
        "representation": Representation.PARITY.value,
        "decoder": "fixed_threshold",
        "validation": classification_metrics(
            y_validation,
            validation_predictions,
        ),
        "test": classification_metrics(
            y_test,
            test_predictions,
        ),
    }


def select_decoder_on_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Tuple[str, object, Dict[str, Dict[str, float]]]:
    """
    Select the decoder using validation accuracy only.

    Test data are never used here.
    """

    candidates = get_decoder_candidates()

    validation_results: Dict[str, Dict[str, float]] = {}

    best_name = None
    best_model = None
    best_score = -np.inf

    for name, candidate in candidates.items():
        model = clone(candidate)

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        metrics = classification_metrics(
            y_validation,
            predictions,
        )

        validation_results[name] = metrics

        score = metrics["accuracy"]

        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    if best_name is None or best_model is None:
        raise RuntimeError(
            "Decoder selection failed."
        )

    return (
        best_name,
        best_model,
        validation_results,
    )


def evaluate_learned_representation(
    representation: Representation,
    train_records: List[Dict[str, Dict[str, int]]],
    validation_records: List[Dict[str, Dict[str, int]]],
    test_records: List[Dict[str, Dict[str, int]]],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    num_qubits: int,
) -> Dict[str, object]:
    """
    Train and evaluate one learned readout representation.

    Workflow:
        training data -> fit candidate decoders
        validation data -> choose decoder
        train + validation -> refit chosen decoder
        test data -> one final evaluation
    """

    X_train = records_to_feature_matrix(
        records=train_records,
        num_qubits=num_qubits,
        representation=representation,
    )

    X_validation = records_to_feature_matrix(
        records=validation_records,
        num_qubits=num_qubits,
        representation=representation,
    )

    X_test = records_to_feature_matrix(
        records=test_records,
        num_qubits=num_qubits,
        representation=representation,
    )

    (
        best_decoder_name,
        best_validation_model,
        validation_results,
    ) = select_decoder_on_validation(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    # Recreate the selected model and refit it on train + validation.
    selected_template = get_decoder_candidates()[
        best_decoder_name
    ]

    final_model = clone(
        selected_template
    )

    X_train_final = np.vstack(
        [
            X_train,
            X_validation,
        ]
    )

    y_train_final = np.concatenate(
        [
            y_train,
            y_validation,
        ]
    )

    final_model.fit(
        X_train_final,
        y_train_final,
    )

    test_predictions = final_model.predict(
        X_test
    )

    test_metrics = classification_metrics(
        y_test,
        test_predictions,
    )

    return {
        "representation": representation.value,
        "feature_dimension": int(
            X_train.shape[1]
        ),
        "selected_decoder": best_decoder_name,
        "validation_candidates": validation_results,
        "selected_validation_accuracy": float(
            validation_results[
                best_decoder_name
            ]["accuracy"]
        ),
        "test": test_metrics,
    }


def evaluate_all_representations(
    train_records: List[Dict[str, Dict[str, int]]],
    validation_records: List[Dict[str, Dict[str, int]]],
    test_records: List[Dict[str, Dict[str, int]]],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    num_qubits: int,
) -> Dict[str, Dict[str, object]]:
    """
    Evaluate R0 plus every learned representation.

    R0L uses the same parity scalar as R0 but allows the decoder
    to learn its decision mapping.
    """

    results: Dict[str, Dict[str, object]] = {}

    results[Representation.PARITY.value] = (
        evaluate_fixed_parity(
            validation_records=validation_records,
            test_records=test_records,
            y_validation=y_validation,
            y_test=y_test,
        )
    )

    learned_representations = [
        Representation.LEARNED_PARITY,
        Representation.SINGLE_Z,
        Representation.ALL_Z,
        Representation.ZENG_AMM,
        Representation.PROBABILITIES,
        Representation.Z_ZZ,
        Representation.PROB_Z_ZZ,
        Representation.FULL,
    ]

    for representation in learned_representations:
        results[representation.value] = (
            evaluate_learned_representation(
                representation=representation,
                train_records=train_records,
                validation_records=validation_records,
                test_records=test_records,
                y_train=y_train,
                y_validation=y_validation,
                y_test=y_test,
                num_qubits=num_qubits,
            )
        )

    return results
