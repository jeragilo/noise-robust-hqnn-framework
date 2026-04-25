"""
Dataset utilities for the Noise-Robust HQNN framework.

This module provides standardized dataset loading and preprocessing
for synthetic and named benchmark datasets used in the thesis.

The purpose of this module is to make experiments reproducible and
consistent across demos and benchmark pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from sklearn.datasets import make_classification, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


@dataclass
class DatasetBundle:
    """
    Container for a processed dataset.

    Attributes:
        name: Human-readable dataset name.
        X_train: Training features.
        X_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.
        feature_names: Optional feature names.
        description: Short dataset description.
    """
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: Optional[list[str]] = None
    description: str = ""


def preprocess_for_quantum(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_features: int = 4,
    test_size: float = 0.30,
    random_state: int = 42,
    use_pca: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard preprocessing for quantum-compatible experiments.

    Steps:
    1. Train/test split.
    2. Standardize features.
    3. Optionally reduce dimensionality using PCA.
    4. Select or reduce features to match the number of qubits/features.

    Why this matters:
    Quantum circuits usually require a small number of input features.
    This function creates a consistent preprocessing pipeline so that
    all models are compared under the same conditions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if use_pca or X_train.shape[1] > n_features:
        pca = PCA(n_components=n_features, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    else:
        X_train = X_train[:, :n_features]
        X_test = X_test[:, :n_features]

    return X_train, X_test, y_train, y_test


def load_synthetic_binary(
    *,
    n_samples: int = 200,
    n_features: int = 4,
    n_informative: int = 2,
    class_sep: float = 1.5,
    random_state: int = 42,
) -> DatasetBundle:
    """
    Load a controlled synthetic binary classification dataset.

    This is useful for baseline experiments where we want a controlled
    environment before moving to named benchmark datasets.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=class_sep,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = preprocess_for_quantum(
        X,
        y,
        n_features=n_features,
        random_state=random_state,
    )

    return DatasetBundle(
        name="Synthetic Binary Classification",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=[f"feature_{i}" for i in range(n_features)],
        description="Controlled synthetic binary dataset generated with sklearn.make_classification.",
    )


def load_iris_binary(
    *,
    class_a: int = 0,
    class_b: int = 1,
    n_features: int = 4,
    random_state: int = 42,
) -> DatasetBundle:
    """
    Load a binary version of the Iris dataset.

    The full Iris dataset has three classes. For binary classification,
    this function selects two classes and maps them to labels 0 and 1.

    This gives the thesis a named, reproducible benchmark dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target

    mask = (y == class_a) | (y == class_b)
    X = X[mask]
    y = y[mask]

    y = np.where(y == class_a, 0, 1)

    X_train, X_test, y_train, y_test = preprocess_for_quantum(
        X,
        y,
        n_features=n_features,
        random_state=random_state,
        use_pca=False,
    )

    return DatasetBundle(
        name=f"Iris Binary ({data.target_names[class_a]} vs {data.target_names[class_b]})",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(data.feature_names[:n_features]),
        description="Binary classification version of the Iris benchmark dataset.",
    )


def load_wdbc(
    *,
    n_features: int = 4,
    random_state: int = 42,
) -> DatasetBundle:
    """
    Load the Wisconsin Diagnostic Breast Cancer dataset.

    This is a real, named medical classification benchmark.
    The original dataset has 30 features, so PCA is used to reduce
    it to a quantum-compatible number of features.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = preprocess_for_quantum(
        X,
        y,
        n_features=n_features,
        random_state=random_state,
        use_pca=True,
    )

    return DatasetBundle(
        name="Wisconsin Diagnostic Breast Cancer",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=[f"PCA_component_{i}" for i in range(n_features)],
        description=(
            "Named medical benchmark dataset from sklearn. "
            "PCA is applied to reduce 30 original features to a quantum-compatible feature size."
        ),
    )
