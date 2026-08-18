import numpy as np

from framework.datasets import load_iris_binary, load_synthetic_binary, load_wdbc


def test_synthetic_dataset_shapes_and_labels():
    bundle = load_synthetic_binary(n_samples=100, n_features=4, random_state=42)

    assert bundle.X_train.shape[1] == 4
    assert bundle.X_test.shape[1] == 4
    assert set(np.unique(bundle.y_train)).issubset({0, 1})
    assert set(np.unique(bundle.y_test)).issubset({0, 1})


def test_synthetic_dataset_is_reproducible():
    first = load_synthetic_binary(n_samples=100, n_features=4, random_state=7)
    second = load_synthetic_binary(n_samples=100, n_features=4, random_state=7)

    assert np.allclose(first.X_train, second.X_train)
    assert np.array_equal(first.y_train, second.y_train)
    assert np.allclose(first.X_test, second.X_test)
    assert np.array_equal(first.y_test, second.y_test)


def test_iris_binary_dataset():
    bundle = load_iris_binary(class_a=0, class_b=1, n_features=4, random_state=42)

    assert bundle.X_train.shape[1] == 4
    assert bundle.X_test.shape[1] == 4
    assert set(np.unique(bundle.y_train)) == {0, 1}
    assert set(np.unique(bundle.y_test)) == {0, 1}


def test_wdbc_uses_requested_feature_dimension():
    bundle = load_wdbc(n_features=4, random_state=42)

    assert bundle.X_train.shape[1] == 4
    assert bundle.X_test.shape[1] == 4
    assert set(np.unique(bundle.y_train)).issubset({0, 1})
    assert set(np.unique(bundle.y_test)).issubset({0, 1})
