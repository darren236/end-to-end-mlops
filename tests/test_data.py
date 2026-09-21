from dataclasses import replace

import pytest

from mlops_demo.data import FEATURE_NAMES, load_dataset, validate_dataset


def test_dataset_matches_contract() -> None:
    dataset = load_dataset()

    assert dataset.features.shape == (150, 4)
    assert tuple(dataset.features.columns) == FEATURE_NAMES
    assert dataset.target_names == ("setosa", "versicolor", "virginica")


def test_validation_rejects_missing_values() -> None:
    dataset = load_dataset()
    invalid_features = dataset.features.copy()
    invalid_features.iloc[0, 0] = None

    with pytest.raises(ValueError, match="missing"):
        validate_dataset(replace(dataset, features=invalid_features))

