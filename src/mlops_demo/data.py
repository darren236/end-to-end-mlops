"""Dataset loading and validation."""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_iris

FEATURE_NAMES = (
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
)


@dataclass(frozen=True)
class Dataset:
    """Validated features, target, and label metadata."""

    features: pd.DataFrame
    target: pd.Series
    target_names: tuple[str, ...]


def load_dataset() -> Dataset:
    """Load Iris from scikit-learn and normalize its feature names."""
    raw = load_iris(as_frame=True)
    features = raw.data.copy()
    features.columns = list(FEATURE_NAMES)
    target = raw.target.astype(int).copy()
    target_names = tuple(str(name) for name in raw.target_names)

    dataset = Dataset(features=features, target=target, target_names=target_names)
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: Dataset) -> None:
    """Fail fast when the training data violates the expected contract."""
    if tuple(dataset.features.columns) != FEATURE_NAMES:
        raise ValueError("Unexpected feature schema")
    if len(dataset.features) != len(dataset.target):
        raise ValueError("Feature and target row counts differ")
    if len(dataset.features) < 100:
        raise ValueError("Dataset is unexpectedly small")
    if dataset.features.isna().any().any() or dataset.target.isna().any():
        raise ValueError("Dataset contains missing values")
    if (dataset.features < 0).any().any():
        raise ValueError("Iris measurements cannot be negative")
    if set(dataset.target.unique()) != set(range(len(dataset.target_names))):
        raise ValueError("Target labels do not match target names")

