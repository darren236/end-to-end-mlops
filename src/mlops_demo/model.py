"""Serializable model artifact and prediction helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    """Everything needed to reproduce inference and monitoring."""

    estimator: Any
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    metrics: dict[str, float]
    baseline: dict[str, dict[str, float]]
    model_version: str
    trained_at: str
    run_id: str

    def predict_proba(self, values: list[float]) -> np.ndarray:
        """Predict one row after enforcing the saved feature contract."""
        if len(values) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, received {len(values)}")
        matrix = pd.DataFrame([np.asarray(values, dtype=float)], columns=self.feature_names)
        return np.asarray(self.estimator.predict_proba(matrix)[0], dtype=float)


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    """Persist a model bundle atomically enough for the local demo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    joblib.dump(bundle, temporary_path)
    temporary_path.replace(path)


def load_bundle(path: Path) -> ModelBundle:
    """Load and type-check a model artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Run `make train` first.")
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Artifact is not a ModelBundle")
    return bundle
