"""Framework-independent helpers for the guided user interface."""

import json
from pathlib import Path

import numpy as np

from mlops_demo.model import ModelBundle


def load_training_summary(path: Path) -> dict[str, object] | None:
    """Load evaluation metadata when a completed training run is available."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def predict_features(
    bundle: ModelBundle,
    features: dict[str, float],
) -> tuple[str, dict[str, float]]:
    """Run one named-feature prediction and return class probabilities."""
    probabilities = bundle.predict_proba([features[name] for name in bundle.feature_names])
    predicted_index = int(probabilities.argmax())
    probability_map = {
        name: round(float(probability), 6)
        for name, probability in zip(bundle.target_names, probabilities, strict=True)
    }
    return bundle.target_names[predicted_index], probability_map


def simulate_prediction_records(
    bundle: ModelBundle,
    sample_count: int,
    mean_shift_std: float,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Create deterministic synthetic traffic for an explainable drift demo."""
    if sample_count < 1:
        raise ValueError("Sample count must be positive")
    random = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for _ in range(sample_count):
        features = {}
        for name, baseline in bundle.baseline.items():
            standard_deviation = float(baseline["std"])
            value = (
                float(baseline["mean"])
                + mean_shift_std * standard_deviation
                + random.normal(0, standard_deviation * 0.15)
            )
            features[name] = max(0.0, float(value))
        records.append({"features": features, "model_version": bundle.model_version})
    return records

