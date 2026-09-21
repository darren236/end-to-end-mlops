import json
from pathlib import Path

import pytest

from mlops_demo.model import load_bundle
from mlops_demo.monitor import calculate_drift
from mlops_demo.ui_support import (
    load_training_summary,
    predict_features,
    simulate_prediction_records,
)


def test_prediction_and_drift_simulation(
    trained_artifact: tuple[Path, dict[str, object]],
) -> None:
    model_path, _ = trained_artifact
    bundle = load_bundle(model_path)
    features = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
    }

    predicted_class, probabilities = predict_features(bundle, features)
    records = simulate_prediction_records(bundle, sample_count=50, mean_shift_std=1.5)
    report = calculate_drift(bundle, records)

    assert predicted_class == "setosa"
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert report["drift_detected"] is True


def test_summary_loading_and_simulation_validation(tmp_path: Path) -> None:
    summary_path = tmp_path / "metrics.json"
    assert load_training_summary(summary_path) is None

    summary_path.write_text(json.dumps({"accuracy": 0.95}), encoding="utf-8")
    assert load_training_summary(summary_path) == {"accuracy": 0.95}

    with pytest.raises(ValueError, match="positive"):
        simulate_prediction_records(object(), sample_count=0, mean_shift_std=0.0)

