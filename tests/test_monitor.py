import json
from pathlib import Path

import pytest

from mlops_demo.model import load_bundle
from mlops_demo.monitor import calculate_drift, read_prediction_log


def test_drift_report_flags_shifted_inputs(
    trained_artifact: tuple[Path, dict[str, object]],
) -> None:
    model_path, _ = trained_artifact
    bundle = load_bundle(model_path)
    record = {
        "features": {
            name: stats["mean"] + 2 * stats["std"] for name, stats in bundle.baseline.items()
        }
    }

    report = calculate_drift(bundle, [record] * 5, threshold=1.0)

    assert report["drift_detected"] is True
    assert all(result["drifted"] for result in report["features"].values())


def test_drift_report_accepts_baseline_inputs(
    trained_artifact: tuple[Path, dict[str, object]],
) -> None:
    model_path, _ = trained_artifact
    bundle = load_bundle(model_path)
    record = {"features": {name: stats["mean"] for name, stats in bundle.baseline.items()}}

    report = calculate_drift(bundle, [record] * 5)

    assert report["drift_detected"] is False


def test_prediction_log_errors_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Send API requests first"):
        read_prediction_log(tmp_path / "missing.jsonl")

    empty_log = tmp_path / "empty.jsonl"
    empty_log.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_prediction_log(empty_log)


def test_prediction_log_round_trip(tmp_path: Path) -> None:
    log_path = tmp_path / "predictions.jsonl"
    record = {"features": {"sepal_length_cm": 5.1}}
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert read_prediction_log(log_path) == [record]


def test_drift_threshold_must_be_positive(
    trained_artifact: tuple[Path, dict[str, object]],
) -> None:
    model_path, _ = trained_artifact
    bundle = load_bundle(model_path)

    with pytest.raises(ValueError, match="positive"):
        calculate_drift(bundle, [], threshold=0)
