import json
from pathlib import Path

from fastapi.testclient import TestClient

from mlops_demo.api import create_app


def test_health_prediction_logging_and_metrics(
    trained_artifact: tuple[Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    model_path, _ = trained_artifact
    log_path = tmp_path / "predictions.jsonl"
    application = create_app(model_path=model_path, prediction_log_path=log_path)

    with TestClient(application) as client:
        root = client.get("/")
        health = client.get("/health")
        prediction = client.post(
            "/predict",
            json={
                "sepal_length_cm": 5.1,
                "sepal_width_cm": 3.5,
                "petal_length_cm": 1.4,
                "petal_width_cm": 0.2,
            },
        )
        metrics = client.get("/metrics")

    assert root.json()["documentation"] == "/docs"
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert prediction.status_code == 200
    assert prediction.json()["predicted_class"] == "setosa"
    assert abs(sum(prediction.json()["class_probabilities"].values()) - 1.0) < 1e-5
    assert "mlops_model_predictions_total" in metrics.text

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["predicted_class"] == "setosa"
    assert record["features"]["sepal_length_cm"] == 5.1


def test_prediction_validation(
    trained_artifact: tuple[Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    model_path, _ = trained_artifact
    application = create_app(model_path=model_path, prediction_log_path=tmp_path / "log.jsonl")

    with TestClient(application) as client:
        response = client.post(
            "/predict",
            json={
                "sepal_length_cm": -1,
                "sepal_width_cm": 3.5,
                "petal_length_cm": 1.4,
                "petal_width_cm": 0.2,
            },
        )

    assert response.status_code == 422
