from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from mlops_demo.ui import STAGES


def test_guided_ui_walkthrough(
    trained_artifact: tuple[Path, dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, _ = trained_artifact
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("METRICS_PATH", str(model_path.with_name("metrics.json")))
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    monkeypatch.setenv("TRACKING_DIR", str(tmp_path / "mlruns"))

    ui_path = Path(__file__).parents[1] / "src" / "mlops_demo" / "ui.py"
    app = AppTest.from_file(ui_path, default_timeout=30).run()

    assert not app.exception
    assert any("From data to a monitored prediction" in block.value for block in app.markdown)
    assert len(app.metric) == 3

    app.sidebar.radio[0].set_value(STAGES[1]).run()
    assert not app.exception
    assert app.button[0].label == "Run training pipeline"

    app.sidebar.radio[0].set_value(STAGES[2]).run()
    assert not app.exception
    assert any(metric.label == "Held-out accuracy" for metric in app.metric)

    app.sidebar.radio[0].set_value(STAGES[3]).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Predicted species" in message.value for message in app.success)

    app.sidebar.radio[0].set_value(STAGES[4]).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Drift detected" in message.value for message in app.error)
