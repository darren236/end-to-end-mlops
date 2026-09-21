from pathlib import Path

import pytest

from mlops_demo.model import load_bundle
from mlops_demo.train import TrainingConfig, train_model


def test_training_creates_approved_bundle(
    trained_artifact: tuple[Path, dict[str, object]],
) -> None:
    model_path, summary = trained_artifact
    bundle = load_bundle(model_path)

    assert summary["quality_gate"] == "passed"
    assert summary["accuracy"] >= 0.90
    assert bundle.metrics["accuracy"] == summary["accuracy"]
    assert bundle.run_id == summary["run_id"]
    assert len(bundle.baseline) == 4


def test_quality_gate_blocks_weak_release(tmp_path: Path) -> None:
    config = TrainingConfig(
        artifacts_dir=tmp_path / "artifacts",
        tracking_dir=tmp_path / "mlruns",
        accuracy_threshold=1.01,
    )

    with pytest.raises(RuntimeError, match="Quality gate failed"):
        train_model(config)

    assert not (config.artifacts_dir / "model.joblib").exists()

