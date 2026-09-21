from pathlib import Path

import pytest

from mlops_demo.train import TrainingConfig, train_model


@pytest.fixture(scope="session")
def trained_artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    workspace = tmp_path_factory.mktemp("training")
    config = TrainingConfig(
        artifacts_dir=workspace / "artifacts",
        tracking_dir=workspace / "mlruns",
    )
    summary = train_model(config)
    return config.artifacts_dir / "model.joblib", summary

