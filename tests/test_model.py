from pathlib import Path

import joblib
import pytest

from mlops_demo.model import load_bundle


def test_bundle_rejects_wrong_feature_count(
    trained_artifact: tuple[Path, dict[str, object]],
) -> None:
    model_path, _ = trained_artifact
    bundle = load_bundle(model_path)

    with pytest.raises(ValueError, match="Expected 4 features"):
        bundle.predict_proba([1.0, 2.0])


def test_load_bundle_reports_missing_or_invalid_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Run `make train` first"):
        load_bundle(tmp_path / "missing.joblib")

    invalid_path = tmp_path / "invalid.joblib"
    joblib.dump({"not": "a model"}, invalid_path)
    with pytest.raises(TypeError, match="not a ModelBundle"):
        load_bundle(invalid_path)

