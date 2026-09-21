"""Reproducible model training, evaluation, and experiment tracking."""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_demo import __version__
from mlops_demo.data import FEATURE_NAMES, load_dataset
from mlops_demo.model import ModelBundle, save_bundle


@dataclass(frozen=True)
class TrainingConfig:
    """Parameters that define a reproducible training run."""

    artifacts_dir: Path = Path("artifacts")
    tracking_dir: Path = Path("mlruns")
    experiment_name: str = "iris-classification"
    test_size: float = 0.2
    random_state: int = 42
    max_iter: int = 500
    accuracy_threshold: float = 0.90


def _baseline_statistics(features: object) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(features[name].mean()),
            "std": float(features[name].std(ddof=0)),
        }
        for name in FEATURE_NAMES
    }


def _configure_tracking(tracking_dir: Path, experiment_name: str) -> str:
    """Configure a supported local SQLite store and return its experiment ID."""
    resolved_tracking_dir = tracking_dir.resolve()
    resolved_tracking_dir.mkdir(parents=True, exist_ok=True)
    database_path = resolved_tracking_dir / "mlflow.db"
    artifact_path = resolved_tracking_dir / "artifacts"
    artifact_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{database_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_path.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def train_model(config: TrainingConfig) -> dict[str, object]:
    """Train, evaluate, gate, save, and track one model run."""
    dataset = load_dataset()
    train_features, test_features, train_target, test_target = train_test_split(
        dataset.features,
        dataset.target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=dataset.target,
    )

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=config.max_iter, random_state=config.random_state),
            ),
        ]
    )

    experiment_id = _configure_tracking(config.tracking_dir, config.experiment_name)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"logistic-regression-seed-{config.random_state}",
    ) as run:
        pipeline.fit(train_features, train_target)
        predictions = pipeline.predict(test_features)
        metrics = {
            "accuracy": float(accuracy_score(test_target, predictions)),
            "f1_macro": float(f1_score(test_target, predictions, average="macro")),
        }
        params = {
            "dataset": "sklearn.datasets.load_iris",
            "dataset_rows": len(dataset.features),
            "test_size": config.test_size,
            "random_state": config.random_state,
            "max_iter": config.max_iter,
            "accuracy_threshold": config.accuracy_threshold,
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        passed = metrics["accuracy"] >= config.accuracy_threshold
        mlflow.set_tag("quality_gate", "passed" if passed else "failed")
        if not passed:
            raise RuntimeError(
                f"Quality gate failed: accuracy {metrics['accuracy']:.3f} "
                f"is below {config.accuracy_threshold:.3f}"
            )

        trained_at = datetime.now(UTC).isoformat()
        bundle = ModelBundle(
            estimator=pipeline,
            feature_names=FEATURE_NAMES,
            target_names=dataset.target_names,
            metrics=metrics,
            baseline=_baseline_statistics(train_features),
            model_version=__version__,
            trained_at=trained_at,
            run_id=run.info.run_id,
        )
        model_path = config.artifacts_dir / "model.joblib"
        save_bundle(bundle, model_path)

        summary: dict[str, object] = {
            **metrics,
            "quality_gate": "passed",
            "model_path": str(model_path),
            "model_version": __version__,
            "run_id": run.info.run_id,
            "trained_at": trained_at,
            "config": {key: str(value) for key, value in asdict(config).items()},
        }
        metrics_path = config.artifacts_dir / "metrics.json"
        metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(model_path), artifact_path="release")
        mlflow.log_artifact(str(metrics_path), artifact_path="release")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--tracking-dir", type=Path, default=Path("mlruns"))
    parser.add_argument("--experiment-name", default="iris-classification")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--accuracy-threshold", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        artifacts_dir=args.artifacts_dir,
        tracking_dir=args.tracking_dir,
        experiment_name=args.experiment_name,
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
        accuracy_threshold=args.accuracy_threshold,
    )
    print(json.dumps(train_model(config), indent=2))


if __name__ == "__main__":
    main()
