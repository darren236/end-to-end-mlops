"""Generate a transparent feature-drift report from prediction logs."""

import argparse
import json
from pathlib import Path

import numpy as np

from mlops_demo.model import ModelBundle, load_bundle


def read_prediction_log(path: Path) -> list[dict[str, object]]:
    """Read newline-delimited prediction records."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction log not found at {path}. Send API requests first.")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("Prediction log is empty")
    return records


def calculate_drift(
    bundle: ModelBundle,
    records: list[dict[str, object]],
    threshold: float = 1.0,
) -> dict[str, object]:
    """Compare live means with training means in baseline standard deviations."""
    if threshold <= 0:
        raise ValueError("Drift threshold must be positive")

    features: dict[str, dict[str, object]] = {}
    for name in bundle.feature_names:
        values = np.asarray([record["features"][name] for record in records], dtype=float)
        baseline = bundle.baseline[name]
        baseline_std = max(float(baseline["std"]), 1e-12)
        live_mean = float(values.mean())
        standardized_shift = abs(live_mean - float(baseline["mean"])) / baseline_std
        features[name] = {
            "training_mean": round(float(baseline["mean"]), 6),
            "live_mean": round(live_mean, 6),
            "mean_shift_std": round(standardized_shift, 6),
            "drifted": standardized_shift >= threshold,
        }

    return {
        "record_count": len(records),
        "threshold_std": threshold,
        "drift_detected": any(result["drifted"] for result in features.values()),
        "features": features,
        "model_version": bundle.model_version,
        "run_id": bundle.run_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--log", type=Path, default=Path("logs/predictions.jsonl"))
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = calculate_drift(
        load_bundle(args.model),
        read_prediction_log(args.log),
        threshold=args.threshold,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.fail_on_drift and report["drift_detected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

