"""Shared prediction-event creation and local persistence."""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from mlops_demo.model import ModelBundle

_LOG_LOCK = threading.Lock()


def build_prediction_record(
    bundle: ModelBundle,
    features: dict[str, float],
    predicted_class: str,
    probabilities: dict[str, float],
) -> dict[str, object]:
    """Create the structured event consumed by the drift monitor."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "features": features,
        "predicted_class": predicted_class,
        "class_probabilities": probabilities,
        "model_version": bundle.model_version,
        "run_id": bundle.run_id,
    }


def append_prediction(path: Path, record: dict[str, object]) -> None:
    """Append one event safely within a single process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")

