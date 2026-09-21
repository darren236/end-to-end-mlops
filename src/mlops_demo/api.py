"""FastAPI inference service with prediction logging and Prometheus metrics."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from mlops_demo import __version__
from mlops_demo.model import ModelBundle, load_bundle
from mlops_demo.telemetry import append_prediction, build_prediction_record


class IrisFeatures(BaseModel):
    """Validated measurements accepted by the prediction endpoint."""

    sepal_length_cm: float = Field(ge=0, le=20, examples=[5.1])
    sepal_width_cm: float = Field(ge=0, le=20, examples=[3.5])
    petal_length_cm: float = Field(ge=0, le=20, examples=[1.4])
    petal_width_cm: float = Field(ge=0, le=20, examples=[0.2])


class PredictionResponse(BaseModel):
    predicted_class: str
    class_probabilities: dict[str, float]
    model_version: str


def create_app(
    model_path: Path | str | None = None,
    prediction_log_path: Path | str | None = None,
) -> FastAPI:
    """Build an isolated application instance for production or tests."""
    resolved_model_path = Path(
        model_path or os.getenv("MODEL_PATH", "artifacts/model.joblib")
    )
    resolved_log_path = Path(
        prediction_log_path or os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl")
    )

    registry = CollectorRegistry()
    request_count = Counter(
        "mlops_api_requests_total",
        "HTTP requests handled by the model API.",
        ("method", "path", "status"),
        registry=registry,
    )
    request_latency = Histogram(
        "mlops_api_request_duration_seconds",
        "HTTP request latency in seconds.",
        ("method", "path"),
        registry=registry,
    )
    prediction_count = Counter(
        "mlops_model_predictions_total",
        "Predictions by output class.",
        ("predicted_class",),
        registry=registry,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.model_bundle = load_bundle(resolved_model_path)
        yield

    application = FastAPI(
        title="Iris MLOps API",
        version=__version__,
        description="A monitored prediction API for the end-to-end MLOps demo.",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def observe_request(request: Request, call_next: object) -> Response:
        started = perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            request_count.labels(request.method, request.url.path, status).inc()
            request_latency.labels(request.method, request.url.path).observe(
                perf_counter() - started
            )

    @application.get("/", tags=["service"])
    def root() -> dict[str, str]:
        return {
            "service": "Iris MLOps API",
            "documentation": "/docs",
            "health": "/health",
            "metrics": "/metrics",
        }

    @application.get("/health", tags=["service"])
    def health(request: Request) -> dict[str, object]:
        bundle: ModelBundle = request.app.state.model_bundle
        return {
            "status": "ok",
            "model_version": bundle.model_version,
            "model_trained_at": bundle.trained_at,
        }

    @application.post("/predict", response_model=PredictionResponse, tags=["model"])
    def predict(payload: IrisFeatures, request: Request) -> PredictionResponse:
        bundle: ModelBundle = request.app.state.model_bundle
        feature_values = payload.model_dump()
        probabilities = bundle.predict_proba(
            [feature_values[name] for name in bundle.feature_names]
        )
        predicted_index = int(probabilities.argmax())
        predicted_class = bundle.target_names[predicted_index]
        probability_map = {
            name: round(float(probability), 6)
            for name, probability in zip(bundle.target_names, probabilities, strict=True)
        }
        prediction_count.labels(predicted_class).inc()
        append_prediction(
            resolved_log_path,
            build_prediction_record(
                bundle,
                feature_values,
                predicted_class,
                probability_map,
            ),
        )
        return PredictionResponse(
            predicted_class=predicted_class,
            class_probabilities=probability_map,
            model_version=bundle.model_version,
        )

    @application.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
