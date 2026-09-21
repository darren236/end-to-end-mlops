FROM python:3.12-slim AS trainer

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[train]"
RUN python -m mlops_demo.train \
    --artifacts-dir /build/artifacts \
    --tracking-dir /tmp/mlruns

FROM python:3.12-slim AS ui-runtime

ENV MODEL_PATH=/app/artifacts/model.joblib \
    METRICS_PATH=/app/artifacts/metrics.json \
    PREDICTION_LOG_PATH=/app/logs/predictions.jsonl \
    TRACKING_DIR=/app/mlruns \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY .streamlit ./.streamlit
RUN python -m pip install --no-cache-dir ".[ui]" \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts /app/logs /app/mlruns \
    && chown -R appuser:appuser /app
COPY --from=trainer --chown=appuser:appuser /build/artifacts /app/artifacts

USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"

CMD ["streamlit", "run", "src/mlops_demo/ui.py", "--server.address=0.0.0.0", "--server.port=8501"]

FROM python:3.12-slim AS runtime

ENV MODEL_PATH=/app/artifacts/model.joblib \
    PREDICTION_LOG_PATH=/app/logs/predictions.jsonl \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts /app/logs \
    && chown -R appuser:appuser /app
COPY --from=trainer --chown=appuser:appuser /build/artifacts /app/artifacts

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "mlops_demo.api:app", "--host", "0.0.0.0", "--port", "8000"]
