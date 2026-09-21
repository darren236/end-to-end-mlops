"""Streamlit walkthrough of the complete MLOps lifecycle."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from mlops_demo.data import load_dataset
from mlops_demo.model import ModelBundle, load_bundle
from mlops_demo.monitor import calculate_drift, read_prediction_log
from mlops_demo.telemetry import append_prediction, build_prediction_record
from mlops_demo.ui_support import (
    load_training_summary,
    predict_features,
    simulate_prediction_records,
)

STAGES = (
    "1 · Explore data",
    "2 · Train model",
    "3 · Review evaluation",
    "4 · Try a prediction",
    "5 · Check drift",
)

FEATURE_LABELS = {
    "sepal_length_cm": "Sepal length (cm)",
    "sepal_width_cm": "Sepal width (cm)",
    "petal_length_cm": "Petal length (cm)",
    "petal_width_cm": "Petal width (cm)",
}

FEATURE_DEFAULTS = {
    "sepal_length_cm": 5.1,
    "sepal_width_cm": 3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm": 0.2,
}

APP_STYLES = """
<style>
    :root {
        --surface-0: #080a09;
        --surface-1: #0d100e;
        --surface-2: #121612;
        --surface-3: #171c18;
        --line: #303630;
        --line-strong: #475047;
        --text: #f4f7f4;
        --muted: #a4aca5;
        --green: #76b900;
        --green-bright: #8fd600;
        --green-soft: rgba(118, 185, 0, 0.13);
    }
    html, body, [class*="css"] {
        font-feature-settings: "ss01" 1, "cv02" 1;
    }
    .block-container {
        max-width: 1200px;
        padding-top: 2.35rem;
        padding-bottom: 5rem;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 10% -5%, rgba(118, 185, 0, 0.13), transparent 31rem),
            radial-gradient(circle at 92% 8%, rgba(48, 78, 52, 0.16), transparent 26rem),
            linear-gradient(rgba(118, 185, 0, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(118, 185, 0, 0.025) 1px, transparent 1px),
            var(--surface-0);
        background-size: auto, auto, 48px 48px, 48px 48px, auto;
    }
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(118, 185, 0, 0.055), transparent 13rem),
            #0b0e0c;
        border-right: 1px solid #2a302b;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 1.4rem;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none;
    }
    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.025em;
    }
    h2 {
        font-size: 2rem !important;
        margin-top: 0.25rem !important;
    }
    h3 {
        font-size: 1.25rem !important;
    }
    p, label, [data-testid="stCaptionContainer"] {
        color: #c5cbc6;
    }
    a {
        color: var(--green-bright) !important;
        text-decoration-color: rgba(143, 214, 0, 0.45) !important;
    }
    hr {
        border-color: #292f2a !important;
    }
    [data-testid="stMetric"] {
        position: relative;
        background: linear-gradient(145deg, #121612 0%, #0e110f 100%);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 1rem 1.15rem 1.1rem;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
        overflow: hidden;
    }
    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: var(--green);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.69rem;
        font-weight: 700;
    }
    [data-testid="stMetricValue"] {
        color: var(--text);
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 6px;
        overflow: hidden;
    }
    [data-testid="stVegaLiteChart"] {
        background: rgba(15, 18, 16, 0.78);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 0.35rem;
    }
    [data-testid="stExpander"] {
        background: rgba(15, 18, 16, 0.82);
        border: 1px solid var(--line) !important;
        border-radius: 6px !important;
    }
    [data-testid="stAlert"] {
        border-radius: 4px;
        border-width: 1px;
    }
    [data-baseweb="slider"] [role="slider"] {
        box-shadow: 0 0 0 2px #080a09, 0 0 0 3px var(--green);
    }
    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div {
        background: #0d100e;
        border-color: var(--line);
    }
    .stButton > button {
        border: 1px solid var(--green);
        border-radius: 4px;
        color: var(--text);
        font-weight: 700;
        min-height: 2.9rem;
        letter-spacing: 0.01em;
        transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
    }
    .stButton > button:hover {
        border-color: var(--green-bright);
        color: var(--green-bright);
    }
    .stButton > button[kind="primary"] {
        background: var(--green);
        color: #080a09;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--green-bright);
        border-color: var(--green-bright);
        color: #080a09;
    }
    [data-testid="stCodeBlock"] {
        border: 1px solid var(--line);
        border-radius: 5px;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-left: 2px solid transparent;
        margin: 0.08rem 0;
        padding: 0.34rem 0.45rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: var(--green-soft);
        border-left-color: var(--green);
        color: var(--text);
    }
    .mlops-side-brand {
        align-items: center;
        display: flex;
        gap: 0.65rem;
        margin: 0.25rem 0 1.9rem;
    }
    .mlops-side-mark {
        align-items: center;
        background: var(--green);
        color: #080a09;
        display: inline-flex;
        font-size: 0.7rem;
        font-weight: 900;
        height: 1.8rem;
        justify-content: center;
        letter-spacing: -0.05em;
        width: 1.8rem;
    }
    .mlops-side-name {
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }
    .mlops-side-kicker {
        color: #778079;
        font-size: 0.62rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    .mlops-status {
        background: #0f150e;
        border: 1px solid #34452c;
        border-left: 3px solid var(--green);
        border-radius: 4px;
        margin: 1.15rem 0 0.75rem;
        padding: 0.85rem 0.9rem;
    }
    .mlops-status-label {
        color: var(--green-bright);
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }
    .mlops-status-value {
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 650;
        margin-top: 0.25rem;
    }
    .mlops-hero {
        position: relative;
        overflow: hidden;
        color: white;
        padding: 2.25rem 2.45rem 2.35rem;
        margin: 0 0 2.15rem 0;
        border: 1px solid #3a423b;
        border-radius: 8px;
        background:
            linear-gradient(110deg, rgba(118, 185, 0, 0.08), transparent 46%),
            radial-gradient(circle at 91% 18%, rgba(118, 185, 0, 0.17), transparent 12rem),
            #0c0f0d;
        box-shadow: 0 20px 54px rgba(0, 0, 0, 0.32);
    }
    .mlops-hero::before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 3px;
        background: linear-gradient(
            90deg, var(--green) 0%, var(--green-bright) 38%, transparent 78%
        );
    }
    .mlops-hero::after {
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        right: -75px;
        top: -120px;
        border: 1px solid rgba(118, 185, 0, 0.18);
        transform: rotate(24deg);
        box-shadow: 0 0 0 32px rgba(118, 185, 0, 0.025), 0 0 0 68px rgba(118, 185, 0, 0.018);
        pointer-events: none;
    }
    .mlops-hero-meta {
        align-items: center;
        display: flex;
        justify-content: space-between;
        position: relative;
        z-index: 1;
    }
    .mlops-eyebrow {
        color: var(--green-bright);
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .mlops-live {
        align-items: center;
        color: #aeb6af;
        display: flex;
        font-size: 0.68rem;
        font-weight: 700;
        gap: 0.45rem;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }
    .mlops-live-dot {
        background: var(--green-bright);
        border-radius: 50%;
        box-shadow: 0 0 0 4px rgba(118, 185, 0, 0.12);
        height: 0.42rem;
        width: 0.42rem;
    }
    .mlops-hero h1 {
        color: white;
        font-size: clamp(2.2rem, 4vw, 3.25rem);
        line-height: 1.02;
        letter-spacing: -0.045em;
        margin: 1.45rem 0 0.8rem;
        max-width: 800px;
        position: relative;
        z-index: 1;
    }
    .mlops-hero p {
        color: #c0c7c1;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 775px;
        margin-bottom: 1.4rem;
        position: relative;
        z-index: 1;
    }
    .mlops-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        position: relative;
        z-index: 1;
    }
    .mlops-pill {
        background: #111512;
        border: 1px solid #3b443c;
        border-radius: 3px;
        color: #dce1dc;
        font-size: 0.68rem;
        font-weight: 650;
        letter-spacing: 0.05em;
        padding: 0.37rem 0.67rem;
        text-transform: uppercase;
    }
</style>
"""


def _model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))


def _metrics_path() -> Path:
    return Path(os.getenv("METRICS_PATH", str(_model_path().with_name("metrics.json"))))


def _prediction_log_path() -> Path:
    return Path(os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl"))


def _load_model_if_available() -> ModelBundle | None:
    path = _model_path()
    return load_bundle(path) if path.exists() else None


def _header() -> None:
    st.markdown(
        """
        <div class="mlops-hero">
          <div class="mlops-hero-meta">
            <div class="mlops-eyebrow">Interactive reference project</div>
            <div class="mlops-live"><span class="mlops-live-dot"></span>Pipeline online</div>
          </div>
          <h1>From data to a monitored prediction</h1>
          <p>Walk through a complete, inspectable ML lifecycle—from validation and tracked
          training to an approved prediction and an observable drift result.</p>
          <div class="mlops-pills">
            <span class="mlops-pill">scikit-learn</span>
            <span class="mlops-pill">MLflow</span>
            <span class="mlops-pill">FastAPI</span>
            <span class="mlops-pill">Prometheus</span>
            <span class="mlops-pill">Docker</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar(bundle: ModelBundle | None) -> str:
    st.sidebar.markdown(
        """
        <div class="mlops-side-brand">
          <span class="mlops-side-mark">ML</span>
          <div>
            <div class="mlops-side-name">PIPELINE LAB</div>
            <div class="mlops-side-kicker">End-to-end MLOps</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.subheader("Pipeline stages")
    stage = st.sidebar.radio(
        "Pipeline stage",
        STAGES,
        key="pipeline_stage",
        label_visibility="collapsed",
    )
    current_step = STAGES.index(stage) + 1
    st.sidebar.progress(current_step / len(STAGES), text=f"Step {current_step} of {len(STAGES)}")
    st.sidebar.divider()
    if bundle:
        st.sidebar.markdown(
            f"""
            <div class="mlops-status">
              <div class="mlops-status-label">● Approved model</div>
              <div class="mlops-status-value">Version {bundle.model_version}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.caption(f"Run `{bundle.run_id[:8]}` · accuracy {bundle.metrics['accuracy']:.1%}")
    else:
        st.sidebar.warning("No approved model yet")
        st.sidebar.caption("Go to step 2 and run the training pipeline.")
    st.sidebar.markdown("[View source on GitHub](https://github.com/darren236/end-to-end-mlops)")
    return stage


def _explore_data() -> None:
    st.header("1. Explore and validate the dataset")
    st.write(
        "The pipeline starts with a known data contract: four non-negative measurements and "
        "one of three Iris species. Training stops immediately if that contract is violated."
    )
    dataset = load_dataset()
    frame = dataset.features.copy()
    frame["species"] = dataset.target.map(dict(enumerate(dataset.target_names)))

    sample_metric, feature_metric, class_metric = st.columns(3)
    sample_metric.metric("Samples", f"{len(frame):,}")
    feature_metric.metric("Input features", len(dataset.features.columns))
    class_metric.metric("Target classes", len(dataset.target_names))

    chart_column, balance_column = st.columns([2, 1])
    with chart_column:
        st.subheader("Feature separation")
        st.scatter_chart(
            frame,
            x="sepal_length_cm",
            y="petal_length_cm",
            color="species",
            height=360,
        )
    with balance_column:
        st.subheader("Class balance")
        counts = frame["species"].value_counts().rename_axis("species").to_frame("samples")
        st.bar_chart(counts, color="#76B900", height=360)

    with st.expander("Inspect sample rows and the enforced schema"):
        st.dataframe(frame.head(12), width="stretch", hide_index=True)
        st.code("\n".join(f"{name}: float >= 0" for name in dataset.features.columns))


def _train_model() -> None:
    st.header("2. Train, track, and apply the quality gate")
    st.write(
        "Choose the repeatable training parameters. The run is logged to MLflow, evaluated "
        "with five-fold cross-validation and a held-out set, then released only if it passes."
    )

    config_column, explanation_column = st.columns([1, 1])
    with config_column:
        test_size = st.slider("Held-out test fraction", 0.15, 0.35, 0.20, 0.05)
        random_state = st.number_input("Random seed", min_value=0, max_value=10_000, value=42)
        threshold = st.slider("Minimum accuracy", 0.50, 1.00, 0.90, 0.01)
    with explanation_column:
        st.info(
            "The seed controls the split and model initialization. The held-out set is never "
            "used for fitting. A failed quality gate remains visible in MLflow but is not promoted."
        )
        st.code(
            "StandardScaler → LogisticRegression\n"
            "5-fold CV → held-out evaluation → gate → artifact",
            language="text",
        )

    if st.button("Run training pipeline", type="primary", width="stretch"):
        from mlops_demo.train import TrainingConfig, train_model

        with st.status("Running the MLOps pipeline…", expanded=True) as status:
            status.write("✓ Dataset loaded and schema validated")
            status.write("Training five cross-validation folds and the final estimator…")
            try:
                summary = train_model(
                    TrainingConfig(
                        artifacts_dir=_model_path().parent,
                        tracking_dir=Path(os.getenv("TRACKING_DIR", "mlruns")),
                        experiment_name="iris-guided-ui",
                        test_size=float(test_size),
                        random_state=int(random_state),
                        accuracy_threshold=float(threshold),
                    )
                )
            except RuntimeError as error:
                status.update(label="Quality gate failed", state="error", expanded=True)
                st.error(str(error))
            else:
                st.session_state["training_summary"] = summary
                status.write("✓ Metrics and model artifact recorded in MLflow")
                status.update(label="Model approved for release", state="complete", expanded=True)
                st.success(
                    f"Passed with {summary['accuracy']:.1%} held-out accuracy. "
                    "Continue to step 3 to inspect the evidence."
                )

    summary = st.session_state.get("training_summary") or load_training_summary(_metrics_path())
    if summary:
        st.divider()
        accuracy_column, f1_column, cv_column, gate_column = st.columns(4)
        accuracy_column.metric("Held-out accuracy", f"{summary['accuracy']:.1%}")
        f1_column.metric("Macro F1", f"{summary['f1_macro']:.1%}")
        cv_column.metric(
            "Cross-validation",
            f"{summary.get('cv_accuracy_mean', 0):.1%}",
            f"± {summary.get('cv_accuracy_std', 0):.1%}",
        )
        gate_column.metric("Quality gate", str(summary["quality_gate"]).upper())


def _review_evaluation(bundle: ModelBundle | None) -> None:
    st.header("3. Review the evaluation evidence")
    if bundle is None:
        st.warning("Train a model in step 2 before reviewing its evaluation.")
        return

    summary = st.session_state.get("training_summary") or load_training_summary(_metrics_path())
    accuracy_column, f1_column, cv_column = st.columns(3)
    accuracy_column.metric("Held-out accuracy", f"{bundle.metrics['accuracy']:.1%}")
    f1_column.metric("Macro F1", f"{bundle.metrics['f1_macro']:.1%}")
    cv_column.metric(
        "5-fold CV accuracy",
        f"{bundle.metrics.get('cv_accuracy_mean', 0):.1%}",
        f"± {bundle.metrics.get('cv_accuracy_std', 0):.1%}",
    )

    if summary and summary.get("confusion_matrix"):
        st.subheader("Held-out confusion matrix")
        class_names = summary["class_names"]
        confusion = pd.DataFrame(
            summary["confusion_matrix"],
            index=[f"Actual {name}" for name in class_names],
            columns=[f"Predicted {name}" for name in class_names],
        )
        st.dataframe(confusion, width="stretch")
        st.caption(
            f"Rows are actual classes and columns are predictions across "
            f"{summary['test_sample_count']} untouched test examples."
        )
    else:
        st.info("Re-run training once to generate the expanded evaluation report.")

    metadata_column, baseline_column = st.columns(2)
    with metadata_column:
        st.subheader("Traceability")
        st.json(
            {
                "model_version": bundle.model_version,
                "mlflow_run_id": bundle.run_id,
                "trained_at": bundle.trained_at,
                "feature_order": list(bundle.feature_names),
            }
        )
    with baseline_column:
        st.subheader("Training baseline")
        baseline = pd.DataFrame(bundle.baseline).T.reset_index(names="feature")
        st.dataframe(baseline, width="stretch", hide_index=True)


def _try_prediction(bundle: ModelBundle | None) -> None:
    st.header("4. Send a prediction through the approved model")
    if bundle is None:
        st.warning("Train a model in step 2 before requesting predictions.")
        return

    st.write("Adjust the measurements, then inspect the full probability distribution.")
    input_columns = st.columns(2)
    features: dict[str, float] = {}
    for index, (name, label) in enumerate(FEATURE_LABELS.items()):
        with input_columns[index % 2]:
            features[name] = st.slider(
                label,
                min_value=0.0,
                max_value=10.0,
                value=FEATURE_DEFAULTS[name],
                step=0.1,
                key=f"prediction_{name}",
            )

    if st.button("Predict species", type="primary", width="stretch"):
        predicted_class, probabilities = predict_features(bundle, features)
        append_prediction(
            _prediction_log_path(),
            build_prediction_record(bundle, features, predicted_class, probabilities),
        )
        st.session_state["last_prediction"] = {
            "class": predicted_class,
            "probabilities": probabilities,
        }

    result = st.session_state.get("last_prediction")
    if result:
        st.success(f"Predicted species: **{str(result['class']).title()}**")
        probability_frame = pd.DataFrame(
            {
                "species": list(result["probabilities"].keys()),
                "probability": list(result["probabilities"].values()),
            }
        ).set_index("species")
        st.bar_chart(probability_frame, color="#76B900", height=320)
        st.caption("This prediction was appended to the local monitoring log used in step 5.")


def _render_drift_report(report: dict[str, object], *, simulated: bool = False) -> None:
    if report["drift_detected"]:
        prefix = (
            "Expected monitoring alert triggered" if simulated else "Monitoring alert triggered"
        )
        st.warning(
            f"{prefix}: at least one feature exceeded the configured drift threshold."
        )
    else:
        st.success("No drift detected for this traffic window.")
    drift_frame = pd.DataFrame(report["features"]).T.reset_index(names="feature")
    st.dataframe(drift_frame, width="stretch", hide_index=True)
    st.caption(
        f"Compared {report['record_count']} records against the saved training baseline. "
        f"Threshold: {report['threshold_std']} standard deviations."
    )


def _check_drift(bundle: ModelBundle | None) -> None:
    st.header("5. Detect feature drift")
    if bundle is None:
        st.warning("Train a model in step 2 before checking drift.")
        return

    st.write(
        "Compare recent feature means with the baseline stored alongside the model. "
        "Use simulation to see how the monitor behaves, or analyze predictions made in step 4."
    )
    source = st.radio("Traffic source", ("Simulated traffic", "Prediction log"), horizontal=True)

    if source == "Simulated traffic":
        sample_count = st.slider("Traffic window size", 10, 200, 50, 10)
        shift = st.slider("Injected mean shift (standard deviations)", 0.0, 2.5, 1.5, 0.1)
        threshold = st.slider("Alert threshold (standard deviations)", 0.5, 2.5, 1.0, 0.1)
        if st.button("Simulate drift alert", type="primary", width="stretch"):
            records = simulate_prediction_records(bundle, sample_count, shift)
            st.session_state["drift_report"] = calculate_drift(
                bundle,
                records,
                threshold=threshold,
            )
    else:
        log_path = _prediction_log_path()
        if log_path.exists():
            line_count = len(log_path.read_text(encoding="utf-8").splitlines())
            st.info(f"Found {line_count} logged prediction(s) at `{log_path}`.")
            if st.button("Analyze prediction log", type="primary", width="stretch"):
                st.session_state["drift_report"] = calculate_drift(
                    bundle,
                    read_prediction_log(log_path),
                )
        else:
            st.info("No prediction log exists yet. Make a prediction in step 4 first.")

    report = st.session_state.get("drift_report")
    if report:
        st.divider()
        _render_drift_report(report, simulated=source == "Simulated traffic")

    with st.expander("What would change in production?"):
        st.markdown(
            "- Store events in a durable, centralized system rather than a local file.\n"
            "- Use time or count windows, minimum sample sizes, and model-version filtering.\n"
            "- Add robust statistical tests, dashboards, and routed alerts.\n"
            "- Join delayed labels to monitor model quality and concept drift."
        )


def main() -> None:
    st.set_page_config(
        page_title="End-to-End MLOps Walkthrough",
        page_icon="🧪",
        layout="wide",
    )
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    _header()
    bundle = _load_model_if_available()
    stage = _sidebar(bundle)
    if stage == STAGES[0]:
        _explore_data()
    elif stage == STAGES[1]:
        _train_model()
    elif stage == STAGES[2]:
        _review_evaluation(bundle)
    elif stage == STAGES[3]:
        _try_prediction(bundle)
    else:
        _check_drift(bundle)


if __name__ == "__main__":
    main()
