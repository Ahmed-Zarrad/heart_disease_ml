"""Doctor-facing web interface for heart-disease risk prediction + SHAP insights.

Run from the project root:

    streamlit run app/streamlit_app.py

The app loads the model saved by ``python -m src.train``.  If no model exists yet
it shows clear instructions instead of crashing.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import streamlit as st

from src import config, explain
from src.data_preprocessing import clean_data, describe_data, load_data, split_xy

st.set_page_config(
    page_title="Heart Disease Risk - Decision Support",
    page_icon="(+)",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_model():
    if not config.MODEL_PATH.exists():
        return None, None
    model = joblib.load(config.MODEL_PATH)
    metadata = (
        joblib.load(config.METADATA_PATH) if config.METADATA_PATH.exists() else {}
    )
    return model, metadata


@st.cache_data(show_spinner=False)
def load_background():
    """Cleaned feature matrix used as the SHAP background / reference set."""
    df = clean_data(load_data())
    X, _ = split_xy(df)
    return X


@st.cache_resource(show_spinner=False)
def build_explainer(_model, background):
    # Leading underscore tells Streamlit not to hash the unhashable model object.
    return explain.make_explainer(_model, background)


# --------------------------------------------------------------------------- #
# Sidebar - patient input form
# --------------------------------------------------------------------------- #
def patient_input_form() -> pd.DataFrame:
    st.sidebar.header("Patient clinical data")
    values = {}
    for col in config.FEATURE_COLUMNS:
        info = config.FEATURE_INFO[col]
        if info["kind"] == "numeric":
            values[col] = st.sidebar.slider(
                info["label"],
                min_value=float(info["min"]),
                max_value=float(info["max"]),
                value=float(info["default"]),
                step=float(info["step"]),
                help=info.get("help"),
            )
        else:  # categorical (incl. FastingBS, stored as 0/1)
            options = list(info["options"].keys())
            values[col] = st.sidebar.selectbox(
                info["label"],
                options=options,
                index=options.index(info["default"]),
                format_func=lambda o, info=info: info["options"][o],
                help=info.get("help"),
            )
    # Build a one-row frame with the exact training column order.
    return pd.DataFrame([values], columns=config.FEATURE_COLUMNS)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.title("Heart Disease Risk - Clinical Decision Support")
    st.caption(
        "Predicts the probability of heart disease from routine clinical "
        "features and explains *why* using SHAP. Research/educational tool - "
        "not a substitute for clinical judgement."
    )

    model, metadata = load_model()
    if model is None:
        st.warning(
            "No trained model found.\n\n"
            "Train one first from the project root:\n\n"
            "```bash\npython -m src.train\n```\n\n"
            f"Expected model file: `{config.MODEL_PATH}`"
        )
        st.stop()

    patient_df = patient_input_form()
    predict = st.sidebar.button("Predict", type="primary", use_container_width=True)

    # --- Model info banner ---
    if metadata:
        cols = st.columns(4)
        cols[0].metric("Model", metadata.get("model_name", "n/a"))
        tm = metadata.get("test_metrics", {})
        cols[1].metric("Test accuracy", f"{tm.get('accuracy', float('nan')):.3f}")
        cols[2].metric("Test ROC-AUC", f"{tm.get('roc_auc', float('nan')):.3f}")
        cols[3].metric("Test recall", f"{tm.get('recall', float('nan')):.3f}")

    if not predict:
        st.info("Enter the patient's data in the sidebar, then click **Predict**.")
        with st.expander("About the dataset"):
            st.json(describe_data())
        st.stop()

    # --- Prediction ---
    proba = float(model.predict_proba(patient_df)[0, 1])
    label = "HEART DISEASE LIKELY" if proba >= 0.5 else "Low risk"

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Prediction")
        st.metric("Risk of heart disease", f"{proba * 100:.1f}%")
        (st.error if proba >= 0.5 else st.success)(label)
        st.progress(min(max(proba, 0.0), 1.0))

    # --- Explanation ---
    background = load_background()
    explainer = build_explainer(model, background)

    with right:
        st.subheader("Why this prediction? (SHAP)")
        try:
            exp = explain.explain(model, explainer, patient_df)
            fig = explain.local_explanation_figure(exp[0])
            st.pyplot(fig, clear_figure=True)
            st.caption(
                "Red bars push the risk **up**, blue bars push it **down**, "
                "starting from the average patient (base value)."
            )
        except Exception as e:  # pragma: no cover - defensive UI guard
            st.error(f"Could not generate SHAP explanation: {e}")

    # --- Global importance ---
    with st.expander("Global feature importance (across the dataset)"):
        try:
            sample = background.sample(
                min(200, len(background)), random_state=config.RANDOM_STATE
            )
            global_exp = explain.explain(model, explainer, sample)
            st.pyplot(explain.global_bar_figure(global_exp), clear_figure=True)
        except Exception as e:  # pragma: no cover
            st.error(f"Could not generate global importance: {e}")

    with st.expander("Patient input (as sent to the model)"):
        st.dataframe(patient_df.T, use_container_width=True)


if __name__ == "__main__":
    main()
