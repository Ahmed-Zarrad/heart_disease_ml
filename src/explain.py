"""SHAP explainability helpers - global feature importance and per-patient reasons.

The fitted model is a ``Pipeline(preprocessor -> classifier)``.  SHAP is applied in
the *transformed* feature space (after imputation / scaling / one-hot encoding),
because that is the space the classifier actually sees.  Helpers here:

* pick the most appropriate SHAP explainer for the model type
  (TreeExplainer > LinearExplainer > model-agnostic Explainer);
* always return a ``shap.Explanation`` for the **positive class** (HeartDisease=1)
  with readable feature names, regardless of SHAP/model version quirks;
* render matplotlib figures the Streamlit app embeds directly.

Requires ``shap`` (see requirements.txt).  Tested against shap >= 0.44.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from . import config
from . import models as model_zoo

# Cap the background set so model-agnostic explainers stay fast.
MAX_BACKGROUND = 100


def _classifier(pipeline):
    return pipeline.named_steps["classifier"]


def _preprocessor(pipeline):
    return pipeline.named_steps["preprocessor"]


def transform_features(pipeline, X) -> np.ndarray:
    """Run X through the preprocessor only, returning a dense float array."""
    Xt = _preprocessor(pipeline).transform(X)
    if hasattr(Xt, "toarray"):  # sparse -> dense (one-hot output)
        Xt = Xt.toarray()
    return np.asarray(Xt, dtype=float)


def get_feature_names(pipeline) -> list[str]:
    """Transformed feature names, prettified for display."""
    raw = list(_preprocessor(pipeline).get_feature_names_out())
    return [clean_name(n) for n in raw]


def clean_name(name: str) -> str:
    """'num__Age' -> 'Age', 'cat__ChestPainType_ASY' -> 'ChestPainType=ASY'."""
    for prefix in ("num__", "cat__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # One-hot columns look like 'Feature_Value'; turn the last '_' into '='.
    for cat in config.CATEGORICAL_FEATURES:
        if name.startswith(cat + "_"):
            return f"{cat}={name[len(cat) + 1:]}"
    return name


def _background(pipeline, X_background) -> np.ndarray:
    Xt = transform_features(pipeline, X_background)
    if len(Xt) > MAX_BACKGROUND:
        # shap.sample keeps it representative and reproducible.
        Xt = shap.sample(Xt, MAX_BACKGROUND, random_state=config.RANDOM_STATE)
    return Xt


def make_explainer(pipeline, X_background):
    """Return the most suitable SHAP explainer for the pipeline's classifier."""
    clf = _classifier(pipeline)
    bg = _background(pipeline, X_background)
    feature_names = get_feature_names(pipeline)

    if model_zoo.is_tree_model(clf):
        return shap.TreeExplainer(clf, bg, feature_names=feature_names)

    if type(clf).__name__ == "LogisticRegression":
        return shap.LinearExplainer(clf, bg, feature_names=feature_names)

    # Model-agnostic fallback (SVM, KNN, ...). Explains P(class=1).
    return shap.Explainer(
        lambda data: clf.predict_proba(data)[:, 1],
        bg,
        feature_names=feature_names,
    )


def _select_positive_class(explanation):
    """Collapse a multi-class Explanation to the positive (disease) class."""
    values = np.asarray(explanation.values)
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        idx = values.shape[-1] - 1  # last column == class 1 for binary problems
        explanation = explanation[..., idx]
    return explanation


def explain(pipeline, explainer, X):
    """Return a positive-class ``shap.Explanation`` for the given raw rows X."""
    Xt = transform_features(pipeline, X)
    explanation = explainer(Xt)
    explanation = _select_positive_class(explanation)
    # Ensure feature names + readable data survive (some explainers drop them).
    explanation.feature_names = get_feature_names(pipeline)
    explanation.data = Xt
    return explanation


# --------------------------------------------------------------------------- #
# Plot helpers (return matplotlib Figures so the app can st.pyplot them)
# --------------------------------------------------------------------------- #
def global_importance_figure(explanation, max_display: int = 12):
    """Beeswarm-style global feature importance across the provided rows."""
    fig = plt.figure(figsize=(7, 5))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title("Global feature importance (SHAP)")
    plt.tight_layout()
    return fig


def global_bar_figure(explanation, max_display: int = 12):
    """Mean |SHAP| bar chart - the simplest 'which features matter' view."""
    fig = plt.figure(figsize=(7, 5))
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.title("Mean absolute SHAP value")
    plt.tight_layout()
    return fig


def local_explanation_figure(single_explanation, max_display: int = 12):
    """Waterfall plot explaining one patient's prediction."""
    fig = plt.figure(figsize=(7, 5))
    shap.plots.waterfall(single_explanation, max_display=max_display, show=False)
    plt.tight_layout()
    return fig
