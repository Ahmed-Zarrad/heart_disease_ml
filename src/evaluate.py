"""Evaluation metrics and diagnostic plots for the final model.

Kept separate from training so your teammate can re-evaluate any saved model
without re-running the whole pipeline, e.g.:

    import joblib
    from src import evaluate
    from src.data_preprocessing import get_train_test
    model = joblib.load("models/best_model.joblib")
    _, X_test, _, y_test = get_train_test()
    print(evaluate.evaluate_model(model, X_test, y_test))
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend - safe for scripts/servers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def _positive_proba(model, X):
    """Return P(HeartDisease=1); fall back to decision_function if needed."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute the standard binary-classification metrics on the test set."""
    y_pred = model.predict(X_test)
    y_proba = _positive_proba(model, X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    return metrics


def classification_text_report(model, X_test, y_test) -> str:
    """Full per-class precision/recall/F1 report as a printable string."""
    y_pred = model.predict(X_test)
    return classification_report(
        y_test, y_pred, target_names=["No disease", "Disease"], zero_division=0
    )


def save_metrics(metrics: dict, path=config.TEST_METRICS_PATH) -> None:
    pd.DataFrame([metrics]).to_csv(path, index=False)


def plot_confusion_matrix(model, X_test, y_test, path=None):
    """Save a confusion-matrix figure and return its path."""
    path = path or (config.REPORTS_DIR / "confusion_matrix.png")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No disease", "Disease"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Test Set")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc_curve(model, X_test, y_test, path=None):
    """Save a ROC-curve figure and return its path (None if no scores)."""
    if _positive_proba(model, X_test) is None:
        return None
    path = path or (config.REPORTS_DIR / "roc_curve.png")
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("ROC Curve - Test Set")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
