"""Generate the requested classification graphs for the heart-disease project.

Run from the project root:

    source .venv/bin/activate
    python -m src.generate_graphs

Outputs are written to ``reports/graphs``.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate, validation_curve
from sklearn.pipeline import Pipeline

from . import config, evaluate, explain, models as model_zoo
from .data_preprocessing import build_preprocessor, clean_data, get_train_test, load_data, split_xy


METRICS = ["accuracy", "precision", "recall", "roc_auc"]


def _load_artifacts():
    model = joblib.load(config.MODEL_PATH)
    metadata = joblib.load(config.METADATA_PATH) if config.METADATA_PATH.exists() else {}
    return model, metadata


def _save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _prepare_data():
    df = clean_data(load_data())
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = get_train_test(df)
    return df, X, y, X_train, X_test, y_train, y_test


def boxplots_by_class(df: pd.DataFrame, out_dir: Path):
    numeric_features = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes = axes.ravel()
    groups = [df[df[config.TARGET] == label] for label in [0, 1]]

    for ax, feature in zip(axes, numeric_features):
        ax.boxplot(
            [groups[0][feature].dropna(), groups[1][feature].dropna()],
            tick_labels=["No disease", "Disease"],
            patch_artist=True,
            boxprops={"facecolor": "#cfe8ff"},
            medianprops={"color": "#1f1f1f"},
        )
        ax.set_title(feature)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Feature distributions grouped by class label", fontsize=14)
    return _save_fig(fig, out_dir / "01_boxplots_by_class.png")


def permutation_importance_plot(model, X_test, y_test, out_dir: Path):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=config.RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    feature_names = list(X_test.columns)
    order = np.argsort(result.importances_mean)[::-1]
    ranked = pd.DataFrame(
        {
            "feature": np.array(feature_names)[order],
            "mean_importance": result.importances_mean[order],
            "std_importance": result.importances_std[order],
        }
    )
    ranked.to_csv(out_dir / "feature_importance_permutation.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        ranked["feature"][::-1],
        ranked["mean_importance"][::-1],
        xerr=ranked["std_importance"][::-1],
        color="#4c78a8",
    )
    ax.set_title("Permutation importance ranking")
    ax.set_xlabel("Decrease in ROC-AUC when shuffled")
    return _save_fig(fig, out_dir / "03_permutation_importance.png")


def scatter_matrix_plot(df: pd.DataFrame, ranked_features: list[str], out_dir: Path):
    features = [f for f in ranked_features if f in config.NUMERIC_FEATURES][:5]
    if len(features) < 2:
        features = config.NUMERIC_FEATURES[:5]
    axes = scatter_matrix(
        df[features],
        diagonal="hist",
        alpha=0.7,
        figsize=(11, 11),
        color="#4c78a8",
    )
    fig = axes[0, 0].get_figure()
    fig.suptitle("Scatter matrix of the most relevant features")
    return _save_fig(fig, out_dir / "02_scatter_matrix.png")


def cross_validated_score_boxplots(X_train, y_train, out_dir: Path):
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    score_store = {metric: {} for metric in METRICS}

    for name, estimator in model_zoo.get_models().items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        results = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=METRICS,
            n_jobs=-1,
        )
        for metric in METRICS:
            score_store[metric][name] = results[f"test_{metric}"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes = axes.ravel()
    for ax, metric in zip(axes, METRICS):
        data = [score_store[metric][name] for name in score_store[metric]]
        labels = list(score_store[metric].keys())
        ax.boxplot(data, tick_labels=labels, patch_artist=True)
        ax.set_title(f"Cross-validated {metric}")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Cross-validated scores across candidate models", fontsize=14)
    return _save_fig(fig, out_dir / "04_cv_score_boxplots.png")


def validation_curve_plot(model, X_train, y_train, out_dir: Path):
    # Best model in this project is Random Forest, so use a depth curve.
    param_name = "classifier__max_depth"
    param_range = [2, 3, 4, 5, 6, 7, None]
    train_scores, valid_scores = validation_curve(
        model,
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    valid_std = valid_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(param_range))
    ax.plot(x, train_mean, marker="o", label="Training ROC-AUC")
    ax.fill_between(x, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.plot(x, valid_mean, marker="o", label="Validation ROC-AUC")
    ax.fill_between(x, valid_mean - valid_std, valid_mean + valid_std, alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in param_range])
    ax.set_xlabel("Random Forest max_depth")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Validation curve")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save_fig(fig, out_dir / "05_validation_curve.png")


def classification_diagnostics(model, X_test, y_test, out_dir: Path):
    evaluate.plot_confusion_matrix(
        model, X_test, y_test, path=out_dir / "06_confusion_matrix.png"
    )
    evaluate.plot_roc_curve(model, X_test, y_test, path=out_dir / "07_roc_curve.png")


def shap_global_and_local(model, X_train, X_test, out_dir: Path):
    background = X_train.sample(min(100, len(X_train)), random_state=config.RANDOM_STATE)
    explainer = explain.make_explainer(model, background)

    global_exp = explain.explain(model, explainer, background)
    _save_fig(
        explain.global_bar_figure(global_exp),
        out_dir / "08_shap_mean_abs_bar.png",
    )

    test_probs = model.predict_proba(X_test)[:, 1]
    idx = int(np.argsort(np.abs(test_probs - np.median(test_probs)))[0])
    row = X_test.iloc[[idx]]
    local_exp = explain.explain(model, explainer, row)
    _save_fig(
        explain.local_explanation_figure(local_exp[0]),
        out_dir / "09_shap_waterfall.png",
    )


def main():
    out_dir = config.GRAPH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model, metadata = _load_artifacts()
    df, X, y, X_train, X_test, y_train, y_test = _prepare_data()

    boxplots_by_class(df, out_dir)

    permutation_importance_plot(model, X_test, y_test, out_dir)
    ranked_features = pd.read_csv(out_dir / "feature_importance_permutation.csv")["feature"].tolist()

    scatter_matrix_plot(df, ranked_features, out_dir)
    cross_validated_score_boxplots(X_train, y_train, out_dir)
    validation_curve_plot(model, X_train, y_train, out_dir)
    classification_diagnostics(model, X_test, y_test, out_dir)
    shap_global_and_local(model, X_train, X_test, out_dir)

    summary = {
        "output_dir": str(out_dir),
        "model_name": metadata.get("model_name", "n/a"),
        "files": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "graph_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()