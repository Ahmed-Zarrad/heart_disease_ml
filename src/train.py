"""Training: K-Fold cross-validation, hyper-parameter tuning and final fit.

This is the entry point your teammate runs to actually *train and test* the
models.  Typical usage from the project root:

    # 1) Compare all candidate models with 5-fold CV (no tuning, fast):
    python -m src.train --compare-only

    # 2) Full run: CV comparison -> tune the best model -> fit -> evaluate -> save:
    python -m src.train

    # 3) Force a specific model and tune it:
    python -m src.train --model "Random Forest"

Outputs (written to ../reports and ../models):
    reports/cv_results.csv      cross-validation comparison table
    reports/test_metrics.csv    held-out test-set metrics of the final model
    reports/*.png               confusion matrix + ROC curve
    models/best_model.joblib    the fitted Pipeline (preprocessor + classifier)
    models/model_metadata.joblib  name, best params, metrics, feature names, ...
"""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from . import config, evaluate
from . import models as model_zoo
from .data_preprocessing import build_preprocessor, get_train_test

# Metrics computed for every fold during cross-validation.
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def make_pipeline(estimator) -> Pipeline:
    """Wrap an estimator together with a fresh preprocessor into one Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def _cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )


def cross_validate_all(X_train, y_train) -> pd.DataFrame:
    """Run K-Fold CV for every candidate model and return a comparison table."""
    cv = _cv_splitter()
    rows = []
    for name, estimator in model_zoo.get_models().items():
        pipe = make_pipeline(estimator)
        results = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1
        )
        row = {"model": name}
        for metric in SCORING:
            row[metric] = results[f"test_{metric}"].mean()
            row[f"{metric}_std"] = results[f"test_{metric}"].std()
        rows.append(row)
        print(
            f"  {name:<20s} "
            + "  ".join(f"{m}={row[m]:.3f}" for m in SCORING)
        )
    table = (
        pd.DataFrame(rows)
        .sort_values(config.SELECTION_METRIC, ascending=False)
        .reset_index(drop=True)
    )
    return table


def tune_model(name: str, X_train, y_train) -> GridSearchCV:
    """GridSearchCV over a single model's hyper-parameter grid."""
    estimator = model_zoo.get_models()[name]
    grid = model_zoo.get_param_grids()[name]
    pipe = make_pipeline(estimator)
    search = GridSearchCV(
        pipe,
        param_grid=grid,
        scoring=config.TUNING_SCORING,
        cv=_cv_splitter(),
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def run(model_name: str | None = None, compare_only: bool = False) -> dict:
    """Full training routine. Returns a metadata dict (also persisted to disk)."""
    print(f"Loading data from {config.DATA_PATH} ...")
    X_train, X_test, y_train, y_test = get_train_test()
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    print(f"\n[1/4] {config.CV_FOLDS}-fold cross-validation comparison")
    cv_table = cross_validate_all(X_train, y_train)
    cv_table.to_csv(config.CV_RESULTS_PATH, index=False)
    print(f"  -> saved CV results to {config.CV_RESULTS_PATH}")

    best_name = model_name or cv_table.iloc[0]["model"]
    print(f"\n[2/4] Selected model: {best_name}")

    if compare_only:
        print("compare-only requested; stopping before tuning.")
        return {"cv_results": cv_table.to_dict(orient="records")}

    print(f"\n[3/4] Hyper-parameter tuning ({best_name}) optimising "
          f"{config.TUNING_SCORING} ...")
    search = tune_model(best_name, X_train, y_train)
    best_pipeline = search.best_estimator_
    print(f"  best CV {config.TUNING_SCORING}: {search.best_score_:.4f}")
    print(f"  best params: {json.dumps(search.best_params_, indent=2)}")

    print("\n[4/4] Evaluating tuned model on the held-out test set ...")
    test_metrics = evaluate.evaluate_model(best_pipeline, X_test, y_test)
    evaluate.save_metrics(test_metrics, config.TEST_METRICS_PATH)
    evaluate.plot_confusion_matrix(best_pipeline, X_test, y_test)
    evaluate.plot_roc_curve(best_pipeline, X_test, y_test)
    for k, v in test_metrics.items():
        print(f"  test {k:<10s}: {v:.4f}")

    # Persist the fitted pipeline + metadata for the Streamlit app.
    joblib.dump(best_pipeline, config.MODEL_PATH)
    metadata = {
        "model_name": best_name,
        "best_params": search.best_params_,
        "cv_score": float(search.best_score_),
        "test_metrics": test_metrics,
        "feature_columns": config.FEATURE_COLUMNS,
        "transformed_feature_names": list(
            best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        ),
    }
    joblib.dump(metadata, config.METADATA_PATH)
    print(f"\nSaved model    -> {config.MODEL_PATH}")
    print(f"Saved metadata -> {config.METADATA_PATH}")
    return metadata


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train heart-disease classifiers.")
    parser.add_argument(
        "--model",
        default=None,
        help="Force a specific model by name (otherwise the best CV model is used).",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only run the CV comparison; skip tuning, evaluation and saving.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run(model_name=args.model, compare_only=args.compare_only)


if __name__ == "__main__":
    main()
