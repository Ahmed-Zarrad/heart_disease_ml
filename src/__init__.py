"""Heart Disease prediction - source package.

Modules
-------
config              : paths, schema and project-wide constants
data_preprocessing  : loading, cleaning and the sklearn preprocessing pipeline
models              : model factory + hyper-parameter grids
train               : K-Fold cross-validation, tuning and final model training
evaluate            : test-set metrics and diagnostic plots
explain             : SHAP explainability helpers (global + per-patient)
"""

__all__ = [
    "config",
    "data_preprocessing",
    "models",
    "train",
    "evaluate",
    "explain",
]
