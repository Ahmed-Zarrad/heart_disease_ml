"""Model factory and hyper-parameter grids for the comparative analysis.

``get_models`` returns fresh, unfitted estimators keyed by a readable name.
``get_param_grids`` returns matching GridSearchCV grids; keys are prefixed with
``classifier__`` because every estimator is wrapped in a Pipeline whose final
step is named ``classifier`` (see ``train.make_pipeline``).

XGBoost is included automatically *iff* the ``xgboost`` package is installed,
so the code never hard-fails if that optional dependency is missing.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config

try:  # optional dependency
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - import guard
    HAS_XGBOOST = False

# Estimator class names that SHAP's fast TreeExplainer supports.
TREE_MODEL_TYPES = {
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
}


def get_models() -> dict:
    """Return the dictionary of candidate classifiers for comparison."""
    rs = config.RANDOM_STATE
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=rs),
        "Decision Tree": DecisionTreeClassifier(random_state=rs),
        "Random Forest": RandomForestClassifier(random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
        "SVM": SVC(probability=True, random_state=rs),
        "KNN": KNeighborsClassifier(),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            random_state=rs,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        )
    return models


def get_param_grids() -> dict:
    """Return GridSearchCV parameter grids (Pipeline-prefixed) per model."""
    grids = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs", "liblinear"],
        },
        "Decision Tree": {
            "classifier__max_depth": [3, 5, 7, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__criterion": ["gini", "entropy"],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_split": [2, 5],
        },
        "Gradient Boosting": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.01, 0.1],
            "classifier__max_depth": [2, 3],
        },
        "SVM": {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["rbf", "linear"],
            "classifier__gamma": ["scale", "auto"],
        },
        "KNN": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
    }
    if HAS_XGBOOST:
        grids["XGBoost"] = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.01, 0.1],
            "classifier__max_depth": [3, 5],
            "classifier__subsample": [0.8, 1.0],
        }
    return grids


def is_tree_model(estimator) -> bool:
    """True if SHAP's TreeExplainer can be used on this estimator."""
    return type(estimator).__name__ in TREE_MODEL_TYPES
