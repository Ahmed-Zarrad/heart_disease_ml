"""Data loading, cleaning and the reusable preprocessing pipeline.

Design notes
------------
* Cleaning (turning physiologically-impossible 0s into NaN) is deterministic and
  leakage-free, so it is applied to the raw frame *before* the train/test split.
* All *learned* transformations (imputation, scaling, one-hot encoding) live inside
  a scikit-learn ``ColumnTransformer`` that is fitted **inside** each CV fold /
  on the training split only - so there is no information leakage from test data.
* The same ``ColumnTransformer`` is bundled with the classifier into a single
  ``Pipeline`` (see ``train.make_pipeline``), which means the Streamlit app can
  feed raw patient values straight in and get a prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Load the raw heart-disease CSV and sanity-check its schema."""
    df = pd.read_csv(path)
    missing = set(config.FEATURE_COLUMNS + [config.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset at {path} is missing expected columns: {sorted(missing)}"
        )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert invalid 0 measurements to NaN so the imputer can handle them.

    This is intentionally the *only* row-level cleaning done outside the pipeline,
    because replacing a sentinel value does not depend on any statistic learned
    from the data and therefore cannot leak information between folds.
    """
    df = df.copy()
    for col in config.ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)
    return df


def split_xy(df: pd.DataFrame):
    """Split a frame into the feature matrix X and the target vector y."""
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the (unfitted) ColumnTransformer.

    Numeric features  -> median imputation + standard scaling.
    Categorical feats -> most-frequent imputation + one-hot encoding.

    ``handle_unknown='ignore'`` makes the encoder robust to categories that the
    app might submit but that never appeared in training.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_train_test(df: pd.DataFrame | None = None):
    """Return a stratified, cleaned train/test split of raw (un-transformed) X.

    The features are returned *un-transformed*; transformation happens inside the
    model pipeline so the held-out test set never influences fitting.
    """
    if df is None:
        df = load_data()
    df = clean_data(df)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def describe_data(df: pd.DataFrame | None = None) -> dict:
    """Lightweight EDA summary used by the app's 'About the data' section."""
    if df is None:
        df = load_data()
    cleaned = clean_data(df)
    return {
        "n_rows": int(len(df)),
        "n_features": len(config.FEATURE_COLUMNS),
        "class_balance": df[config.TARGET].value_counts().to_dict(),
        "missing_after_cleaning": cleaned[config.FEATURE_COLUMNS]
        .isna()
        .sum()
        .to_dict(),
    }
