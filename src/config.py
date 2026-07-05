"""Project-wide configuration: paths, schema and constants.

Everything that the rest of the code base needs to agree on lives here so that
preprocessing, training, evaluation, explainability and the Streamlit app all
read from a single source of truth.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
GRAPH_DIR = REPORTS_DIR / "graphs"

DATA_PATH = DATA_DIR / "heart.csv"
MODEL_PATH = MODELS_DIR / "best_model.joblib"          # fitted sklearn Pipeline
METADATA_PATH = MODELS_DIR / "model_metadata.joblib"   # name, params, metrics, etc.
CV_RESULTS_PATH = REPORTS_DIR / "cv_results.csv"
TEST_METRICS_PATH = REPORTS_DIR / "test_metrics.csv"

# Create the writable folders on import so first runs never crash on a missing
# directory.  (data/ is expected to already contain heart.csv.)
for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility / experiment settings
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20      # held-out test fraction
CV_FOLDS = 5          # K in K-Fold cross-validation
TUNING_SCORING = "roc_auc"   # metric GridSearchCV optimises
SELECTION_METRIC = "roc_auc"  # metric used to pick the best model from CV

# --------------------------------------------------------------------------- #
# Dataset schema  (Kaggle: fedesoriano/heart-failure-prediction)
# --------------------------------------------------------------------------- #
TARGET = "HeartDisease"

NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns where a literal 0 is physiologically impossible and therefore encodes
# a *missing* measurement (RestingBP has 1 such row, Cholesterol has ~172).
# These are converted to NaN during cleaning and imputed inside the pipeline.
ZERO_AS_MISSING = ["RestingBP", "Cholesterol"]

# --------------------------------------------------------------------------- #
# Human-readable metadata used by the Streamlit UI to build input widgets and
# to show clinicians meaningful labels instead of raw codes.
# --------------------------------------------------------------------------- #
FEATURE_INFO = {
    "Age": {
        "kind": "numeric", "label": "Age (years)",
        "min": 18, "max": 100, "default": 54, "step": 1,
        "help": "Patient age in years.",
    },
    "Sex": {
        "kind": "categorical", "label": "Sex",
        "options": {"M": "Male", "F": "Female"}, "default": "M",
        "help": "Biological sex of the patient.",
    },
    "ChestPainType": {
        "kind": "categorical", "label": "Chest pain type",
        "options": {
            "TA": "Typical Angina",
            "ATA": "Atypical Angina",
            "NAP": "Non-Anginal Pain",
            "ASY": "Asymptomatic",
        },
        "default": "ASY",
        "help": "Type of chest pain reported by the patient.",
    },
    "RestingBP": {
        "kind": "numeric", "label": "Resting blood pressure (mm Hg)",
        "min": 80, "max": 220, "default": 130, "step": 1,
        "help": "Resting blood pressure on admission.",
    },
    "Cholesterol": {
        "kind": "numeric", "label": "Serum cholesterol (mg/dl)",
        "min": 100, "max": 610, "default": 240, "step": 1,
        "help": "Serum cholesterol. A value of 0 in the raw data means 'not measured'.",
    },
    "FastingBS": {
        "kind": "categorical", "label": "Fasting blood sugar > 120 mg/dl",
        "options": {0: "No (<= 120 mg/dl)", 1: "Yes (> 120 mg/dl)"}, "default": 0,
        "numeric_value": True,  # encoded as 0/1 in the numeric block
        "help": "Whether fasting blood sugar exceeds 120 mg/dl.",
    },
    "RestingECG": {
        "kind": "categorical", "label": "Resting ECG result",
        "options": {
            "Normal": "Normal",
            "ST": "ST-T wave abnormality",
            "LVH": "Left ventricular hypertrophy",
        },
        "default": "Normal",
        "help": "Resting electrocardiogram result.",
    },
    "MaxHR": {
        "kind": "numeric", "label": "Maximum heart rate achieved",
        "min": 60, "max": 210, "default": 140, "step": 1,
        "help": "Maximum heart rate achieved during exercise testing.",
    },
    "ExerciseAngina": {
        "kind": "categorical", "label": "Exercise-induced angina",
        "options": {"N": "No", "Y": "Yes"}, "default": "N",
        "help": "Angina induced by exercise.",
    },
    "Oldpeak": {
        "kind": "numeric", "label": "Oldpeak (ST depression)",
        "min": -3.0, "max": 7.0, "default": 1.0, "step": 0.1,
        "help": "ST depression induced by exercise relative to rest.",
    },
    "ST_Slope": {
        "kind": "categorical", "label": "ST slope (peak exercise)",
        "options": {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"},
        "default": "Flat",
        "help": "Slope of the peak exercise ST segment.",
    },
}
