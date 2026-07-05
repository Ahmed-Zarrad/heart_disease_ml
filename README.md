# Heart Disease Prediction with Explainable AI

Data Mining & Machine Learning course project — M.Sc. in Artificial Intelligence and Data Engineering.

Early detection of heart disease as a **binary classification** task, with a strong
focus on **interpretability**: a Streamlit web app lets a clinician enter a patient's
clinical data, returns a risk probability, and uses **SHAP** to explain *why*.

- **Dataset:** [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (fedesoriano) — 918 patients, 11 clinical features + binary target. A copy is already included at [`data/heart.csv`](data/heart.csv).
- **ML tasks:** preprocessing → classification → comparative analysis → Explainable AI.
- **Current artifacts:** deterministic train/test splits are saved as [`data/heart_train.csv`](data/heart_train.csv) and [`data/heart_test.csv`](data/heart_test.csv); the latest trained model and reports are in [`models/`](models/) and [`reports/`](reports/).

---

## Project structure

```
heart_disease_ml/
├── data/heart.csv              # the dataset (already included)
├── data/heart_train.csv        # generated training split
├── data/heart_test.csv         # generated test split
├── src/
│   ├── config.py               # paths, schema, constants, UI metadata
│   ├── data_preprocessing.py   # loading, cleaning, leakage-free preprocessor
│   ├── models.py               # model factory + hyper-parameter grids
│   ├── train.py                # K-Fold CV, tuning, final fit  (run this)
│   ├── evaluate.py             # test metrics + confusion/ROC plots
│   └── explain.py              # SHAP global + per-patient explanations
│   └── generate_graphs.py      # end-to-end chart generation for the report
├── app/streamlit_app.py        # doctor-facing web interface
├── models/                     # saved model + metadata (created by training)
├── reports/                    # CV table, test metrics, figures (created by training)
│   └── graphs/                 # classification graphs generated for the report
├── run_pipeline.py             # convenience wrapper around src.train
└── requirements.txt
```

---

## Setup

```bash
cd heart_disease_ml
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> Python 3.10+ recommended. `xgboost` is optional — if it isn't installed the
> pipeline simply runs without that model.

---

## How the work is split

This repository is the **coding part**. The training/testing experiments are run
on top of it:

| Part | Who | Command / file |
|------|-----|----------------|
| Code: preprocessing, models, CV/tuning, evaluation, SHAP, web app | **(coding)** | `src/`, `app/` |
| Run experiments, K-Fold CV, hyper-parameter tuning, analyse & report results | **(training & testing)** | `python -m src.train`, `reports/` |

Everything below is what the training/testing side runs.

---

## 1. Train & compare models

```bash
# Compare all candidate models with 5-fold cross-validation (fast, no tuning):
python -m src.train --compare-only

# Full run: CV comparison -> tune the best model -> evaluate on test -> save:
python -m src.train

# Force / tune a specific model:
python -m src.train --model "Random Forest"
```

Models compared: Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, SVM, KNN (+ XGBoost if installed).

**Outputs**

- `data/heart_train.csv`, `data/heart_test.csv` — deterministic stratified split of the dataset
- `reports/cv_results.csv` — cross-validation comparison (accuracy, precision, recall, F1, ROC-AUC ± std)
- `reports/test_metrics.csv` — held-out test-set metrics of the final model
- `reports/confusion_matrix.png`, `reports/roc_curve.png`
- `models/best_model.joblib` — the fitted `Pipeline` (preprocessor + classifier)
- `models/model_metadata.joblib` — model name, best params, metrics, feature names

Current run summary:

- Best model: Random Forest
- Test accuracy: 0.8804
- Test ROC-AUC: 0.9324
- Test recall: 0.9216

## 2. Launch the explainable web app

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

Enter a patient's data in the sidebar → get a risk probability → see a SHAP
waterfall plot explaining the individual prediction, plus global feature importance.

For a dedicated step-by-step guide, see [`STREAMLIT_RUN.md`](STREAMLIT_RUN.md).

## 3. Generate report graphs

```bash
source .venv/bin/activate
python -m src.generate_graphs
```

This writes the requested classification visuals to [`reports/graphs/`](reports/graphs/) including class-grouped box plots, correlation plots, permutation importance, cross-validation boxplots, a validation curve, confusion matrix, ROC curve, SHAP importance, and a SHAP waterfall plot.

---

## Methodology notes (for the report)

- **Cleaning:** `RestingBP == 0` (1 row) and `Cholesterol == 0` (~172 rows) are
  physiologically impossible and treated as **missing**, then median-imputed
  *inside* the pipeline.
- **No leakage:** imputation, scaling and one-hot encoding live in a
  `ColumnTransformer` that is fitted only on training folds / the training split.
- **Validation:** stratified 5-fold cross-validation for model selection and
  `GridSearchCV` tuning; a 20% stratified hold-out test set for the final, honest
  performance estimate.
- **Model selection metric:** ROC-AUC (configurable in `src/config.py`).
- **Explainability:** SHAP — `TreeExplainer` for tree models, `LinearExplainer`
  for logistic regression, a model-agnostic explainer otherwise.

All knobs (random seed, test size, CV folds, scoring) are in
[`src/config.py`](src/config.py).
