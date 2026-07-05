# How to Run the Streamlit App

This project includes a Streamlit app for heart-disease risk prediction and SHAP-based explainability.

## Prerequisites

- Python environment created in `.venv`
- Project dependencies installed with `pip install -r requirements.txt`
- A trained model saved at `models/best_model.joblib`

If you do not have a trained model yet, run:

```bash
source .venv/bin/activate
python -m src.train
```

## Start the app

From the project root:

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

If you want to force a specific port, you can use:

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py --server.port 8501
```

## What the app does

- Lets you enter a patient's clinical values in the sidebar
- Predicts the heart-disease risk probability
- Shows a SHAP waterfall plot for the individual prediction
- Shows a global SHAP feature-importance plot
- Displays the current patient input in a table

## Expected result

When the model is present, the app should open with:

- A model summary banner
- A prediction button in the sidebar
- A risk score and classification label after prediction
- SHAP explanations for the selected patient

## Troubleshooting

- If the app says no model is found, run `python -m src.train` first.
- If Streamlit cannot load the page, confirm the server is still running in the terminal.
- If you change the code and the browser looks stale, reload the page or restart Streamlit.
- If the environment is missing packages, activate `.venv` and reinstall dependencies.
