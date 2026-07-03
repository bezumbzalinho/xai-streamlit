# shap_streamlit

This directory contains the Streamlit app used to explore SHAP explanations for trained models.

## Contents
- `main.py` : App entry point and page routing.
- `models/` : Model artifacts, config files, and test datasets used by the dashboard.
- `src/` : Reusable loading, comparison, filtering, and visualization utilities.
- `requirements.txt` : Python dependencies for the Streamlit app.

## Purpose
The app standardizes model loading, reconstructs features from the saved processing pipeline, and supports side-by-side comparison of model predictions and SHAP outputs.
