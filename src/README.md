# src

This directory contains the reusable Python modules behind the SHAP dashboard.

## Contents
- `comparison_service.py` : Cached model loading and comparison workspace helpers.
- `config_loader.py` : Project path and YAML configuration loading helpers.
- `data_loader.py` : Test metadata and dataset loading utilities.
- `metadata_filters.py` : Metadata filtering and misclassification explorer controls.
- `model_loader.py` : Model discovery, config parsing, and pipeline reconstruction.
- `prediction_utils.py` : Prediction frame creation and metadata merging.
- `shap_analyzer.py` : SHAP explanation generation from loaded models and datasets.
- `shap_visualizer.py` : SHAP plots and comparison visualizations.
- `ui_pages.py` : Streamlit page rendering logic.

## Purpose
These modules keep the dashboard logic organized, reusable, and easier to extend when comparing models with different feature sets.
