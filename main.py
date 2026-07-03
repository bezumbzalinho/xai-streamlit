import os
import streamlit as st

from shap_streamlit.src.comparison_service import discover_available_models, load_comparison_workspace
from shap_streamlit.src.config_loader import get_project_base_path
from shap_streamlit.src.ui_pages import (
    render_summary_page,
    render_feature_dependence_page,
    render_instance_analysis_page,
)

def setup_environment():
    os.environ["ENV_TYPE"] = "aws"
    return get_project_base_path()

def main():
    st.set_page_config(page_title="SHAP Model Analysis Dashboard", layout="wide")
    st.title("SHAP Model Analysis Dashboard")

    base_path = setup_environment()
    model_names = discover_available_models(base_path)

    with st.spinner("Bootstrapping comparison workspace..."):
        workspace = load_comparison_workspace(base_path, tuple(model_names))

    page = st.sidebar.radio("Page", ["SHAP Summary", "Feature Dependence", "Instance Analysis"])

    if page == "SHAP Summary":
        render_summary_page(workspace)
    elif page == "Feature Dependence":
        render_feature_dependence_page(workspace)
    elif page == "Instance Analysis":
        render_instance_analysis_page(workspace)

if __name__ == "__main__":
    main()