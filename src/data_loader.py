import os
import pandas as pd
from typing import List, Dict
from cpd.core.data.dataset import DatasetH5
import streamlit as st

@st.cache_data(show_spinner=False)
def load_test_metadata(base_path: str, model_name: str) -> pd.DataFrame:
    app_base_path = os.path.join(base_path, "shap_streamlit")
    samples_base = os.path.join(app_base_path, "models", model_name)
    test_csv_path = os.path.join(samples_base, "test_dataset.csv")
    test_df = pd.read_csv(test_csv_path, delimiter=';')
    true_labels = test_df.header_name.apply(lambda x: 0 if x.endswith('E') else 1).values
    test_df["true_label"] = true_labels
    return test_df.copy()


@st.cache_data(show_spinner=False)
def load_file_paths(base_path: str, model_name: str) -> Dict[str, List[str]]:
    """
    Load test file paths from the appropriate samples folder.
    """
    test_df = load_test_metadata(base_path, model_name)

    test_files = test_df.header_h5_file_path.tolist()
    test_folder = os.path.join(base_path, "sample_files", "semantic_layer", "test")

    all_test_filepaths = [os.path.join(test_folder, os.path.basename(f)) for f in test_files]
    test_filepaths = [path for path in all_test_filepaths if os.path.isfile(path)]

    if not test_filepaths:
        raise FileNotFoundError(
            f"No valid test .h5 files found for model '{model_name}' under '{test_folder}'."
        )

    return {
        "test_filepaths": test_filepaths,
    }

@st.cache_resource(show_spinner=False)
def load_datasets(filepaths: Dict[str, List[str]]) -> Dict[str, DatasetH5]:
    """
    Load test datasets.

    Parameters:
    - filepaths (Dict): Dictionary with test filepaths

    Returns:
    - Dict with test datasets
    """
    ds_test = DatasetH5(
        source=filepaths["test_filepaths"],
        use_multiprocessing=True,
        use_cache=True
    )

    return {
        "test": ds_test
    }


@st.cache_resource(show_spinner=False)
def load_model_inputs(base_path: str, model_name: str):
    metadata = load_test_metadata(base_path, model_name)
    filepaths = load_file_paths(base_path, model_name)
    datasets = load_datasets(filepaths)

    # Keep metadata aligned with dataset sources after skipping missing files.
    available_files = {os.path.basename(path) for path in filepaths["test_filepaths"]}
    metadata = metadata[
        metadata["header_h5_file_path"].astype(str).apply(lambda x: os.path.basename(x) in available_files)
    ].reset_index(drop=True)

    return metadata, datasets["test"] # TODO: handle multiple datasets if needed
