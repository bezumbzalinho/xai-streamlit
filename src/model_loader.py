import os
from xgboost import XGBClassifier
from typing import Dict, List, Tuple, Any
from cpd.core.data.can_fd_messages import LinkID
from cpd.core.data.process.base_data_process_step import ProcessingPipeline
from shap_streamlit.src.config_loader import read_config

def load_model(model_name: str, base_path: str) -> Tuple[XGBClassifier, ProcessingPipeline, Dict[str, Any]]:
    """
    Load XGBoost model and its config.

    Parameters:
    - model_name (str): Name of the model
    - base_path (str): Project base path

    Returns:
    - Tuple of (model, config)
    """
    config = read_config(os.path.join(base_path, "models", model_name, "config.yaml"))

    pipeline = ProcessingPipeline.initiate_preprocessing_from_config(
        config["machine_learning"]["preprocessing"]
    )
    
    model = XGBClassifier()
    model._estimator_type = "classifier"
    model.load_model(os.path.join(base_path, "models", model_name, "model.xgb"))

    return model, pipeline, config

def get_features(pipeline_process: ProcessingPipeline, model_params: Dict[str, Any]):
    """
    Generate feature names for the model.

    Parameters:
    - model_name (str): Name of the model
    - config (Dict): Model configuration

    Returns:
    - Tuple of (feature_names, tap_range)
    """
    tap_range = tuple(model_params["tap_range"])
    feature_names = []  
    for link in model_params["links"]:
        link = LinkID(link)
        names = pipeline_process.get_features_name(
            tap_range=tap_range, link_id=link
        )
        for name in names:
            feature_names.append(f"{name}_{link.name}")

    return feature_names, tap_range, model_params["links"]

def discover_models(
    base_path: str,
    models_subdir: str = "models",
    require_files: bool = True,
    include_hidden: bool = False,
    model_filename: str = "model.xgb",
    config_filename: str = "config.yaml"
) -> List[str]:
    """
    Discover available model folders under base_path/models_subdir.

    Parameters:
    - base_path: project root
    - models_subdir: subdirectory containing model folders
    - require_files: if True, only include folders that contain both model + config
    - model_filename: model file to check
    - config_filename: config file to check

    Returns:
    - list of model folder names
    """
    root = os.path.join(base_path, models_subdir)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Models directory not found: {root}")

    model_names: List[str] = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        if entry.startswith(".") and not include_hidden:
            continue
        if require_files:
            model_path = os.path.join(full, model_filename)
            config_path = os.path.join(full, config_filename)
            if not (os.path.isfile(model_path) and os.path.isfile(config_path)):
                continue
        model_names.append(entry)
    if not model_names:
        raise RuntimeError(f"No models discovered in {root} (require_files={require_files}).")
    return model_names