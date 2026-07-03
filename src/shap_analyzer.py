import numpy as np
import shap
from typing import Dict, List, Any, Tuple
from xgboost import XGBClassifier
from cpd.core.data.can_fd_messages import LinkID
from cpd.core.data.dataset import DatasetH5
from cpd.core.data.process.base_data_process_step import ProcessingPipeline
from shap import Explanation

def generate_shap_explanation(
    model: XGBClassifier,
    dataset: DatasetH5,
    links: List[str],
    pipeline: ProcessingPipeline,
    tap_range: Tuple[int, int],
    feature_names: List[str],
    feature_selection_columns: List[int] = None
) -> Explanation:
    """
    Generate SHAP explanation for a model.

    Parameters:
    - model: XGBoost model
    - dataset: Dataset to explain
    - links: Links to use for explanation
    - feature_names: Names of features

    Returns:
    - SHAP explanation object
    """
    # Get samples
    links_id = [LinkID(link) for link in links]
    links_values = [LinkID(link).value for link in links]
    raw_samples = dataset.get_windows_by_links(links=links_values, tap_range=tap_range)
    sample_names = list(raw_samples.keys())

    # Process samples
    X_sample = []
    for filename in raw_samples.keys():
        aux_sample = raw_samples[filename][0]
        aux_output = np.array(
            [pipeline(aux_sample[link.name].reshape(1, 72, -1, 2), link_id=link.value) for link in links_id]
        )
        X_sample.append(aux_output.flatten())
    X_sample = np.array(X_sample)

    if feature_selection_columns is not None:
        X_sample = X_sample[:, feature_selection_columns]
        feature_names = [feature_names[i] for i in feature_selection_columns]

    # Validate feature count against model to avoid XGBoost predict shape errors.
    expected_features = model.get_booster().num_features()
    
    if X_sample.shape[1] != expected_features:
        raise ValueError(
            "Feature count mismatch between model and samples. "
            f"Model expects {expected_features}, got {X_sample.shape[1]}. "
            "Check feature selection, pipeline config, and tap/link settings."
        )

    # Generate SHAP values
    explainer = shap.TreeExplainer(model)
    explainer_values = explainer(X_sample)

    # Create and return explanation object
    explanation = Explanation(
        explainer_values.values,
        explainer_values.base_values,
        data=X_sample,
        feature_names=feature_names
    )

    return explanation, X_sample, sample_names, raw_samples

