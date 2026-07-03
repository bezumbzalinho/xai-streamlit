import numpy as np
import pandas as pd

def build_prediction_frame(
    model_name: str,
    model,
    X_samples: np.ndarray,
    sample_names: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute prediction columns and return them keyed by header_name.
    """
    if len(sample_names) != len(X_samples):
        raise ValueError(
            f"Prediction inputs are misaligned: {len(sample_names)} sample names for {len(X_samples)} samples."
        )

    proba = model.predict_proba(X_samples)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pd.DataFrame(
        {
            "header_name": sample_names,
            f"predicted_proba_{model_name}": proba,
            f"predicted_label_{model_name}": pred,
        }
    )


def merge_predictions_into_metadata(
    metadata: pd.DataFrame,
    model_name: str,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    merged = metadata.merge(predictions, on="header_name", how="left", validate="one_to_one")
    pred_col = f"predicted_label_{model_name}"
    if "true_label" in merged.columns:
        merged[f"misclassified_{model_name}"] = merged["true_label"] != merged[pred_col]
    return merged


def update_model_predictions(model_name: str, model, X_samples: np.ndarray):
    raise RuntimeError(
        "update_model_predictions is deprecated. Use build_prediction_frame() and merge_predictions_into_metadata()."
    )