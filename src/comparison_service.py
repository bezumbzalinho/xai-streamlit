from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from shap import Explanation

from shap_streamlit.src.data_loader import load_model_inputs
from shap_streamlit.src.model_loader import discover_models, get_features, load_model
from shap_streamlit.src.prediction_utils import build_prediction_frame, merge_predictions_into_metadata
from shap_streamlit.src.shap_analyzer import generate_shap_explanation


def _feature_domain(feature_name: str) -> str:
    if not feature_name:
        return "UNKNOWN"
    return feature_name.split("_", 1)[0]


def _build_domain_importance_frame(
    model_name: str,
    explanation: Explanation,
    feature_names: List[str],
) -> pd.DataFrame:
    shap_values = np.asarray(explanation.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    domain_frame = pd.DataFrame(
        {
            "feature_name": feature_names,
            "feature_domain": [_feature_domain(name) for name in feature_names],
            "mean_abs_shap": mean_abs,
        }
    )

    aggregated = (
        domain_frame.groupby("feature_domain", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), feature_count=("feature_name", "size"))
        .sort_values("mean_abs_shap", ascending=False)
    )
    aggregated["model_name"] = model_name
    return aggregated


def _feature_link(feature_name: str) -> str:
    if not feature_name or "_" not in feature_name:
        return "UNKNOWN"
    parts = feature_name.split("_")
    return f"{parts[-2]}_{parts[-1]}"


@dataclass
class ModelArtifacts:
    name: str
    model: Any
    pipeline: Any
    config: Dict[str, Any]
    feature_names: List[str]
    explanation: Explanation
    samples_array: np.ndarray
    sample_names: List[str]
    raw_samples: Dict[str, Any]
    tap_range: Tuple[int, int]
    links: List[str]
    metadata: pd.DataFrame
    domain_importance: pd.DataFrame

    def sample_index(self, sample_name: str) -> int:
        return self.sample_names.index(sample_name)


class ModelComparisonWorkspace:
    def __init__(self, artifacts: Dict[str, ModelArtifacts]):
        self._artifacts = artifacts

    @property
    def model_names(self) -> List[str]:
        return list(self._artifacts.keys())

    @property
    def model_data(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "model": artifact.model,
                "feature_names": artifact.feature_names,
                "explanation": artifact.explanation,
                "samples_array": artifact.samples_array,
                "sample_names": artifact.sample_names,
                "raw_samples": artifact.raw_samples,
                "tap_range": artifact.tap_range,
                "links": artifact.links,
                "pipeline": artifact.pipeline,
                "metadata": artifact.metadata,
                "domain_importance": artifact.domain_importance,
            }
            for name, artifact in self._artifacts.items()
        }

    def get_model(self, model_name: str) -> ModelArtifacts:
        return self._artifacts[model_name]

    def get_common_samples(self, model_a: str, model_b: str) -> List[str]:
        left = set(self._artifacts[model_a].sample_names)
        right = set(self._artifacts[model_b].sample_names)
        return sorted(left.intersection(right))

    def get_pair_metadata(self, model_a: str, model_b: str) -> pd.DataFrame:
        pred_cols_a = [
            "header_name",
            "true_label",
            f"predicted_label_{model_a}",
            f"predicted_proba_{model_a}",
            f"misclassified_{model_a}",
        ]
        pred_cols_b = [
            "header_name",
            f"predicted_label_{model_b}",
            f"predicted_proba_{model_b}",
            f"misclassified_{model_b}",
        ]

        left = self._artifacts[model_a].metadata[[col for col in pred_cols_a if col in self._artifacts[model_a].metadata.columns]].copy()
        right = self._artifacts[model_b].metadata[[col for col in pred_cols_b if col in self._artifacts[model_b].metadata.columns]].copy()

        passthrough_cols = [
            col
            for col in self._artifacts[model_a].metadata.columns
            if col not in left.columns and col != "header_name"
        ]
        if passthrough_cols:
            left = left.merge(
                self._artifacts[model_a].metadata[["header_name", *passthrough_cols]],
                on="header_name",
                how="left",
                validate="one_to_one",
            )

        merged = left.merge(right, on="header_name", how="inner", validate="one_to_one")
        pred_a = f"predicted_label_{model_a}"
        pred_b = f"predicted_label_{model_b}"
        merged["prediction_disagreement"] = (
            merged[pred_a].notna() & merged[pred_b].notna() & (merged[pred_a] != merged[pred_b])
        )
        merged["prediction_agreement"] = (
            merged[pred_a].notna() & merged[pred_b].notna() & (merged[pred_a] == merged[pred_b])
        )
        return merged.sort_values("header_name").reset_index(drop=True)

    def get_disagreement_samples(self, model_a: str, model_b: str) -> pd.DataFrame:
        pair_df = self.get_pair_metadata(model_a, model_b)
        return pair_df[pair_df["prediction_disagreement"]].reset_index(drop=True)

    def get_prediction_contingency(self, model_a: str, model_b: str) -> pd.DataFrame:
        pair_df = self.get_pair_metadata(model_a, model_b)
        pred_a = f"predicted_label_{model_a}"
        pred_b = f"predicted_label_{model_b}"
        return pd.crosstab(
            pair_df[pred_a],
            pair_df[pred_b],
            rownames=[model_a],
            colnames=[model_b],
            dropna=False,
        )

    def get_common_domain_importance(self, model_names: Iterable[str] | None = None) -> pd.DataFrame:
        selected_names = list(model_names or self.model_names)
        frames = [self._artifacts[name].domain_importance for name in selected_names]
        if not frames:
            return pd.DataFrame(columns=["feature_domain", "mean_abs_shap", "feature_count", "model_name"])
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _aggregate_metric(values: pd.Series, metric: str) -> float:
        if metric == "mean":
            return float(values.mean())
        if metric == "median":
            return float(values.median())
        if metric == "max":
            return float(values.max())
        if metric == "p90":
            return float(values.quantile(0.9))
        raise ValueError(f"Unsupported metric: {metric}")

    def get_feature_importance(
        self,
        model_name: str,
        *,
        metric: str = "mean",
        group_by: str = "feature",
        top_k: int | None = None,
    ) -> pd.DataFrame:
        artifact = self._artifacts[model_name]
        shap_values = np.asarray(artifact.explanation.values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]

        abs_values = np.abs(shap_values)
        per_feature = pd.DataFrame(
            {
                "feature_name": artifact.feature_names,
                "value": [self._aggregate_metric(pd.Series(abs_values[:, i]), metric) for i in range(abs_values.shape[1])],
            }
        )
        per_feature["feature_domain"] = per_feature["feature_name"].map(_feature_domain)
        per_feature["feature_link"] = per_feature["feature_name"].map(_feature_link)
        per_feature["model_name"] = model_name

        if group_by == "feature":
            out = per_feature.rename(columns={"feature_name": "group_value"})
        elif group_by == "domain":
            out = (
                per_feature.groupby("feature_domain", as_index=False)
                .agg(value=("value", "sum"), feature_count=("feature_name", "size"))
                .rename(columns={"feature_domain": "group_value"})
            )
        elif group_by == "link":
            out = (
                per_feature.groupby("feature_link", as_index=False)
                .agg(value=("value", "sum"), feature_count=("feature_name", "size"))
                .rename(columns={"feature_link": "group_value"})
            )
        else:
            raise ValueError(f"Unsupported group_by: {group_by}")

        if "feature_count" not in out.columns:
            out["feature_count"] = 1

        out["metric"] = metric
        out["group_by"] = group_by
        out["model_name"] = model_name
        out = out.sort_values("value", ascending=False).reset_index(drop=True)
        if top_k is not None and top_k > 0:
            out = out.head(top_k).copy()
        return out

    def get_common_importance(
        self,
        model_names: Iterable[str] | None = None,
        *,
        metric: str = "mean",
        group_by: str = "domain",
        top_k: int | None = None,
    ) -> pd.DataFrame:
        selected_names = list(model_names or self.model_names)
        frames = [
            self.get_feature_importance(name, metric=metric, group_by=group_by, top_k=top_k)
            for name in selected_names
        ]
        if not frames:
            return pd.DataFrame(columns=["group_value", "value", "feature_count", "metric", "group_by", "model_name"])
        return pd.concat(frames, ignore_index=True)

    def get_top_feature_indices(self, model_name: str, k: int = 10, metric: str = "mean") -> List[int]:
        artifact = self._artifacts[model_name]
        shap_values = np.asarray(artifact.explanation.values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]

        abs_values = np.abs(shap_values)
        if metric == "mean":
            scores = abs_values.mean(axis=0)
        elif metric == "median":
            scores = np.median(abs_values, axis=0)
        elif metric == "max":
            scores = abs_values.max(axis=0)
        elif metric == "p90":
            scores = np.quantile(abs_values, 0.9, axis=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        ranked = np.argsort(scores)[::-1]
        return ranked[: max(1, k)].tolist()

    def get_common_feature_names(self, model_a: str, model_b: str) -> List[str]:
        left = set(self._artifacts[model_a].feature_names)
        right = set(self._artifacts[model_b].feature_names)
        return sorted(left.intersection(right))


@st.cache_resource(show_spinner=False)
def load_model_artifacts(base_path: str, model_name: str) -> ModelArtifacts:
    app_base_path = f"{base_path}/shap_streamlit"
    model, pipeline, config = load_model(model_name, app_base_path)
    feature_names, tap_range, links = get_features(pipeline, config["machine_learning"]["model_params"])

    metadata, dataset = load_model_inputs(base_path, model_name)
    explanation, samples_array, sample_names, raw_samples = generate_shap_explanation(
        model,
        dataset,
        links,
        pipeline,
        tap_range,
        feature_names,
        None,
    )

    predictions = build_prediction_frame(model_name, model, samples_array, sample_names)
    enriched_metadata = merge_predictions_into_metadata(metadata, model_name, predictions)
    domain_importance = _build_domain_importance_frame(model_name, explanation, feature_names)

    return ModelArtifacts(
        name=model_name,
        model=model,
        pipeline=pipeline,
        config=config,
        feature_names=feature_names,
        explanation=explanation,
        samples_array=samples_array,
        sample_names=sample_names,
        raw_samples=raw_samples,
        tap_range=tap_range,
        links=links,
        metadata=enriched_metadata,
        domain_importance=domain_importance,
    )


@st.cache_resource(show_spinner=False)
def load_comparison_workspace(base_path: str, model_names: Tuple[str, ...]) -> ModelComparisonWorkspace:
    artifacts = {model_name: load_model_artifacts(base_path, model_name) for model_name in model_names}
    return ModelComparisonWorkspace(artifacts)


def discover_available_models(base_path: str) -> List[str]:
    return discover_models(f"{base_path}/shap_streamlit", require_files=False, include_hidden=False)