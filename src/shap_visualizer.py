import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import shap
from typing import List, Dict, Tuple
from shap import Explanation
import numpy as np

def plot_shap_summary(
    explanation: Explanation,
    feature_names: List[str],
    title: str,
    max_display: int = 20,
    figsize=(15, 8)
):
    """
    Plot SHAP summary (beeswarm) for a model and return a matplotlib Figure.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    shap.summary_plot(
        explanation,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
        plot_size=figsize,
        color_bar=True
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig

def plot_shap_dependence(
    explanation: Explanation,
    feature_idx: int,
    feature_names: List[str],
    title: str,
    figsize=(10, 6),
):
    """
    Create an interactive SHAP dependence scatter plot (Plotly) for a feature.

    Args:
        explanation: shap.Explanation with .values (n_samples, n_features) and .data.
        feature_idx: index of the feature to plot on x-axis.
        feature_names: list of feature names.
        title: figure title.
        figsize: (width, height) in inches; converted to pixels (~96 dpi) for Plotly layout.

    Returns:
        plotly.graph_objects.Figure
    """
    shap_values = explanation.values
    if shap_values.ndim == 3:
        if shap_values.shape[0] == explanation.data.shape[0]:
            shap_values = shap_values[:, :, 0]
        else:
            shap_values = shap_values[0]

    X = np.asarray(explanation.data)
    if feature_idx >= X.shape[1]:
        raise IndexError(f"feature_idx {feature_idx} out of bounds for data with {X.shape[1]} features.")

    x_vals = X[:, feature_idx]
    y_vals = shap_values[:, feature_idx]

    fig = px.scatter(
        x=x_vals,
        y=y_vals,
        labels={
            "x": feature_names[feature_idx],
            "y": "SHAP value"
        },
        hover_data={
            feature_names[feature_idx]: x_vals,
            "SHAP value": y_vals
        }
    )

    fig.update_traces(marker=dict(size=8, opacity=0.75), selector=dict(mode="markers"))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")

    width_px = int(figsize[0] * 96)
    height_px = int(figsize[1] * 96)
    fig.update_layout(
        title=title,
        width=width_px,
        height=height_px,
        template="plotly_white",
        margin=dict(l=60, r=30, t=80, b=60),
    )

    return fig

def plot_shap_waterfall(
    explanation: Explanation,
    instance_idx: int,
    title: str = "",
    max_display: int = 20,
    figsize=(10, 8)
):
    """
    Plot SHAP waterfall plot for a specific instance.
    """
    fig, ax = plt.subplots(figsize=figsize)
    shap.plots.waterfall(
        explanation[instance_idx],
        show=False,
        max_display=max_display
    )
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_common_domain_importance(domain_df: pd.DataFrame, title: str):
    if domain_df.empty:
        return px.bar(title=title)

    order = (
        domain_df.groupby("feature_domain")["mean_abs_shap"]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    fig = px.bar(
        domain_df,
        x="feature_domain",
        y="mean_abs_shap",
        color="model_name",
        barmode="group",
        category_orders={"feature_domain": list(order)},
        hover_data={"feature_count": True, "mean_abs_shap": ":.4f"},
        labels={
            "feature_domain": "Feature domain",
            "mean_abs_shap": "Mean |SHAP|",
            "model_name": "Model",
        },
        title=title,
    )
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=70, b=40))
    return fig


def plot_group_importance(group_df: pd.DataFrame, title: str, x_label: str, y_label: str):
    if group_df.empty:
        return px.bar(title=title)

    order = (
        group_df.groupby("group_value")["value"]
        .max()
        .sort_values(ascending=False)
        .index
    )

    fig = px.bar(
        group_df,
        x="group_value",
        y="value",
        color="model_name",
        barmode="group",
        category_orders={"group_value": list(order)},
        hover_data={"feature_count": True, "metric": True},
        labels={
            "group_value": x_label,
            "value": y_label,
            "model_name": "Model",
        },
        title=title,
    )
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=70, b=40))
    return fig

def generate_signal_inspector_figs(
    raw_samples: dict,
    sample_name: str,
    *,
    start_tap: int = 10,
    window_size: int = 72,
    window_shift: int = 50000,
    sampling_rate: int = 96,
    remove_dc: bool = True,
    plot_method: str = "ABS",
    y_limits: Tuple[float, float] = None,
    num_columns: int = 3
) -> Dict[str, plt.Figure]:
    """
    Build Inspector tap plots for a given raw sample.

    Parameters:
        raw_samples: dict returned by DatasetH5.get_windows_by_links(...)
                     Structure: { sample_name: [ { link_id: np.ndarray, 'window_idx': int, ... }, ... ] }
        sample_name: key inside raw_samples
        start_tap: first tap
        window_size: number of taps in window
        window_shift: shift param used originally (only for display metadata)
        sampling_rate: Hz
        remove_dc: remove dc component
        plot_method: one of InspectorPlotMethod values (string)
        y_limits: tuple applied to every axis, or None for no limits
        num_columns: grid columns for taps

    Returns:
        dict: mapping link (or class) -> matplotlib Figure
    """
    from copy import deepcopy
    from cpd.core.data.inspector import Inspector, InspectorPlotMethod

    if sample_name not in raw_samples:
        raise KeyError(f"Sample '{sample_name}' not found in raw_samples.")

    # Take first window for that sample
    sample_window = deepcopy(raw_samples[sample_name][0])
    # Remove window_idx if present (Inspector does not expect it)
    sample_window.pop("window_idx", None)

    inspector = Inspector(
        sample=sample_window,
        start_tap=start_tap,
        window_shift=window_shift,
        window_size=window_size,
        sampling_rate=sampling_rate,
        remove_dc=remove_dc,
        plot_method=getattr(InspectorPlotMethod, plot_method),
    )

    figs = inspector.plot_taps(num_columns=num_columns)

    # Apply y-limits only if provided
    if y_limits is not None:
        for link_name, fig in figs.items():
            for ax in fig.axes:
                ax.set_ylim(*y_limits)
    return figs