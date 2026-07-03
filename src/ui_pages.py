import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components

from shap_streamlit.src.comparison_service import ModelComparisonWorkspace

from shap import decision_plot
from shap_streamlit.src.metadata_filters import render_metadata_filters
from shap_streamlit.src.shap_visualizer import (
    generate_signal_inspector_figs,
    plot_group_importance,
    plot_shap_dependence,
    plot_shap_summary,
    plot_shap_waterfall,
)

def _importance_y_label(metric: str) -> str:
    return {
        "mean": "Mean |SHAP|",
        "median": "Median |SHAP|",
        "max": "Max |SHAP|",
        "p90": "P90 |SHAP|",
    }.get(metric, f"{metric} |SHAP|")


def _importance_x_label(group_by: str) -> str:
    return {
        "feature": "Feature",
        "domain": "Feature domain",
        "link": "Link",
    }.get(group_by, group_by)


def _slice_explanation(explanation, mask: pd.Series):
    if mask is None or mask.empty:
        return explanation
    keep_idx = mask[mask].index.to_list()
    return explanation[keep_idx]


def _summary_mask_for_model(
    metadata: pd.DataFrame,
    model_name: str,
    sample_scope: str,
    selected_subclasses: list[str],
    selected_descriptions: list[str],
):
    mask = pd.Series(True, index=metadata.index)
    if sample_scope == "Misclassified":
        col = f"misclassified_{model_name}"
        if col in metadata.columns:
            mask = mask & metadata[col]
    elif sample_scope == "False positives":
        col = f"false_positive_{model_name}"
        if col in metadata.columns:
            mask = mask & metadata[col]
    elif sample_scope == "False negatives":
        col = f"false_negative_{model_name}"
        if col in metadata.columns:
            mask = mask & metadata[col]
    elif sample_scope == "Correct":
        col = f"misclassified_{model_name}"
        if col in metadata.columns:
            mask = mask & (~metadata[col])

    subclass_col = "subclass_inside_outside" if "subclass_inside_outside" in metadata.columns else None
    if selected_subclasses and subclass_col:
        mask = mask & metadata[subclass_col].astype(str).isin(selected_subclasses)

    if selected_descriptions and "header_description" in metadata.columns:
        mask = mask & metadata["header_description"].astype(str).isin(selected_descriptions)

    return mask


def _render_force_plot_many(explanation, feature_names, *, max_rows: int = 1000, height: int = 340):
    values = np.asarray(explanation.values)
    data = np.asarray(explanation.data)
    if values.ndim == 3:
        values = values[:, :, 0]

    rows = min(max_rows, len(values))
    if rows == 0:
        st.info("No samples available for force plot with current filter.")
        return

    base_values = np.asarray(explanation.base_values)
    base_value = float(np.mean(base_values))

    force = shap.force_plot(
        base_value,
        values[:rows, :],
        data[:rows, :],
        feature_names=feature_names,
        matplotlib=False,
        show=False,
    )
    components.html(shap.getjs() + force.html(), height=height, scrolling=True)


def _prepare_force_plot_inputs(
    explanation,
    feature_names: list[str],
    *,
    top_n: int,
):
    values = np.asarray(explanation.values)
    data = np.asarray(explanation.data)
    if values.ndim == 3:
        values = values[:, :, 0]

    mean_abs = np.abs(values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]
    values = values[:, top_indices]
    data = data[:, top_indices]
    selected_feature_names = [feature_names[i] for i in top_indices]

    return values, data, selected_feature_names


def _render_filtered_force_plot(
    explanation,
    feature_names: list[str],
    *,
    top_n: int,
    start_idx: int,
    end_idx: int,
    height: int = 340,
):
    values, data, selected_feature_names = _prepare_force_plot_inputs(
        explanation,
        feature_names,
        top_n=top_n,
    )

    if len(values) == 0:
        st.info("No samples available for force plot with current filter.")
        return

    start_idx = max(0, min(start_idx, len(values) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(values)))
    values = values[start_idx:end_idx, :]
    data = data[start_idx:end_idx, :]

    base_values = np.asarray(explanation.base_values)
    base_value = float(np.mean(base_values))
    force = shap.force_plot(
        base_value,
        values,
        data,
        feature_names=selected_feature_names,
        matplotlib=False,
        show=False,
    )
    html = (
        "<div style='background:#ffffff;color:#111111;padding:8px;border-radius:8px;'>"
        + shap.getjs()
        + force.html()
        + "</div>"
    )
    components.html(html, height=height, scrolling=True)


def render_summary_page(workspace: ModelComparisonWorkspace):
    st.header("SHAP Summary Plots")
    model_data = workspace.model_data
    models = list(model_data.keys())

    with st.expander("Summary Plot (all or filtered samples)", expanded=True):
        ctl_cols = st.columns(5)
        with ctl_cols[0]:
            selected_models = st.multiselect("Models", models, default=models)
        with ctl_cols[1]:
            sample_scope = st.selectbox("Sample scope", ["All", "Misclassified", "False positives", "False negatives", "Correct"], index=0)
        with ctl_cols[2]:
            max_display = st.slider("Top features in summary", min_value=5, max_value=50, value=20, step=1)

        meta_source_model = selected_models[0] if selected_models else models[0]
        meta_source = model_data[meta_source_model]["metadata"]
        with ctl_cols[3]:
            subclass_values = (
                sorted(meta_source["subclass_inside_outside"].dropna().astype(str).unique())
                if "subclass_inside_outside" in meta_source.columns
                else []
            )
            selected_subclasses = st.multiselect("Subclass", subclass_values)
        with ctl_cols[4]:
            desc_values = (
                sorted(meta_source["header_description"].dropna().astype(str).unique())
                if "header_description" in meta_source.columns
                else []
            )
            selected_descriptions = st.multiselect("Header description", desc_values)

        chosen_models = selected_models or models
        cols = st.columns(len(chosen_models))
        filtered_explanations = {}
        for c, m in zip(cols, chosen_models):
            with c:
                metadata = model_data[m]["metadata"]
                mask = _summary_mask_for_model(metadata, m, sample_scope, selected_subclasses, selected_descriptions)
                filtered_count = int(mask.sum())
                st.subheader(m)
                st.caption(f"Filtered samples: {filtered_count} / {len(metadata)}")
                if filtered_count == 0:
                    st.warning("No samples for current filter.")
                    continue
                filtered_exp = _slice_explanation(model_data[m]["explanation"], mask)
                filtered_explanations[m] = filtered_exp
                fig = plot_shap_summary(
                    filtered_exp,
                    model_data[m]["feature_names"],
                    f"SHAP Feature Importance - {m}",
                    max_display=max_display,
                )
                st.pyplot(fig)

        st.markdown("**Force plot (many predictions)**")
        filters_active = (
            sample_scope != "All" or bool(selected_subclasses) or bool(selected_descriptions)
        )
        if not filters_active:
            st.info("Apply a sample scope, subclass, or header description filter to enable the force plot.")
        else:
            force_cols = st.columns(2)
            with force_cols[0]:
                force_model = st.selectbox("Force plot model", chosen_models)
            with force_cols[1]:
                top_n = st.slider(
                    "Top features",
                    min_value=5,
                    max_value=30,
                    value=15,
                    step=1,
                )

            if force_model in filtered_explanations:
                total_samples = len(filtered_explanations[force_model].values)
                window_cols = st.columns(2)
                with window_cols[0]:
                    start_idx = st.number_input(
                        "Start sample index",
                        min_value=0,
                        max_value=max(0, total_samples - 1),
                        value=0,
                        step=1,
                    )
                max_end = min(total_samples, int(start_idx) + 2000)
                default_end = min(total_samples, int(start_idx) + 1000)
                with window_cols[1]:
                    end_idx = st.number_input(
                        "End sample index (exclusive)",
                        min_value=int(start_idx) + 1,
                        max_value=max_end,
                        value=default_end,
                        step=1,
                    )
                st.caption(
                    f"Showing samples [{int(start_idx)}, {int(end_idx)}) out of {total_samples}. "
                    "Maximum window size is 2000 samples."
                )

                _render_filtered_force_plot(
                    filtered_explanations[force_model],
                    model_data[force_model]["feature_names"],
                    top_n=top_n,
                    start_idx=int(start_idx),
                    end_idx=int(end_idx),
                )
            else:
                st.info("Force plot unavailable: no samples for this model with the current filter.")

    with st.expander("Grouped Importance (domain/link)", expanded=True):
        ctl_cols = st.columns(3)
        with ctl_cols[0]:
            metric = st.selectbox("Importance metric", ["mean", "median", "p90", "max"], index=0, key="group_metric")
        with ctl_cols[1]:
            group_by = st.selectbox("Group by", ["domain", "link"], index=0)
        with ctl_cols[2]:
            selected_models_group = st.multiselect("Models", models, default=models, key="group_models")

        chosen_models_group = selected_models_group or models
        group_df = workspace.get_common_importance(
            chosen_models_group,
            metric=metric,
            group_by=group_by,
            top_k=None,
        )
        x_label = _importance_x_label(group_by)
        y_label = _importance_y_label(metric)

        st.plotly_chart(
            plot_group_importance(group_df, f"Grouped importance by {x_label.lower()}", x_label, y_label),
            use_container_width=True,
        )

        st.subheader("Top Contributions Per Model") if len(chosen_models_group) > 1 else st.subheader("Top Contributions")
        top_cols = st.columns(len(chosen_models_group))
        for c, m in zip(top_cols, chosen_models_group):
            with c:
                top_df = workspace.get_feature_importance(m, metric=metric, group_by=group_by, top_k=10)
                st.markdown(f"**{m}**")
                st.dataframe(top_df[["group_value", "value", "feature_count"]], use_container_width=True, hide_index=True)


def _render_dependence_pair(workspace: ModelComparisonWorkspace, model_a: str, feature_a: str, model_b: str, feature_b: str):
    model_data = workspace.model_data
    idx_a = model_data[model_a]["feature_names"].index(feature_a)
    idx_b = model_data[model_b]["feature_names"].index(feature_b)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{model_a}: {feature_a}**")
        fig_a = plot_shap_dependence(
            model_data[model_a]["explanation"],
            idx_a,
            model_data[model_a]["feature_names"],
            f"Dependence - {model_a} - {feature_a}",
        )
        st.plotly_chart(fig_a, use_container_width=True)
    with col_b:
        st.markdown(f"**{model_b}: {feature_b}**")
        fig_b = plot_shap_dependence(
            model_data[model_b]["explanation"],
            idx_b,
            model_data[model_b]["feature_names"],
            f"Dependence - {model_b} - {feature_b}",
        )
        st.plotly_chart(fig_b, use_container_width=True)


def render_feature_dependence_page(workspace: ModelComparisonWorkspace):
    st.header("Feature Dependence Analysis")
    model_data = workspace.model_data
    model_names = list(model_data.keys())

    ctl_cols = st.columns(3)
    with ctl_cols[0]:
        view_mode = st.radio("Mode", ["Single model (top-k)", "Compare two models"], horizontal=True)
    with ctl_cols[1]:
        importance_metric = st.selectbox("Importance metric", ["mean", "median", "p90", "max"], index=0)
    with ctl_cols[2]:
        top_k = st.slider("Top-k important features", min_value=3, max_value=40, value=12, step=1)

    if view_mode == "Single model (top-k)":
        col_left, col_right = st.columns([1, 2])
        with col_left:
            selected_model = st.selectbox("Select model", model_names)
            top_indices = workspace.get_top_feature_indices(selected_model, k=top_k, metric=importance_metric)
            feats = model_data[selected_model]["feature_names"]
            top_feats = [feats[i] for i in top_indices]
            selected_feature = st.selectbox("Feature (top-k)", top_feats)
            st.caption("Only the top-k most important features are shown.")
        with col_right:
            feat_idx = feats.index(selected_feature)
            fig = plot_shap_dependence(
                model_data[selected_model]["explanation"],
                feat_idx,
                feats,
                f"Dependence - {selected_model} - {selected_feature}",
            )
            st.plotly_chart(fig, use_container_width=True)
        return

    compare_cols = st.columns(3)
    with compare_cols[0]:
        m1 = st.selectbox("Model A", model_names, index=0)
    with compare_cols[1]:
        m2_default = 1 if len(model_names) > 1 else 0
        m2 = st.selectbox("Model B", model_names, index=m2_default)
    with compare_cols[2]:
        compare_mode = st.selectbox("Compare strategy", ["Same feature name", "Top-feature by rank"])

    if m1 == m2:
        st.warning("Select two different models for comparison mode.")
        return

    if compare_mode == "Same feature name":
        common_feats = workspace.get_common_feature_names(m1, m2)
        if not common_feats:
            st.warning("No shared feature names found between the selected models. Use 'Top-feature by rank' mode.")
            return
        selected_feature = st.selectbox("Shared feature", common_feats)
        _render_dependence_pair(workspace, m1, selected_feature, m2, selected_feature)
        return

    top_idx_m1 = workspace.get_top_feature_indices(m1, k=top_k, metric=importance_metric)
    top_idx_m2 = workspace.get_top_feature_indices(m2, k=top_k, metric=importance_metric)
    f1 = model_data[m1]["feature_names"]
    f2 = model_data[m2]["feature_names"]
    max_rank = min(len(top_idx_m1), len(top_idx_m2), top_k)
    rank = st.slider("Rank to compare", min_value=1, max_value=max_rank, value=1, step=1)

    feature_a = f1[top_idx_m1[rank - 1]]
    feature_b = f2[top_idx_m2[rank - 1]]
    st.caption(f"Comparing top-{rank} feature from each model.")
    _render_dependence_pair(workspace, m1, feature_a, m2, feature_b)


def _render_prediction_badge(model_name: str, sample_meta_row, *, label_map=None):
    gt = sample_meta_row["true_label"] if "true_label" in sample_meta_row else None
    pred_col = f"predicted_label_{model_name}"
    proba_col = f"predicted_proba_{model_name}"
    pred = sample_meta_row[pred_col] if pred_col in sample_meta_row else None
    proba = sample_meta_row[proba_col] if proba_col in sample_meta_row else None

    if label_map:
        gt_display = label_map.get(gt, gt)
        pred_display = label_map.get(pred, pred)
    else:
        gt_display = gt
        pred_display = pred

    if (gt is not None) and (pred is not None):
        correct = gt == pred
        status = "Correct" if correct else "Misclassified"
        color = "#2e7d32" if correct else "#c62828"
    else:
        status = "Prediction"
        color = "#455a64"

    html = f"""
    <div style="
        border:1px solid {color};
        border-radius:6px;
        padding:8px 10px;
        background:rgba(0,0,0,0.03);
        ">
        <strong style="color:{color}">{model_name}</strong><br/>
        <span style="font-size:16px;">
            Ground Truth: <b>{gt_display}</b><br/>
            Prediction: <b>{pred_display}</b>{f" (p={proba:.2f})" if proba is not None else ""}<br/>
            <span style="color:{color}; font-weight:600;">{status}</span>
        </span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_raw_signal(model_data, sample_name, selected_models):
    st.markdown("**Amplitude scale (leave blank or zero for auto scaling):**")
    max_amp = st.number_input("Amplitude scale (max abs value)", min_value=0, value=0, step=1)
    y_limits = (-max_amp, max_amp) if max_amp > 0 else None

    cols = st.number_input("Columns", 1, 6, 3)
    method = st.selectbox("Plot method", ["REAL", "IMAG", "ABS", "PHASE"], index=0)
    source_model = st.selectbox("Source model (for model tap range)", selected_models, index=0)
    raw_samples = model_data[source_model]["raw_samples"]
    if sample_name not in raw_samples:
        st.error("Sample not found in raw_samples.")
        return
    figs = generate_signal_inspector_figs(
        raw_samples,
        sample_name,
        start_tap=model_data[source_model]["tap_range"][0],
        window_size=72,
        window_shift=50000,
        sampling_rate=96,
        remove_dc=True,
        plot_method=method,
        y_limits=y_limits,
        num_columns=cols,
    )
    for link_name, fig in figs.items():
        st.markdown(f"**Link: {link_name}**")
        st.pyplot(fig)


def _build_decision_plot_figure(*, base_value, shap_values, features, feature_names, title: str):
    plt.figure(figsize=(10, 6))
    decision_plot(
        base_value=base_value,
        shap_values=shap_values,
        features=features,
        feature_names=feature_names,
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def render_instance_analysis_page(workspace: ModelComparisonWorkspace):
    st.header("Instance-level Comparison")
    model_data = workspace.model_data
    model_names = list(model_data.keys())

    selected_models = st.multiselect("Select two models", model_names, default=model_names[:2])

    if len(selected_models) != 2:
        st.warning("Select exactly two models.")
        return
    m1, m2 = selected_models

    pair_metadata = workspace.get_pair_metadata(m1, m2)
    disagreement_df = workspace.get_disagreement_samples(m1, m2)
    filtered_pair = pair_metadata
    with st.expander("Metadata Preview & Filtering"):
        st.subheader("Prediction Contingency Matrix")
        st.dataframe(workspace.get_prediction_contingency(m1, m2), use_container_width=True)
        st.caption(f"Prediction disagreements: {len(disagreement_df)} of {len(pair_metadata)} shared samples")
        filter_result = render_metadata_filters(pair_metadata, m1, m2, show_editor=True)
        candidate = filter_result.get("filtered_df")
        if candidate is not None and not candidate.empty:
            filtered_pair = candidate

    quick_filter = st.selectbox(
        "Quick sample scope",
        [
            "Filtered metadata",
            "Prediction disagreement only",
            f"False positives ({m1})",
            f"False negatives ({m1})",
            f"False positives ({m2})",
            f"False negatives ({m2})",
            "Any misclassified",
            "All shared samples",
        ],
    )

    if quick_filter == "Filtered metadata":
        candidate_df = filtered_pair
    elif quick_filter == "Prediction disagreement only":
        candidate_df = disagreement_df
    elif quick_filter == "Any misclassified":
        miss_a = f"misclassified_{m1}"
        miss_b = f"misclassified_{m2}"
        mask = pd.Series(False, index=filtered_pair.index)
        if miss_a in filtered_pair.columns:
            mask = mask | filtered_pair[miss_a]
        if miss_b in filtered_pair.columns:
            mask = mask | filtered_pair[miss_b]
        candidate_df = filtered_pair[mask]
    elif quick_filter == f"False positives ({m1})":
        col = f"false_positive_{m1}"
        candidate_df = filtered_pair[filtered_pair[col]] if col in filtered_pair.columns else filtered_pair.iloc[0:0]
    elif quick_filter == f"False negatives ({m1})":
        col = f"false_negative_{m1}"
        candidate_df = filtered_pair[filtered_pair[col]] if col in filtered_pair.columns else filtered_pair.iloc[0:0]
    elif quick_filter == f"False positives ({m2})":
        col = f"false_positive_{m2}"
        candidate_df = filtered_pair[filtered_pair[col]] if col in filtered_pair.columns else filtered_pair.iloc[0:0]
    elif quick_filter == f"False negatives ({m2})":
        col = f"false_negative_{m2}"
        candidate_df = filtered_pair[filtered_pair[col]] if col in filtered_pair.columns else filtered_pair.iloc[0:0]
    else:
        candidate_df = pair_metadata

    common = candidate_df["header_name"].dropna().astype(str).tolist()

    with st.container():
        st.subheader("Select Sample for Analysis")

        col_filter, col_select = st.columns([2, 3])
        with col_filter:
            filter_text = st.text_input("Filter samples (substring match)", "")
            candidates = [s for s in common if filter_text.lower() in s.lower()] if filter_text else common
            if not candidates:
                st.error("No samples match filter.")
                return
        with col_select:
            pretty = lambda n: os.path.basename(n)
            sample_name = st.selectbox("Sample", candidates, format_func=pretty)

        sample_meta_row = None
        if not pair_metadata.empty and "header_name" in pair_metadata.columns:
            sample_meta = pair_metadata[pair_metadata["header_name"] == sample_name]
            if not sample_meta.empty:
                sample_meta_row = sample_meta.iloc[0]
                light_cols = [
                    c
                    for c in [
                        "header_name",
                        "true_label",
                        f"predicted_label_{m1}",
                        f"predicted_label_{m2}",
                        "prediction_disagreement",
                        "subclass_inside_outside",
                        "header_description",
                        "all_comments",
                    ]
                    if c in sample_meta.columns
                ]
                st.dataframe(sample_meta[light_cols], use_container_width=True)

    if sample_meta_row is not None:
        badge_cols = st.columns(2)
        for col, m in zip(badge_cols, selected_models):
            with col:
                _render_prediction_badge(m, sample_meta_row)

    idx1 = model_data[m1]["sample_names"].index(sample_name)
    idx2 = model_data[m2]["sample_names"].index(sample_name)

    def correct_icon(model_name):
        if sample_meta_row is None or "true_label" not in sample_meta_row.index:
            return ""
        pred_col = f"predicted_label_{model_name}"
        if pred_col not in sample_meta_row.index:
            return ""
        gt = sample_meta_row["true_label"]
        pred = sample_meta_row[pred_col]
        return "✅" if gt == pred else "❌"

    with st.expander("Waterfall Plots", expanded=True):
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.subheader(f"{m1} {correct_icon(m1)}")
            fig1 = plot_shap_waterfall(
                model_data[m1]["explanation"],
                idx1,
                f"{m1} - {os.path.basename(sample_name)}",
            )
            st.pyplot(fig1)
        with col_w2:
            st.subheader(f"{m2} {correct_icon(m2)}")
            fig2 = plot_shap_waterfall(
                model_data[m2]["explanation"],
                idx2,
                f"{m2} - {os.path.basename(sample_name)}",
            )
            st.pyplot(fig2)

    with st.expander("Decision Plots", expanded=False):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.subheader(f"{m1} {correct_icon(m1)}")
            fig1 = _build_decision_plot_figure(
                base_value=model_data[m1]["explanation"].base_values[idx1],
                shap_values=model_data[m1]["explanation"].values[idx1],
                features=model_data[m1]["explanation"].data[idx1],
                feature_names=model_data[m1]["feature_names"],
                title=f"{m1} - {os.path.basename(sample_name)}",
            )
            st.pyplot(fig1)
            plt.close(fig1)
        with col_d2:
            st.subheader(f"{m2} {correct_icon(m2)}")
            fig2 = _build_decision_plot_figure(
                base_value=model_data[m2]["explanation"].base_values[idx2],
                shap_values=model_data[m2]["explanation"].values[idx2],
                features=model_data[m2]["explanation"].data[idx2],
                feature_names=model_data[m2]["feature_names"],
                title=f"{m2} - {os.path.basename(sample_name)}",
            )
            st.pyplot(fig2)
            plt.close(fig2)

    show_signal = st.checkbox("Show raw signal (Inspector)", value=False)
    if show_signal:
        _render_raw_signal(model_data, sample_name, selected_models)
