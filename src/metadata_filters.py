import streamlit as st
import pandas as pd
from typing import Dict

def add_status_columns(df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    work = df.copy()
    miss_a = f"misclassified_{model_a}"
    miss_b = f"misclassified_{model_b}"
    if miss_a not in work.columns or miss_b not in work.columns:
        return work
    work["a_miss_b_hit"] = work[miss_a] & (~work[miss_b])
    work["b_miss_a_hit"] = work[miss_b] & (~work[miss_a])
    work["both_miss"] = work[miss_a] & work[miss_b]
    work["both_hit"] = (~work[miss_a]) & (~work[miss_b])
    pred_a = f"predicted_label_{model_a}"
    pred_b = f"predicted_label_{model_b}"
    if pred_a in work.columns and pred_b in work.columns:
        work["prediction_disagreement"] = (
            work[pred_a].notna() & work[pred_b].notna() & (work[pred_a] != work[pred_b])
        )
        work["prediction_agreement"] = (
            work[pred_a].notna() & work[pred_b].notna() & (work[pred_a] == work[pred_b])
        )
    if "true_label" in work.columns:
        pred_cols = [pred_a, pred_b]
        for pred_col in pred_cols:
            if pred_col in work.columns:
                suffix = pred_col.replace("predicted_label_", "")
                work[f"false_positive_{suffix}"] = (work[pred_col] == 1) & (work["true_label"] == 0)
                work[f"false_negative_{suffix}"] = (work[pred_col] == 0) & (work["true_label"] == 1)
    return work

def build_scenario_labels(model_a: str, model_b: str) -> Dict[str, str]:
    return {
        "prediction_disagreement": f"{model_a} and {model_b} predict different labels",
        "prediction_agreement": f"{model_a} and {model_b} predict the same label",
        "a_miss_b_hit": f"{model_a} misclassified AND {model_b} correct",
        "b_miss_a_hit": f"{model_b} misclassified AND {model_a} correct",
        "both_miss": f"Both {model_a} and {model_b} misclassified",
        "both_hit": f"Both {model_a} and {model_b} correct",
    }

def render_metadata_filters(
    meta_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    show_editor: bool = True,
    initial_query: str = "",
) -> Dict[str, any]:
    """
    Render filtering controls and display the filtered metadata.
    Returns filtered_df (ALL original columns + added status columns) and applied_filters.
    """
    if meta_df is None or meta_df.empty:
        st.warning("Metadata dataframe is empty or missing.")
        return {"filtered_df": pd.DataFrame(), "applied_filters": {}}

    df = add_status_columns(meta_df, model_a, model_b)
    scenario_labels = build_scenario_labels(model_a, model_b)

    st.subheader("Metadata Filters")
    cols_top = st.columns(4)
    with cols_top[0]:
        text_search = st.text_input("Search header_name (substring)", "")
    with cols_top[1]:
        true_labels = sorted(df["true_label"].unique()) if "true_label" in df.columns else []
        selected_true_labels = st.multiselect("True labels", true_labels)
    with cols_top[2]:
        scenario = st.selectbox(
            "Classification scenario",
            ["(none)"] + list(scenario_labels.keys()),
            format_func=lambda k: k if k == "(none)" else scenario_labels[k]
        )
    with cols_top[3]:
        custom_query = st.text_input(
            "Custom pandas query",
            value=initial_query,
            help=f"Example: is_person_outside == False and true_label == 1"
        )

    cols_extra = st.columns(4)
    with cols_extra[0]:
        subclass_col = "subclass_inside_outside" if "subclass_inside_outside" in df.columns else "subclasses" if "subclasses" in df.columns else None
        subclass_values = sorted(df[subclass_col].dropna().astype(str).unique()) if subclass_col else []
        selected_subclasses = st.multiselect("Subclass", subclass_values)
    with cols_extra[1]:
        desc_values = sorted(df["header_description"].dropna().astype(str).unique()) if "header_description" in df.columns else []
        selected_descriptions = st.multiselect("Header description", desc_values)
    with cols_extra[2]:
        error_type = st.selectbox(
            "Error explorer",
            ["(none)", "Any misclassified", f"False positives ({model_a})", f"False negatives ({model_a})", f"False positives ({model_b})", f"False negatives ({model_b})"],
        )
    with cols_extra[3]:
        selected_instances = st.multiselect(
            "Specific instances",
            sorted(df["header_name"].dropna().astype(str).tolist()) if "header_name" in df.columns else [],
        )

    work = df

    # Text search
    if text_search and "header_name" in work.columns:
        s = text_search.lower()
        work = work[work["header_name"].astype(str).str.lower().str.contains(s)]

    # True label filter
    if selected_true_labels and "true_label" in work.columns:
        work = work[work["true_label"].isin(selected_true_labels)]

    if selected_subclasses and subclass_col and subclass_col in work.columns:
        work = work[work[subclass_col].astype(str).isin(selected_subclasses)]

    if selected_descriptions and "header_description" in work.columns:
        work = work[work["header_description"].astype(str).isin(selected_descriptions)]

    if selected_instances and "header_name" in work.columns:
        work = work[work["header_name"].astype(str).isin(selected_instances)]

    # Scenario filter (FIXED)
    if scenario != "(none)" and scenario in work.columns:
        mask = work[scenario]
        if mask.dtype == bool:
            work = work[mask]
        else:
            st.warning(f"Scenario column '{scenario}' is not boolean; ignoring scenario filter.")

    if error_type != "(none)":
        miss_a = f"misclassified_{model_a}"
        miss_b = f"misclassified_{model_b}"
        fp_a = f"false_positive_{model_a}"
        fn_a = f"false_negative_{model_a}"
        fp_b = f"false_positive_{model_b}"
        fn_b = f"false_negative_{model_b}"
        if error_type == "Any misclassified":
            mask = pd.Series(False, index=work.index)
            if miss_a in work.columns:
                mask = mask | work[miss_a]
            if miss_b in work.columns:
                mask = mask | work[miss_b]
            work = work[mask]
        elif error_type == f"False positives ({model_a})" and fp_a in work.columns:
            work = work[work[fp_a]]
        elif error_type == f"False negatives ({model_a})" and fn_a in work.columns:
            work = work[work[fn_a]]
        elif error_type == f"False positives ({model_b})" and fp_b in work.columns:
            work = work[work[fp_b]]
        elif error_type == f"False negatives ({model_b})" and fn_b in work.columns:
            work = work[work[fn_b]]

    # Custom query
    if custom_query:
        try:
            queried = work.query(custom_query)
            # Ensure result is DataFrame
            if isinstance(queried, pd.Series):
                st.warning("Custom query returned a Series; ignoring.")
            else:
                work = queried
        except Exception as e:
            st.warning(f"Query error: {e}")

    st.caption(f"Filtered samples: {len(work)}")

    # Create a display-safe copy without heavy columns (e.g. 'yaml')
    heavy_cols = [c for c in ["yaml"] if c in work.columns]
    work_display = work.drop(columns=heavy_cols) if heavy_cols else work

    if show_editor:
        st.markdown("Explore rows below, then copy a header_name to use in the analysis section.")
        primary_cols = [
            "header_name",
            "header_description",
            "subclass_inside_outside",
            "true_label",
            "prediction_disagreement",
            f"predicted_label_{model_a}",
            f"predicted_proba_{model_a}",
            f"predicted_label_{model_b}",
            f"predicted_proba_{model_b}",
            f"false_positive_{model_a}",
            f"false_negative_{model_a}",
            f"false_positive_{model_b}",
            f"false_negative_{model_b}",
            "a_miss_b_hit",
            "b_miss_a_hit",
            "both_miss",
            "both_hit",
        ]
        display_cols = [c for c in primary_cols if c in work_display.columns] + [
            c for c in work_display.columns if c not in primary_cols
        ]
        st.data_editor(
            work_display,
            use_container_width=True,
            hide_index=True,
            key="meta_editor",
            column_order=display_cols,
            disabled=True,  # read-only
        )
        if heavy_cols:
            st.caption(f"Excluded heavy columns: {', '.join(heavy_cols)}")
    else:
        st.dataframe(work_display, use_container_width=True)

    return {
        "filtered_df": work,
        "applied_filters": dict(
            text_search=text_search,
            true_labels=selected_true_labels,
            subclasses=selected_subclasses,
            descriptions=selected_descriptions,
            instances=selected_instances,
            error_type=error_type,
            scenario=scenario,
            custom_query=custom_query
        )
    }