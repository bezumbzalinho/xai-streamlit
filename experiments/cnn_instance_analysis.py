from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from shap_streamlit.src.shap_visualizer import plot_shap_waterfall
from shap_cnn_fft import (
	build_cnn_inputs_from_h5,
	build_existing_file_records,
	build_fft_feature_extractor,
	build_fft_head_model,
	build_shap_tensor,
	ensure_shap_keras_compat,
	find_fft_layer,
	flatten_fft_tensor,
	infer_true_label,
	load_model_and_config,
	to_predicted_label,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="CNN FFT SHAP instance analysis (waterfall + decision plot)."
	)
	parser.add_argument(
		"--model-dir",
		type=Path,
		default=Path("/home/ctw04958/projects/child-presence/shap_streamlit/models/unleashed-steed-161"),
		help="Directory containing model.keras, config.yaml and test_dataset.csv",
	)
	parser.add_argument(
		"--scope",
		type=str,
		default="misclassified",
		choices=["all", "misclassified", "fp", "fn", "subclass-misclassified"],
		help="Subset used to select the instance.",
	)
	parser.add_argument(
		"--subclass",
		type=str,
		default="",
		help="Required when --scope subclass-misclassified.",
	)
	parser.add_argument(
		"--target-header",
		type=str,
		default="",
		help="Select a specific sample by header_name (preferred).",
	)
	parser.add_argument(
		"--sample-index",
		type=int,
		default=0,
		help="Fallback index inside selected scope when --target-header is not provided.",
	)
	parser.add_argument("--background-size", type=int, default=8)
	parser.add_argument("--chunk-size", type=int, default=1)
	parser.add_argument("--shap-nsamples", type=int, default=64)
	parser.add_argument("--max-display", type=int, default=30)
	return parser.parse_args()


def select_scope_mask(df: pd.DataFrame, scope: str, subclass: str) -> np.ndarray:
	mis_mask = (df["true_label"].to_numpy() != df["predicted_label"].to_numpy())
	fp_mask = (df["true_label"].to_numpy() == 0) & (df["predicted_label"].to_numpy() == 1)
	fn_mask = (df["true_label"].to_numpy() == 1) & (df["predicted_label"].to_numpy() == 0)

	mask = np.ones(len(df), dtype=bool)
	if scope == "misclassified":
		mask = mis_mask
	elif scope == "fp":
		mask = fp_mask
	elif scope == "fn":
		mask = fn_mask
	elif scope == "subclass-misclassified":
		if not subclass:
			raise ValueError("--scope subclass-misclassified requires --subclass.")
		if "subclass_inside_outside" not in df.columns:
			raise ValueError("Column 'subclass_inside_outside' not found in test dataset.")
		subclass_mask = df["subclass_inside_outside"].astype(str).to_numpy() == subclass
		mask = mis_mask & subclass_mask

	return mask


def resolve_target_index(df_scope: pd.DataFrame, target_header: str, sample_index: int) -> int:
	if target_header:
		row = df_scope[df_scope["header_name"].astype(str) == target_header]
		if row.empty:
			raise ValueError(f"target-header '{target_header}' not found in selected scope.")
		return int(row.index[0])

	if sample_index < 0 or sample_index >= len(df_scope):
		raise ValueError(f"sample-index {sample_index} out of range for scope size {len(df_scope)}")

	return int(df_scope.index[sample_index])


def run() -> None:
	args = parse_args()
	model_dir = args.model_dir.resolve()

	model, _, params = load_model_and_config(model_dir)
	test_csv_path = model_dir / "test_dataset.csv"
	df_test = pd.read_csv(test_csv_path, delimiter=";")
	df_test["true_label"] = infer_true_label(df_test)

	link_ids = params["links"]
	window_size = int(params["window_size"])
	input_shape_0 = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
	n_taps_model_input = int(input_shape_0[2])
	full_tap_range_for_model_input = (1, n_taps_model_input)
	expected_shape = (window_size, n_taps_model_input, 2)

	file_records = build_existing_file_records(df_test, model_dir)
	x_inputs, aligned_rows = build_cnn_inputs_from_h5(
		file_records=file_records,
		link_ids=link_ids,
		expected_shape=expected_shape,
		full_tap_range_for_model_input=full_tap_range_for_model_input,
	)

	aligned_df = df_test.iloc[aligned_rows["row_idx"].to_numpy()].reset_index(drop=True)
	pred_raw = model.predict(x_inputs, verbose=0)
	aligned_df["predicted_label"] = to_predicted_label(pred_raw)

	scope_mask = select_scope_mask(aligned_df, scope=args.scope, subclass=args.subclass)
	df_scope = aligned_df[scope_mask].copy()
	if df_scope.empty:
		raise ValueError(f"No samples found for scope='{args.scope}' and subclass='{args.subclass}'.")

	target_global_idx = resolve_target_index(df_scope, args.target_header, args.sample_index)
	target_row = aligned_df.iloc[target_global_idx]

	fft_extractor = build_fft_feature_extractor(model=model, params=params)
	fft_features = fft_extractor.predict(x_inputs, verbose=0)
	fft_head_model = build_fft_head_model(model=model, params=params, fft_feature_shape=fft_features.shape)
	fft_layer = find_fft_layer(model)

	ensure_shap_keras_compat()
	scope_indices = np.flatnonzero(scope_mask)
	n_bg = min(max(1, args.background_size), scope_indices.size)
	chunk_size = max(1, args.chunk_size)
	shap_nsamples = max(1, args.shap_nsamples)

	background_idx = scope_indices[:n_bg]
	background = fft_features[background_idx]
	to_explain = fft_features[target_global_idx : target_global_idx + 1]

	print(
		f"Running CNN instance analysis with scope={args.scope}, background_size={n_bg}, "
		f"chunk_size={chunk_size}, shap_nsamples={shap_nsamples}"
	)
	print(
		"Target sample: "
		f"header_name={target_row.get('header_name', '<unknown>')} | "
		f"true_label={target_row.get('true_label')} | "
		f"predicted_label={target_row.get('predicted_label')}"
	)

	explainer = shap.GradientExplainer(fft_head_model, background)
	shap_tensor = build_shap_tensor(
		explainer=explainer,
		to_explain=to_explain,
		chunk_size=chunk_size,
		shap_nsamples=shap_nsamples,
	)

	sample_period_raw = float(params.get("target_sampling_rate", 96))
	sample_period_s = sample_period_raw / 1000.0 if sample_period_raw > 1.0 else sample_period_raw
	fs_hz = 1.0 / sample_period_s
	n_fft_time = int(window_size)
	negative_freq = bool(getattr(fft_layer, "negative_freq", params.get("negative_freq", False)))
	tap_start = int(params.get("tap_range", [1, 30])[0])

	flat_shap, feature_names = flatten_fft_tensor(
		tensor=shap_tensor,
		link_ids=link_ids,
		tap_start=tap_start,
		fs_hz=fs_hz,
		n_fft_time=n_fft_time,
		negative_freq=negative_freq,
	)
	flat_features, _ = flatten_fft_tensor(
		tensor=to_explain,
		link_ids=link_ids,
		tap_start=tap_start,
		fs_hz=fs_hz,
		n_fft_time=n_fft_time,
		negative_freq=negative_freq,
	)

	bg_pred = np.asarray(fft_head_model.predict(background, verbose=0)).reshape(-1)
	base_value = float(bg_pred.mean())

	explanation = shap.Explanation(
		values=flat_shap,
		base_values=np.array([base_value], dtype=np.float32),
		data=flat_features,
		feature_names=feature_names,
	)

	target_header = str(target_row.get("header_name", "selected_sample"))
	waterfall_title = f"CNN FFT SHAP Waterfall - {target_header}"
	fig_waterfall = plot_shap_waterfall(
		explanation=explanation,
		instance_idx=0,
		title=waterfall_title,
		max_display=max(1, args.max_display),
	)
	plt.show()
	plt.close(fig_waterfall)

	plt.figure(figsize=(12, 6))
	shap.decision_plot(
		base_value=base_value,
		shap_values=flat_shap[0],
		features=flat_features[0],
		feature_names=feature_names,
		show=False,
	)
	plt.title(f"CNN FFT SHAP Decision Plot - {target_header}")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	run()
