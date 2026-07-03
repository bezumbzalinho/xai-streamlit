from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import shap

from shap_streamlit.src.shap_visualizer import plot_shap_dependence
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


DEFAULT_FEATURES = [
	"HM_TX1RX2_tap11_fft28",
	"HM_TX1RX2_tap11_fft19",
	"HM_TX1RX2_tap14_fft28",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="CNN FFT feature dependence plots for selected features on misclassified samples."
	)
	parser.add_argument(
		"--model-dir",
		type=Path,
		default=Path("/home/ctw04958/projects/child-presence/shap_streamlit/models/unleashed-steed-161"),
		help="Directory containing model.keras, config.yaml and test_dataset.csv",
	)
	parser.add_argument(
		"--background-size",
		type=int,
		default=8,
		help="Background samples used by SHAP GradientExplainer.",
	)
	parser.add_argument(
		"--max-explain",
		type=int,
		default=0,
		help="Limit misclassified samples explained by SHAP (0 means all).",
	)
	parser.add_argument(
		"--chunk-size",
		type=int,
		default=8,
		help="Chunk size for SHAP computation.",
	)
	parser.add_argument(
		"--shap-nsamples",
		type=int,
		default=64,
		help="Number of samples used internally by SHAP GradientExplainer.",
	)
	parser.add_argument(
		"--feature",
		dest="features",
		action="append",
		default=None,
		help=(
			"Feature prefix to plot, e.g. HM_TX1RX2_tap11_fft28. "
			"Can be passed multiple times."
		),
	)
	return parser.parse_args()


def _resolve_feature_name(feature_query: str, feature_names: Sequence[str]) -> str:
	matches = [name for name in feature_names if name.startswith(feature_query)]
	if len(matches) == 1:
		return matches[0]
	if not matches:
		raise ValueError(f"Feature '{feature_query}' not found in FFT feature names.")
	raise ValueError(f"Feature '{feature_query}' matched multiple features: {matches[:5]}")


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

	mis_mask = aligned_df["true_label"].to_numpy() != aligned_df["predicted_label"].to_numpy()
	selected_idx = np.flatnonzero(mis_mask)
	if selected_idx.size == 0:
		raise ValueError("No misclassified samples found.")
	if args.max_explain > 0:
		selected_idx = selected_idx[: args.max_explain]

	fft_extractor = build_fft_feature_extractor(model=model, params=params)
	fft_features = fft_extractor.predict(x_inputs, verbose=0)
	fft_head_model = build_fft_head_model(model=model, params=params, fft_feature_shape=fft_features.shape)
	fft_layer = find_fft_layer(model)

	ensure_shap_keras_compat()
	n_explain = int(selected_idx.size)
	n_bg = min(max(1, args.background_size), n_explain)
	chunk_size = min(max(1, args.chunk_size), n_explain)
	shap_nsamples = max(1, args.shap_nsamples)

	background_idx = selected_idx[:n_bg]
	background = fft_features[background_idx]
	to_explain = fft_features[selected_idx]

	print(
		f"Running CNN FFT feature dependence on misclassified samples with background_size={n_bg}, "
		f"n_explain={n_explain}, chunk_size={chunk_size}, shap_nsamples={shap_nsamples}"
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

	explanation = shap.Explanation(
		values=flat_shap,
		base_values=np.zeros(flat_shap.shape[0], dtype=np.float32),
		data=flat_features,
		feature_names=feature_names,
	)

	feature_queries = args.features or DEFAULT_FEATURES
	resolved_feature_names = [_resolve_feature_name(query, feature_names) for query in feature_queries]

	for rank, feature_name in enumerate(resolved_feature_names, start=1):
		feature_idx = feature_names.index(feature_name)
		fig_dep = plot_shap_dependence(
			explanation=explanation,
			feature_idx=int(feature_idx),
			feature_names=feature_names,
			title=f"CNN FFT Feature Dependence #{rank}: {feature_name} (Misclassified Samples)",
			figsize=(11, 6),
		)
		fig_dep.show()


if __name__ == "__main__":
	run()
