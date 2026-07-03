from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from cpd.core.data.can_fd_messages import LinkID
from shap_cnn_fft import (
	build_cnn_inputs_from_h5,
	build_existing_file_records,
	build_fft_feature_extractor,
	build_fft_head_model,
	build_shap_tensor,
	ensure_shap_keras_compat,
	find_fft_layer,
	infer_true_label,
	load_model_and_config,
	to_predicted_label,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Range-Doppler style FFT-SHAP visualization (Link x Tap x Hz). "
			"Color represents SHAP contribution."
		)
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
		help="Subset used for SHAP computation.",
	)
	parser.add_argument(
		"--subclass",
		type=str,
		default="",
		help="Required when --scope subclass-misclassified.",
	)
	parser.add_argument("--background-size", type=int, default=8)
	parser.add_argument("--max-explain", type=int, default=0, help="0 means all selected samples.")
	parser.add_argument("--chunk-size", type=int, default=8)
	parser.add_argument("--shap-nsamples", type=int, default=64)
	parser.add_argument(
		"--sample-index",
		type=int,
		default=-1,
		help=(
			"Index inside selected scope. -1 means aggregate all selected samples using --aggregate."
		),
	)
	parser.add_argument(
		"--aggregate",
		type=str,
		default="mean_abs",
		choices=["mean_abs", "mean_signed", "max_abs"],
		help="Aggregation used when --sample-index is -1.",
	)
	parser.add_argument(
		"--feature-display",
		type=str,
		default="log1p",
		choices=["raw", "log1p"],
		help="How feature values are displayed in heatmap (for visualization only).",
	)
	parser.add_argument(
		"--color-scale",
		type=str,
		default="robust",
		choices=["robust", "full"],
		help="Color range mode: robust uses percentiles to reduce outlier dominance.",
	)
	# parser.add_argument(
	# 	"--png-out",
	# 	type=Path,
	# 	default=Path("range_doppler_fft_shap_heatmap.png"),
	# 	help="Output PNG file for heatmap figure.",
	# )
	# parser.add_argument(
	# 	"--no-show",
	# 	action="store_true",
	# 	help="If set, do not open figures interactively.",
	# )
	return parser.parse_args()


def select_scope_indices(df: pd.DataFrame, scope: str, subclass: str) -> np.ndarray:
	mis_mask = (df["true_label"].to_numpy() != df["predicted_label"].to_numpy())
	fp_mask = (df["true_label"].to_numpy() == 0) & (df["predicted_label"].to_numpy() == 1)
	fn_mask = (df["true_label"].to_numpy() == 1) & (df["predicted_label"].to_numpy() == 0)

	base_scope_mask = np.ones(len(df), dtype=bool)
	if scope == "misclassified":
		base_scope_mask = mis_mask
	elif scope == "fp":
		base_scope_mask = fp_mask
	elif scope == "fn":
		base_scope_mask = fn_mask
	elif scope == "subclass-misclassified":
		if not subclass:
			raise ValueError("--scope subclass-misclassified requires --subclass")
		if "subclass_inside_outside" not in df.columns:
			raise ValueError("Column 'subclass_inside_outside' not found in test dataset")
		subclass_mask = df["subclass_inside_outside"].astype(str).to_numpy() == subclass
		base_scope_mask = mis_mask & subclass_mask

	selected_idx = np.flatnonzero(base_scope_mask)
	if selected_idx.size == 0:
		raise ValueError(f"No samples found for scope='{scope}' and subclass='{subclass}'.")

	return selected_idx


def build_frequency_axis_hz(
	*,
	n_fft_out: int,
	n_fft_time: int,
	fs_hz: float,
	negative_freq: bool,
) -> np.ndarray:
	if not negative_freq:
		k = np.arange(1, n_fft_out + 1, dtype=np.float32)
	else:
		freq_bins = n_fft_out // 2
		k = np.concatenate([
			np.arange(-freq_bins, 0, dtype=np.float32),
			np.arange(1, freq_bins + 1, dtype=np.float32),
		])

	return (k * fs_hz) / float(n_fft_time)


def build_contribution_cube(shap_tensor: np.ndarray, mode: str) -> np.ndarray:
	if shap_tensor.ndim != 4:
		raise ValueError(f"Expected SHAP tensor with 4 dims [samples, links, taps, fft], got {shap_tensor.shape}")

	if mode == "mean_abs":
		return np.mean(np.abs(shap_tensor), axis=0)
	if mode == "max_abs":
		return np.max(np.abs(shap_tensor), axis=0)
	if mode == "mean_signed":
		return np.mean(shap_tensor, axis=0)

	raise ValueError(f"Unsupported aggregate mode: {mode}")


def build_feature_cube(feature_tensor: np.ndarray, mode: str) -> np.ndarray:
	if feature_tensor.ndim != 4:
		raise ValueError(
			f"Expected feature tensor with 4 dims [samples, links, taps, fft], got {feature_tensor.shape}"
		)

	if mode in {"mean_abs", "mean_signed"}:
		return np.mean(feature_tensor, axis=0)
	if mode == "max_abs":
		return np.max(feature_tensor, axis=0)

	raise ValueError(f"Unsupported aggregate mode: {mode}")


def _build_frequency_tick_indices(hz_axis: np.ndarray) -> np.ndarray:
	n_fft = hz_axis.shape[0]
	center = n_fft // 2

	idx = set(range(0, n_fft, 6))
	idx.update(range(max(0, center - 8), min(n_fft, center + 9), 2))
	idx.update({0, n_fft - 1})

	idx_sorted = np.array(sorted(idx), dtype=int)
	if idx_sorted.size > 14:
		# Keep density around center if too crowded.
		keep = set(idx_sorted[::2])
		keep.update(range(max(0, center - 6), min(n_fft, center + 7), 2))
		keep.update({0, n_fft - 1})
		idx_sorted = np.array(sorted(keep), dtype=int)

	return idx_sorted


def plot_link_heatmaps(
	feature_cube: np.ndarray,
	shap_cube: np.ndarray,
	link_ids: Sequence[int],
	tap_start: int,
	hz_axis: np.ndarray,
	*,
	title: str,
	feature_display: str,
	color_scale: str,
	# png_out: Path,
	# show: bool,
) -> None:
	n_links = len(link_ids)
	fig, axes = plt.subplots(
		1,
		n_links,
		figsize=(5.2 * n_links, 4.8),
		sharey=True,
		constrained_layout=True,
	)
	if n_links == 1:
		axes = [axes]

	if feature_display == "log1p":
		feature_display_cube = np.log1p(np.maximum(feature_cube, 0.0))
		color_label = "Feature value (log1p scale)"
	else:
		feature_display_cube = feature_cube
		color_label = "Feature value"

	if color_scale == "robust":
		feature_vmin = float(np.percentile(feature_display_cube, 1.0))
		feature_vmax = float(np.percentile(feature_display_cube, 99.0))
	else:
		feature_vmin = float(np.min(feature_display_cube))
		feature_vmax = float(np.max(feature_display_cube))
	if np.isclose(feature_vmin, feature_vmax):
		feature_vmax = feature_vmin + 1e-6

	for ax, li in zip(axes, range(n_links)):
		arr = feature_display_cube[li]  # [tap, fft]
		shap_arr = shap_cube[li]
		abs_shap = np.abs(shap_arr)
		n_taps, n_fft = arr.shape
		tap_axis = np.arange(tap_start, tap_start + n_taps)
		x_extent = [float(hz_axis[0]), float(hz_axis[-1]), float(tap_axis[0] - 0.5), float(tap_axis[-1] + 0.5)]

		im = ax.imshow(
			arr,
			aspect="auto",
			origin="lower",
			cmap="viridis",
			vmin=feature_vmin,
			vmax=feature_vmax,
			extent=x_extent,
			interpolation="nearest",
		)

		if np.any(abs_shap > 0):
			levels = np.quantile(abs_shap, [0.80, 0.92, 0.98])
			levels = np.unique(levels[levels > 0])
			if levels.size > 0:
				X, Y = np.meshgrid(hz_axis, tap_axis)
				ax.contour(
					X,
					Y,
					abs_shap,
					levels=levels,
					colors="white",
					linewidths=[0.6, 1.0, 1.4][: levels.size],
				)

			# flat_idx = np.argsort(abs_shap.reshape(-1))[-3:]
			# for flat_pos in flat_idx:
			# 	ti, fi = np.unravel_index(flat_pos, abs_shap.shape)
			# 	value = shap_arr[ti, fi]
			# 	ax.text(
			# 		fi,
			# 		ti,
			# 		f"{value:+.3f}",
			# 		fontsize=7,
			# 		color="white",
			# 		ha="center",
			# 		va="center",
			# 		bbox={"boxstyle": "round,pad=0.15", "fc": "black", "ec": "none", "alpha": 0.55},
			# 	)

		ax.set_title(LinkID(link_ids[li]).name)
		ax.set_xlabel("Frequency (Hz)")
		ax.set_ylabel("Tap")

		tick_idx = _build_frequency_tick_indices(hz_axis)
		ax.set_xticks(hz_axis[tick_idx])
		ax.set_xticklabels([f"{hz_axis[i]:.2f}" for i in tick_idx], rotation=45, ha="right")

		tap_tick_idx = np.linspace(0, n_taps - 1, min(6, n_taps)).astype(int)
		ax.set_yticks(tap_axis[tap_tick_idx])
		ax.set_yticklabels([str(tap_axis[i]) for i in tap_tick_idx])
		ax.text(
			0.02,
			0.98,
			f"Color: feature value\nContours: SHAP impact",
			transform=ax.transAxes,
			fontsize=7,
			va="top",
			ha="left",
			color="white",
			bbox={"boxstyle": "round,pad=0.2", "fc": "black", "ec": "none", "alpha": 0.45},
		)

	fig.suptitle(title)
	cbar = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.02, label=color_label)
	cbar.set_ticks([feature_vmin, feature_vmax])
	cbar.set_ticklabels(["Low", "High"])
	
	plt.show()

	# png_out = png_out.resolve()
	# fig.savefig(str(png_out), dpi=180, bbox_inches="tight")
	# print(f"Heatmap saved to: {png_out}")

	# if show:
	# 	plt.show()
	# else:
	# 	plt.close(fig)


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

	selected_idx = select_scope_indices(aligned_df, scope=args.scope, subclass=args.subclass)
	if args.max_explain > 0:
		selected_idx = selected_idx[: args.max_explain]

	if selected_idx.size == 0:
		raise ValueError("No selected samples after applying max-explain.")

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
		f"Running FFT-SHAP map with scope={args.scope}, background_size={n_bg}, "
		f"n_explain={n_explain}, chunk_size={chunk_size}, shap_nsamples={shap_nsamples}"
	)
	if args.aggregate.startswith("mean"):
		print(f"Feature/SHAP maps are normalized by sample count (N={n_explain}) via mean aggregation.")
	else:
		print("Using max aggregation (not normalized by sample count).")

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
	hz_axis = build_frequency_axis_hz(
		n_fft_out=shap_tensor.shape[3],
		n_fft_time=n_fft_time,
		fs_hz=fs_hz,
		negative_freq=negative_freq,
	)

	tap_start = int(params.get("tap_range", [1, 30])[0])
	if args.sample_index >= 0:
		if args.sample_index >= shap_tensor.shape[0]:
			raise ValueError(
				f"sample-index {args.sample_index} out of range for selected set size {shap_tensor.shape[0]}"
			)
		feature_cube = to_explain[args.sample_index]
		contrib_cube = shap_tensor[args.sample_index]
		title_scope = f"sample={args.sample_index}"
	else:
		feature_cube = build_feature_cube(to_explain, mode=args.aggregate)
		contrib_cube = build_contribution_cube(shap_tensor, mode=args.aggregate)
		title_scope = f"aggregate={args.aggregate} over {shap_tensor.shape[0]} samples"

	title = (
		"FFT Range-Doppler Map | "
		f"scope={args.scope} | {title_scope} | fs={fs_hz:.3f}Hz | N={n_fft_time}"
	)
	# show = not args.no_show

	plot_link_heatmaps(
		feature_cube=feature_cube,
		shap_cube=contrib_cube,
		link_ids=link_ids,
		tap_start=tap_start,
		hz_axis=hz_axis,
		title=title,
		feature_display=args.feature_display,
		color_scale=args.color_scale,
		# png_out=args.png_out,
		# show=show,
	)


if __name__ == "__main__":
	run()
