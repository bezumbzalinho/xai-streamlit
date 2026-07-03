from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import yaml
from tensorflow import keras

from cpd.core.data.can_fd_messages import LinkID
from cpd.core.data.dataset import DatasetH5
from cpd.core.ml.models.tensorflow.custom_layers import (
	TFFFTFeatureExtraction,
	TFGPUFFTFeatureExtraction,
	TFTapRangeSelection,
)

warnings.filterwarnings(
	"ignore",
	message=r"The structure of `inputs` doesn't match the expected structure\..*",
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="CNN SHAP analysis for all/misclassified/FP/FN samples."
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
		help="Limit samples explained by SHAP (0 means all).",
	)
	parser.add_argument(
		"--chunk-size",
		type=int,
		default=8,
		help="Chunk size for SHAP computation to control memory usage.",
	)
	parser.add_argument(
		"--shap-nsamples",
		type=int,
		default=64,
		help="Number of samples used internally by SHAP GradientExplainer.",
	)
	parser.add_argument(
		"--max-display",
		type=int,
		default=25,
		help="Maximum number of top features shown in SHAP summary plot.",
	)
	parser.add_argument(
		"--subclass",
		type=str,
		default="",
		help="Optional subclass filter for additional misclassified summary plot.",
	)
	parser.add_argument(
		"--scope",
		type=str,
		default="all",
		choices=["all", "misclassified", "fp", "fn", "subclass-misclassified"],
		help=(
			"Subset used for SHAP computation. "
			"Use 'subclass-misclassified' with --subclass to explain only that slice."
		),
	)
	return parser.parse_args()


def load_model_and_config(model_dir: Path) -> Tuple[keras.Model, Dict, Dict]:
	model_path = model_dir / "model.keras"
	config_path = model_dir / "config.yaml"

	custom_objects = {
		"TFTapRangeSelection": TFTapRangeSelection,
		"CPD>TFTapRangeSelection": TFTapRangeSelection,
	}

	model = keras.models.load_model(
		str(model_path),
		custom_objects=custom_objects,
		compile=False,
		safe_mode=False,
	)

	with open(config_path, "r", encoding="utf-8") as file:
		config = yaml.safe_load(file)

	params = config["machine_learning"]["model_params"]
	return model, config, params


def build_per_tap_feature_extractor(model: keras.Model, params: Dict) -> keras.Model:
	link_ids = params["links"]
	n_links = len(link_ids)
	tap_range = tuple(params.get("tap_range", [1, 30]))
	n_taps = int(tap_range[1]) - int(tap_range[0]) + 1
	per_tap_output = model.get_layer("per_tap_model").output
	x = keras.ops.reshape(per_tap_output, (-1, n_links, n_taps, per_tap_output.shape[-1]))

	inputs = model.inputs if isinstance(model.inputs, list) else [model.inputs]
	return keras.Model(inputs=inputs, outputs=x, name="cnn_per_tap_feature_extractor")


def build_per_tap_head_model(model: keras.Model, params: Dict, feature_shape: Tuple[int, ...]) -> keras.Model:
	n_anchors = int(params.get("n_anchors", 2))
	n_links = len(params["links"])
	share_across_anchors = bool(params.get("share_across_anchors", True))
	n_taps = int(feature_shape[2])

	features = keras.Input(shape=feature_shape[1:], dtype=tf.float32, name="per_tap_features")
	x = features

	if share_across_anchors:
		x = keras.ops.reshape(x, (-1, n_links // n_anchors * n_taps, x.shape[3]))
		x = model.get_layer("per_anchor_model")(x)
		x = keras.ops.reshape(x, (-1, n_anchors, x.shape[1]))
	else:
		x = keras.ops.reshape(x, (-1, n_anchors, n_links // n_anchors * n_taps, x.shape[3]))
		x = keras.ops.stack(
			[model.get_layer(f"per_anchor_model_{i}")(x[:, i]) for i in range(n_anchors)],
			axis=1,
		)

	x = keras.ops.max(x, axis=1)
	return keras.Model(inputs=features, outputs=x, name="cnn_per_tap_head")


def infer_true_label(df: pd.DataFrame) -> pd.Series:
	if "true_classes" in df.columns:
		return pd.to_numeric(df["true_classes"], errors="coerce").fillna(0).astype(int)

	if "true_label" in df.columns:
		return pd.to_numeric(df["true_label"], errors="coerce").fillna(0).astype(int)

	if "header_name" in df.columns:
		# Fallback compatible with the XGBoost notebook convention.
		return df["header_name"].astype(str).apply(lambda value: 0 if value.endswith("E") else 1)

	raise ValueError(
		"Could not infer true labels. Provide 'true_classes' or 'true_label' columns in test_dataset.csv."
	)


def build_existing_file_records(df_test: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
	if "header_h5_file_path" not in df_test.columns:
		raise ValueError("Column 'header_h5_file_path' not found in test_dataset.csv")

	project_root = model_dir.parents[2]
	test_folder = project_root / "sample_files" / "semantic_layer" / "test"

	records: List[Dict] = []
	for idx, row in df_test.reset_index(drop=True).iterrows():
		h5_name = Path(str(row["header_h5_file_path"]))
		h5_path = test_folder / h5_name.name
		if h5_path.is_file():
			records.append(
				{
					"row_idx": idx,
					"h5_path": str(h5_path),
					"file_stem": h5_path.stem,
				}
			)

	if not records:
		raise FileNotFoundError("No existing .h5 files were found from test_dataset.csv entries.")

	return pd.DataFrame.from_records(records)


def _safe_array_from_window(
	win: Dict,
	link_id: int,
	expected_shape: Tuple[int, int, int],
) -> Optional[np.ndarray]:
	link_name = LinkID(link_id).name
	arr = np.asarray(win[link_name])
	if arr.shape != expected_shape:
		return None
	return arr


def build_cnn_inputs_from_h5(
	file_records: pd.DataFrame,
	link_ids: Sequence[int],
	expected_shape: Tuple[int, int, int],
	full_tap_range_for_model_input: Tuple[int, int],
) -> Tuple[List[np.ndarray], pd.DataFrame]:
	dataset = DatasetH5(
		source=file_records["h5_path"].tolist(),
		use_multiprocessing=True,
		use_cache=True,
	)
	raw = dataset.get_windows_by_links(
		links=list(link_ids),
		tap_range=full_tap_range_for_model_input,
	)

	stem_to_row = dict(zip(file_records["file_stem"], file_records["row_idx"]))
	per_link_arrays: Dict[int, List[np.ndarray]] = {link: [] for link in link_ids}
	used_rows: List[int] = []

	for sample_name in sorted(raw.keys()):
		sample = raw[sample_name]
		if not sample:
			continue

		win = sample[0]
		row_idx = stem_to_row.get(str(sample_name))
		if row_idx is None:
			continue

		temp: Dict[int, np.ndarray] = {}
		valid = True
		for link in link_ids:
			arr = _safe_array_from_window(win, link, expected_shape)
			if arr is None:
				valid = False
				break
			temp[link] = arr

		if not valid:
			continue

		for link in link_ids:
			per_link_arrays[link].append(temp[link])
		used_rows.append(int(row_idx))

	if not used_rows:
		raise RuntimeError("No valid windows were built from available .h5 files.")

	x_inputs = [np.stack(per_link_arrays[link], axis=0) for link in link_ids]
	aligned_rows = pd.DataFrame({"row_idx": used_rows})
	return x_inputs, aligned_rows


def ensure_shap_keras_compat() -> None:
	if not hasattr(tf.keras.backend, "learning_phase"):
		tf.keras.backend.learning_phase = lambda: 0

	if not hasattr(tf.keras.backend, "set_learning_phase"):
		tf.keras.backend.set_learning_phase = lambda value: None


def normalize_shap_per_input(raw_sv: object, n_inputs: int) -> List[np.ndarray]:
	if isinstance(raw_sv, list) and len(raw_sv) == n_inputs and not isinstance(raw_sv[0], list):
		return raw_sv

	if isinstance(raw_sv, list) and len(raw_sv) > 0 and isinstance(raw_sv[0], list):
		return raw_sv[0]

	raise ValueError(f"Unexpected SHAP output format: {type(raw_sv)}")


def normalize_shap_single_input(raw_sv: object) -> np.ndarray:
	if isinstance(raw_sv, list):
		if len(raw_sv) == 0:
			raise ValueError("Received empty SHAP output list.")
		if isinstance(raw_sv[0], list):
			if len(raw_sv[0]) == 0:
				raise ValueError("Received nested empty SHAP output list.")
			return np.asarray(raw_sv[0][0])
		return np.asarray(raw_sv[0])

	return np.asarray(raw_sv)


def find_fft_layer(model: keras.Model) -> keras.layers.Layer:
	for layer in model.layers:
		if isinstance(layer, (TFFFTFeatureExtraction, TFGPUFFTFeatureExtraction)):
			return layer
	for layer in model.layers:
		if "fft_feature_extraction" in layer.name:
			return layer
	raise ValueError("Could not find FFT feature extraction layer in model.")


def sorted_block_layers(model: keras.Model) -> List[keras.layers.Layer]:
	blocks = [layer for layer in model.layers if layer.name.startswith("block_")]

	def block_idx(layer: keras.layers.Layer) -> int:
		return int(layer.name.split("_")[-1])

	return sorted(blocks, key=block_idx)


def get_global_pool_layers(model: keras.Model) -> Tuple[keras.layers.Layer, keras.layers.Layer]:
	avg_layer = None
	max_layer = None
	for layer in model.layers:
		if isinstance(layer, keras.layers.GlobalAveragePooling1D) and avg_layer is None:
			avg_layer = layer
		if isinstance(layer, keras.layers.GlobalMaxPooling1D) and max_layer is None:
			max_layer = layer
	if avg_layer is None or max_layer is None:
		raise ValueError("Could not find GlobalAveragePooling1D/GlobalMaxPooling1D layers in model.")
	return avg_layer, max_layer


def build_fft_feature_extractor(model: keras.Model, params: Dict) -> keras.Model:
	n_links = len(params["links"])
	tap_range = tuple(params.get("tap_range", [1, 30]))
	n_taps = int(tap_range[1]) - int(tap_range[0]) + 1

	fft_out = find_fft_layer(model).output  # [batch * links, fft_bins, taps, channels]
	x = keras.ops.transpose(fft_out, [0, 2, 1, 3])  # [batch * links, taps, fft_bins, channels]
	x = keras.ops.reshape(x, (-1, n_links, n_taps, x.shape[2]))  # [batch, links, taps, fft_bins]

	inputs = model.inputs if isinstance(model.inputs, list) else [model.inputs]
	return keras.Model(inputs=inputs, outputs=x, name="cnn_fft_feature_extractor")


def build_fft_head_model(model: keras.Model, params: Dict, fft_feature_shape: Tuple[int, ...]) -> keras.Model:
	n_anchors = int(params.get("n_anchors", 2))
	n_links = len(params["links"])
	n_taps = int(fft_feature_shape[2])
	share_across_anchors = bool(params.get("share_across_anchors", True))
	pooling_layer_type = str(params.get("pooling_layer_type", "avg"))

	blocks = sorted_block_layers(model)
	avg_pool, max_pool = get_global_pool_layers(model)

	fft_features = keras.Input(shape=fft_feature_shape[1:], dtype=tf.float32, name="fft_features")
	x = keras.ops.reshape(fft_features, (-1, n_taps, fft_features.shape[3]))  # [batch*links, taps, fft]
	x = keras.ops.reshape(x, (-1, fft_features.shape[3], 1))  # [batch*links*taps, fft, 1]

	for block in blocks:
		x = block(x)

	if pooling_layer_type == "avg":
		x = avg_pool(x)
	elif pooling_layer_type == "max":
		x = max_pool(x)
	elif pooling_layer_type == "concat":
		x = keras.ops.concatenate([avg_pool(x), max_pool(x)], axis=-1)
	else:
		raise ValueError(f"Unsupported pooling_layer_type '{pooling_layer_type}'")

	x = model.get_layer("per_tap_model")(x)
	x = keras.ops.reshape(x, (-1, n_links, n_taps, x.shape[1]))

	if share_across_anchors:
		x = keras.ops.reshape(x, (-1, n_links // n_anchors * n_taps, x.shape[3]))
		x = model.get_layer("per_anchor_model")(x)
		x = keras.ops.reshape(x, (-1, n_anchors, x.shape[1]))
	else:
		x = keras.ops.reshape(x, (-1, n_anchors, n_links // n_anchors * n_taps, x.shape[3]))
		x = keras.ops.stack(
			[model.get_layer(f"per_anchor_model_{i}")(x[:, i]) for i in range(n_anchors)],
			axis=1,
		)

	x = keras.ops.max(x, axis=1)
	return keras.Model(inputs=fft_features, outputs=x, name="cnn_fft_head")


def build_shap_tensor(
	explainer: shap.GradientExplainer,
	to_explain: np.ndarray,
	chunk_size: int,
	shap_nsamples: int,
) -> np.ndarray:
	return build_shap_tensor_from_per_tap(
		explainer=explainer,
		to_explain=to_explain,
		chunk_size=chunk_size,
		shap_nsamples=shap_nsamples,
	)


def _fft_idx_to_hz_label(
	fft_idx: int,
	*,
	n_fft_out: int,
	n_fft_time: int,
	fs_hz: float,
	negative_freq: bool,
) -> str:
	if not negative_freq:
		k = fft_idx + 1
	else:
		freq_bins = n_fft_out // 2
		if fft_idx < freq_bins:
			k = -(freq_bins - fft_idx)
		else:
			k = (fft_idx - freq_bins) + 1

	f_hz = (k * fs_hz) / float(n_fft_time)
	return f"{f_hz:+.2f}Hz" if negative_freq else f"{f_hz:.2f}Hz"


def flatten_fft_tensor(
	tensor: np.ndarray,
	link_ids: Sequence[int],
	tap_start: int,
	*,
	fs_hz: float,
	n_fft_time: int,
	negative_freq: bool,
) -> Tuple[np.ndarray, List[str]]:
	n_links = len(link_ids)
	n_taps = tensor.shape[2]
	n_fft_out = tensor.shape[3]

	feature_names = []
	for link_id in link_ids:
		link_name = LinkID(link_id).name
		for tap_idx in range(n_taps):
			tap_label = tap_start + tap_idx
			for fft_idx in range(n_fft_out):
				hz_label = _fft_idx_to_hz_label(
					fft_idx,
					n_fft_out=n_fft_out,
					n_fft_time=n_fft_time,
					fs_hz=fs_hz,
					negative_freq=negative_freq,
				)
				feature_names.append(
					f"{link_name}_tap{tap_label}_fft{fft_idx}({hz_label})"
				)

	expected_features = n_links * n_taps * n_fft_out
	flat = tensor.reshape(tensor.shape[0], -1)
	if flat.shape[1] != expected_features:
		raise ValueError(
			f"Unexpected flattened feature size {flat.shape[1]}, expected {expected_features}"
		)

	return flat, feature_names


def to_predicted_label(model_output: np.ndarray) -> np.ndarray:
	output = np.asarray(model_output)
	if output.ndim == 2 and output.shape[1] == 1:
		output = output[:, 0]
	return (output >= 0.5).astype(int)


def plot_feature_summary_matrix(
	shap_matrix: np.ndarray,
	feature_matrix: np.ndarray,
	feature_names: Sequence[str],
	title: str,
	max_display: int,
) -> None:
	if shap_matrix.size == 0 or shap_matrix.shape[0] == 0:
		print(f"Skipping plot '{title}' (empty subset).")
		return

	plt.figure(figsize=(10, 5))
	shap.summary_plot(
		shap_matrix,
		features=feature_matrix,
		feature_names=list(feature_names),
		max_display=max_display,
		plot_type="dot",
		show=False,
	)
	plt.title(title)
	plt.tight_layout()
	plt.show()


def build_link_summary_matrix(
	explainer: shap.GradientExplainer,
	to_explain: Sequence[np.ndarray],
	n_inputs: int,
	chunk_size: int,
	shap_nsamples: int,
) -> np.ndarray:
	n_explain = to_explain[0].shape[0]
	link_summary = np.zeros((n_explain, n_inputs), dtype=np.float32)

	for start in range(0, n_explain, chunk_size):
		end = min(start + chunk_size, n_explain)
		chunk = [x[start:end] for x in to_explain]
		raw_chunk = explainer.shap_values(chunk, nsamples=shap_nsamples)
		chunk_per_input = normalize_shap_per_input(raw_chunk, n_inputs=n_inputs)

		for input_idx, shap_values in enumerate(chunk_per_input):
			link_summary[start:end, input_idx] = np.mean(
				np.abs(shap_values),
				axis=tuple(range(1, shap_values.ndim)),
			)

		print(f"SHAP progress: {end}/{n_explain}")

	return link_summary


def build_link_summary_matrix_from_per_tap(
	explainer: shap.GradientExplainer,
	to_explain: np.ndarray,
	n_links: int,
	chunk_size: int,
	shap_nsamples: int,
) -> np.ndarray:
	n_explain = to_explain.shape[0]
	link_summary = np.zeros((n_explain, n_links), dtype=np.float32)

	for start in range(0, n_explain, chunk_size):
		end = min(start + chunk_size, n_explain)
		chunk = to_explain[start:end]
		raw_chunk = explainer.shap_values(chunk, nsamples=shap_nsamples)
		shap_values = normalize_shap_single_input(raw_chunk)

		if shap_values.ndim < 2:
			raise ValueError(f"Unexpected SHAP tensor rank for per-tap input: {shap_values.shape}")

		if shap_values.shape[1] != n_links:
			raise ValueError(
				f"Expected SHAP links dimension={n_links}, but got shape {shap_values.shape}."
			)

		reduce_axes = tuple(range(2, shap_values.ndim))
		if reduce_axes:
			link_summary[start:end, :] = np.mean(np.abs(shap_values), axis=reduce_axes)
		else:
			link_summary[start:end, :] = np.abs(shap_values)

		print(f"SHAP progress: {end}/{n_explain}")

	return link_summary


def build_shap_tensor_from_per_tap(
	explainer: shap.GradientExplainer,
	to_explain: np.ndarray,
	chunk_size: int,
	shap_nsamples: int,
) -> np.ndarray:
	n_explain = to_explain.shape[0]
	shap_tensor = np.zeros_like(to_explain, dtype=np.float32)

	for start in range(0, n_explain, chunk_size):
		end = min(start + chunk_size, n_explain)
		chunk = to_explain[start:end]
		raw_chunk = explainer.shap_values(chunk, nsamples=shap_nsamples)
		shap_values = normalize_shap_single_input(raw_chunk)

		if shap_values.shape != chunk.shape:
			raise ValueError(f"Expected SHAP chunk shape {chunk.shape}, got {shap_values.shape}")

		shap_tensor[start:end] = shap_values
		print(f"SHAP progress: {end}/{n_explain}")

	return shap_tensor


def flatten_per_tap_tensor(
	tensor: np.ndarray,
	link_ids: Sequence[int],
	tap_start: int,
	feats_per_tap: int,
) -> Tuple[np.ndarray, List[str]]:
	n_links = len(link_ids)
	n_taps = tensor.shape[2]

	feature_names = []
	for _, link_id in enumerate(link_ids):
		link_name = LinkID(link_id).name
		for tap_idx in range(n_taps):
			tap_label = tap_start + tap_idx
			for feat_idx in range(feats_per_tap):
				feature_names.append(f"{link_name}_tap{tap_label}_f{feat_idx}")

	expected_features = n_links * n_taps * feats_per_tap
	flat = tensor.reshape(tensor.shape[0], -1)
	if flat.shape[1] != expected_features:
		raise ValueError(
			f"Unexpected flattened feature size {flat.shape[1]}, expected {expected_features}"
		)

	return flat, feature_names


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

	feature_extractor = build_per_tap_feature_extractor(model=model, params=params)
	per_tap_features = feature_extractor.predict(x_inputs, verbose=0)
	head_model = build_per_tap_head_model(model=model, params=params, feature_shape=per_tap_features.shape)

	mis_mask = (aligned_df["true_label"].to_numpy() != aligned_df["predicted_label"].to_numpy())
	fp_mask = (aligned_df["true_label"].to_numpy() == 0) & (aligned_df["predicted_label"].to_numpy() == 1)
	fn_mask = (aligned_df["true_label"].to_numpy() == 1) & (aligned_df["predicted_label"].to_numpy() == 0)

	print(f"Total aligned samples: {len(aligned_df)}")
	print(f"Misclassified: {int(mis_mask.sum())}")
	print(f"False Positives: {int(fp_mask.sum())}")
	print(f"False Negatives: {int(fn_mask.sum())}")

	base_scope_mask = np.ones(len(aligned_df), dtype=bool)
	if args.scope == "misclassified":
		base_scope_mask = mis_mask
	elif args.scope == "fp":
		base_scope_mask = fp_mask
	elif args.scope == "fn":
		base_scope_mask = fn_mask
	elif args.scope == "subclass-misclassified":
		if not args.subclass:
			raise ValueError("--scope subclass-misclassified requires --subclass.")
		if "subclass_inside_outside" not in aligned_df.columns:
			raise ValueError("Column 'subclass_inside_outside' not found in test dataset.")
		subclass_mask = aligned_df["subclass_inside_outside"].astype(str).to_numpy() == args.subclass
		base_scope_mask = mis_mask & subclass_mask

	selected_idx = np.flatnonzero(base_scope_mask)
	if selected_idx.size == 0:
		raise ValueError(f"No samples found for scope='{args.scope}' and subclass='{args.subclass}'.")

	print(f"Selected scope: {args.scope}")
	print(f"Samples in scope before max-explain: {selected_idx.size}")

	ensure_shap_keras_compat()

	if args.max_explain > 0:
		selected_idx = selected_idx[: args.max_explain]

	n_explain = int(selected_idx.size)
	n_bg = min(max(1, args.background_size), n_explain)
	chunk_size = min(max(1, args.chunk_size), n_explain)
	shap_nsamples = max(1, args.shap_nsamples)
	max_display = max(1, args.max_display)

	background_idx = selected_idx[:n_bg]
	background = per_tap_features[background_idx]
	to_explain = per_tap_features[selected_idx]
	aligned_df = aligned_df.iloc[selected_idx].reset_index(drop=True)

	print(
		f"Running SHAP with background_size={n_bg}, n_explain={n_explain}, "
		f"chunk_size={chunk_size}, shap_nsamples={shap_nsamples}"
	)

	explainer = shap.GradientExplainer(head_model, background)
	shap_tensor = build_shap_tensor_from_per_tap(
		explainer=explainer,
		to_explain=to_explain,
		chunk_size=chunk_size,
		shap_nsamples=shap_nsamples,
	)

	feats_per_tap = int(per_tap_features.shape[3])
	tap_start = int(params.get("tap_range", [1, 30])[0])
	flat_shap, feature_names = flatten_per_tap_tensor(
		tensor=shap_tensor,
		link_ids=link_ids,
		tap_start=tap_start,
		feats_per_tap=feats_per_tap,
	)
	flat_features, _ = flatten_per_tap_tensor(
		tensor=to_explain,
		link_ids=link_ids,
		tap_start=tap_start,
		feats_per_tap=feats_per_tap,
	)

	if args.scope == "all":
		title = "CNN SHAP Feature Importance - All Samples"
	elif args.scope == "misclassified":
		title = "CNN SHAP Feature Importance - Misclassified Samples"
	elif args.scope == "fp":
		title = "CNN SHAP Feature Importance - False Positives"
	elif args.scope == "fn":
		title = "CNN SHAP Feature Importance - False Negatives"
	else:
		title = f"CNN SHAP Feature Importance - {args.subclass} - Misclassified"

	plot_feature_summary_matrix(
		shap_matrix=flat_shap,
		feature_matrix=flat_features,
		feature_names=feature_names,
		title=title,
		max_display=max_display,
	)


if __name__ == "__main__":
	run()
