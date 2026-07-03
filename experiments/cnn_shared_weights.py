from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from keras.src.initializers import HeNormal
from tensorflow import keras

from cpd.core.data.can_fd_messages import LinkID
from cpd.core.data.dataset import DatasetBase
from cpd.core.ml.labeler.base_labeler import BaseLabeler
from cpd.core.ml.labeler.VSD_labeler import VSDLabeler
from cpd.core.ml.models.base_model import EvaluationResult
from cpd.core.ml.models.tensorflow.custom_layers import (
    TFFFTFeatureExtraction,
    TFGPUFFTFeatureExtraction,
    TFTapRangeSelection,
)
from cpd.core.ml.models.tensorflow.tensorflow_model import TensorflowModel
from cpd.core.ml.models.tensorflow.utils import (
    set_trainable_by_type,
    transfer_model_weights,
)

MEMORY_TYPE = Dict[str, Union[float, np.ndarray]]


def conv_block(
    x: tf.Tensor, out_channels: int, stride: int, initializer: keras.Initializer, activation: str, name: str
):
    layers = keras.Sequential(
        [
            keras.layers.Conv1D(
                out_channels, 3, strides=stride, use_bias=False, padding="same", kernel_initializer=initializer
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation),
        ],
        name=name,
    )
    if isinstance(x, list):
        return [layers(i) for i in x]
    else:
        return layers(x)


def dw_separable(
    x: tf.Tensor, out_channels: int, stride: int, initializer: keras.Initializer, activation: str, name: str
):
    in_channels = x[0].shape[-1] if isinstance(x, list) else x.shape[-1]
    layers = keras.Sequential(
        [
            keras.layers.Conv1D(
                in_channels,
                groups=in_channels,
                kernel_size=3,
                strides=stride,
                use_bias=False,
                padding="same",
                kernel_initializer=initializer,
            ),  # equivalent to DepthwiseConv1D without requiring DEPTHWISE_CONV_2D TFLM op
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation),
            keras.layers.Conv1D(
                out_channels, kernel_size=1, strides=1, use_bias=False, padding="same", kernel_initializer=initializer
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation),
        ],
        name=name,
    )
    if isinstance(x, list):
        return [layers(i) for i in x]
    else:
        return layers(x)


def irb(
    x: tf.Tensor,
    out_channels: int,
    stride: int,
    expand: int,
    initializer: keras.Initializer,
    activation: str,
    name: str,
):
    in_channels = x[0].shape[-1] if isinstance(x, list) else x.shape[-1]

    layers = keras.Sequential(
        [
            # 1x1 conv1d, BN, ReLU
            keras.layers.Conv1D(
                in_channels * expand,
                kernel_size=1,
                strides=1,
                use_bias=False,
                padding="same",
                kernel_initializer=initializer,
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation),
            # 3x3 dwise s=s, BN, ReLU
            keras.layers.Conv1D(
                in_channels * expand,
                groups=in_channels * expand,
                kernel_size=3,
                strides=stride,
                use_bias=False,
                padding="same",
                kernel_initializer=initializer,
            ),  # equivalent to DepthwiseConv1D without requiring DEPTHWISE_CONV_2D TFLM op
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation),
            # linear 1x1 conv2d, BN, ReLU
            keras.layers.Conv1D(
                out_channels, kernel_size=1, strides=1, use_bias=False, padding="same", kernel_initializer=initializer
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation),
        ],
        name=name,
    )
    if in_channels == out_channels and stride == 1:
        # add residual connection
        if isinstance(x, list):
            return [layers(i) + i for i in x]
        else:
            return layers(x) + x
    else:
        if isinstance(x, list):
            return [layers(i) for i in x]
        else:
            return layers(x)


class CNNSharedWeights(TensorflowModel):
    """CNN architecture which consists of two steps: a tap-wise feature extractor with shared weights across all taps,
    and an anchor-wise classifier with shared weights across both anchors. The classification results from all
    anchors are combined using max pooling."""

    def __init__(
        self,
        links: List[LinkID] = [1, 2, 3, 4],
        tap_range: Tuple[int, int] = (1, 30),
        labeler: BaseLabeler = VSDLabeler(),
        lr: float = 0.001,
        optimizer: str = "adam",
        epochs: int = 5,
        batch_size: int = 32,
        val_split: float = 0.1,
        window_size: int = 72,
        window_shift: int = 24,
        target_sampling_rate: int = 96,
        early_stopping_rounds: int = 5,
        fft_bins: int = 16,
        fft_negative_freq: bool = True,
        feats_per_tap: int = 4,
        pooling_layer_type: str = "avg",
        pertap_model_config: List[Tuple[int, int]] = None,
        peranchor_model_config: List[int] = None,
        block: str = "dw_separable",
        activation: str = "relu",
        share_across_anchors: bool = True,
        **kwargs,
    ):
        """
        Args:
            links (List[LinkID], optional): FBD6 links to use during the training or inference. Defaults to [2, 4].
            tap_range (Tuple[int, int], optional): Defines the tap range. Defaults to (1, 30), which means 0:29.
            labeler (BaseLabeler, optional): Labeler for generating the target class. Defaults to VSDLabeler().
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            optimizer (str, optional): Optimizer to use. Defaults to "adam".
            epochs (int, optional): Number of epochs for training. Defaults to 5.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            val_split (float, optional): Validation dataset size. Defaults to 0.1.
            window_size (int, optional): Windows Size used to generate the dataset for input format. Defaults to 72.
            window_shift (int, optional): Number of samples to skip to catch the next window. Defaults to 24.
            target_sampling_rate (int, optional): Sensor sampling frequency. Defaults to 96.
            early_stopping_rounds (int, optional): Early stopping rounds. Defaults to 5.
            fft_bins (int, optional): Number of frequency bins to use from the FFT. Defaults to 16.
            fft_negative_freq (bool, optional): Whether to include negative frequencies in FFT output. Defaults to True.
            feats_per_tap (int, optional): Number of features per tap. Defaults to 4.
            pooling_layer_type (str, optional): Pooling layer type ("avg" or "max"). Defaults to "avg".
            pertap_model_config (List[Tuple[int, int]], optional): List tuples of (out_channels, stride) for
                convolutional blocks in per-tap feature extractor. Defaults to None.
            peranchor_model_config (List[int], optional): List of units for per-anchor model dense layers.
                Defaults to None.
            block (str, optional): Main building block to use for the per-tap model. Depthwise Separable Convolutions
                (MobileNet): "dw_separable", or Inverted Residual Blocks (MobileNetV2): "irb".
            activation (str, optional): Activation function to use. Defaults to "relu".
            share_across_anchors (bool, optional): Whether to share classifier weights across anchors. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        self.fft_bins = fft_bins
        self.fft_negative_freq = fft_negative_freq
        self.feats_per_tap = feats_per_tap
        if pooling_layer_type not in ["avg", "max", "concat"]:
            raise ValueError(
                f"Unsupported pooling_layer_type {pooling_layer_type}, must be 'avg' or 'max', or 'concat'"
            )
        self.pooling_layer_type = pooling_layer_type
        self.pertap_model_config = (
            pertap_model_config if pertap_model_config is not None else [(16, 2), (32, 1), (64, 2), (64, 1), (128, 2)]
        )
        self.peranchor_model_config = peranchor_model_config if peranchor_model_config is not None else [16]
        self.share_across_anchors = share_across_anchors
        self.n_anchors = 2
        self.activation = activation
        self.block = block
        super().__init__(
            links=links,
            tap_range=tap_range,
            labeler=labeler,
            epochs=epochs,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            val_split=val_split,
            window_size=window_size,
            window_shift=window_shift,
            target_sampling_rate=target_sampling_rate,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} object, "
            + f"_n_rr: '{self.window_size}', "
            + f"lr: '{self.lr}', "
            + f"optimizer: '{self.optimizer}', "
            + f"{super().__repr__()}>"
        )

    def get_model_specific_tap_range(self, *args, **kwargs) -> Tuple:
        """
        Get the model specific tap_range.
        Currently due to the numpy/tf preprocessing layers and differences,
        the LR is cutting the taps directly in the preprocessing.
        However, the CNN needs the full range as input despite a smaller tap_range provided,
        and the taps are only cut inside the model

        Returns:
            Tuple: tap range tuple
        """
        return (1, 30)

    def build_model(
        self,
        ipbasis_map: bool = False,
        tflm_fft: bool = False,
        per_anchor: bool = False,
        *args,
        **kwargs,
    ) -> keras.Model:
        """
        Build the shared weights CNN model.

        Args:
            pipeline (List[Union[keras.layers.Layer, List[Tuple[type, dict]]]], optional):
                A list describing the preprocessing pipeline. Each element can be a keras layer or a list of
                (layer class, parameters) tuples for feature concatenation.
            ipbasis_map (bool, optional): If False, use link names for input layers; otherwise, use IPB names.
            per_anchor (bool, optional): Output per-anchor results without max-pooling over anchors
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            keras.Model: Compiled Keras model
        """
        if self.links not in [[1, 2, 3, 4], [2, 4]]:
            raise ValueError("link configuration not supported, needs either 1 or 2 links per anchor")

        n_anchors = self.n_anchors
        n_links = len(self.links)
        n_taps = self.tap_range[1] - self.tap_range[0] + 1

        inputs = []
        for link_id in self.links:
            input_layer = keras.Input(
                shape=(self.window_size, 30, 2),
                batch_size=1,
                dtype=tf.float32,
                name=LinkID(link_id).ipb_name() if ipbasis_map else LinkID(link_id).name,
            )
            inputs.append(input_layer)

        # tap range selection
        x = []
        for input in inputs:
            x.append(TFTapRangeSelection(self.tap_range)(input))

        # stack links -> shape [batch_size * links, window_size, taps, 2 complex components]
        x = keras.ops.stack(x, axis=1)
        x = keras.ops.reshape(x, (-1, *x.shape[2:]))

        # mean removal
        x -= keras.ops.mean(x, axis=1, keepdims=True)

        # FFT
        fft_layer = TFFFTFeatureExtraction if tflm_fft else TFGPUFFTFeatureExtraction
        x = fft_layer(freq_bins=self.fft_bins, negative_freq=self.fft_negative_freq)(x)

        # per-tap feature extraction with shared weights
        x = keras.ops.transpose(x, [0, 2, 1, 3])
        x = keras.ops.reshape(x, (-1, x.shape[2], 1))  # fold taps and links into batch dim
        x = self._per_tap_model(x, unstack=n_links if tflm_fft else None)
        x = keras.ops.reshape(x, (-1, n_links, n_taps, x.shape[1]))  # get links and taps back out

        # per-anchor classification
        if self.share_across_anchors:
            x = keras.ops.reshape(x, (-1, n_links // n_anchors * n_taps, x.shape[3]))  # fold anchors into batch dim
            x = self._per_anchor_model(x)
            x = keras.ops.reshape(x, (-1, n_anchors, x.shape[1]))  # get anchors back
        else:
            x = keras.ops.reshape(x, (-1, n_anchors, n_links // n_anchors * n_taps, x.shape[3]))
            x = keras.ops.stack(
                [self._per_anchor_model(x[:, i], name=f"per_anchor_model_{i}") for i in range(n_anchors)],
                axis=1,
            )

        if per_anchor:
            return keras.Model(inputs, x)

        # max pooling across anchors
        x = keras.ops.max(x, axis=1)

        return keras.Model(inputs, x)

    def _per_tap_model(self, x, name="per_tap_model", unstack=None):
        initializer = HeNormal()

        if unstack:
            # for TFLM conversion, it may be beneficial not to process the whole batch at once
            # (x.shape[0] = n_links * n_taps).
            # Splitting here reduces RAM usage at the cost of needing more ROM (duplicate ops).
            x = keras.ops.split(x, unstack, axis=0)

        x = conv_block(
            x,
            self.pertap_model_config[0][0],
            self.pertap_model_config[0][1],
            initializer,
            self.activation,
            name="block_0",
        )
        for i in range(1, len(self.pertap_model_config)):
            if self.block == "dw_separable":
                x = dw_separable(
                    x,
                    self.pertap_model_config[i][0],
                    self.pertap_model_config[i][1],
                    initializer,
                    self.activation,
                    name=f"block_{i}",
                )
            elif self.block == "irb":
                x = irb(
                    x,
                    self.pertap_model_config[i][0],
                    self.pertap_model_config[i][1],
                    6 if i > 1 else 1,
                    initializer,
                    self.activation,
                    name=f"block_{i}",
                )
            else:
                raise ValueError(f"building block {self.block} not supported")

        if unstack:
            x = keras.ops.concatenate(x, axis=0)

        # concat pooling (-> combines both avg and max pooling)
        if self.pooling_layer_type == "avg":
            x = keras.layers.GlobalAvgPool1D()(x)
        elif self.pooling_layer_type == "max":
            x = keras.layers.GlobalMaxPool1D()(x)
        elif self.pooling_layer_type == "concat":
            x = keras.ops.concatenate([keras.layers.GlobalAvgPool1D()(x), keras.layers.GlobalMaxPool1D()(x)], axis=-1)
        else:
            raise ValueError(f"pooling layer type {self.pooling_layer_type} not supported")

        return keras.Sequential(
            [
                keras.layers.Dense(self.feats_per_tap, kernel_initializer=initializer, use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.Activation(self.activation),
            ],
            name=name,
        )(x)

    def _per_anchor_model(self, x, name="per_anchor_model"):
        initializer = HeNormal()

        return keras.Sequential(
            [
                keras.layers.Flatten(),
            ]
            + [
                layer
                for units in self.peranchor_model_config
                for layer in [
                    keras.layers.Dense(units, kernel_initializer=initializer, use_bias=False),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation(self.activation),
                ]
            ]
            + [
                keras.layers.Dense(1, activation="sigmoid", kernel_initializer=initializer),
            ],
            name=name,
        )(x)

    def evaluate_by_anchor(self, dataset: DatasetBase) -> dict[str, EvaluationResult]:
        """
        Evaluate the model on the given dataset, providing separate results for each anchor.

        Args:
            dataset (DatasetBase): Dataset to evaluate
        Returns:
            dict[str, EvaluationResult]: Result of the evaluation by constraint name
        """
        tf_dataset = self.get_tf_dataset(
            dataset=dataset,
            links=self.links,
            tap_range=self.model_specific_tap_range,
            window_size=self.window_size,
            window_shift=self.window_shift,
            target_sampling_rate=self.target_sampling_rate,
            labeler=self.labeler,
            shuffle=self.shuffle_data,
            preprocessing_func=self.pipeline_process,
        )
        per_anchor_model = self.build_model(per_anchor=True)
        transfer_model_weights(self._model, per_anchor_model)
        predictions_per_anchor = per_anchor_model.predict(tf_dataset["feature"], batch_size=self.batch_size, verbose=0)

        results_per_anchor = {}
        for i, anchor in enumerate(["VM", "HM"]):
            result = EvaluationResult()
            predictions = predictions_per_anchor[:, i]
            result.predictions = predictions.tolist()
            result.filenames = tf_dataset["filename"]
            result.true_classes = tf_dataset["class"]
            result.true_classes_labels = [self.labeler.get_label_name_by_int(y) for y in tf_dataset["class"]]
            result.predicted_classes = self.predict_class(predictions).tolist()
            result.predicted_classes_labels.extend(
                [self.labeler.get_label_name_by_int(y) for y in result.predicted_classes]
            )
            result.window_idx = tf_dataset["window_idx"]
            result.subclasses = tf_dataset["subclass"]
            results_per_anchor[anchor] = result

        return results_per_anchor

    def freeze(self):
        self._model.get_layer("per_tap_model").trainable = False
        classifiers = (
            [self._model.get_layer("per_anchor_model")]
            if self.share_across_anchors
            else [self._model.get_layer(f"per_anchor_model_{i}" for i in range(self.n_anchors))]
        )
        for classifier in classifiers:
            set_trainable_by_type(classifier, keras.layers.BatchNormalization, False)

    def unfreeze(self):
        self._model.get_layer("per_tap_model").trainable = True
        classifiers = (
            [self._model.get_layer("per_anchor_model")]
            if self.share_across_anchors
            else [self._model.get_layer(f"per_anchor_model_{i}" for i in range(self.n_anchors))]
        )
        for classifier in classifiers:
            set_trainable_by_type(classifier, keras.layers.BatchNormalization, True)
