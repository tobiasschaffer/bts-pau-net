"""TensorFlow implementation of the 3-D PAU-Net used in the experiments."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    Conv3D,
    GroupNormalization,
    Input,
    SpatialDropout3D,
    UpSampling3D,
)

DEFAULT_INPUT_SHAPE = (4, 96, 192, 160)
BLOCK_TYPES = ("plain", "resnet", "resnetn", "preact", "preact_short")


def dice_loss(epsilon: float = 1e-8):
    """Return the soft Dice loss used in the original notebooks."""

    def loss(y_true, y_pred):
        y_true_float = tf.cast(y_true, tf.float32)
        intersection = K.sum(K.abs(y_true_float * y_pred), axis=(-3, -2, -1))
        denominator = K.sum(
            K.square(y_true_float) + K.square(y_pred), axis=(-3, -2, -1)
        )
        return 1.0 - K.mean(2.0 * intersection / (denominator + epsilon), axis=(0, 1))

    loss.__name__ = "dice_loss"
    return loss


def thresholded_dice(y_true, y_pred):
    """Mean hard Dice coefficient across batch and output channels."""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=(-3, -2, -1))
    denominator = K.sum(y_true, axis=(-3, -2, -1)) + K.sum(
        y_pred, axis=(-3, -2, -1)
    )
    return K.mean(2.0 * intersection / (denominator + 1e-8), axis=(0, 1))


def channel_dice(channel: int, name: str):
    """Create a named hard-Dice metric for one output channel."""

    def metric(y_true, y_pred):
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        y_true_float = tf.cast(y_true, tf.float32)
        intersection = K.sum(
            K.abs(y_true_float * y_pred_binary), axis=(-3, -2, -1)
        )
        denominator = K.sum(y_true_float, axis=(-3, -2, -1)) + K.sum(
            y_pred_binary, axis=(-3, -2, -1)
        )
        return K.mean(2.0 * intersection / (denominator + 1e-8), axis=0)[channel]

    metric.__name__ = f"dice_{name.lower()}"
    return metric


def residual_block(inputs, filters: int, block_type: str = "preact"):
    """Build one of the residual blocks evaluated in the notebooks."""
    if block_type not in BLOCK_TYPES:
        raise ValueError(f"Unknown block type {block_type!r}; choose from {BLOCK_TYPES}")

    if block_type == "plain":
        x = Conv3D(filters, 3, padding="same", data_format="channels_first")(inputs)
        return Activation("relu")(x)
    if block_type == "resnet":
        x = Conv3D(filters, 3, padding="same", data_format="channels_first")(inputs)
        x = Activation("relu")(x)
        x = Conv3D(filters, 3, padding="same", data_format="channels_first")(x)
        return Add()([x, inputs])
    if block_type == "resnetn":
        x = Conv3D(filters, 3, padding="same", data_format="channels_first")(inputs)
        x = GroupNormalization(groups=8, axis=1)(x)
        x = Activation("relu")(x)
        x = Conv3D(filters, 3, padding="same", data_format="channels_first")(x)
        x = GroupNormalization(groups=8, axis=1)(x)
        return Activation("relu")(Add()([x, inputs]))
    if block_type == "preact":
        x = GroupNormalization(groups=8, axis=1)(inputs)
        x = Activation("relu")(x)
        x = Conv3D(filters, 3, padding="same", data_format="channels_first")(x)
        x = GroupNormalization(groups=8, axis=1)(x)
        x = Activation("relu")(x)
        x = Conv3D(filters, 3, padding="same", data_format="channels_first")(x)
        return Add()([x, inputs])

    x = GroupNormalization(groups=8, axis=1)(inputs)
    x = Activation("relu")(x)
    x = Conv3D(filters, 3, padding="same", data_format="channels_first")(x)
    return Add()([x, inputs])


def build_pau_net(
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
    output_channels: int = 3,
    block_type: str = "preact",
    channel_names: Sequence[str] | None = None,
) -> tf.keras.Model:
    """Build and compile the overshooting 3-D PAU-Net.

    The decoder produces a segmentation at twice the spatial resolution of the
    input crop, matching the architecture in the supplied notebooks.
    """
    if len(input_shape) != 4:
        raise ValueError("input_shape must be (channels, depth, height, width)")
    channels, *spatial = input_shape
    if channels % 4 or any(size % 16 for size in spatial):
        raise ValueError("Channels must be divisible by 4 and spatial sizes by 16")

    inputs = Input(tuple(input_shape))
    x = Conv3D(32, 3, padding="same", data_format="channels_first")(inputs)
    x = SpatialDropout3D(0.2, data_format="channels_first")(x)

    x0 = residual_block(x, 32, block_type)
    x = Conv3D(64, 3, strides=2, padding="same", data_format="channels_first")(x0)

    overshoot = UpSampling3D(2, data_format="channels_first")(x0)
    overshoot = Conv3D(16, 3, padding="same", data_format="channels_first")(overshoot)
    overshoot = residual_block(overshoot, 16, block_type)

    x = residual_block(x, 64, block_type)
    x1 = residual_block(x, 64, block_type)
    x = Conv3D(128, 3, strides=2, padding="same", data_format="channels_first")(x1)
    x = residual_block(x, 128, block_type)
    x = residual_block(x, 128, block_type)

    x = Conv3D(64, 1, data_format="channels_first")(x)
    x = Add()([UpSampling3D(2, data_format="channels_first")(x), x1])
    x = residual_block(x, 64, block_type)
    x = Conv3D(32, 1, data_format="channels_first")(x)
    x = Add()([UpSampling3D(2, data_format="channels_first")(x), x0])
    x = residual_block(x, 32, block_type)
    x = Conv3D(16, 1, data_format="channels_first")(x)
    x = Add()([UpSampling3D(2, data_format="channels_first")(x), overshoot])
    x = residual_block(x, 16, block_type)
    outputs = Conv3D(
        output_channels, 1, activation="sigmoid", data_format="channels_first"
    )(x)

    model = tf.keras.Model(inputs, outputs, name=f"pau_net_{output_channels}ch")
    names = list(channel_names or [str(index) for index in range(output_channels)])
    metrics = [thresholded_dice] + [
        channel_dice(index, name) for index, name in enumerate(names)
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=dice_loss(), metrics=metrics)
    return model
