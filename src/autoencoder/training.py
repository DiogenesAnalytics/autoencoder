"""All things related to training autoencoders."""
from typing import Callable
from typing import Tuple
from typing import Union

import keras
import tensorflow as tf

from .model.base import BaseAutoencoder


def build_anomaly_loss_function(
    anomalous_data: tf.Tensor,
    model: Union[keras.Model, BaseAutoencoder],
    axis: Tuple[int, ...] = (1, 2, 3),
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Closure that sets up the custom anomaly detection loss function."""
    # check model type
    if isinstance(model, BaseAutoencoder):
        # get keras.Model object
        model = model.model

    # create function
    def anomaly_diff(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculates mean training/anomalous data reconstruction error difference."""
        # calculate the dynamic mean reconstruction error on training data
        train_reconstruction_errors = tf.reduce_mean(
            tf.square(y_true - y_pred), axis=axis
        )
        dynamic_threshold = tf.reduce_mean(train_reconstruction_errors)

        # calculate the mean reconstruction error on anomalous data
        anomalous_data_pred = model(anomalous_data, training=False)
        anomalous_data_errors = tf.reduce_mean(
            tf.square(anomalous_data - anomalous_data_pred), axis=axis
        )
        anomalous_data_mean = tf.reduce_mean(anomalous_data_errors)

        # calculate the difference
        return dynamic_threshold - anomalous_data_mean

    # optimize with tf.function
    optimized_func: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = tf.function(
        anomaly_diff
    )

    # get wrapped function
    return optimized_func


def build_encode_dim_loss_function(
    encode_dim: int,
    regularization_factor: float = 0.001,
    axis: Tuple[int, ...] = (1, 2, 3),
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Closure that sets up the custom encode dim loss function."""
    # calculate the encoding dim loss
    encode_dim_loss = encode_dim * regularization_factor

    # create function
    def penalize_encode_dimension(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Penalizes loss with additional encoding dimension value."""
        # calculate the dynamic mean reconstruction error on training data
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=axis)

        # calculate penalized loss
        return reconstruction_loss + encode_dim_loss

    # optimize with tf.function
    optimized_func: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = tf.function(
        penalize_encode_dimension
    )

    # get wrapped function
    return optimized_func
