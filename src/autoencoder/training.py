"""All things related to training autoencoders."""
from dataclasses import dataclass
from typing import Callable
from typing import Optional
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


@dataclass
class build_encode_dim_loss_function:  # noqa
    """Decorator factory class to build a penalized loss function."""

    encode_dim: int
    regularization_factor: float = 0.001
    axis: Tuple[int, ...] = (1, 2, 3)

    def __call__(
        self,
        penalized_loss: Optional[Callable[[tf.Tensor, int, float], tf.Tensor]] = None,
    ) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        """Call decorator to build custom penalized loss function."""
        return self.decorate(penalized_loss)

    @staticmethod
    def default_penalty(loss: tf.Tensor, encode: int, reg: float) -> tf.Tensor:
        """Calculate the default penalty for the encoding dimension."""
        return loss + (loss * encode * reg)

    def decorate(
        self,
        penalized_loss: Optional[Callable[[tf.Tensor, int, float], tf.Tensor]] = None,
    ) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        """Decorator that builds the complete penalized loss function."""
        # check for none
        if penalized_loss is None:
            # get default
            penalized_loss = self.default_penalty

        # create function
        def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            """Calculate reconstruction error and apply penalty."""
            # calculate the dynamic mean reconstruction error on training data
            reconstruction_loss = tf.reduce_mean(
                tf.square(y_true - y_pred), axis=self.axis
            )

            # calculate penalized loss
            return penalized_loss(
                reconstruction_loss, self.encode_dim, self.regularization_factor
            )

        # optimize with tf.function
        optimized_func: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = tf.function(
            custom_loss
        )

        # get wrapped function
        return optimized_func
