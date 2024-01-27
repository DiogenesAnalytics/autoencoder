"""A simple autoencoder to get you started."""
__all__ = [
    "RndmNDAE",
    "RndmNDParams",
]
from dataclasses import dataclass
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Optional

import tensorflow as tf
from keras import layers

from .base import BaseAutoencoder
from .base import BaseModelParams
from .base import Decode
from .base import Inputs
from .base import Latent
from .base import MetaLayer


def random_output(x: tf.Tensor) -> tf.Tensor:
    """Function to generate a tensor of random floats the same shape as the input."""
    return tf.random.uniform(shape=tf.shape(x), dtype=tf.float32)


@dataclass
class RndmNDParams(BaseModelParams):
    """Layer parameters class for random dummy autoencoder."""

    # default values from: https://blog.keras.io/building-autoencoders-in-keras.html
    default_parameters: ClassVar[Dict[str, MetaLayer]] = {
        "l0": Inputs(layers.InputLayer, {"input_shape": (28, 28, 1)}),
        "l1": Latent(layers.Lambda, {"function": lambda x: x}),
        "l2": Decode(layers.Lambda, {"function": random_output}),
    }

    # setup instance layer params
    l0: Optional[Dict[str, Any]] = None
    l1: Optional[Dict[str, Any]] = None
    l2: Optional[Dict[str, Any]] = None


class RndmNDAE(BaseAutoencoder):
    """A dummy autoencoder that randomly generates predicted reconstructions."""

    _default_config = RndmNDParams()

    def __init__(self, model_config: Optional[RndmNDParams] = None) -> None:
        """Overrided base constructor to set the layer params class used."""
        # call super
        super().__init__(model_config=model_config)
