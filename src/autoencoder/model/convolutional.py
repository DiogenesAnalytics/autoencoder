"""A more convoluted autoencoder."""
from dataclasses import dataclass
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Optional

from keras import layers

from .base import BaseAutoencoder
from .base import BaseModelParams
from .base import Decode
from .base import Encode
from .base import Inputs
from .base import MetaLayer


@dataclass
class ConvolutionalParams(BaseModelParams):
    """Model layer parameters class for convolutional autoencoder."""

    # default values from: https://blog.keras.io/building-autoencoders-in-keras.html
    default_parameters: ClassVar[Dict[str, MetaLayer]] = {
        "l0": Inputs(layers.InputLayer, {"input_shape": (28, 28, 1)}),
        "l1": Encode(
            layers.Conv2D,
            {
                "filters": 16,
                "kernel_size": (3, 3),
                "activation": "relu",
                "padding": "same",
            },
        ),
        "l2": Encode(layers.MaxPooling2D, {"pool_size": (2, 2), "padding": "same"}),
        "l3": Encode(
            layers.Conv2D,
            {
                "filters": 8,
                "kernel_size": (3, 3),
                "activation": "relu",
                "padding": "same",
            },
        ),
        "l4": Encode(layers.MaxPooling2D, {"pool_size": (2, 2), "padding": "same"}),
        "l5": Encode(
            layers.Conv2D,
            {
                "filters": 8,
                "kernel_size": (3, 3),
                "activation": "relu",
                "padding": "same",
            },
        ),
        "l6": Encode(layers.MaxPooling2D, {"pool_size": (2, 2), "padding": "same"}),
        "l7": Decode(
            layers.Conv2D,
            {
                "filters": 8,
                "kernel_size": (3, 3),
                "activation": "relu",
                "padding": "same",
            },
        ),
        "l8": Decode(layers.UpSampling2D, {"size": (2, 2)}),
        "l9": Decode(
            layers.Conv2D,
            {
                "filters": 8,
                "kernel_size": (3, 3),
                "activation": "relu",
                "padding": "same",
            },
        ),
        "l10": Decode(layers.UpSampling2D, {"size": (2, 2)}),
        "l11": Decode(
            layers.Conv2D,
            {
                "filters": 16,
                "kernel_size": (3, 3),
                "activation": "relu",
            },
        ),
        "l12": Decode(layers.UpSampling2D, {"size": (2, 2)}),
        "l13": Decode(
            layers.Conv2D,
            {
                "filters": 1,
                "kernel_size": (3, 3),
                "activation": "sigmoid",
                "padding": "same",
            },
        ),
    }

    # setup instance layer params
    l0: Optional[Dict[str, Any]] = None
    l1: Optional[Dict[str, Any]] = None
    l2: Optional[Dict[str, Any]] = None
    l3: Optional[Dict[str, Any]] = None
    l4: Optional[Dict[str, Any]] = None
    l5: Optional[Dict[str, Any]] = None
    l6: Optional[Dict[str, Any]] = None
    l7: Optional[Dict[str, Any]] = None
    l8: Optional[Dict[str, Any]] = None
    l9: Optional[Dict[str, Any]] = None
    l10: Optional[Dict[str, Any]] = None
    l11: Optional[Dict[str, Any]] = None
    l12: Optional[Dict[str, Any]] = None
    l13: Optional[Dict[str, Any]] = None


class ConvolutionalAutoencoder(BaseAutoencoder):
    """A more convoluted autoencoder."""

    _default_config = ConvolutionalParams()

    def __init__(self, model_config: Optional[ConvolutionalParams] = None) -> None:
        """Overrided base constructor to set the layer params class used."""
        # call super
        super().__init__(model_config=model_config)
