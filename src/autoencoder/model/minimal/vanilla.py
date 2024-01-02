"""A simple autoencoder to get you started."""
__all__ = [
    "MinAE",
    "MinParams",
]
from dataclasses import dataclass
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Optional

from keras import layers

from ..base import BaseAutoencoder
from ..base import BaseModelParams
from ..base import Decode
from ..base import Encode
from ..base import Inputs
from ..base import MetaLayer


@dataclass
class MinParams(BaseModelParams):
    """Layer parameters class for minimal autoencoder."""

    # default values from: https://blog.keras.io/building-autoencoders-in-keras.html
    default_parameters: ClassVar[Dict[str, MetaLayer]] = {
        "l0": Inputs(layers.InputLayer, {"input_shape": (784,)}),
        "l1": Encode(layers.Dense, {"units": 32, "activation": "relu"}),
        "l2": Decode(layers.Dense, {"units": 784, "activation": "sigmoid"}),
    }

    # setup instance layer params
    l0: Optional[Dict[str, Any]] = None
    l1: Optional[Dict[str, Any]] = None
    l2: Optional[Dict[str, Any]] = None


class MinAE(BaseAutoencoder):
    """A simple autoencoder to get you started."""

    _default_config = MinParams()

    def __init__(self, model_config: Optional[MinParams] = None) -> None:
        """Overrided base constructor to set the layer params class used."""
        # call super
        super().__init__(model_config=model_config)
