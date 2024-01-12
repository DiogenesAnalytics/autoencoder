"""A simple autoencoder to get you started."""
__all__ = [
    "MinNDAE",
    "MinNDParams",
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
class MinNDParams(BaseModelParams):
    """Layer parameters class for minimal autoencoder."""

    # default values from: https://blog.keras.io/building-autoencoders-in-keras.html
    default_parameters: ClassVar[Dict[str, MetaLayer]] = {
        "l0": Inputs(layers.InputLayer, {"input_shape": (28, 28, 1)}),
        "l1": Encode(layers.Flatten, {}),
        "l2": Encode(layers.Dense, {"units": 32, "activation": "relu"}),
        "l3": Decode(layers.Dense, {"units": 28 * 28, "activation": "sigmoid"}),
        "l4": Decode(layers.Reshape, {"target_shape": (28, 28, 1)}),
    }

    # setup instance layer params
    l0: Optional[Dict[str, Any]] = None
    l1: Optional[Dict[str, Any]] = None
    l2: Optional[Dict[str, Any]] = None
    l3: Optional[Dict[str, Any]] = None
    l4: Optional[Dict[str, Any]] = None


class MinNDAE(BaseAutoencoder):
    """A simple autoencoder to get you started."""

    _default_config = MinNDParams()

    def __init__(self, model_config: Optional[MinNDParams] = None) -> None:
        """Overrided base constructor to set the layer params class used."""
        # call super
        super().__init__(model_config=model_config)
