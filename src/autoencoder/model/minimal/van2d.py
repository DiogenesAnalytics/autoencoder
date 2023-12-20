"""A simple autoencoder to get you started."""
__all__ = [
    "Minimal2DAutoencoder",
    "Min2DParams",
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
class Min2DParams(BaseModelParams):
    """Layer parameters class for minimal autoencoder."""

    # default values from: https://blog.keras.io/building-autoencoders-in-keras.html
    default_parameters: ClassVar[Dict[str, MetaLayer]] = {
        "l0": Inputs(layers.InputLayer, {"input_shape": (28, 28)}),
        "l1": Encode(layers.Flatten, {}),
        "l2": Encode(layers.Dense, {"units": 32, "activation": "relu"}),
        "l3": Decode(layers.Dense, {"units": 28 * 28, "activation": "sigmoid"}),
        "l4": Decode(layers.Reshape, {"target_shape": (28, 28)}),
    }

    # setup instance layer params
    l0: Optional[Dict[str, Any]] = None
    l1: Optional[Dict[str, Any]] = None
    l2: Optional[Dict[str, Any]] = None
    l3: Optional[Dict[str, Any]] = None
    l4: Optional[Dict[str, Any]] = None


class Minimal2DAutoencoder(BaseAutoencoder):
    """A simple autoencoder to get you started."""

    _default_config = Min2DParams()

    def __init__(self, model_config: Optional[Min2DParams] = None) -> None:
        """Overrided base constructor to set the layer params class used."""
        # call super
        super().__init__(model_config=model_config)
