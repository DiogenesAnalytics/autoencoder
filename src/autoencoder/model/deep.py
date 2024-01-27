"""Getting deep into autoencoders."""
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
from .base import Latent
from .base import MetaLayer


@dataclass
class DeepParams(BaseModelParams):
    """Model layer parameters class for deep autoencoder."""

    # default values from: https://blog.keras.io/building-autoencoders-in-keras.html
    default_parameters: ClassVar[Dict[str, MetaLayer]] = {
        "l0": Inputs(layers.InputLayer, {"input_shape": (784,)}),
        "l1": Encode(layers.Dense, {"units": 128, "activation": "relu"}),
        "l2": Encode(layers.Dense, {"units": 64, "activation": "relu"}),
        "l3": Latent(layers.Dense, {"units": 32, "activation": "relu"}),
        "l4": Decode(layers.Dense, {"units": 64, "activation": "relu"}),
        "l5": Decode(layers.Dense, {"units": 128, "activation": "relu"}),
        "l6": Decode(layers.Dense, {"units": 784, "activation": "sigmoid"}),
    }

    # setup instance layer params
    l0: Optional[Dict[str, Any]] = None
    l1: Optional[Dict[str, Any]] = None
    l2: Optional[Dict[str, Any]] = None
    l3: Optional[Dict[str, Any]] = None
    l4: Optional[Dict[str, Any]] = None
    l5: Optional[Dict[str, Any]] = None
    l6: Optional[Dict[str, Any]] = None


class DeepAE(BaseAutoencoder):
    """Getting deep into autoencoders."""

    _default_config = DeepParams()

    def __init__(self, model_config: Optional[DeepParams] = None) -> None:
        """Overrided base constructor to set the layer params class used."""
        # call super
        super().__init__(model_config=model_config)
