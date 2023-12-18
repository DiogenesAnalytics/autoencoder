"""Autoencoder base class."""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from inspect import signature
from typing import Any
from typing import Dict
from typing import Generator
from typing import Iterator
from typing import Optional
from typing import Tuple

import keras
from keras.layers import Layer


@dataclass
class MetaLayer:
    """Container for a keras Layer and its kwargs."""

    layer: Layer
    params: Dict[str, Any]

    def __iter__(self) -> Iterator[Any]:
        """Define iteration behavior."""
        return iter((self.layer, self.params))


class Encode(MetaLayer):
    """Designate a meta layer as an ecoding layer."""


class Decode(MetaLayer):
    """Designate a meta layer as a decoding layer."""


class Inputs(MetaLayer):
    """Designate a meta layer as an input layer."""


class BaseModelParams(ABC):
    """Autoencoder model layer hyperparameters configuration base class."""

    @abstractmethod
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """Defines the argument type that the constructor should accept."""
        pass

    @property
    @abstractmethod
    def default_parameters(self) -> Dict[str, MetaLayer]:
        """Defines the required default layer parameters attribute."""
        # NOTE: this dictionary sets layer order used to build the keras.Model
        pass

    def __post_init__(self) -> None:
        """Store updated params and get sequence index slices."""
        # get updated parameters for instance
        self._instance_parameters = self._build_instance_params()

    def _filter_layer_attrs(self) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Filter out layer attributes from class instance."""
        # get constructor signature
        init_sig = signature(self.__class__.__init__)

        # loop over layer_name/params in defaults
        for layer_id in self.default_parameters.keys():
            # now find corresponding layer_id in constructor args
            assert (
                layer_id in init_sig.parameters.keys()
            ), "Constructor arguments must match default_parameter dict keys."

            # finally get value of constructor args
            yield layer_id, self.__dict__[layer_id]

    def _update_layer_params(self) -> Generator[MetaLayer, None, None]:
        """Update default layer parameters values."""
        # get layer instance attrs and their values
        for attr, value in self._filter_layer_attrs():
            # get meta layer object
            meta_layer = self.default_parameters[attr]

            # get copy of default params
            default_params_copy = meta_layer.params.copy()

            # check if none
            if value is not None:
                # update default with any user supplied kwargs
                default_params_copy |= value

            # add labels
            if "name" not in default_params_copy:
                # just use attribute from constructor signature
                default_params_copy["name"] = attr
            else:
                # get name
                old_name = default_params_copy["name"]

                # add id to front of name
                default_params_copy["name"] = f"{attr}.{old_name}"

            # get type of meta layer
            layer_type = type(meta_layer)

            # generate new meta layer object
            yield layer_type(meta_layer.layer, default_params_copy)

    def _build_instance_params(self) -> Tuple[MetaLayer, ...]:
        """Create mutable sequence of layer params for instance."""
        return tuple(self._update_layer_params())


@dataclass
class BaseAutoencoder(ABC):
    """Autoencoder base class."""

    model_config: Optional[BaseModelParams] = None

    def __post_init__(self) -> None:
        """Setup autoencoder model."""
        # check if default config used
        if self.model_config is None:
            # get default
            self.model_config = self._default_config

        # build model ...
        self.model = self._build_model()

    @property
    @abstractmethod
    def _default_config(self) -> BaseModelParams:
        """Defines the default layer parameters attribute."""
        pass

    def _build_model(self) -> keras.Model:
        """Assemple autoencoder from encoder/decoder submodels."""
        # chcek model config exists
        assert self.model_config is not None

        # setup layer reference caches
        encode_layers, decode_layers, all_layers = [], [], []

        # loop over meta_layers
        for meta_layer in self.model_config._instance_parameters:
            # get components
            layer_constructor, params = meta_layer

            # instantiate layer
            layer = layer_constructor(**params)

            # check layer type
            if isinstance(meta_layer, Encode):
                # add to encode layers
                encode_layers.append(layer)

            elif isinstance(meta_layer, Decode):
                # add to encode layers
                decode_layers.append(layer)

            elif isinstance(meta_layer, Inputs):
                # add to total autoencoder layer
                all_layers.append(layer)

            else:
                # get unknown type
                wrong_type = type(meta_layer)

                # notify user
                raise TypeError(
                    f"Layer type must be Encode, Decode, or Inputs not {wrong_type}."
                )

        # build encoder/decoder models ...
        self._encoder = keras.Sequential(encode_layers, name="Encoder")
        self._decoder = keras.Sequential(decode_layers, name="Decoder")

        # add encoder/decoder models to all layers
        all_layers += [self._encoder, self._decoder]

        # ... and finally the full autoencoder
        return keras.Sequential(all_layers, name=f"{self.__class__.__name__}")

    def summary(self, **kwargs: Any) -> None:
        """Wrapper for Keras model.summary method."""
        self.model.summary(**kwargs)

    def compile(self, **kwargs: Any) -> None:
        """Wrapper for Keras model.compile method."""
        self.model.compile(**kwargs)

    def fit(self, **kwargs: Any) -> None:
        """Wrapper for the Keras model.fit method."""
        self.model.fit(**kwargs)
