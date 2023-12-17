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
            # unpack default parameters
            layer, params = self.default_parameters[attr]

            # get copy of default params
            default_params_copy = params.copy()

            # check if none
            if value is not None:
                # update default with any user supplied kwargs
                default_params_copy |= value

            # generate
            yield MetaLayer(layer, default_params_copy)

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
        # get pointer to instance parameters
        assert self.model_config is not None
        inst_params = self.model_config._instance_parameters

        # build model ...
        return keras.Sequential([layer(**params) for layer, params in inst_params])

    def summary(self, **kwargs: Any) -> None:
        """Wrapper for Keras model.summary method."""
        self.model.summary(**kwargs)

    def compile(self, **kwargs: Any) -> None:
        """Wrapper for Keras model.compile method."""
        self.model.compile(**kwargs)

    def fit(self, **kwargs: Any) -> None:
        """Wrapper for the Keras model.fit method."""
        self.model.fit(**kwargs)
