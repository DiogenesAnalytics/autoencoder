"""Autoencoder base class."""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Generator
from typing import Optional
from typing import Tuple

import keras
from keras.layers import Layer
from typing_extensions import TypeAlias


# custom types
DefaultParams: TypeAlias = Dict[str, Tuple[Layer, Dict[str, Any]]]


class BaseLayerParams(ABC):
    """Autoencoder layers hyperparameters configuration base class."""

    def __post_init__(self) -> None:
        """Store updated params and get sequence index slices."""
        # get updated parameters for instance
        self._instance_parameters = self._build_instance_params()

    def _filter_layer_attrs(self) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Filter out layer attributes from class instance."""
        # get all attributes and values in class instance namespace
        for attr, value in self.__dict__.items():
            # make sure attribute name is in default parameters
            if attr in self.default_parameters.keys():
                # generate tuple pairs
                yield attr, value

    def _update_layer_params(
        self,
    ) -> Generator[Tuple[Layer, Dict[str, Any]], None, None]:
        """Update default layer parameters values."""
        # get layer instance attrs and their values
        for attr, value in self._filter_layer_attrs():
            # unpack default parameters
            layer, params = self.default_parameters[attr]

            # check if none
            if value is not None:
                # merge instance onto default
                params |= value

            # generate
            yield layer, params

    def _build_instance_params(self) -> Tuple[Tuple[Layer, Dict[str, Any]], ...]:
        """Create mutable sequence of layer params for instance."""
        return tuple(self._update_layer_params())

    @property
    @abstractmethod
    def default_parameters(self) -> DefaultParams:
        """Defines the required default layer parameters attribute."""
        pass


@dataclass
class BaseAutoencoder(ABC):
    """Autoencoder base class."""

    model_config: Optional[BaseLayerParams] = None

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
    def _default_config(self) -> BaseLayerParams:
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
