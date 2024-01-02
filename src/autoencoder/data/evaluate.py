"""Tools for evaluating an autoencoder's perfomance on a dataset."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from ..model.base import BaseAutoencoder
from .visualize import plot_anomalous_images
from .visualize import plot_error_distribution


@dataclass
class AutoencoderEvaluator:
    """Class for probing dataset using trained autoencoder."""

    ae: Union[tf.keras.Model, BaseAutoencoder]
    dataset: tf.data.Dataset
    axis: Tuple[int, ...] = (1, 2, 3)

    def __post_init__(self) -> None:
        """Calculate and store errors, and threshold."""
        # get the reconstrution error
        self.errors: List[float] = list(self.gen_reconstruction_error())

        # store threshold
        self.threshold = self.calc_95th_threshold(self.errors)

    @staticmethod
    def has_file_paths(dataset: tf.data.Dataset) -> Optional[List[str]]:
        """See if tf.data.Dataset has file_paths attribute."""
        # see if tensorflow dataset has custom file_paths attr
        return dataset.file_paths if hasattr(dataset, "file_paths") else None

    @staticmethod
    def check_batch_type(
        batch: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]
    ) -> tf.Tensor:
        """Determine if batch is a tensor or a tuple pair of tensors."""
        # check if tuple
        if isinstance(batch, tuple):
            batch = batch[0]

        # get batch back
        return batch

    @staticmethod
    def calc_95th_threshold(
        errors: Union[Tuple[float, ...], List[float]]
    ) -> Union[float, Any]:
        """Calculate threshold for anomalous data using 95th percentile."""
        return np.percentile(errors, 95)

    @staticmethod
    def calc_max_threshold(
        errors: Union[Tuple[float, ...], List[float]]
    ) -> Union[float, Any]:
        """Calculate threshold for anomalous data using maximum value."""
        return np.max(errors)

    def gen_batch_predictions(
        self, axis: Optional[Tuple[int, ...]] = None
    ) -> Generator[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], None, None]:
        """Generate pairs of inputs, predictions, and their MSE."""
        # loop over batches of data
        for batch in tqdm(self.dataset, desc="calculating reconstruction error"):
            # extract actual inputs from inside batch container
            inputs = self.check_batch_type(batch)

            # generate predictions
            predictions = self.ae.predict(x=inputs, verbose=0)

            # measure MSE
            mse = tf.reduce_mean(tf.square(inputs - predictions), axis=self.axis)

            # update errors list
            yield (inputs, predictions, mse)

    def gen_reconstruction_error(self) -> Generator[Any, None, None]:
        """Create list of reconstruction errors for a tf.data.Dataset."""
        # loop over batches of data
        for _inputs, _predictions, mse in self.gen_batch_predictions():
            # update errors list
            yield from mse.numpy()

    def view_error_distribution(
        self, title: str = "Reconstruction Error Distribution", bins: int = 10**3
    ) -> None:
        """Plot the reconstruction error distribution."""
        plot_error_distribution(self.errors, self.threshold, bins, title)

    def view_anomalous_images(
        self,
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot anomalous image inputs with their reconstructed outputs."""
        # Use the trained autoencoder to predict and calculate reconstruction error
        for batch_idx, (inputs, predictions, mse) in enumerate(
            self.gen_batch_predictions()
        ):
            # now build, display, and optionally save images
            plot_anomalous_images(
                inputs, predictions, mse, self.threshold, batch_idx, output_path
            )
