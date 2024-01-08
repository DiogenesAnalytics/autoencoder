"""Tools for evaluating an autoencoder's perfomance on a dataset."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

from ..model.base import BaseAutoencoder
from .visualize import plot_error_distribution


@dataclass
class ReconstructionError:
    """Class for generating the reconstruction error for a dataset."""

    ae: Union[tf.keras.Model, BaseAutoencoder]
    dataset: tf.data.Dataset
    axis: Tuple[int, ...] = (1, 2, 3)

    def _check_data_attrs_set(self) -> None:
        """Make sure errors and threshold attributes have been set."""
        assert all(
            [hasattr(self, "errors"), hasattr(self, "threshold")]
        ), "errors/threshold attributes should be set before running this method."

    @staticmethod
    def get_file_paths(dataset: tf.data.Dataset) -> Optional[List[str]]:
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

    def calculate_error(self, file_paths: Optional[List[str]] = None) -> None:
        """Calculate and store errors, and threshold."""
        # check file paths
        if file_paths is None:
            # get file paths from dataset
            file_paths = self.get_file_paths(self.dataset)

        # get the reconstrution error
        self.errors = pd.DataFrame(
            data=self.gen_reconstruction_error(),
            columns=["reconstruction_error"],
            index=file_paths,
        )

        # store 95th threshold
        self.threshold = self.calc_95th_threshold(
            self.errors["reconstruction_error"].values.tolist()
        )

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

    def _plot_error_distribution(
        self,
        title: str,
        label: str,
        density: bool,
        additional_data: Optional[List["ReconstructionError"]],
        additional_labels: Optional[List[str]],
        alphas: Optional[List[float]],
        bins: Optional[List[int]],
    ) -> None:
        """Handles all histogram plots."""
        # make sure attrs set
        self._check_data_attrs_set()

        # setup list of data and labels
        error_data = [self.errors["reconstruction_error"].values.tolist()]

        # check for more data
        if additional_data is not None:
            # add in other data supplied
            error_data += [
                ds.errors["reconstruction_error"].tolist() for ds in additional_data
            ]

        # chek for more labels
        if additional_labels is not None:
            # get error labels
            error_labels = [label] + additional_labels

        else:
            # otherwise don't use any
            error_labels = None

        # check for alphas
        if alphas is None:
            # determine alpha value
            alph_val = 0.5 if len(error_data) > 1 else 1

            # build alphas list
            alphas = [alph_val] * len(error_data)

        # check for bins
        if bins is None:
            # build default bins list
            bins = [10**3] * len(error_data)

        # now plot
        plot_error_distribution(
            errors=error_data,
            threshold=self.threshold,
            title=title,
            bins=bins,
            alphas=alphas,
            labels=error_labels,
            density=density,
        )

    def histogram(
        self,
        title: str = "Reconstruction Error Histogram",
        label: str = "threshold_source",
        additional_data: Optional[List["ReconstructionError"]] = None,
        additional_labels: Optional[List[str]] = None,
        alphas: Optional[List[float]] = None,
        bins: Optional[List[int]] = None,
    ) -> None:
        """Plot the reconstruction error as a histogram."""
        self._plot_error_distribution(
            title=title,
            bins=bins,
            label=label,
            additional_data=additional_data,
            additional_labels=additional_labels,
            density=False,
            alphas=alphas,
        )

    def probability_distribution(
        self,
        title: str = "Reconstruction Error Probability Distribution",
        label: str = "threshold_source",
        additional_data: Optional[List["ReconstructionError"]] = None,
        additional_labels: Optional[List[str]] = None,
        alphas: Optional[List[float]] = None,
        bins: Optional[List[int]] = None,
    ) -> None:
        """Plot the reconstruction error as a probability distribution."""
        self._plot_error_distribution(
            title=title,
            bins=bins,
            label=label,
            additional_data=additional_data,
            additional_labels=additional_labels,
            density=True,
            alphas=alphas,
        )

    def save(self, output_path: Union[str, Path]) -> None:
        """Save object instance data to path."""
        # creat path obj
        output_path = Path(output_path)

        # make sure path doesn't exist
        assert (
            not output_path.exists()
        ), "Save method expects a new directory not an existing one."

        # now create it
        output_path.mkdir()

        # save errors dataframe
        self.errors.to_csv(output_path / "errors.csv", columns=["reconstruction_error"])

        # open new JSON file
        with open(output_path / "threshold.json", "w") as outfile:
            # save threshold with pretty print
            json.dump({"threshold": self.threshold}, outfile, indent=4)

    def load(self, input_path: Union[str, Path]) -> None:
        """Load previously saved object instance data."""
        # create path obj
        input_path = Path(input_path)

        # make sure path doesn't exist
        assert input_path.exists(), "Load method cannot find directory path."

        # open threshold file
        with open(input_path / "threshold.json") as infile:
            # update threshold
            self.threshold = json.load(infile)["threshold"]

        # now get errors attribute
        self.errors = pd.read_csv(input_path / "errors.csv")
