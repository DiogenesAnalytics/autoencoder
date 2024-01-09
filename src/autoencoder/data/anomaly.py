"""Tools for evaluating an autoencoder's perfomance on a dataset."""
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
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
class AnomalyDetector:
    """Class for generating the reconstruction error for a dataset."""

    ae: Union[tf.keras.Model, BaseAutoencoder]
    dataset: tf.data.Dataset
    axis: Tuple[int, ...] = (1, 2, 3)

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

        # toggle file paths exist attr
        self._file_paths_exist = True if file_paths is not None else False

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

    def _check_data_attrs_set(self) -> None:
        """Make sure errors and threshold attributes have been set."""
        assert all(
            [hasattr(self, "errors"), hasattr(self, "threshold")]
        ), "errors/threshold attributes must be set before running this method."

    def _plot_error_distribution(
        self,
        title: str,
        label: str,
        density: bool,
        additional_data: Optional[List["AnomalyDetector"]],
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
        additional_data: Optional[List["AnomalyDetector"]] = None,
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
        additional_data: Optional[List["AnomalyDetector"]] = None,
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
        output_path.mkdir(parents=True)

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

    @staticmethod
    def _build_mask_filter(
        less_than: Optional[float] = None,
        greater_than: Optional[float] = None,
        less_equal: bool = False,
        greater_equal: bool = False,
    ) -> Callable[[float], bool]:
        """Builds the mask required for filtering reconstruciton error DataFrame."""
        # make sure at least one is set
        assert any([less_than, greater_than]), "Must pass at least one condition."

        # make sure both are not equal
        assert not all(
            [less_equal, greater_equal]
        ), "Must pass only one equals condition."

        # build wrapped function
        def condition(x: float) -> bool:
            """For filtering pandas Dataframe."""
            return (
                less_than is None or (x < less_than or (less_equal and x <= less_than))
            ) and (
                greater_than is None
                or (x > greater_than or (greater_equal and x >= greater_than))
            )

        # return wrapped function
        return condition

    def _check_file_paths_available(self) -> None:
        """Make sure that file_path were not empty or missing from the dataset."""
        assert (
            self._file_paths_exist
        ), "Cannot run method due to missing file_paths data in calculate_error method."

    def _filter_file_paths(
        self,
        less_than: Optional[float] = None,
        greater_than: Optional[float] = None,
        less_equal: bool = False,
        greater_equal: bool = False,
    ) -> Generator[Tuple[str, float], None, None]:
        """Scan through errors DataFrame and find matches."""
        # build mask
        condition = self._build_mask_filter(
            less_than, greater_than, less_equal, greater_equal
        )

        # Build mask for matching conditions
        mask = self.errors["reconstruction_error"].apply(condition)

        # iterate file_path, error paris
        for index, row in self.errors[mask].iterrows():
            yield str(index), row["reconstruction_error"]

    @staticmethod
    def _create_labels_dict(invert: bool = False) -> Dict[str, str]:
        """Simply create a dictionary of labels based on desired state."""
        return {
            "below": "pass" if not invert else "fail",
            "above": "fail" if not invert else "pass",
        }

    def _select_subset_error_data(
        self,
        threshold: Optional[float],
        below: bool,
        above: bool,
        include_threshold_below: bool,
        include_threshold_above: bool,
        invert_labels: bool,
        process_func: Callable[[str, float, str, int], None],
    ) -> None:
        """Get subset of reconstruction error data."""
        # make sure attrs set
        self._check_data_attrs_set()

        # make sure only one threshold option is included
        assert not all(
            [include_threshold_above, include_threshold_below]
        ), "Must pass only one threshold inclusion option."

        # set threshold
        threshold = threshold if threshold is not None else self.threshold

        # get labels
        labels = self._create_labels_dict(invert=invert_labels)

        # check if below threshold should be listed
        if below:
            # loop and print
            for i, (file_path, error) in enumerate(
                self._filter_file_paths(
                    less_than=threshold, less_equal=include_threshold_below
                )
            ):
                # process data
                process_func(file_path, error, labels["below"], i)

        # check for above
        if above:
            # loop and print
            for j, (file_path, error) in enumerate(
                self._filter_file_paths(
                    greater_than=threshold, greater_equal=include_threshold_above
                )
            ):
                # process data
                process_func(file_path, error, labels["above"], j)

    @staticmethod
    def print_out_results(path: str, error: float, label: str, idx: int) -> None:
        """Simply prints out selected error data."""
        print(f"{label} {error:0.4f} {path}")

    def list(
        self,
        threshold: Optional[float] = None,
        below: bool = True,
        above: bool = False,
        include_threshold_below: bool = False,
        include_threshold_above: bool = True,
        invert_labels: bool = False,
    ) -> None:
        """List out all file paths that match threshold conditions."""
        # check file_paths exist
        self._check_file_paths_available()

        # print out selected data
        self._select_subset_error_data(
            threshold=threshold,
            below=below,
            above=above,
            include_threshold_below=include_threshold_below,
            include_threshold_above=include_threshold_above,
            invert_labels=invert_labels,
            process_func=self.print_out_results,
        )

    @staticmethod
    def _build_copy_func(
        destination_path: Union[str, Path]
    ) -> Callable[[str, float, str, int], None]:
        """Wrapper that builds the copy func for specified destination."""
        # get path obj
        destination_path = Path(destination_path)

        # build function
        def copy_path_func(path: str, error: float, label: str, idx: int) -> None:
            """Copies file from source to destination and names it using idx value."""
            # get path obj
            source_path = Path(path)

            # new name
            new_name = f"{idx}_{error}_{source_path.name}{source_path.suffix}"

            # copy
            shutil.copy(path, destination_path / f"{label}/{new_name}")

        # get function
        return copy_path_func

    def extract(
        self,
        destination_path: Union[str, Path],
        threshold: Optional[float] = None,
        below: bool = True,
        above: bool = False,
        include_threshold_below: bool = False,
        include_threshold_above: bool = True,
        invert_labels: bool = False,
    ) -> None:
        """Extract files that match threshold conditions to destination path."""
        # make sure attrs set
        self._check_data_attrs_set()

        # check file_paths exist
        self._check_file_paths_available()

        # get path obj
        destination_path = Path(destination_path)

        # make sure directory is not taken
        assert (
            not destination_path.exists()
        ), "Cannot use an existing directory for destination of copy() method."

        # create label dirs
        labels = self._create_labels_dict(invert=invert_labels)

        # loop over threshold state
        for state, used in zip(["below", "above"], [below, above], strict=True):
            # check label is used
            if used:
                # create subdir from label assigned to threshold state
                subdir = destination_path / labels[state]

                # mkdir including parents
                subdir.mkdir(parents=True)

        # build the copy function
        copy_path_func = self._build_copy_func(destination_path)

        # print out selected data
        self._select_subset_error_data(
            threshold=threshold,
            below=below,
            above=above,
            include_threshold_below=include_threshold_below,
            include_threshold_above=include_threshold_above,
            invert_labels=invert_labels,
            process_func=copy_path_func,
        )
