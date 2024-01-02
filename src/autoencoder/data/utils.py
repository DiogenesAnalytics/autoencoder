"""Various tools for handling data."""
from pathlib import Path
from typing import Any
from typing import Tuple
from typing import Union

import tensorflow as tf
from keras.utils import image_dataset_from_directory


def preprocess_image_tensors(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Simply normalize image values to interval [0, 1] and return as (x, x) pair."""
    return (x / 255.0,) * 2


def prepare_autoencoder_image_dataset(
    data_dir: Union[str, Path],
    **kwargs: Any,
) -> Union[tf.data.Dataset, Tuple[tf.data.Dataset, tf.data.Dataset]]:
    """Load image dataset specifically for use in autoencoder training/testing."""
    # set labels to none
    kwargs["labels"] = None

    # get dataset
    dataset = image_dataset_from_directory(directory=str(data_dir), **kwargs)

    # check if train/val split is being used
    if "subset" in kwargs and kwargs["subset"] == "both":
        # split into train/val
        train, val = dataset

        # preprocess
        processed: Union[tf.data.Dataset, Tuple[tf.data.Dataset, tf.data.Dataset]] = (
            train.map(preprocess_image_tensors),
            val.map(preprocess_image_tensors),
        )

        # update file paths
        processed[0].file_paths = train.file_paths
        processed[1].file_paths = val.file_paths

    else:
        # preprocess
        processed = dataset.map(preprocess_image_tensors)

        # update file paths
        processed.file_paths = dataset.file_paths

    # return final dataset(s), preprocessed, and with file_paths attribute
    return processed


def train_val_split_image_dataset(
    data_dir: Union[str, Path],
    validation_split: float = 0.2,
    **kwargs: Any,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """By default returns a train/val split image dataset for autoencoders."""
    # update subset in kwargs
    kwargs["subset"] = "both"

    # call autoencoder image dataset builder
    return prepare_autoencoder_image_dataset(
        str(data_dir), validation_split=validation_split, **kwargs
    )
