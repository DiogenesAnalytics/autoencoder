"""Tools for evaluating autoencoder performance."""
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm


def preprocess_images(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Simply normalize image values to interval [0, 1] and return as (x, x) pairs."""
    return (x / 255.0,) * 2


def evaluate_image_dataset(
    path: Union[str, Path],
    ae: tf.keras.Model,
    resize_dims: Optional[Tuple[int, int]] = None,
    color_mode: str = "rgb",
) -> pd.DataFrame:
    """Evaluate autoencoder model on an entire dataset."""
    # get entire dataset
    x_train = tf.keras.utils.image_dataset_from_directory(
        directory=str(path),
        labels=None,
        batch_size=1,
        image_size=resize_dims,
        color_mode=color_mode,
        shuffle=False,
    )

    # create dataset from image paths
    x_train_paths = tf.data.Dataset.from_tensor_slices(x_train.file_paths)

    # normalize, and create x/y pairs, and pair with image file paths
    x_train_processed = x_train.zip((x_train.map(preprocess_images), x_train_paths))

    # setup tqdm iterator
    x_train_prog = tqdm(x_train_processed, total=len(x_train_processed))

    # get DataFrame of paths/scores
    return pd.DataFrame.from_records(
        (
            {"path": path.numpy().decode(), "error": ae.evaluate(x, y, verbose=0)}
            for (x, y), path in x_train_prog
        ),
        index="path",
    )
