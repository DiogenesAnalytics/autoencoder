"""For displaying/plotting/graphing and any other data visualization need."""
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.typing import NDArray


def training_loss_history(
    history: Dict[str, List[float]], model_name: str = "Autoencoder"
) -> None:
    """Plot training history."""
    # loop over history keras.callbacks.History dict
    for key, value in history.items():
        # add to plot
        plt.plot(value, label=key)

    plt.title(f"{model_name}: Training History")
    plt.ylabel("Model Error")
    plt.xlabel("No. Epochs")
    plt.legend(loc="upper left")
    plt.show()


def compare_image_predictions(
    original: NDArray[Any],
    predicted: NDArray[Any],
    display_num: int = 10,
) -> None:
    """Creates a MatplotLib figure with two rows: original images, and predicted."""
    # create figure
    plt.figure(figsize=(20, 4))

    # loop over 10 images
    for i in range(display_num):
        # Display original
        ax = plt.subplot(2, display_num, i + 1)
        plt.imshow(original[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, display_num, i + 1 + display_num)
        plt.imshow(predicted[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # final plt
    plt.show()


def plot_error_distribution(
    errors: List[List[float]],
    threshold: float,
    bins: int,
    title: str,
    density: bool = False,
    labels: Optional[List[str]] = None,
) -> None:
    """Plot a simple histogram for the reconstruction error."""
    # calculate alpha
    alpha = 0.5 if len(errors) > 1 else None

    # build histogram
    plt.hist(x=errors, alpha=alpha, bins=bins, density=density)

    # add title
    plt.title(title)

    # label axes
    plt.xlabel("Reconstruction Error")
    plt.ylabel("# Samples")

    # set legend
    if labels is not None:
        plt.legend(labels)

    # plotting threshold
    plt.axvline(x=threshold, color="r", linestyle="dashed", linewidth=2)

    # display histogram
    plt.show()


def plot_anomalous_images(
    inputs: tf.Tensor,
    predictions: tf.Tensor,
    mse: tf.Tensor,
    threshold: float,
    batch_id: int,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Build, display, and optionally save anomalous images."""
    # loop over indexes of anomalous tensors
    for anomaly_idx in tf.where(mse > threshold):
        # convert to scalar
        scalar_idx = anomaly_idx[0]

        # plot original
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title(f"Original Image (Error: {mse[scalar_idx]:.4f})")
        plt.imshow(
            tf.clip_by_value(tf.convert_to_tensor(inputs)[scalar_idx], 0.0, 1.0).numpy()
            * 255.0,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        # plot reconstructed
        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Image")
        plt.imshow(
            tf.clip_by_value(
                tf.convert_to_tensor(predictions)[scalar_idx], 0.0, 1.0
            ).numpy()
            * 255.0,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        # save if configured
        if output_path is not None:
            # create file name
            file_name = f"anomaly_{batch_id}_{scalar_idx}.png"

            # save to outputpath
            plt.savefig(Path(output_path) / file_name)

        # show
        plt.show()

    # close after plot
    plt.close()
