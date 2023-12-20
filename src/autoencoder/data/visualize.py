"""For displaying/plotting/graphing and any other data visualization need."""
from typing import Any
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
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
