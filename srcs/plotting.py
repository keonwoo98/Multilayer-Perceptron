"""Plot the learning curves recorded during training."""

import matplotlib
import matplotlib.pyplot as plt


def plot_history(history, save_path="learning_curves.png"):
    """Draw loss and accuracy curves (training vs validation) side by side."""
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["loss"], label="training loss")
    ax_loss.plot(epochs, history["val_loss"], label="validation loss")
    ax_loss.set_xlabel("epochs")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["acc"], label="training acc")
    ax_acc.plot(epochs, history["val_acc"], label="validation acc")
    ax_acc.set_xlabel("epochs")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"> learning curves saved to '{save_path}'")
    # Only open a window with an interactive backend (avoids a warning when
    # running head-less, e.g. on an evaluation server).
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
