# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["set_style", "savefig", "plot_losses"]

# Font Sizes to use
SMALLER = 9
SMALL_SIZE = 12
MEDIUM_SIZE = 14
MEDIUM_SIZE_a = 16
BIGGER_SIZE = 18


# %%


def set_style():
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("text.latex", preamble=r"\usepackage{txfonts}")

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": ["Computer Modern"],
            "legend.frameon": False,
            "legend.handlelength": 2,
            # "xtick.top": True,
            # "ytick.right": True,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "figure.autolayout": False,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.05,
            "figure.constrained_layout.w_pad": 0.05,
            "figure.constrained_layout.hspace": 0,
            "figure.constrained_layout.wspace": 0.15,
            "axes.labelpad": 1,
            # "xtick.direction": "in",
            # "ytick.direction": "in",
            "xtick.major.pad": 3,
            "ytick.major.pad": 3,
        }
    )


# Function to save images
def savefig(filename=None, dpi=600, remove_bg=False):
    if filename:
        parent_dir = Path(filename).parent

        try:
            parent_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            1
        else:
            print(f"New folder created {parent_dir}")

        plt.savefig(
            fname=filename,
            bbox_inches="tight",
            dpi=dpi,
            transparent=remove_bg,
        )
    else:
        plt.show()


# Function to plot the loss curve for validation and training sets
def plot_losses(
    loss_lists,
    labels=["Train Loss", "Val Loss"],
    skip=5,
    logscale=False,
    fname=False,
    remove_bg=False,
):
    fig, ax = plt.subplots()
    epochs = len(loss_lists[0])

    assert len(loss_lists) == len(labels), (
        "Number of loss curves must match number of labels"
    )

    lengths = [len(loss) for loss in loss_lists]
    assert all(length == lengths[0] for length in lengths), (
        "All loss curves must have the same length."
    )

    x = np.arange(skip + 1, epochs + 1)

    for i, loss in enumerate(loss_lists):
        y = np.array(loss[skip:]).reshape(-1)
        (line,) = ax.plot(x, y, label=labels[i])

        # Find minimum and mark it
        full_loss = np.array(loss)
        min_idx = np.argmin(full_loss)
        min_val = full_loss[min_idx]
        ax.plot(min_idx + 1, min_val, "*", markersize=12, color=line.get_color())

    if logscale:
        ax.set_yscale("log")

    ax.set_xlim([0, epochs + 1])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    if fname:
        savefig(fname, remove_bg=remove_bg)
    else:
        plt.show()
