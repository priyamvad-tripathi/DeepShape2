# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["set_style", "savefig", "plot_losses", "plot_uvcoverage"]

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
            pass
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
    labels=None,
    skip=0,
    logscale=False,
    fname=False,
    remove_bg=False,
    xlabel="Epoch",
    ylabel="Loss",
):
    """Plot one or several loss curves and mark the minimum of each.

    Parameters
    ----------
    loss_lists : sequence of floats, or sequence of sequences of floats
        Either a single curve (``[0.9, 0.7, ...]`` or a 1-D array) or several
        curves (``[train, val]``, a 2-D array, ...).
    labels : str, sequence of str, or None
        A single label, one label per curve, or ``None`` to use sensible
        defaults ("Loss" for one curve, "Train Loss"/"Val Loss" for two).
    skip : int
        Number of leading epochs to leave out of the plot.
    """
    # --- normalise inputs ------------------------------------------------
    if len(loss_lists) == 0:
        raise ValueError("No loss curves given.")

    # A scalar first element means we were handed a single curve.
    if np.ndim(loss_lists[0]) == 0:
        loss_lists = [loss_lists]
    loss_lists = [np.asarray(loss).reshape(-1) for loss in loss_lists]

    if labels is None:
        defaults = {1: ["Loss"], 2: ["Train Loss", "Val Loss"]}
        labels = defaults.get(
            len(loss_lists), [f"Loss {i + 1}" for i in range(len(loss_lists))]
        )
    elif isinstance(labels, str):
        labels = [labels]
    else:
        labels = list(labels)

    assert len(loss_lists) == len(labels), (
        "Number of loss curves must match number of labels"
    )

    lengths = [len(loss) for loss in loss_lists]
    assert all(length == lengths[0] for length in lengths), (
        "All loss curves must have the same length."
    )

    # --- plot -------------------------------------------------------------
    fig, ax = plt.subplots()
    epochs = lengths[0]
    x = np.arange(skip + 1, epochs + 1)

    for loss, label in zip(loss_lists, labels):
        y = loss[skip:]
        (line,) = ax.plot(x, y, label=label)

        # Find minimum and mark it
        min_idx = np.argmin(y)
        ax.plot(x[min_idx], y[min_idx], "*", markersize=12, color=line.get_color())

    if logscale:
        ax.set_yscale("log")

    ax.set_xlim([0, epochs + 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    if fname:
        savefig(fname, remove_bg=remove_bg)
    else:
        plt.show()


def plot_uvcoverage(vis_list, ax=None, plot_file=None, title="UV coverage", **kwargs):
    """Standard plot of uv coverage

    :param vis_list:
    :param plot_file:
    :param kwargs:
    :return:
    """

    for ivis, ovis in enumerate(vis_list):
        gvis = ovis.where(ovis["flags"] == 0)
        bvis = ovis.where(ovis["flags"] > 0)
        u = np.array(gvis.uvw_lambda.sel(spatial="u").data.flat)
        v = np.array(gvis.uvw_lambda.sel(spatial="v").data.flat)
        if ivis == 0:
            plt.plot(u, v, ".", color="b", markersize=0.2, label="Unflagged")
        else:
            plt.plot(
                u,
                v,
                ".",
                color="b",
                markersize=0.2,
            )

        plt.plot(-u, -v, ".", color="b", markersize=0.2)
        u = np.array(bvis.uvw_lambda.sel(spatial="u").data.flat)
        v = np.array(bvis.uvw_lambda.sel(spatial="v").data.flat)
        if ivis == 0:
            plt.plot(u, v, ".", color="r", markersize=0.2, label="Flagged")
        else:
            plt.plot(u, v, ".", color="r", markersize=0.2)
        plt.plot(-u, -v, ".", color="r", markersize=0.2)
    plt.xlabel("U (wavelengths)")
    plt.ylabel("V (wavelengths)")
    plt.legend()
    plt.title(title)
    if plot_file is not None:
        plt.savefig(plot_file)
    plt.show(block=False)
