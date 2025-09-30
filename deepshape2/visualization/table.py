import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc_context
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .base import MEDIUM_SIZE, SMALL_SIZE, savefig, set_style

set_style()

__all__ = ["plot"]


def normalize_images_shape(images, subtitles=None):
    if isinstance(images, list):
        images = np.array(images)
        if subtitles is not None:
            subtitles = np.array(subtitles)

    if images.ndim == 2:
        images = images[np.newaxis, np.newaxis]
        if subtitles is not None:
            subtitles = subtitles[np.newaxis, np.newaxis]
    elif images.ndim == 3:
        images = images[np.newaxis]
        if subtitles is not None:
            subtitles = subtitles[np.newaxis]

    return images, subtitles


def get_vmin_vmax(images, same_scale, scale_row=None):
    cols = images.shape[1]
    vmin = np.ones(cols) * np.inf
    vmax = np.ones(cols) * -np.inf

    for col in range(cols):
        for row in same_scale:
            img = images[scale_row, col] if scale_row is not None else images[row, col]
            vmin[col] = min(vmin[col], np.min(img))
            vmax[col] = max(vmax[col], np.max(img))
    return vmin, vmax


def plot(images, **kwargs):
    titles = kwargs.get("titles", None)
    max_imgs = kwargs.get("max_imgs", 8)
    cmap = kwargs.get("cmap", "inferno")
    cbar = kwargs.get("cbar", False)
    caption = kwargs.get("caption", None)
    same_scale = kwargs.get("same_scale", 0)
    fname = kwargs.get("fname", None)
    text = kwargs.get("text", None)
    text_row = kwargs.get("text_row", None)
    scale_row = kwargs.get("scale_row", None)
    suptitle = kwargs.get("suptitle", None)
    swap = kwargs.get("swap", False)
    remove_bg = kwargs.get("remove_bg", False)
    subtitles = kwargs.get("subtitles", None)
    size_fac = kwargs.get("size_fac", 1)

    images, subtitles = normalize_images_shape(images, subtitles)

    if swap:
        images = np.swapaxes(images, 0, 1)
        if subtitles is not None:
            subtitles = np.swapaxes(subtitles, 0, 1)

    rows = min(max_imgs, images.shape[0])
    cols = images.shape[1]

    if isinstance(titles, str):
        titles = [titles]
    if titles and len(titles) != cols:
        raise ValueError("Title list should match number of columns")

    if not isinstance(same_scale, int):
        vmin, vmax = get_vmin_vmax(images, same_scale, scale_row)

    with rc_context(
        rc={
            "axes.labelpad": 1,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    ):
        figsize = (
            (2.5 * cols * size_fac, rows * 2.5 * size_fac)
            if cbar
            else (2 * cols * size_fac, rows * 2 * size_fac)
        )
        fig, axs = plt.subplots(rows, cols, squeeze=False, figsize=figsize)
        axs = np.atleast_2d(axs)

        for r in range(rows):
            for c in range(cols):
                ax = axs[r, c]
                if isinstance(same_scale, int) or r not in same_scale:
                    im = ax.imshow(images[r, c], cmap=cmap)
                else:
                    im = ax.imshow(images[r, c], cmap=cmap, vmin=vmin[c], vmax=vmax[c])

                if titles and r == 0:
                    ax.set_title(titles[c], size=MEDIUM_SIZE)
                if subtitles is not None:
                    ax.set_title(subtitles[r, c], size=MEDIUM_SIZE)

                if cbar:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
                    fmt.set_powerlimits((-3, 2))
                    colbar = fig.colorbar(
                        im, cax=cax, orientation="vertical", format=fmt
                    )
                    colbar.ax.tick_params(labelsize=SMALL_SIZE)
                    colbar.ax.yaxis.get_offset_text().set(size=SMALL_SIZE)

        if text is not None and text_row is not None:
            for ax, txt in zip(axs[text_row], text):
                ax.text(
                    0.65,
                    0.8,
                    txt,
                    size=MEDIUM_SIZE,
                    color="white",
                    transform=ax.transAxes,
                )

        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        if caption:
            for cap, ax in zip(caption, axs[:, 0]):
                ax.set_ylabel(cap, fontsize=MEDIUM_SIZE)

        # plt.subplots_adjust(
        #     hspace=0.02 if cbar else 0.01, wspace=0.25 if cbar else 0.05
        # )

        if suptitle:
            fig.suptitle(suptitle)

        if fname:
            savefig(fname, remove_bg=remove_bg)
        else:
            plt.show()
