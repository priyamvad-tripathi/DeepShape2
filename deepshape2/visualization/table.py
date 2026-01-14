import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc_context
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .base import MEDIUM_SIZE, SMALL_SIZE, savefig, set_style

set_style()

__all__ = ["plot", "plot_log_image"]


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
    """Compute vmin/vmax per column from the provided same_scale rows."""
    cols = images.shape[1]
    vmin = np.ones(cols) * np.inf
    vmax = np.ones(cols) * -np.inf

    for col in range(cols):
        for row in same_scale:
            img = images[scale_row, col] if scale_row is not None else images[row, col]
            vmin[col] = min(vmin[col], np.nanmin(img))
            vmax[col] = max(vmax[col], np.nanmax(img))
    return vmin, vmax


def plot(
    images,
    titles=None,
    max_imgs=8,
    cmap="inferno",
    cbar=False,
    caption=None,
    same_scale=0,
    fname=None,
    text=None,
    text_row=None,
    scale_row=None,
    suptitle=None,
    swap=False,
    remove_bg=False,
    subtitles=None,
    size_fac=1,
    scale_ranges=None,  # Optional: list of (vmin,vmax) per column
    return_scales=False,
):
    """
    Plot images in a grid with optional scaling per column for specific rows only.

    Parameters
    ----------
    same_scale : int or list
        Rows over which to apply same scaling.
    scale_ranges : list of (vmin, vmax) or None
        Optional predefined scale per column.

    Returns
    -------
    used_scales : list
        List of (vmin, vmax) used for each column. None if no scaling applied.
    """
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

    # Compute vmin/vmax per column
    used_scales = [None] * cols
    if scale_ranges is not None:
        used_scales = scale_ranges
    elif not isinstance(same_scale, int):
        vmin, vmax = get_vmin_vmax(images, same_scale, scale_row)
        used_scales = [(vmin[c], vmax[c]) for c in range(cols)]

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

                # Apply scaling only to rows in same_scale
                scale = None
                if isinstance(same_scale, int) or r not in same_scale:
                    scale = None
                else:
                    scale = used_scales[c]

                if scale is not None:
                    im = ax.imshow(
                        images[r, c], cmap=cmap, vmin=scale[0], vmax=scale[1]
                    )
                else:
                    im = ax.imshow(images[r, c], cmap=cmap)

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

        if suptitle:
            fig.suptitle(suptitle)

        savefig(fname, remove_bg=remove_bg)
    if return_scales:
        return used_scales


# %%
def plot_log_image(im, figsize=(8, 8), cmap="inferno", fname=None, remove_bg=False):
    im = np.clip(im, 0, None)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Display the image in log scale but preserve actual flux scaling
    im_disp = ax.imshow(
        im + 1e-9,
        norm=LogNorm(vmin=max(im.min(), 1e-9), vmax=im.max()),
        cmap=cmap,
        origin="lower",
    )

    # Remove all axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # --- Custom colorbar on the right ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 2))

    colbar = fig.colorbar(im_disp, cax=cax, orientation="vertical", format=fmt)
    colbar.set_label("Flux [Jy]", fontsize=MEDIUM_SIZE, labelpad=10)
    colbar.ax.tick_params(labelsize=SMALL_SIZE)
    colbar.ax.yaxis.get_offset_text().set(size=SMALL_SIZE)

    if fname:
        savefig(fname, remove_bg=remove_bg)
    else:
        plt.show()
