import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

from .base import SMALL_SIZE, savefig, set_style

set_style()

__all__ = ["probability_distribution_metric"]


# %%
def probability_distribution_metric(
    metric_list,
    metric_name="PSNR",
    model_names=["VAE-MHA", "VAE-CNN", "CAT"],
    fname=None,
    show_median=True,
    shade_percentile=(25, 75),
    clip_left=None,  # custom x-axis left bound for plotting (not for KDE)
    clip_right=None,  # custom x-axis right bound for plotting (not for KDE)
):
    """
    Plot KDE-based probability distributions for a metric from multiple models.
    KDE is computed over the full data range but plotted only within user-specified
    x-axis limits (clip_left, clip_right). Median ± IQR are shown in the legend.
    """
    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = sns.color_palette("tab10", n_colors=len(metric_list))

    for metric, model_name, color in zip(metric_list, model_names, colors):
        # Compute statistics
        median_val = np.median(metric)
        lower_q = np.percentile(metric, shade_percentile[0])
        upper_q = np.percentile(metric, shade_percentile[1])
        sigma_low = median_val - lower_q
        sigma_high = upper_q - median_val

        # Legend label
        label_text = (
            rf"$\mathrm{{{model_name}}}:~{median_val:.2f}"
            rf"^{{+{sigma_high:.2f}}}_{{-{sigma_low:.2f}}}$"
        )

        # Compute KDE on full metric range
        kde = gaussian_kde(metric)
        x_vals = np.linspace(metric.min(), metric.max(), 400)
        y_vals = kde(x_vals)
        y_vals = np.clip(y_vals, 0, None)

        # Normalize so area under curve ≈ 1
        area = np.trapz(y_vals, x_vals)
        if area > 0:
            y_vals /= area

        # Apply plotting limits only visually
        if clip_left is not None or clip_right is not None:
            mask = np.ones_like(x_vals, dtype=bool)
            if clip_left is not None:
                mask &= x_vals >= clip_left
            if clip_right is not None:
                mask &= x_vals <= clip_right
            x_vals, y_vals = x_vals[mask], y_vals[mask]

        # Plot filled KDE and curve
        ax.fill_between(x_vals, y_vals, alpha=0.2, color=color)
        ax.plot(x_vals, y_vals, color=color, linewidth=2, label=label_text)

        # Draw median line
        if show_median:
            ax.axvline(median_val, color=color, linestyle="--", linewidth=1.5)

    # Labels and formatting
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Probability density")
    ax.legend(fontsize=SMALL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)

    # Apply final axis limits (optional)
    if clip_left is not None or clip_right is not None:
        ax.set_xlim(left=clip_left, right=clip_right)

    savefig(fname)
