# %%
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

from deepshape2.visualization.base import SMALL_SIZE, savefig, set_style

set_style()

__all__ = ["probability_distribution_metric", "binned_boxplot"]


colors = sns.color_palette("tab10", n_colors=3)
fill_alpha = 0.25  # transparency for both KDE and boxplots


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
    labels = []

    for metric, model_name, color in zip(metric_list, model_names, colors):
        # Compute statistics
        median_val = np.median(metric)
        lower_q = np.percentile(metric, shade_percentile[0])
        upper_q = np.percentile(metric, shade_percentile[1])
        sigma_low = median_val - lower_q
        sigma_high = upper_q - median_val

        # Legend label
        labels.append(
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
        ax.fill_between(x_vals, y_vals, alpha=fill_alpha, color=color)
        ax.plot(x_vals, y_vals, color=color, linewidth=1.5, alpha=0.8)

        # Draw median line
        if show_median:
            ax.axvline(median_val, color=color, linestyle="--", linewidth=1, alpha=0.8)

    # Labels and formatting
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Probability density")
    ax.set_ylim(bottom=0)

    handles = [
        mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.4, label=name)
        for color, name in zip(colors, labels)
    ]
    ax.legend(handles=handles, fontsize=SMALL_SIZE)

    # Apply final axis limits (optional)
    if clip_left is not None or clip_right is not None:
        ax.set_xlim(left=clip_left, right=clip_right)

    savefig(fname)


# %%
def binned_boxplot(
    stat_values,
    metric_list,
    model_names=["VAE-MHA", "VAE-CNN", "CAT"],
    n_bins=5,
    metric_name="PSNR",
    stat_name="Flux",
    fname=None,
    bottom_y_label="Density",
):
    """
    Plots binned boxplots of metrics per model and a KDE of the binning variable.

    Parameters
    ----------
    stat_values : np.ndarray
        1D array of the statistic to bin (length N_galaxies).
    metric_list : list of np.ndarray
        List of 1D arrays (length N_galaxies) of metrics for each model.
    model_names : list of str
        Names of the models.
    n_bins : int
        Number of bins for the statistic.
    metric_name : str
        Label for metric y-axis.
    stat_name : str
        Label for statistic x-axis.
    fname : str or None
        If provided, save figure to this file.
    """
    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(metric_list))]

    # Compute bin edges and centers
    bin_edges = np.linspace(stat_values.min(), stat_values.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(6, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Colors
    colors = sns.color_palette("tab10", n_colors=len(metric_list))

    # Small offset for each model within a bin
    total_width = 0.4 * (bin_edges[1] - bin_edges[0])
    width = total_width / len(metric_list)
    offsets = np.linspace(
        -total_width / 2 + width / 2, total_width / 2 - width / 2, len(metric_list)
    )

    for i, (metric, model_name, color, offset) in enumerate(
        zip(metric_list, model_names, colors, offsets)
    ):
        binned_metrics = []
        positions = []
        for j in range(n_bins):
            mask = (stat_values >= bin_edges[j]) & (stat_values < bin_edges[j + 1])
            binned_metrics.append(metric[mask])
            positions.append(bin_centers[j] + offset)

        # Boxplot without outliers, small width
        bp = ax_top.boxplot(
            binned_metrics,
            positions=positions,
            widths=width * 0.8,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=color, linewidth=2),
            boxprops=dict(facecolor=color, alpha=0.4),
        )

        for patch in bp["boxes"]:
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)

    ax_top.set_ylabel(metric_name)
    ax_top.grid(True, linestyle="--", alpha=0.5)

    # Add legend manually
    for color, model_name in zip(colors, model_names):
        ax_top.plot([], [], color=color, label=model_name, linewidth=6)
    ax_top.legend(loc="upper right")

    # Bottom: KDE of the statistic
    kde = gaussian_kde(stat_values)
    x_vals = np.linspace(stat_values.min(), stat_values.max(), 500)
    y_vals = kde(x_vals)
    y_vals /= y_vals.max()  # normalize for plotting
    ax_bottom.fill_between(x_vals, y_vals, alpha=0.3, color="gray")
    ax_bottom.plot(x_vals, y_vals, color="black", linewidth=1.5)
    ax_bottom.set_xlabel(stat_name)
    ax_bottom.set_ylabel(bottom_y_label)
    ax_bottom.grid(True, linestyle="--", alpha=0.5)
    ax_bottom.set_ylim(bottom=0)

    # Set x-ticks at bin centers
    ax_bottom.set_xticks(bin_centers)
    ax_bottom.set_xticklabels([f"{int(c)}" for c in bin_centers])

    savefig(fname)
