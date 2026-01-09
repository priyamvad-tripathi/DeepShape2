# %%
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

from deepshape2.visualization.base import SMALL_SIZE, savefig, set_style

set_style()

__all__ = ["probability_distribution_metric", "binned_boxplot", "metric_dependence"]


colors = sns.color_palette("tab10", n_colors=3)
fill_alpha = 0.25  # transparency for both KDE and boxplots


# %%
def probability_distribution_metric(
    metric_list,
    metric_name="PSNR [dB]",
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
        metric = np.asarray(metric)
        metric = metric[~np.isnan(metric)]

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
    metric_matrix,
    model_names=["VAE-MHA", "VAE-CNN", "CAT"],
    metric_names=(r"$\Delta \epsilon \,[\times 100]$", "PSNR [dB]"),
    stat_name=r"Flux $[\mu\mathrm{Jy}]$",
    bin_edges=None,
    fname=None,
    legend=False,
    logx=False,
):
    """
    Plots binned boxplots of metrics per model and a KDE of the binning variable.
    Works with metric_matrix shaped (n_metrics, n_models, N_galaxies).
    """

    n_metrics, n_models, n_galaxies = metric_matrix.shape

    # -----------------------------
    # Compute bin edges and centers
    # -----------------------------
    bin_edges = np.asarray(bin_edges)
    n_bins = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # -----------------------------
    # Figure and axes
    # -----------------------------l
    fig, axes = plt.subplots(
        n_metrics + 1,
        1,
        figsize=(6, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3] * n_metrics + [1], "hspace": 0},
    )

    ax_boxes = axes[:-1]
    ax_kde = axes[-1]

    # -----------------------------
    # Set axis scale
    # -----------------------------
    if logx:
        stat_min = max(1e-2, stat_values.min())
        stat_max = stat_values.max()
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlim(stat_min, stat_max)
    else:
        for ax in axes:
            ax.set_xscale("linear")
            ax.set_xlim(stat_values.min(), stat_values.max())

    fig.canvas.draw()

    # -----------------------------
    # Width and separation in physical units
    # -----------------------------
    width_cm = 0.35  # box width on paper
    sep_cm = 0.42  # separation between boxes
    fig_width_in = fig.get_size_inches()[0]
    bbox = ax_boxes[0].get_position()
    axis_width_in = fig_width_in * bbox.width
    width_frac = (width_cm / 2.54) / axis_width_in
    sep_frac = (sep_cm / 2.54) / axis_width_in

    x0, x1 = ax_boxes[0].get_xlim()

    if not logx:
        data_span = x1 - x0
        widths_per_bin = [width_frac * data_span] * n_bins
        sep_data = sep_frac * data_span
    else:
        trans = ax_boxes[0].transData
        inv = ax_boxes[0].transData.inverted()
        widths_per_bin = []
        for c in bin_centers:
            cx_disp = trans.transform((c, 0))[0]
            left_disp = cx_disp - (width_frac * axis_width_in * fig.dpi) / 2
            right_disp = cx_disp + (width_frac * axis_width_in * fig.dpi) / 2
            x_left = inv.transform((left_disp, 0))[0]
            x_right = inv.transform((right_disp, 0))[0]
            widths_per_bin.append(x_right - x_left)
        sep_data = 10 ** (sep_frac * (np.log10(x1) - np.log10(x0)))

    # -----------------------------
    # Draw boxplots for each metric
    # -----------------------------
    for ax, metric_idx, metric_name in zip(ax_boxes, range(n_metrics), metric_names):
        for model_idx, (model_name, color) in enumerate(zip(model_names, colors)):
            binned_metrics = []
            positions = []
            shift_index = model_idx - (n_models - 1) / 2  # symmetric shift
            for j, center in enumerate(bin_centers):
                mask = (stat_values >= bin_edges[j]) & (stat_values < bin_edges[j + 1])
                stat_in_bin = stat_values[mask]
                metric_in_bin = metric_matrix[metric_idx, model_idx, mask]

                # Remove NaNs in metric, keeping corresponding stat
                valid = ~np.isnan(metric_in_bin)
                stat_in_bin = stat_in_bin[valid]
                metric_in_bin = metric_in_bin[valid]

                binned_metrics.append(metric_in_bin)

                if not logx:
                    pos = center + shift_index * sep_data
                else:
                    pos = center * (sep_data**shift_index)
                positions.append(pos)

            ax.boxplot(
                binned_metrics,
                positions=positions,
                widths=widths_per_bin,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color=color, linewidth=2),
                boxprops=dict(facecolor=color, edgecolor=color, alpha=fill_alpha),
                whiskerprops=dict(color=color, alpha=0.8, linewidth=1.5),
                capprops=dict(color=color, alpha=0.8, linewidth=1.5),
            )

        ax.set_ylabel(metric_name)
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        if legend and metric_idx == 0:
            handles = [
                mpatches.Patch(
                    facecolor=color, edgecolor=color, label=name, alpha=fill_alpha
                )
                for color, name in zip(colors, model_names)
            ]
            ax.legend(handles=handles, frameon=False)

    # -----------------------------
    # Bottom: KDE of the statistic
    # -----------------------------
    kde = gaussian_kde(stat_values)
    x_vals = np.linspace(stat_values.min(), stat_values.max(), 500)
    y_vals = kde(x_vals)
    ax_kde.fill_between(x_vals, y_vals, alpha=0.3, color="gray")
    ax_kde.plot(x_vals, y_vals, color="black", linewidth=1, alpha=0.8)
    ax_kde.set_xlabel(stat_name)
    ax_kde.set_ylabel("Arbitrary")
    ax_kde.set_yscale("log")
    ax_kde.set_ylim(bottom=1e-4)
    ax_kde.minorticks_off()

    # -----------------------------
    # Tick styling for log axis
    # -----------------------------
    if logx:
        stat_max = np.max(stat_values)
        major_ticks = [1e-2, 1e-1, stat_max]
        ax_kde.set_xticks(major_ticks)
        ax_kde.set_xticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
        ax_kde.minorticks_off()
    else:
        ax_kde.set_xscale("linear")
        ax_kde.set_xlim(left=10, right=200)
        ax_kde.set_xticks(bin_edges)
        ax_kde.set_xticklabels([f"{int(e)}" for e in bin_edges])

    # Save figure
    savefig(fname)


# %%
def metric_dependence(
    metric_list,
    prop_list,
    bin_edges_list,
    metric_names=["PSNR [dB]", r"$\Delta \epsilon$"],
    prop_names=[r"Peak flux $[\mu\mathrm{Jy}]$", r"Size [arcsec]"],
    colors=("#DC143C", "#008080"),
    markers=("o", "s"),
    fname=None,
    metric_lims_list=None,
):
    """
    Plot metric dependence for two metrics using manually defined bins.

    Parameters
    ----------
    metric_list : list of array-like
        Two metric arrays
    prop_list : list of array-like
        Two input properties
    bin_edges_list : list of array-like
        Explicit bin edges for each property
    metric_names : list of str
        Names of the metrics for each row
    prop_names : list of str
        Names of the properties
    colors : tuple
        Colors for each metric row
    markers : tuple
        Markers for each metric row
    fname : str or None
        Output filename
    metric_lims_list : list of tuple or None
        Optional y limits per metric
    """

    def binned_quantiles_manual(x, y, bins):
        centers = 0.5 * (bins[:-1] + bins[1:])
        q25 = np.full(len(centers), np.nan)
        q50 = np.full(len(centers), np.nan)
        q75 = np.full(len(centers), np.nan)

        for i in range(len(centers)):
            if i < len(centers) - 1:
                mask = (x >= bins[i]) & (x < bins[i + 1])
            else:
                mask = (x >= bins[i]) & (x <= bins[i + 1])

            if np.any(mask):
                q25[i], q50[i], q75[i] = np.percentile(y[mask], [25, 50, 75])

        return centers, q25, q50, q75

    n_metrics = len(metric_list)

    if metric_lims_list is None:
        metric_lims_list = [(None, None)] * n_metrics

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        n_metrics, 3, width_ratios=[3, 3, 1.5], wspace=0.03, hspace=0.05
    )

    axes = []

    for i, metric in enumerate(metric_list):
        color = colors[i]
        marker = markers[i]

        stats = [
            binned_quantiles_manual(prop, metric, bins)
            for prop, bins in zip(prop_list, bin_edges_list)
        ]

        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1], sharey=ax1)
        ax3 = fig.add_subplot(gs[i, 2], sharey=ax1)

        # Share x axes vertically
        if i > 0:
            ax1.sharex(axes[0][0])
            ax2.sharex(axes[0][1])

        # First property
        c, q25, q50, q75 = stats[0]
        ax1.plot(c, q50, color=color, marker=marker)
        ax1.fill_between(c, q25, q75, color=color, alpha=0.3)
        ax1.set_ylabel(metric_names[i])
        ax1.set_ylim(*metric_lims_list[i])

        # Second property
        c, q25, q50, q75 = stats[1]
        ax2.plot(c, q50, color=color, marker=marker)
        ax2.fill_between(c, q25, q75, color=color, alpha=0.3)
        plt.setp(ax2.get_yticklabels(), visible=False)

        # KDE panel
        kde = gaussian_kde(metric)
        y_grid = np.linspace(metric.min(), metric.max(), 300)
        kde_vals = kde(y_grid)

        metric_med = np.median(metric)
        q25, q75 = np.percentile(metric, [25, 75])

        (line_kde,) = ax3.plot(kde_vals, y_grid, color=color)
        ax3.axhline(metric_med, color=color, linestyle="--")
        ax3.set_xlim(left=0)
        plt.setp(ax3.get_yticklabels(), visible=False)

        # Add median/IQR label at the top of KDE
        label_text = f"{metric_names[i]}=${metric_med:.2f} ^{{+{q75 - metric_med:.2f}}}_{{-{metric_med - q25:.2f}}}$"
        ax3.text(
            0.98,
            0.95,
            label_text,
            ha="right",
            va="top",
            transform=ax3.transAxes,
            fontsize=SMALL_SIZE,
        )

        # X labels only on bottom row
        if i == n_metrics - 1:
            ax1.set_xlabel(prop_names[0])
            ax2.set_xlabel(prop_names[1])
            ax3.set_xlabel("Probability")

        axes.append((ax1, ax2, ax3))

    # Save figure
    savefig(fname)
