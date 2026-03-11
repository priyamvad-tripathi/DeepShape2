# %%
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import container, gridspec
from scipy import stats
from scipy.stats import linregress as lin

from deepshape2.visualization.base import savefig, set_style

set_style()

__all__ = ["plot_bias", "contour_plot", "plot_residual_slope"]


# %%
def plot_bias(
    ypred,
    ytest,
    colors=("blue", "orange"),
    power=1e4,
    bad_index=None,
    bias_line=True,
    ellipticity_cutoff=0.7,
    lim=0.7,
):
    """
    Scatter plot of ellipticity residuals with linear bias lines
    and error estimates on slope and intercept.
    """
    ypred = np.array(ypred)
    ytest = np.array(ytest)
    assert ypred.shape == ytest.shape, "ypred and ytest must have same shape"

    # Remove bad indices
    if bad_index is not None:
        mask = ~np.array(bad_index, dtype=bool)
        ypred = ypred[mask]
        ytest = ytest[mask]

    delta = ypred - ytest
    exp = int(np.log10(power))

    fig, ax = plt.subplots(figsize=(6, 4))

    for i in range(2):
        x = ytest[:, i]
        y = delta[:, i]

        # Apply ellipticity cutoff
        msk = np.abs(x) < ellipticity_cutoff
        x = x[msk]
        y = y[msk]

        # Linear regression
        res = lin(x, y)
        m = res.slope
        c = res.intercept
        m_err = res.stderr
        c_err = res.intercept_stderr

        # Print with errors
        print(
            f"Component e{i + 1}: m = {m:.3e} ± {m_err:.3e}, c = {c:.3e} ± {c_err:.3e}"
        )

        # Legend label (old-style formatting)
        cap = r"$\hat{m}_1,\hat{c}_1$" if i == 0 else r"$\hat{m}_2,\hat{c}_2$"
        label = (
            "{"
            + cap
            + rf"$ \times 10^{exp}$"
            + "= {"
            + rf"{m * power:.1f}$\pm${m_err * power:.1f}"
            + rf", {c * power:.1f}$\pm${c_err * power:.1f}"
            + "}"
        )

        # Scatter
        ax.scatter(
            x,
            y,
            color=colors[i],
            s=5,
            alpha=0.5,
            label=label,
        )

        # Bias line
        if bias_line:
            ax.axline(
                xy1=(0, c),
                slope=m,
                color=colors[i],
                linestyle="--",
                linewidth=1,
            )

    ax.set(
        xlabel=r"$\epsilon^T$",
        ylabel=r"$\hat{\epsilon}-\epsilon^T$",
        xlim=(-1, 1),
        ylim=(-lim, lim),
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    lgnd = ax.legend(fontsize=9)
    lgnd.legend_handles[0]._sizes = [30]
    lgnd.legend_handles[1]._sizes = [30]

    fig.tight_layout()
    plt.show()


# %%
def contour_plot(
    ytest,
    ypred_list,
    legends,
    **kwargs,
):
    """
    Generates a contour plot comparing true values and predicted values from two shape measurement methods. Also plots the 1D residual distribution.
    # Fig 7 in DeepShape Paper

    Parameters:
    ------------
    ytest (array-like): True values of the ellipticity. Dim = [N_obs].
    ypred_list (list of array-like): A list of ellipticity measurements using two methods. Dim=[2, N_obs].
    legends (list of str): A list of legend labels for the two methods.

    **kwargs : dict, optional
        Additional keyword arguments:
        - lim (float): The limit for the y-axis. Default is 0.3.
        - cmaps (list of str): List of colormap names for the contour plots. Default is ["Reds", "Blues"].
        - colors (list of str): List of colors for the slope lines and KDE plots. Default is ["firebrick", "blue"].
        - fname (str or None): Filename to save the plot. If None, the plot is shown. Default is None.
    """

    lim = kwargs.get("lim", 0.3)
    cmaps = kwargs.get("cmaps", ["Reds", "Blues"])
    colors = kwargs.get("colors", ["firebrick", "blue"])
    fname = kwargs.get("fname", None)
    remove_bg = kwargs.get("remove_bg", False)

    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

    fig = plt.figure(
        figsize=(6, 4.2),
    )  # Set background canvas colour to White instead of grey default
    fig.patch.set_facecolor("white")

    ax = plt.subplot(gs[0, 0])

    ax.set_xlim(-1, 1)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\epsilon_1^{\mathrm{true}}$")
    ax.set_ylabel(r"$\hat{\epsilon}_1-\epsilon^{\mathrm{true}}_1$")
    ax.axhline(0, color="black", linestyle=":", linewidth=1.2)

    axr = plt.subplot(gs[0, 1], sharey=ax)
    axr.get_xaxis().set_visible(False)
    axr.get_yaxis().set_visible(False)
    axr.spines["right"].set_visible(False)
    axr.spines["top"].set_visible(False)
    axr.spines["bottom"].set_visible(False)
    axr.axhline(0, color="black", linestyle=":", linewidth=1.2)

    # remove_edge_ticks(ax, which="major", axis="x")
    # remove_edge_ticks(ax, which="minor")

    ax.set_xticks([-1, -0.5, 0, 0.5, 1])

    # For each measurement values
    for ny, ypred in enumerate(ypred_list):
        delta = ypred - ytest

        sns.kdeplot(
            x=ytest,
            y=delta,
            fill=True,
            ax=ax,
            cmap=cmaps[ny],
            levels=3,
            alpha=0.6,
        )

        # Calculate slope and intercept
        res = lin(ytest, delta)
        ax.axline(
            xy1=(0, res.intercept),
            slope=res.slope,
            color=colors[ny],
            linewidth=1.5,
            linestyle="--",
        )

        kde = stats.gaussian_kde(delta)
        yy = np.linspace(-lim, lim, 1000)
        axr.plot(kde(yy), yy, color=colors[ny])

    handles = [
        mpatches.Patch(facecolor=plt.cm.Reds(100), label=legends[0]),
        mpatches.Patch(facecolor=plt.cm.Blues(100), label=legends[1]),
    ]
    ax.legend(handles=handles)

    savefig(fname, remove_bg=remove_bg)


# %%
def plot_residual_slope(
    factor, y_pred, y_true, bins, bin_centers, min_count=10, fname=None
):
    markers = ["o", "s"]

    fig = plt.figure(figsize=(6, 4.2))
    ax = fig.add_subplot(111)
    plt.sca(ax)

    for comp in range(2):
        res = y_pred[:, comp] - y_true[:, comp]
        x = y_true[:, comp]

        slopes = []
        slope_errs = []

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (factor >= lo) & (factor < hi)

            if mask.sum() < min_count:
                slopes.append(np.nan)
                slope_errs.append(np.nan)
                continue

            x_bin = x[mask]
            res_bin = res[mask]

            m, c = np.polyfit(x_bin, res_bin, 1)

            res_fit = m * x_bin + c
            sigma2 = np.var(res_bin - res_fit)
            m_err = np.sqrt(sigma2 / np.sum((x_bin - x_bin.mean()) ** 2))

            slopes.append(np.abs(m))
            slope_errs.append(m_err)

        ax.errorbar(
            bin_centers,
            slopes,
            yerr=slope_errs,
            lw=1.5,
            marker=markers[comp],
            capsize=3,
            label=rf"$\hat{{m}}_{{{comp + 1}}}$",
        )

    ax.set(
        xlabel=r"$f$",
        ylabel=r"$|\hat{m}|$",
    )

    handles, labels = ax.get_legend_handles_labels()
    handles = [
        h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles
    ]
    ax.legend(handles, labels)

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    savefig(fname)
