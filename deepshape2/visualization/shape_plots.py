# %%
import matplotlib.pyplot as plt
import numpy as np

from deepshape2.visualization.base import set_style

set_style()

__all__ = ["plot_bias"]


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

    Parameters
    ----------
    ypred : array-like, shape (N,2)
        Predicted ellipticities.
    ytest : array-like, shape (N,2)
        True ellipticities.
    colors : tuple
        Colors for the two ellipticity components.
    power : float
        Scaling factor for slope/intercept in the legend.
    bad_index : array-like, optional
        Boolean array of points to exclude.
    bias_line : bool
        Whether to plot linear bias lines.
    ellipticity_cutoff : float
        Maximum absolute value of x for plotting.
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

    def fit_line(x, y):
        """Fit linear regression and return slope, intercept, and stderr."""
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        residuals = y - (slope * x + intercept)
        stderr = np.std(residuals) / np.sqrt(len(x))
        return slope, intercept, stderr

    fig, ax = plt.subplots(figsize=(6, 4))

    for i in range(2):
        x = ytest[:, i]
        y = delta[:, i]

        # Apply ellipticity cutoff
        mask = np.abs(x) < ellipticity_cutoff
        x, y = x[mask], y[mask]

        # Fit linear bias
        m, c, stderr = fit_line(x, y)

        # Print slope and intercept
        print(f"Component e{i + 1}: slope = {m:.5f}, intercept = {c:.5f}")

        # Scatter points
        label = (
            rf"$e{i + 1}: m={m * power:.2f}\pm{stderr * power:.2f}, c={c * power:.2f}$"
        )
        ax.scatter(x, y, color=colors[i], s=5, alpha=0.5, label=label)

        # Bias line
        if bias_line:
            x_line = np.array([-ellipticity_cutoff, ellipticity_cutoff])
            y_line = m * x_line + c
            ax.plot(x_line, y_line, color=colors[i], linestyle="--", linewidth=1)

    ax.set(
        xlabel=r"$\epsilon^T$",
        ylabel=r"$\hat{\epsilon}-\epsilon^T$",
        xlim=(-1, 1),
        ylim=(-lim, lim),
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)

    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()
