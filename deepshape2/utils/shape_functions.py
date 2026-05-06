# %%
import galsim
import numpy as np
from astropy.stats import circmean, circstd, rayleightest

__all__ = [
    "RMSE",
    "position_angle",
    "ellipticity_modulus",
    "convert_g_to_e",
    "convert_e_to_g",
    "axisratio_pa_to_shape",
    "wrap_angle",
    "pa_delta2",
    "compute_circular_stats",
]


# %% Functions
def RMSE(ypred, ytrue):
    return np.sqrt(np.mean((ypred - ytrue) ** 2))


def position_angle(y):
    y = np.asarray(y)

    if y.ndim == 1:  # single shape (2,)
        e1, e2 = y
        return 0.5 * np.arctan2(e2, e1)

    # multiple shapes (N, 2)
    e1 = y[:, 0]
    e2 = y[:, 1]
    return 0.5 * np.arctan2(e2, e1)


def ellipticity_modulus(shape):
    """Compute ellipticity magnitude"""

    shape = np.asarray(shape)
    if shape.ndim == 1:  # single shape (2,)
        return np.sqrt(shape[0] ** 2 + shape[1] ** 2)
    return np.sqrt(np.sum(shape**2, axis=1))


def convert_g_to_e(shape_g):
    """Convert shear (g1, g2) to ellipticity (e1, e2) using galsim.Shear."""

    shape_e = np.zeros_like(shape_g)

    for ns, shape in enumerate(shape_g):
        g1, g2 = shape
        shear = galsim.Shear(g1=g1, g2=g2)
        shape_e[ns] = [shear.e1, shear.e2]

    return shape_e


def convert_e_to_g(shape_e):
    """Convert ellipticity (e1, e2) to shear (g1, g2) using galsim.Shear."""

    shape_g = np.zeros_like(shape_e)

    for ns, shape in enumerate(shape_e):
        e1, e2 = shape
        shear = galsim.Shear(e1=e1, e2=e2)
        shape_g[ns] = [shear.g1, shear.g2]
    return shape_g


def axisratio_pa_to_shape(a, b, pa, complement=True, rad=False):

    a = np.asarray(a)
    b = np.asarray(b)
    pa = np.asarray(pa)

    if rad:
        pa = np.degrees(pa)

    if complement:
        pa = 90 - np.asarray(pa)

    q = b / a

    g = np.empty((len(q), 2))

    for i, (qi, beta) in enumerate(zip(q, pa)):
        s = galsim.Shear(q=qi, beta=beta * galsim.degrees)
        g[i, 0] = s.e1
        g[i, 1] = s.e2

    return g


def wrap_angle(x, period="half"):
    """
    Wrap angles (radians) to a symmetric interval.

    Parameters
    ----------
    x : array-like
        Input angles in radians.
    period : {'half', 'full'} or float
        - 'half' → wrap to [-pi/2, pi/2)
        - 'full' → wrap to [-pi, pi)
        - float  → custom period P, wraps to [-P/2, P/2)

    Returns
    -------
    ndarray
        Wrapped angles.
    """
    x = np.asarray(x, dtype=float)

    if period == "half":
        P = np.pi
    elif period == "full":
        P = 2 * np.pi
    elif np.isscalar(period):
        P = float(period)
    else:
        raise ValueError("period must be 'half', 'full', or a float")

    return (x + P / 2) % P - P / 2


def pa_delta2(pa_method, pa_ref):
    """
    Compute residuals between two PA arrays (radians).
    Wraps inputs to [-pi/2, pi/2) first, doubles, then wraps difference.
    Returns delta2 in radians, in [-pi, pi).
    """
    pa_method = wrap_angle(pa_method, period="half")
    pa_ref = wrap_angle(pa_ref, period="half")
    return wrap_angle(2.0 * pa_method - 2.0 * pa_ref, period="full")


def compute_circular_stats(pa_method, pa_ref):
    """
    Full circular statistics for a paired PA comparison (spin-2).

    Parameters
    ----------
    pa_method, pa_ref : arrays of PA in radians, [-pi/2, pi/2)

    Returns
    -------
    dict with:
        R            : mean resultant length [0, 1]
        mean_offset  : circular mean of residuals in degrees
        circ_std     : circular std of residuals in degrees
        rayleigh_p   : Rayleigh test p-value (H0: uniform distribution)
        delta_deg    : per-source residuals in degrees, for plotting
        n            : sample size
    """
    delta2 = pa_delta2(pa_method, pa_ref)

    circ_mean_2 = circmean(delta2)
    circ_std_2 = circstd(delta2)

    mean_offset_deg = np.rad2deg(circ_mean_2) / 2.0
    circ_std_deg = np.rad2deg(circ_std_2) / 2.0

    R = np.abs(np.mean(np.exp(1j * np.asarray(delta2))))
    n = len(delta2)

    rayleigh_p = rayleightest(delta2)

    return dict(
        R=R,
        mean_offset=mean_offset_deg,
        circ_std=circ_std_deg,
        rayleigh_p=rayleigh_p,
        delta_deg=np.rad2deg(delta2) / 2.0,
        n=n,
    )
