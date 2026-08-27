# %%
import math

import galsim
import numpy as np
from astropy.stats import circmean, circstd, rayleightest

__all__ = [
    "RMSE",
    "shape_galsim",
    "position_angle",
    "ellipticity_modulus",
    "convert_g_to_e",
    "convert_e_to_g",
    "axisratio_pa_to_shape",
    "wrap_angle",
    "pa_delta2",
    "compute_circular_stats",
    "gaussian_weighted_moments",
    "measure_array",
]

# %% Shape measurement functions


def _shape_galsim_single(image: np.ndarray, e1=True):
    """Compute ellipticity and validity flag for a single 2D galaxy image using GalSim."""

    NPIX = image.shape[0]
    image_galsim = galsim.Image(image)

    try:
        shape = galsim.hsm.FindAdaptiveMom(
            image_galsim,
            guess_centroid=galsim.PositionD(NPIX // 2, NPIX // 2),
            strict=False,
        )
    except galsim.errors.GalSimHSMError:
        new_params = galsim.hsm.HSMParams(
            max_mom2_iter=2000, convergence_threshold=0.1, bound_correct_wt=2.0
        )
        shape = galsim.hsm.FindAdaptiveMom(
            image_galsim,
            guess_centroid=galsim.PositionD(NPIX // 2, NPIX // 2),
            strict=False,
            hsmparams=new_params,
        )

    # Convert status → validity (True = good, False = bad)
    valid = True if shape.moments_status == 0 else False

    if e1:
        return np.array([shape.observed_shape.e1, shape.observed_shape.e2]), valid

    g = np.array([shape.observed_shape.g1, shape.observed_shape.g2])
    return g, valid


def shape_galsim(images, e1=True):
    """
    Compute galaxy shapes using GalSim adaptive moments.

    Returns
    -------
    g : ndarray
    valid : ndarray or int
        True = good measurement, False = bad
    """
    if images.ndim > 3:
        images = np.squeeze(images)

    if images.ndim == 2:
        return _shape_galsim_single(images, e1=e1)

    elif images.ndim == 3:
        n_images = images.shape[0]
        g_all = np.zeros((n_images, 2), dtype=float)
        valid_all = np.zeros(n_images, dtype=bool)

        for i in range(n_images):
            g, valid = _shape_galsim_single(images[i], e1=e1)
            g_all[i] = g
            valid_all[i] = valid

        return g_all, valid_all

    else:
        raise ValueError(
            f"Invalid input shape {images.shape}. Expected 2D or 3D array (after squeezing)."
        )


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
    """Convert reduced shear (g1, g2) to distortion (e1, e2).
    |e| = 2|g| / (1 + |g|^2), phase preserved. Accepts (2,) or (N, 2).
    """
    g = np.asarray(shape_g, dtype=float)
    if g.shape[-1] != 2:
        raise ValueError(f"expected last axis of length 2, got {g.shape}")
    gsq = np.sum(g**2, axis=-1, keepdims=True)
    return g * (2.0 / (1.0 + gsq))


def convert_e_to_g(shape_e):
    """Convert distortion (e1, e2) to reduced shear (g1, g2).
    |g| = |e| / (1 + sqrt(1 - |e|^2)), phase preserved. Accepts (2,) or (N, 2).
    """
    e = np.asarray(shape_e, dtype=float)
    if e.shape[-1] != 2:
        raise ValueError(f"expected last axis of length 2, got {e.shape}")
    esq = np.sum(e**2, axis=-1, keepdims=True)
    return e / (1.0 + np.sqrt(np.clip(1.0 - esq, 0.0, None)))


def axisratio_pa_to_shape(a, b, pa, complement=True, rad=False):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pa = np.asarray(pa, dtype=float)

    scalar = (a.ndim == 0) and (b.ndim == 0) and (pa.ndim == 0)

    a, b, pa = np.atleast_1d(a, b, pa)
    a, b, pa = np.broadcast_arrays(a, b, pa)

    if rad:
        pa = np.degrees(pa)

    if complement:
        pa = 90 - pa

    q = b / a

    g = np.empty((q.size, 2))

    for i, (qi, beta) in enumerate(zip(q.ravel(), pa.ravel())):
        s = galsim.Shear(q=qi, beta=beta * galsim.degrees)
        g[i, 0] = s.e1
        g[i, 1] = s.e2

    return g[0] if scalar else g


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


# %%
def gaussian_weighted_moments(img, sigma_init_px, n_iter=40, tol=1e-8):
    """
    HSM-style adaptive moments with an elliptical Gaussian weight.

    Returns (xc, yc, Qxx, Qyy, Qxy) in pixel units, deconvolved from the
    weight function under the Gaussian assumption:
        Q_measured^-1 = Q_true^-1 + Q_weight^-1
    """
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx].astype(float)

    iy, ix = np.unravel_index(np.argmax(img), img.shape)
    xc, yc = float(ix), float(iy)
    Q = np.array([[sigma_init_px**2, 0.0], [0.0, sigma_init_px**2]])

    for _ in range(n_iter):
        Qinv = np.linalg.inv(Q)
        dx, dy = xx - xc, yy - yc
        r2 = Qinv[0, 0] * dx * dx + 2 * Qinv[0, 1] * dx * dy + Qinv[1, 1] * dy * dy
        w = np.exp(-0.5 * np.clip(r2, 0, 200))

        wi = img * w
        norm = wi.sum()

        xc_new = (wi * xx).sum() / norm
        yc_new = (wi * yy).sum() / norm
        dx, dy = xx - xc_new, yy - yc_new
        Qm = np.array(
            [
                [(wi * dx * dx).sum() / norm, (wi * dx * dy).sum() / norm],
                [(wi * dx * dy).sum() / norm, (wi * dy * dy).sum() / norm],
            ]
        )

        # Deconvolve the weight; fall back to the matched-weight result
        # (Q = 2 Q_m) if the subtraction goes non-positive-definite.
        try:
            Qt = np.linalg.inv(np.linalg.inv(Qm) - np.linalg.inv(Q))
            if np.linalg.det(Qt) <= 0 or Qt[0, 0] <= 0 or Qt[1, 1] <= 0:
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            Qt = 2.0 * Qm

        shift = math.hypot(xc_new - xc, yc_new - yc)
        dQ = np.abs(Qt - Q).max() / np.abs(Q).max()
        xc, yc, Q = xc_new, yc_new, Qt
        if shift < tol and dQ < tol:
            break

    return xc, yc, Q[0, 0], Q[1, 1], Q[0, 1]


def measure_array(img, pixscale):
    """Adaptive-moment shape of a single PSF stamp. Mirrors measure_psf()."""
    img = np.asarray(img, dtype=np.float64)
    ny, nx = img.shape

    # Initial guess: 0.4" beam, the ILT ballpark.
    sigma_init = max(0.4 / 2.3548 / pixscale, 1.5)
    xc, yc, Qxx, Qyy, Qxy = gaussian_weighted_moments(img, sigma_init)

    T = Qxx + Qyy
    tr, det = T, Qxx * Qyy - Qxy**2
    disc = max(tr * tr / 4.0 - det, 0.0)
    lam1, lam2 = tr / 2.0 + math.sqrt(disc), tr / 2.0 - math.sqrt(disc)
    f = 2.3548 * pixscale

    return {
        "e1": (Qxx - Qyy) / T,
        "e2": 2.0 * Qxy / T,
        "|e|": math.hypot((Qxx - Qyy) / T, 2.0 * Qxy / T),
        "fwhm_maj_asec": f * math.sqrt(lam1),
        "fwhm_min_asec": f * math.sqrt(max(lam2, 0.0)),
        "pa_deg": 0.5 * math.degrees(math.atan2(2.0 * Qxy, Qxx - Qyy)),
        "T_arcsec2": T * pixscale**2,
        "peak": float(img.max()),
        "min_sidelobe": float(img.min()),
        "dx_px": xc - (nx / 2.0 - 0.5),
        "dy_px": yc - (ny / 2.0 - 0.5),
    }
