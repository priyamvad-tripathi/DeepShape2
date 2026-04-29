# %%
import galsim
import numpy as np

__all__ = [
    "RMSE",
    "position_angle",
    "ellipticity_modulus",
    "convert_g_to_e",
    "axisratio_pa_to_shape",
]


# %% Functions
def RMSE(ypred, ytrue):
    return np.sqrt(np.mean((ypred - ytrue) ** 2))


def position_angle(y):
    e1 = y[:, 0]
    e2 = y[:, 1]
    alpha = 0.5 * np.arctan2(e2, e1)
    return alpha


def ellipticity_modulus(y):
    """Compute ellipticity magnitude"""
    return np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)


def convert_g_to_e(shape_g):
    """Convert shear (g1, g2) to ellipticity (e1, e2) using galsim.Shear."""

    shape_e = np.zeros_like(shape_g)

    for ns, shape in enumerate(shape_g):
        g1, g2 = shape
        shear = galsim.Shear(g1=g1, g2=g2)
        shape_e[ns] = [shear.e1, shear.e2]

    return shape_e


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
