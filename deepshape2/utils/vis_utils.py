import numpy as np

__all__ = ["split_band", "combine_channels"]


def split_band(nu_c, del_nu, n):
    """
    Split a band of total width del_nu centred on nu_c into n equal sub-bands.

    Returns
    -------
    centres : (n,) array of sub-band central frequencies
    widths  : (n,) array of sub-band bandwidths (all equal here)
    """
    width = del_nu / n
    nu_lo = nu_c - del_nu / 2.0
    centres = nu_lo + (np.arange(n) + 0.5) * width
    widths = np.full(n, width)
    return centres, widths


def combine_channels(image, weights):
    """
    Weighted mean of images.

    Parameters
    ----------
    image : ndarray, shape (N_frequency, Npix, Npix)
        Image array.

    weights : ndarray, shape (N_time, N_baseline, N_frequency, N_polarisation)
        4D weights, where axis 2 corresponds to the frequency/channel
        dimension of the image array.

    Returns
    -------
    ndarray, shape (Npix, Npix)
        Weighted mean image.
    """
    if weights.ndim != 4:
        raise ValueError(f"weights must be 4D, got shape {weights.shape}")

    if image.ndim != 3:
        raise ValueError(f"image must be 3D, got shape {image.shape}")

    if weights.shape[2] != image.shape[0]:
        raise ValueError(
            f"weights axis 2 ({weights.shape[2]}) must match "
            f"number of frequency channels ({image.shape[0]})"
        )

    # Sum weights over all axes except the frequency/channel axis
    w = weights.sum(axis=(0, 1, 3))

    return np.tensordot(w, image, axes=(0, 0)) / w.sum()
