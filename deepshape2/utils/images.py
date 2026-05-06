import galsim
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u
from numpy.fft import ifftshift, irfftn, rfftn

__all__ = [
    "extract_image",
    "extract_multiple",
    "shape_galsim",
    "get_stamps",
    "process_stamp",
    "centers_to_limits",
    "print_peak",
    "extract_cutouts",
    "calculate_dirty_peak",
]

# %% Image extraction functions


def extract_image(arr, NPIX=128, center=None, relative=True, switch_xy=True):
    """
    Extract a crop of size NPIX x NPIX from a 2D or 3D array.
    - If arr is 2D: returns (NPIX, NPIX).
    - If arr is 3D: returns (batch, NPIX, NPIX).

    Parameters
    ----------
    arr : ndarray
        Input array of shape (H, W) or (B, H, W).
    NPIX : int
        Size of the square crop.
    center : tuple or list or None
        Crop center. If arr is 3D, can be (y,x) applied to all images,
        or a list of length B with per-image centers.
    relative : bool
        If True, center is relative to image center. If False, absolute coords.
    switch_xy : bool
        If True, interpret center as (x,y). Otherwise (y,x).

    Returns
    -------
    ndarray
        Cropped array of shape (NPIX, NPIX) or (B, NPIX, NPIX).
    """

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, ...]  # add batch dimension
        squeeze_out = True
    elif arr.ndim == 3:
        squeeze_out = False
    else:
        raise ValueError("Input must be 2D or 3D array")

    B, H, W = arr.shape

    # handle center input
    if center is None:
        centers = [(H // 2, W // 2)] * B
    elif isinstance(center[0], (int, float)):  # single center for all
        cy, cx = (center[1], center[0]) if switch_xy else (center[0], center[1])
        centers = [(cy, cx)] * B
    else:  # list of per-image centers
        if len(center) != B:
            raise ValueError("Length of center list must match batch size")
        centers = []
        for c in center:
            cy, cx = (c[1], c[0]) if switch_xy else (c[0], c[1])
            centers.append((cy, cx))

    out = np.full((B, NPIX, NPIX), np.nan, dtype=float)

    for b in range(B):
        cy, cx = centers[b]

        if relative and center is not None:
            cy = H // 2 + cy
            cx = W // 2 + cx

        start_y = int(cy - NPIX // 2)
        end_y = start_y + NPIX
        start_x = int(cx - NPIX // 2)
        end_x = start_x + NPIX

        # Clip bounds
        src_start_y = max(start_y, 0)
        src_end_y = min(end_y, H)
        src_start_x = max(start_x, 0)
        src_end_x = min(end_x, W)

        dst_start_y = src_start_y - start_y
        dst_end_y = dst_start_y + (src_end_y - src_start_y)
        dst_start_x = src_start_x - start_x
        dst_end_x = dst_start_x + (src_end_x - src_start_x)

        # Copy valid data
        if src_start_y < src_end_y and src_start_x < src_end_x:
            out[b, dst_start_y:dst_end_y, dst_start_x:dst_end_x] = arr[
                b, src_start_y:src_end_y, src_start_x:src_end_x
            ]

    return out[0] if squeeze_out else out


def extract_multiple(arr, centers, NPIX=128, relative=True, switch_xy=True):
    centers = np.asarray(centers)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must be of shape (B, 2)")

    crops = []

    for i, center in enumerate(centers):
        crop = extract_image(
            arr, NPIX=NPIX, center=[center], relative=relative, switch_xy=switch_xy
        )
        crops.append(crop)

    return np.stack(crops, axis=0)


def centers_to_limits(centers, stamp_size):
    x_c, y_c = centers[:, 0], centers[:, 1]
    x0 = x_c - stamp_size // 2
    x1 = x0 + stamp_size
    y0 = y_c - stamp_size // 2
    y1 = y0 + stamp_size

    limits = np.stack([x0, x1, y0, y1], axis=1)

    return limits.astype(int)


# %% Stamp processing functions


def get_stamps(stamp_list, NPIX=128):
    resized = []
    for stamp in stamp_list:
        resized.append(process_stamp(stamp, NPIX))
    return np.stack(resized)


def process_stamp(stamp, NPIX):
    size = stamp.shape[0]
    if size == NPIX:
        return stamp
    elif size > NPIX:
        start = (size - NPIX) // 2
        return stamp[start : start + NPIX, start : start + NPIX]
    else:
        pad = (NPIX - size) // 2
        pad_extra = (NPIX - size) % 2
        return np.pad(
            stamp,
            ((pad, pad + pad_extra), (pad, pad + pad_extra)),
            mode="constant",
        )


# %% Shape measurement functions


def _shape_galsim_single(image: np.ndarray):
    """Compute ellipticity and moments status for a single 2D galaxy image using GalSim."""

    NPIX = image.shape[0]
    image_galsim = galsim.Image(image)
    try:
        shape = galsim.hsm.FindAdaptiveMom(
            image_galsim,
            guess_centroid=galsim.PositionD(NPIX // 2, NPIX // 2),
            strict=False,
        )
    except galsim.errors.GalSimHSMError:
        # Retry with relaxed parameters if initial estimation fails
        new_params = galsim.hsm.HSMParams(
            max_mom2_iter=2000, convergence_threshold=0.1, bound_correct_wt=2.0
        )
        shape = galsim.hsm.FindAdaptiveMom(
            image_galsim,
            guess_centroid=galsim.PositionD(NPIX // 2, NPIX // 2),
            strict=False,
            hsmparams=new_params,
        )

    g = np.array([shape.observed_shape.g1, shape.observed_shape.g2])
    return g, shape.moments_status


def shape_galsim(images):
    """
    Compute galaxy shapes using GalSim adaptive moments.

    Parameters
    ----------
    images : numpy.ndarray
        Input image or array of images.
    Returns
    -------
    g : numpy.ndarray
        Ellipticity components:
            - Shape (2,) for a single image
            - Shape (N, 2) for a batch
    status : numpy.ndarray or int
        Moment status code(s):
            - int for a single image
            - shape (N,) for a batch
    """
    if images.ndim > 3:
        images = np.squeeze(images)

    # Handle single image
    if images.ndim == 2:
        return _shape_galsim_single(images)

    # Handle batch of images
    elif images.ndim == 3:
        n_images = images.shape[0]
        g_all = np.zeros((n_images, 2), dtype=float)
        status_all = np.zeros(n_images, dtype=int)

        for i in range(n_images):
            g, status = _shape_galsim_single(images[i])
            g_all[i] = g
            status_all[i] = status

        return g_all, status_all

    else:
        raise ValueError(
            f"Invalid input shape {images.shape}. Expected 2D or 3D array (after squeezing)."
        )


def print_peak(img, title="Image"):
    """Print and return the coordinates of the peak (max value) in a 2D image array."""
    peak_idx = np.unravel_index(np.argmax(img), img.shape)
    print(f"Peak position of {title}: {peak_idx}, value: {img[peak_idx]:.3e}")
    return peak_idx


# %%
def extract_cutouts(image, w2, ra, dec, size=128):

    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    result = np.full((len(coords), size, size), np.nan, dtype=float)

    for i, coord in enumerate(coords):
        try:
            cut = Cutout2D(
                data=image,
                position=coord,
                wcs=w2,
                size=(size, size),
                mode="partial",
                fill_value=np.nan,
            )
            result[i] = cut.data

        except Exception:
            # Only happens if coordinate is completely outside WCS
            # In that case, we can just leave the cutout as NaNs
            continue

    return result


def calculate_dirty_peak(img, psf):
    """Calculate the peak value of corresponding dirty image"""

    # ---- Normalize shapes ----
    single = False
    if img.ndim == 2:
        img = img[None, ...]
        single = True

    if psf.ndim == 2:
        psf = psf[None, ...]  # broadcast later

    # ---- Broadcast PSF if needed ----
    if psf.shape[0] == 1 and img.shape[0] > 1:
        psf = np.repeat(psf, img.shape[0], axis=0)

    if psf.shape != img.shape:
        raise ValueError(f"Incompatible shapes: img {img.shape}, psf {psf.shape}")

    # ---- FFTs ----
    img_f = rfftn(img, axes=(-2, -1))
    psf_f = rfftn(ifftshift(psf, axes=(-2, -1)), axes=(-2, -1))

    dirty = irfftn(img_f * psf_f, s=img.shape[-2:], axes=(-2, -1))

    peaks = dirty.max(axis=(-2, -1)).astype(np.float32)

    return peaks[0] if single else peaks
