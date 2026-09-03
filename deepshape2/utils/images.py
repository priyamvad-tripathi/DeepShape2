import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from numpy.fft import ifftshift, irfftn, rfftn

__all__ = [
    "extract_image",
    "extract_multiple",
    "get_stamps",
    "process_stamp",
    "centers_to_limits",
    "print_peak",
    "extract_cutouts",
    "calculate_dirty_peak",
    "extract_stamps",
    "peak_brightness",
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

    for _, center in enumerate(centers):
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


def extract_stamps(
    image,
    centers,
    npix=128,
    switch_xy=True,
    fill=np.nan,
    dtype=None,
    chunk=512,
    return_valid=False,
):
    """
    Cut N npix x npix stamps out of a large 2D image by fancy indexing.

    Parameters
    ----------
    image : (H, W) ndarray or h5py Dataset
        Datasets are read into memory once; per-stamp h5py fancy indexing is
        orders of magnitude slower.
    centers : (N, 2) array of ints
        Absolute pixel positions. (x, y) if switch_xy, else (y, x).
    npix : int
        Stamp size. The centre lands at index npix // 2, matching
        `start = c - npix // 2`.
    fill : float
        Value for pixels falling outside the image.
    dtype : np.dtype or None
        Output dtype. Defaults to the image dtype (float32 stays float32).
    chunk : int
        Stamps per gather, to bound peak temporary memory.
    return_valid : bool
        Also return a (N,) bool array, True where the stamp is fully inside.

    Returns
    -------
    (N, npix, npix) ndarray, and optionally the validity mask.
    """
    if not isinstance(image, np.ndarray):
        image = image[:]  # h5py Dataset -> memory, once
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got {image.shape}")

    centers = np.asarray(centers)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"centers must be (N, 2), got {centers.shape}")

    cx, cy = (
        (centers[:, 0], centers[:, 1]) if switch_xy else (centers[:, 1], centers[:, 0])
    )
    cx = np.rint(cx).astype(np.int64)
    cy = np.rint(cy).astype(np.int64)

    H, W = image.shape
    half = npix // 2
    y0, x0 = cy - half, cx - half
    n = len(centers)

    if dtype is None:
        dtype = image.dtype
    if np.isnan(fill) and not np.issubdtype(dtype, np.floating):
        raise ValueError("fill=nan needs a floating dtype")

    inside = (x0 >= 0) & (x0 + npix <= W) & (y0 >= 0) & (y0 + npix <= H)
    out = np.empty((n, npix, npix), dtype=dtype)
    span = np.arange(npix, dtype=np.int64)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        rows = y0[s:e, None] + span  # (b, npix)
        cols = x0[s:e, None] + span
        ok_r, ok_c = (rows >= 0) & (rows < H), (cols >= 0) & (cols < W)

        block = image[
            np.clip(rows, 0, H - 1)[:, :, None], np.clip(cols, 0, W - 1)[:, None, :]
        ].astype(dtype, copy=False)
        if not (ok_r.all() and ok_c.all()):
            block = block.copy()
            block[~(ok_r[:, :, None] & ok_c[:, None, :])] = fill
        out[s:e] = block

    return (out, inside) if return_valid else out


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


NPIX = 128
NPAD = 256
CHUNK = 512


def peak_brightness(iso, psf, chunk=CHUNK, npad=NPAD, npix=NPIX):
    """
    Peak of each isolated stamp convolved with the PSF, in Jy/beam.

    iso is Jy/pixel and psf is peak-normalised, so the convolution is
    exactly the noiseless dirty image of that source in isolation.
    """
    n = len(iso)
    o = (npad - npix) // 2
    psf_f = rfftn(ifftshift(psf), axes=(0, 1))
    out = np.empty(n, dtype=np.float32)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        pad = np.zeros((e - s, npad, npad), dtype=np.float32)
        pad[:, o : o + npix, o : o + npix] = iso[s:e]
        conv = irfftn(rfftn(pad, axes=(1, 2)) * psf_f, s=(npad, npad), axes=(1, 2))
        out[s:e] = conv.max(axis=(1, 2))
    return out
