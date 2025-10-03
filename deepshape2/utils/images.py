import galsim
import numpy as np

__all__ = [
    "extract_image",
    "extract_multiple",
    "shape_galsim",
    "get_stamps",
    "process_stamp",
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


def shape_galsim(image, NPIX=128):
    """
    Predict the shape of a galaxy using adaptive moments.

    Parameters:
    ----------
    image (numpy.ndarray): The input image array containing the galaxy.
    NPIX (int, optional): The size of the image in pixels. Default is 128.

    Returns:
    tuple: A tuple containing:
        - g (numpy.ndarray): An array with two elements representing the ellipticity components (g1, g2).
        - moments_status (int): The status of the moments calculation (0 if successful, non-zero otherwise).

    Notes:
    This function uses the GalSim library to estimate the adaptive moments of the input image.
    If the initial moments estimation fails, it retries with modified parameters.
    """

    im_size = NPIX
    # create a galsim version of the data
    image_galsim = galsim.Image(image)
    # estimate the moments of the observation image
    shape = galsim.hsm.FindAdaptiveMom(
        image_galsim,
        guess_centroid=galsim.PositionD(im_size // 2, im_size // 2),
        strict=False,
    )
    if shape.error_message:
        new_params = galsim.hsm.HSMParams(
            max_mom2_iter=2000, convergence_threshold=0.1, bound_correct_wt=2.0
        )
        shape = image_galsim.FindAdaptiveMom(strict=False, hsmparams=new_params)
    g = np.array([shape.observed_shape.g1, shape.observed_shape.g2])

    return g, shape.moments_status
