# %% Imports
import logging
import time
import warnings

import numpy as np
import xarray
from astropy import units as u
from dask import compute, delayed
from ska_sdp_func_python.visibility import subtract_visibility

from ..simulation.visibilities import (
    make_dirty_image_and_psf,
    predict_visibilities_from_array,
    rephase_visibility,
)
from ..utils.io import load_config
from ..utils.misc import post_step

# Disable warnings and logging from external libraries
warnings.warn = lambda *args, **kwargs: None
logging.getLogger().addHandler(logging.NullHandler())
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = ["get_facets", "residual_facet_image"]

cfg = load_config()
NPIX_SKY = cfg["NPIX_SKY"]
SCALE_RADIANS = cfg["SCALE_RADIANS"]


# %% Functions
def _process_facet(vis: xarray.Dataset, galaxy_location, NPIX_facet=256):
    """
    Process a single galaxy location:
    recenter visibilities, invert to get dirty and PSF facets.
    """
    vis_recentered = rephase_visibility(vis, galaxy_location)
    return make_dirty_image_and_psf(vis_recentered, NPIX=NPIX_facet, do_wstacking=False)


def process_batch(vis, loc_batch, NPIX_facet):
    dirty_batch, psf_batch = [], []
    for loc in loc_batch:
        dirty, psf = _process_facet(vis, loc, NPIX_facet)
        dirty_batch.append(dirty)
        psf_batch.append(psf)
    return np.stack(dirty_batch), np.stack(psf_batch)


def chunked_iterable(seq, size):
    """Yield successive chunks from a sequence."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def get_facets(vis, galaxy_locations, NPIX_facet=256, batch_size=64, client=None):
    dirty_all, psf_all = [], []
    n = len(galaxy_locations)
    total_batches = (n + batch_size - 1) // batch_size  # ceiling division
    start = time.time()

    for i in range(0, n, batch_size):
        loc_batch = galaxy_locations[i : i + batch_size]

        delayed_tasks = [
            delayed(_process_facet)(vis, loc, NPIX_facet) for loc in loc_batch
        ]
        results = compute(*delayed_tasks)

        dirty, psf = zip(*results)
        dirty_all.append(np.stack(dirty))
        psf_all.append(np.stack(psf))

        current_batch = i // batch_size + 1
        post_step(
            f"processing batch {current_batch}/{total_batches}", start, client=client
        )

    return np.concatenate(dirty_all), np.concatenate(psf_all)


def residual_facet_image(
    vis_original: xarray.Dataset, reconstructed_facet: np.ndarray, galaxy_location
):
    if reconstructed_facet.ndim != 2:
        raise ValueError("Reconstructed_facet must be a 2D array")

    # --- Compute pixel offsets from image centre ---
    x_pix, y_pix = galaxy_location
    dx = x_pix - NPIX_SKY // 2.0
    dy = y_pix - NPIX_SKY // 2.0

    # --- Compute new phasecentre SkyCoord ---
    pointing_centre = vis_original.attrs["phasecentre"]

    offset_ra = -dx * SCALE_RADIANS * u.rad
    offset_dec = dy * SCALE_RADIANS * u.rad

    gal_center_skycoord = pointing_centre.spherical_offsets_by(offset_ra, offset_dec)

    # Residual visibilities & image
    vis_model = predict_visibilities_from_array(
        image_array=reconstructed_facet,
        ra_deg=gal_center_skycoord.ra.deg,
        dec_deg=gal_center_skycoord.dec.deg,
    )

    vis_facet = rephase_visibility(vis_original, galaxy_location)

    vis_residual = subtract_visibility(vis_facet, vis_model)

    image_residual = make_dirty_image_and_psf(
        vis_residual, NPIX=reconstructed_facet.shape[0], do_psf=False
    )

    return image_residual
