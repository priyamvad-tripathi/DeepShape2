# %% Load Modules
import warnings

import astropy.units as u
import galsim
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

# from astropy.wcs import WCS
from dask import compute, delayed

from deepshape2.utils import load_config, process_stamp

warnings.filterwarnings("ignore", category=UserWarning)

__all__ = [
    "generate_patch_locations",
    "random_patch",
    "simulate_wide_field",
    "filter_patch_by_flux",
    "filter_patch_by_size",
    "compute_pixel_coordinates",
]

# %% Default Parameters
cfg = load_config()

# Pixel size in different units
SCALE_RADIANS = cfg["SCALE_RADIANS"]
SCALE_DEGREES = SCALE_RADIANS * 180 / np.pi
SCALE_ARCSEC = SCALE_DEGREES * 3600

# FFT parameters for GalSim
FFTBIGSIZE = cfg["FFTBIGSIZE"]
big_fft_params = galsim.GSParams(maximum_fft_size=FFTBIGSIZE)

# Cache Sersic Indexes for faster galsim implementation
sersic_indexes = np.linspace(0.7, 2, 100)

# Sky center
RA0 = cfg["RA0"]
DEC0 = cfg["DEC0"]


# Size of wide field in degrees in flat sky approximation along one axis
SKY_SIZE = 20
NPIX_SKY = cfg["NPIX_SKY"]

NPIX_STAMP = 256

TRECS_DIR = cfg["TRECS_DIR"]

# %% Catalogue Processing Functions


def generate_patch_locations(
    sky_size=SKY_SIZE, patch_size=1.0, max_patches=100, seed=43
):
    """Pre-select non-overlapping random patch locations (center coordinates)."""
    if seed is not None:
        np.random.seed(seed)

    x_min, x_max = -sky_size / 2, sky_size / 2
    y_min, y_max = -sky_size / 2, sky_size / 2

    chosen_centers = []
    attempts = 0
    while len(chosen_centers) < max_patches and attempts < max_patches * 10:
        cx = np.random.uniform(x_min + patch_size / 2, x_max - patch_size / 2)
        cy = np.random.uniform(y_min + patch_size / 2, y_max - patch_size / 2)

        # Check overlap: distance between centers < patch_size
        overlap = any(
            abs(cx - c[0]) < patch_size and abs(cy - c[1]) < patch_size
            for c in chosen_centers
        )
        if not overlap:
            chosen_centers.append((cx, cy))
        attempts += 1

    if len(chosen_centers) < max_patches:
        print(f"Warning: Only {len(chosen_centers)} non-overlapping patches found.")

    return chosen_centers


def random_patch(center, catalogue_type, patch_size=1.0):
    """
    Extract a patch around a given center (cx, cy) in flat-sky coordinates.
    Returns the galaxies in the patch and RA/Dec of the patch center.
    """
    catalogue = pd.read_pickle({TRECS_DIR} / f"catalog_{catalogue_type}.pkl")

    cx, cy = center
    x0, y0 = cx - patch_size / 2, cy - patch_size / 2
    x1, y1 = cx + patch_size / 2, cy + patch_size / 2

    # Select galaxies inside patch
    patch = catalogue[
        (catalogue["x"] >= x0)
        & (catalogue["x"] < x1)
        & (catalogue["y"] >= y0)
        & (catalogue["y"] < y1)
    ].copy()

    # Drop galaxies too close to patch edge
    margin = SCALE_DEGREES * 130
    mask = (
        (patch["x"] - margin >= x0)
        & (patch["x"] + margin <= x1)
        & (patch["y"] - margin >= y0)
        & (patch["y"] + margin <= y1)
    )
    patch = patch[mask].copy()

    # Drop old RA/Dec if present
    for col in ["ra", "dec"]:
        if col in patch.columns:
            patch.drop(columns=col, inplace=True)

    return patch


def filter_patch_by_flux(
    patch: pd.DataFrame, flux_min: float = 1e-6, flux_max: float = 200e-6
) -> pd.DataFrame:
    mask = pd.Series(True, index=patch.index)
    if flux_min is not None:
        mask &= patch["flux"] >= flux_min
    if flux_max is not None:
        mask &= patch["flux"] <= flux_max
    return patch[mask].copy()


def filter_patch_by_size(
    patch: pd.DataFrame,
    size_min: float = None,
    size_max: float = 6,
) -> pd.DataFrame:
    mask = pd.Series(True, index=patch.index)
    if size_min is not None:
        mask &= patch["size"] >= size_min
    if size_max is not None:
        mask &= patch["size"] <= size_max
    return patch[mask].copy()


# %% Wide-field Simulation Functions


def compute_pixel_coordinates(patch, patch_center_flat, NPIX_SKY=NPIX_SKY):
    """
    Compute pixel positions and RA/Dec of galaxies relative to patch center.

    Parameters
    ----------
    patch : pd.DataFrame
        Must have columns 'x', 'y' in flat-sky degrees
    patch_center_flat : tuple
        (x, y) coordinates of patch center in flat-sky degrees
    NPIX_SKY : int
        Size of the square wide-field image in pixels

    Returns
    -------
    patch_out : pd.DataFrame
        Copy of patch with 'pix_x', 'pix_y', 'RA', 'Dec'
    patch_center_ra_dec : np.ndarray
        RA/Dec of patch center
    """
    cx, cy = patch_center_flat
    patch_out = patch.copy()

    # Offsets in pixels relative to patch center
    dx = ((patch_out["x"] - cx) / SCALE_DEGREES).astype(int)
    dy = ((patch_out["y"] - cy) / SCALE_DEGREES).astype(int)

    # Pixel coordinates
    patch_out["pix_x"] = dx + NPIX_SKY // 2
    patch_out["pix_y"] = dy + NPIX_SKY // 2

    # Remove flat-sky columns
    patch_out = patch_out.drop(columns=["x", "y"], errors="ignore")

    # RA/Dec of patch center using _xy_to_radec
    origin = SkyCoord(RA0 * u.deg, DEC0 * u.deg, frame="icrs")
    patch_center_coord = origin.spherical_offsets_by(
        d_lat=cy * u.deg, d_lon=-cx * u.deg
    )
    patch_center_ra_dec = np.array(
        [patch_center_coord.ra.deg, patch_center_coord.dec.deg]
    )

    return patch_out, patch_center_ra_dec


def _simulate_galaxy(
    row, simple=False, min_flux=10e-6, npix_stamp=NPIX_STAMP, scale=SCALE_ARCSEC
):
    flux = row["flux"]
    scale_length = row["size"]
    e1 = row["e1"]
    e2 = row["e2"]

    if simple:
        sersic_index = 1
    elif "sersic_index" in row and not np.isnan(row["sersic_index"]):
        sersic_index = row["sersic_index"]
    else:
        sersic_index = np.random.choice(sersic_indexes)

    hlr = scale_length * 1.6783469900166605
    gal = galsim.Sersic(
        n=sersic_index, half_light_radius=hlr, gsparams=big_fft_params, flux=flux
    )

    e_tot = galsim.Shear(e1=e1, e2=e2)
    gal_true = gal.shear(e_tot)

    nx = gal_true.getGoodImageSize(pixel_scale=scale)
    bounds = galsim.BoundsI(0, nx - 1, 0, nx - 1)
    stamp = galsim.ImageF(bounds, scale=scale)

    gal_true.drawImage(stamp, center=galsim.PositionI(nx // 2, nx // 2))
    stamp.replaceNegative(replace_value=0)

    if flux < min_flux:
        isolated_stamp = 0
    else:
        isolated_stamp = process_stamp(stamp.array.copy(), NPIX=npix_stamp)

    return stamp, sersic_index, isolated_stamp


def simulate_wide_field(patch, NPIX_SKY=NPIX_SKY, **kwargs):
    verbosity = kwargs.get("verbosity", 0)
    simple = kwargs.get("simple", False)
    min_flux = kwargs.get("min_flux", 10e-6)
    npix_stamp = kwargs.get("npix_stamp", NPIX_STAMP)

    # Step 1: Initialize wide-field image
    bounds = galsim.BoundsI(0, NPIX_SKY - 1, 0, NPIX_SKY - 1)
    field = galsim.ImageF(bounds, scale=SCALE_ARCSEC)

    if verbosity > 0:
        print(
            f"Simulating wide-field of size {NPIX_SKY}x{NPIX_SKY} with {len(patch)} galaxies"
        )
        print(
            f"Intensity: [{np.min(patch['flux']) * 1e6:0.2f},{np.max(patch['flux']) * 1e6:0.2f}] uJy"
        )
        print(
            f"Scale length: [{np.min(patch['size']):0.2f},{np.max(patch['size']):0.2f}] arcsec"
        )

    # Step 2: Run dask to simulate galaxies in parallel
    def simulate_batch(batch, simple=simple, min_flux=min_flux, npix_stamp=npix_stamp):
        return [
            _simulate_galaxy(
                row, simple=simple, min_flux=min_flux, npix_stamp=npix_stamp
            )
            for row in batch
        ]

    cols = ["flux", "size", "e1", "e2"]
    rows = patch[cols].to_dict(orient="records")

    # chunk rows into groups of 100
    chunk_size = 100
    tasks = [
        delayed(simulate_batch)(
            rows[i : i + chunk_size],
            simple=simple,
            min_flux=min_flux,
            npix_stamp=npix_stamp,
        )
        for i in range(0, len(rows), chunk_size)
    ]

    # Compute results
    results = compute(*tasks)
    results = [r for batch in results for r in batch]

    # Extract sersic indexes
    patch["sersic_index"] = np.stack([r[1] for r in results])

    # Extract isolated stamps and corresponding flux mask
    mask = np.array([isinstance(row[-1], np.ndarray) for row in results])
    patch["flux_mask"] = mask
    isolated_stamps = np.stack(
        [row[-1] for row in results if isinstance(row[-1], np.ndarray)]
    )

    # Step 3: Add full galaxy stamps to wide-field image
    for result, cx, cy in zip(results, patch["pix_x"], patch["pix_y"]):
        stamp, *_ = result
        stamp_size = stamp.array.shape[0]
        half_nx = stamp_size // 2

        # Compute bounds for stamp placement in field
        x_min = cx - half_nx
        x_max = x_min + stamp_size - 1
        y_min = cy - half_nx
        y_max = y_min + stamp_size - 1

        # Create a new GalSim image with bounds at the desired field coordinates
        stamp_img = galsim.ImageF(galsim.BoundsI(x_min, x_max, y_min, y_max))

        # Copy the original stamp into the new image
        stamp_img.array[:, :] = stamp.array.copy()

        # Add to field
        bounds = stamp_img.bounds & field.bounds
        field[bounds] += stamp_img[bounds]

    sky_array = field.array.copy()

    return sky_array, patch, isolated_stamps
