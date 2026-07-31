# %% Load Modules
import warnings
from pathlib import Path

import astropy.units as u
import dask.array as da
import galsim
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

# from astropy.wcs import WCS
from dask import compute, delayed

from ..utils import load_config, process_stamp

warnings.filterwarnings("ignore", category=UserWarning)

__all__ = [
    "generate_patch_locations",
    "random_patch",
    "simulate_wide_field",
    "filter_patch_by_flux",
    "filter_patch_by_size",
    "compute_pixel_coordinates",
    "simulate_isolated_stamps",
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

DEFAULT_STAMP_HALF_WIDTH_PIXELS = 130

# Size of wide field in degrees in flat sky approximation along one axis
SKY_SIZE = 20
NPIX_SKY = cfg["NPIX_SKY"]
NPIX_STAMP = 128
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


def load_catalogue(
    catalogue: pd.DataFrame | str | Path | None = None,
    catalogue_type: str | None = None,
) -> pd.DataFrame:
    """Return a catalogue from an in-memory frame, a pickle path, or a type name."""
    if isinstance(catalogue, pd.DataFrame):
        return catalogue
    if catalogue is not None:
        return pd.read_pickle(catalogue)
    if catalogue_type is None:
        raise ValueError("Provide either 'catalogue' or 'catalogue_type'.")
    return pd.read_pickle(TRECS_DIR / f"catalog_{catalogue_type}.pkl")


def random_patch(
    center: tuple[float, float],
    catalogue: pd.DataFrame | str | Path | None = None,
    catalogue_type: str | None = None,
    patch_size_degrees: float = 1.0,
    pixel_scale_degrees: float = SCALE_DEGREES,
    stamp_half_width_pixels: int = DEFAULT_STAMP_HALF_WIDTH_PIXELS,
) -> pd.DataFrame:
    """Select galaxies inside a square patch of flat-sky coordinates.

    The patch is centred on ``center`` and spans ``patch_size_degrees`` on a
    side. Galaxies within ``stamp_half_width_pixels`` of an edge are excluded so
    that a full stamp can be drawn around every galaxy returned.

    Any pre-existing ``ra``/``dec`` columns are dropped, since they refer to the
    original catalogue projection rather than this patch.

    Raises:
        ValueError: if no catalogue source is given, or if the edge margin
            leaves no usable area inside the patch.
    """
    catalogue = load_catalogue(catalogue, catalogue_type)

    cx, cy = center
    x0, y0 = cx - patch_size_degrees / 2, cy - patch_size_degrees / 2
    x1, y1 = cx + patch_size_degrees / 2, cy + patch_size_degrees / 2

    # Select galaxies inside patch
    patch = catalogue[
        (catalogue["x"] >= x0)
        & (catalogue["x"] < x1)
        & (catalogue["y"] >= y0)
        & (catalogue["y"] < y1)
    ].copy()

    # Drop galaxies too close to patch edge
    margin = pixel_scale_degrees * 130
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


def compute_pixel_coordinates(
    patch, patch_center_flat, scale=SCALE_DEGREES, NPIX_SKY=NPIX_SKY
):
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
    dx = ((patch_out["x"] - cx) / scale).astype(int)
    dy = ((patch_out["y"] - cy) / scale).astype(int)

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
    row,
    simple=False,
    min_flux=10e-6,
    npix_stamp=NPIX_STAMP,
    scale=SCALE_ARCSEC,
    reduced_shear=False,
):
    """
    Simulate a single galaxy stamp using GalSim.
    Parameters
    ----------
    row : pd.Series
        Must have 'flux', 'size', 'e1', 'e2'. Optionally
        'sersic_index' for fixed Sersic index.
    simple : bool
        If True, force Sersic index = 1 (default False)
    min_flux : float
        Galaxies below this flux get no stamp (default 10e-6)
    npix_stamp : int
        Output stamp size (default NPIX_STAMP)
    scale : float
        Pixel scale in arcsec (default SCALE_ARCSEC)
    reduced_shear : bool
        If True, interpret e1/e2 as reduced shear g1/g2 instead of ellipticity distortion
        original deep_set /wide_set catlogue use g1/g2 while new 150 MHz catalogue uses e1/e2.
        This flag allows to switch between the two (default False)

    Returns
    -------
    stamp : galsim.ImageF
        The full galaxy stamp (may be larger than npix_stamp)
    sersic_index : float
        The Sersic index used for this galaxy
    isolated_stamp : np.ndarray or int
        The isolated stamp cropped to npix_stamp, or 0 if flux < min_flux
    """

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

    if reduced_shear:
        e_tot = galsim.Shear(g1=e1, g2=e2)
    else:
        e_tot = galsim.Shear(e1=e1, e2=e2)
    gal_true = gal.shear(e_tot)

    nx = gal_true.getGoodImageSize(pixel_scale=scale)
    nx = min(nx, 16384)  # limit to avoid memory issues with very large galaxies
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
    scale = kwargs.get("scale", SCALE_ARCSEC)
    stamps_only = kwargs.get("stamps_only", False)

    # Step 1: Initialize wide-field image
    bounds = galsim.BoundsI(0, NPIX_SKY - 1, 0, NPIX_SKY - 1)
    field = galsim.ImageF(bounds, scale=scale)

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
    def simulate_batch(
        batch, simple=simple, min_flux=min_flux, npix_stamp=npix_stamp, scale=scale
    ):
        return [
            _simulate_galaxy(
                row,
                simple=simple,
                min_flux=min_flux,
                npix_stamp=npix_stamp,
                scale=scale,
            )
            for row in batch
        ]

    cols = ["flux", "size", "e1", "e2"]
    rows = patch[cols].to_dict(orient="records")

    # chunk rows into groups of 250
    chunk_size = 128
    tasks = [
        delayed(simulate_batch)(
            rows[i : i + chunk_size],
            simple=simple,
            min_flux=min_flux,
            npix_stamp=npix_stamp,
            scale=scale,
        )
        for i in range(0, len(rows), chunk_size)
    ]

    # Compute results
    results = compute(*tasks)
    results = [r for batch in results for r in batch]

    if not stamps_only:
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

    if stamps_only:
        # Blended stamps — dask crops from sky, only for bright sources
        bright_mask = np.array([isinstance(row[-1], np.ndarray) for row in results])
        bright_cx = patch["pix_x"][bright_mask].values
        bright_cy = patch["pix_y"][bright_mask].values

        half = npix_stamp // 2
        darr = da.from_array(sky_array, chunks=(5000, 5000))

        crops = [
            darr[cy - half : cy + half, cx - half : cx + half]
            for cx, cy in zip(bright_cx, bright_cy)
        ]
        blended_stamps = np.stack(da.compute(*crops))

        return isolated_stamps, blended_stamps

    return sky_array, patch, isolated_stamps


# %% Isolated stamp simulation functions


def _isolated_only(
    row, simple=False, min_flux=50e-6, npix_stamp=NPIX_STAMP, scale=SCALE_ARCSEC
):

    _, sersic_index, isolated_stamp = _simulate_galaxy(
        row,
        simple=simple,
        min_flux=min_flux,
        npix_stamp=npix_stamp,
        scale=scale,
    )
    return sersic_index, isolated_stamp


def simulate_isolated_stamps(patch, **kwargs):
    """
    Simulate isolated (unblended) postage stamps for every galaxy in `patch`,
    in parallel with dask. No wide-field image is built.

    Uses exactly the same drawing code as simulate_wide_field
    (_simulate_galaxy -> process_stamp), so the stamps are pixel-consistent
    with the ones produced for the blended dataset.

    Parameters
    ----------
    patch : pd.DataFrame
        Needs columns 'flux', 'size', 'e1', 'e2'. If a 'sersic_index' column
        is present it is used, otherwise one is drawn at random per galaxy
        (as in simulate_wide_field). No positional columns are required.

    Keyword Arguments
    -----------------
    simple : bool          force n_sersic = 1 (default False)
    min_flux : float       galaxies below this flux get no stamp (default 50e-6)
    npix_stamp : int       output stamp size (default NPIX_STAMP)
    scale : float          pixel scale in arcsec (default SCALE_ARCSEC)
    chunk_size : int       galaxies per dask task (default 128)
    verbosity : int

    Returns
    -------
    isolated_stamps : np.ndarray, shape (n_kept, npix_stamp, npix_stamp)
    patch_out : pd.DataFrame
        The rows of `patch` that produced a stamp, in the same order as
        `isolated_stamps`, with the realised 'sersic_index' filled in.
    """
    verbosity = kwargs.get("verbosity", 0)
    simple = kwargs.get("simple", False)
    min_flux = kwargs.get("min_flux", 50e-6)
    npix_stamp = kwargs.get("npix_stamp", NPIX_STAMP)
    scale = kwargs.get("scale", SCALE_ARCSEC)
    chunk_size = kwargs.get("chunk_size", 128)

    if verbosity > 0:
        print(f"Simulating {len(patch)} isolated stamps of size {npix_stamp}")
        print(
            f"Intensity: [{np.min(patch['flux']) * 1e6:0.2f},"
            f"{np.max(patch['flux']) * 1e6:0.2f}] uJy"
        )
        print(
            f"Scale length: [{np.min(patch['size']):0.2f},"
            f"{np.max(patch['size']):0.2f}] arcsec"
        )

    def simulate_batch(batch):
        return [
            _isolated_only(
                row,
                simple=simple,
                min_flux=min_flux,
                npix_stamp=npix_stamp,
                scale=scale,
            )
            for row in batch
        ]

    cols = ["flux", "size", "e1", "e2"]
    if "sersic_index" in patch.columns:
        cols.append("sersic_index")
    rows = patch[cols].to_dict(orient="records")

    tasks = [
        delayed(simulate_batch)(rows[i : i + chunk_size])
        for i in range(0, len(rows), chunk_size)
    ]

    results = compute(*tasks)
    results = [r for batch in results for r in batch]

    patch_out = patch.copy()
    patch_out["sersic_index"] = np.array([r[0] for r in results])
    mask = np.array([isinstance(r[-1], np.ndarray) for r in results])
    patch_out["flux_mask"] = mask

    kept = [r[-1] for r in results if isinstance(r[-1], np.ndarray)]
    if len(kept) == 0:
        isolated_stamps = np.empty((0, npix_stamp, npix_stamp), dtype="float32")
    else:
        isolated_stamps = np.stack(kept)

    return isolated_stamps, patch_out[mask].copy()
