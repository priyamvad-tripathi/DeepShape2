# %% Load Modules
import argparse
import time

import astropy.units as u
import galsim
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from colorist import Color
from dask import compute, delayed
from dask.distributed import Client, LocalCluster

from deepshape2.utils import load_config, load_h5, post_step, process_stamp


# %% Set up argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Sky simulation arguments")

    parser.add_argument(
        "-w",
        "--n-workers",
        type=int,
        default=50,
        help="Number of Dask workers to use (default: 50)",
    )

    parser.add_argument(
        "-nf",
        "--n-patches",
        type=int,
        default=50,
        help="Number of simulated patches (each 1 deg2 wide) (default: 50)",
    )

    return parser.parse_args()


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

# Sky centre
RA0 = cfg["RA0"]
DEC0 = cfg["DEC0"]

origin = SkyCoord(ra=RA0 * u.deg, dec=DEC0 * u.deg, frame="icrs")
offset_frame = SkyOffsetFrame(origin=origin)

# Size of wide field in degrees in flat sky approximation along one axis
SKY_SIZE = 20
NPIX_SKY = cfg["NPIX_SKY"]

DATA_DIR = cfg["DATA_DIR"]

NPIX_STAMP = 256
# %% Catalogue Functions


def generate_patch_locations(
    sky_size=SKY_SIZE, patch_size=1.0, max_patches=100, seed=43
):
    """Pre-select non-overlapping random patch locations."""
    if seed is not None:
        np.random.seed(seed)

    x_min, x_max = -sky_size / 2, sky_size / 2
    y_min, y_max = -sky_size / 2, sky_size / 2

    chosen_regions = []
    locations = []

    attempts = 0
    while len(chosen_regions) < max_patches and attempts < max_patches * 10:
        rand_x0 = np.random.uniform(x_min, x_max - patch_size)
        rand_y0 = np.random.uniform(y_min, y_max - patch_size)
        x1, y1 = rand_x0 + patch_size, rand_y0 + patch_size

        # Check overlap
        overlap = any(
            (
                rand_x0 < xr + patch_size
                and x1 > xr
                and rand_y0 < yr + patch_size
                and y1 > yr
            )
            for xr, yr in chosen_regions
        )
        if not overlap:
            chosen_regions.append((rand_x0, rand_y0))
            locations.append((rand_x0, rand_y0))
        attempts += 1

    if len(chosen_regions) < max_patches:
        print(f"Warning: Only {len(chosen_regions)} non-overlapping patches found.")

    return locations


def random_patch(location, catalogue_type="wide", patch_size=1.0):
    catalogue = pd.read_pickle(f"/scratch/tripathi/TRECS/catalog_{catalogue_type}.pkl")

    # Choose patch location
    x0, y0 = location
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    # Compute patch centre in flat-sky coords
    cx = x0 + patch_size / 2.0
    cy = y0 + patch_size / 2.0

    # Select galaxies inside patch
    patch = catalogue[
        (catalogue["x"] >= x0)
        & (catalogue["x"] < x1)
        & (catalogue["y"] >= y0)
        & (catalogue["y"] < y1)
    ].copy()

    #  Drop galaxies which are too close to the edge
    mask = (
        (patch["x"] - SCALE_DEGREES * 128 >= x0)
        & (patch["x"] + SCALE_DEGREES * 128 <= x1)
        & (patch["y"] - SCALE_DEGREES * 128 >= y0)
        & (patch["y"] + SCALE_DEGREES * 128 <= y1)
    )
    patch = patch[mask].copy()

    # Drop old RA/Dec columns if present
    for col in ["ra", "dec"]:
        if col in patch.columns:
            patch.drop(columns=col, inplace=True)

    # x,y are offsets in degrees relative to RA0,Dec0
    offset_coords = SkyCoord(
        lon=patch["x"].values * u.deg, lat=patch["y"].values * u.deg, frame=offset_frame
    )

    # Convert back to ICRS RA,Dec for galaxies
    icrs_coords = offset_coords.icrs
    patch["RA"] = icrs_coords.ra.deg
    patch["Dec"] = icrs_coords.dec.deg

    # Convert patch centre to RA,Dec
    centre_offset = SkyCoord(lon=cx * u.deg, lat=cy * u.deg, frame=offset_frame)
    centre_icrs = centre_offset.icrs
    centre_ra = centre_icrs.ra.deg
    centre_dec = centre_icrs.dec.deg

    return patch, (centre_ra, centre_dec)


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


# %% Galaxy Image Simulation
def compute_pixel_coordinates(patch, bottom_left):
    x0, y0 = bottom_left
    patch_out = patch.copy()

    # Integer pixel positions relative to bottom-left corner
    patch_out["pix_x"] = ((patch_out["x"] - x0) / SCALE_DEGREES).astype(int)
    patch_out["pix_y"] = ((patch_out["y"] - y0) / SCALE_DEGREES).astype(int)

    # Convert pixel positions back to RA/Dec
    pix_x_deg = x0 + patch_out["pix_x"].values * SCALE_DEGREES
    pix_y_deg = y0 + patch_out["pix_y"].values * SCALE_DEGREES
    offset_coords = SkyCoord(
        lon=pix_x_deg * u.deg,
        lat=pix_y_deg * u.deg,
        frame=offset_frame,  # Must be defined externally as the same origin
    )
    icrs_coords = offset_coords.icrs
    patch_out["RA_pix"] = icrs_coords.ra.deg
    patch_out["Dec_pix"] = icrs_coords.dec.deg

    return patch_out


def simulate_galaxy(row, simple=False, min_flux=10e-6):
    flux = row["flux"]
    scale_length = row["size"]
    e1 = row["e1"]
    e2 = row["e2"]
    pos = galsim.PositionI(row["pix_x"], row["pix_y"])

    if simple:
        sersic_index = 1
    else:
        sersic_index = np.random.choice(sersic_indexes)

    hlr = scale_length * 1.6783469900166605
    gal = galsim.Sersic(
        n=sersic_index, half_light_radius=hlr, gsparams=big_fft_params, flux=flux
    )

    e_tot = galsim.Shear(g1=e1, g2=e2)
    gal_true = gal.shear(e_tot)

    nx = gal_true.getGoodImageSize(pixel_scale=SCALE_ARCSEC)

    stamp = gal_true.drawImage(
        nx=nx,
        ny=nx,
        scale=SCALE_ARCSEC,
        center=pos,
    )
    stamp.replaceNegative(replace_value=0)

    if flux < min_flux:
        isolated_stamp = 0
    else:
        isolated_stamp = process_stamp(stamp.array.copy(), NPIX=NPIX_STAMP)

    return stamp, sersic_index, isolated_stamp


def simulate_wide_field(patch, bottom_left, NPIX_SKY=NPIX_SKY, **kwargs):
    verbosity = kwargs.get("verbosity", 0)
    simple = kwargs.get("simple", False)
    min_flux = kwargs.get("min_flux", 10e-6)

    # Step 1: Compute pixel coordinates & RA/Dec
    patch_out = compute_pixel_coordinates(patch, bottom_left)

    # Step 2: Initialize wide-field image
    field = galsim.ImageF(NPIX_SKY, NPIX_SKY, scale=SCALE_ARCSEC)

    if verbosity > 0:
        print(
            f"Simulating wide-field of size {NPIX_SKY}x{NPIX_SKY} with {len(patch_out)} galaxies"
        )
        print(
            f"Intensity: [{np.min(patch_out['flux']) * 1e6:0.2f},{np.max(patch_out['flux']) * 1e6:0.2f}] uJy"
        )
        print(
            f"Scale length: [{np.min(patch_out['size']):0.2f},{np.max(patch_out['size']):0.2f}] arcsec"
        )

    # Step 1: Run dask to simulate galaxies in parallel
    def simulate_batch(batch, simple=simple, min_flux=min_flux):
        return [simulate_galaxy(row, simple=simple, min_flux=min_flux) for row in batch]

    rows = patch_out.to_dict(orient="records")

    # chunk rows into groups of 100
    chunk_size = 100
    tasks = [
        delayed(simulate_batch)(rows[i : i + chunk_size], simple=simple)
        for i in range(0, len(rows), chunk_size)
    ]

    # Compute results
    results = compute(*tasks)
    results = [r for batch in results for r in batch]

    # Extract sersic indexes
    patch_out["sersic_index"] = np.stack([r[1] for r in results])

    # Extract isolated stamps and corresponding flux mask
    mask = np.array([isinstance(row[-1], np.ndarray) for row in results])
    patch_out["flux_mask"] = mask
    isolated_stamps = np.stack(
        [row[-1] for row in results if isinstance(row[-1], np.ndarray)]
    )

    # Step 3: Add full galaxy stamps to wide-field image
    for stamp, *_ in results:
        bounds = stamp.bounds & field.bounds
        field[bounds] += stamp[bounds]

    sky_array = field.array.copy()

    return sky_array, patch_out, isolated_stamps


# %% Test the wide-field simulation
if __name__ == "__main__":
    args = parse_args()

    NUM_DASK_WORKERS = args.n_workers
    NUM_PATCHES = args.n_patches
    MIN_FLUX = 10e-6  # Min flux in Jy for extracting stamps

    data = load_h5(DATA_DIR + "sky.h5", mode="a", delete_if_exists=True)
    data.attrs["min_flux_for_stamps"] = MIN_FLUX

    start = time.time()
    with (
        LocalCluster(
            n_workers=NUM_DASK_WORKERS,
            processes=True,
            threads_per_worker=1,
            scheduler_port=8786,
            memory_limit=0,
        ) as cluster,
        Client(cluster) as client,
    ):
        print(client.dashboard_link)

        locations = generate_patch_locations()[:NUM_PATCHES]

        for nl, location in enumerate(locations):
            print(
                f"{Color.GREEN}Simulating patch {nl + 1}/{NUM_PATCHES} at location ({location[0]:.3f}, {location[1]:.3f}){Color.OFF}"
            )

            group = data.create_group(f"patch_{nl + 1:03d}")

            # Extract galaxies at patch location
            patch, centre = random_patch(location)
            patch = filter_patch_by_size(filter_patch_by_flux(patch))
            print(f"Number of galaxies in patch: {len(patch)}")

            # Simulate wide-field image of the patch
            sky_array, patch_out, isolated_stamps = simulate_wide_field(
                patch, location, min_flux=MIN_FLUX
            )

            print(
                f"Number of bright galaxies (flux>=10uJy): {patch_out['flux_mask'].sum()}"
            )

            # Save results
            patch_rec = patch_out.to_records(index=False)
            group.create_dataset("patch_df", data=patch_rec, compression="gzip")

            group.create_dataset(
                "sky", data=sky_array, compression="gzip", chunks=(1024, 1024)
            )
            group.attrs["centre"] = centre

            group.create_dataset(
                "isolated_stamps",
                data=isolated_stamps,
                compression="gzip",
                chunks=(1, 256, 256),
            )

            post_step("field image simulation", start, client, data)

    data.close()
