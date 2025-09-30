# %% Load Modules
import astropy.units as u
import galsim
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from tqdm import tqdm

from deepshape2.utils import extract_image, load_config
from deepshape2.visualization import plot

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


# %% Catalogue Functions
def random_patch(catalogue_type="wide", patch_size=1.0, seed=43):
    catalogue = pd.read_pickle(f"/scratch/tripathi/TRECS/catalog_{catalogue_type}.pkl")

    if seed is not None:
        np.random.seed(seed)

    # Randomly choose patch lower-left corner
    x_min, x_max = catalogue["x"].min(), catalogue["x"].max()
    y_min, y_max = catalogue["y"].min(), catalogue["y"].max()
    rand_x0 = np.random.uniform(x_min, x_max - patch_size)
    rand_y0 = np.random.uniform(y_min, y_max - patch_size)
    x0, x1 = rand_x0, rand_x0 + patch_size
    y0, y1 = rand_y0, rand_y0 + patch_size

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

    return patch, (centre_ra, centre_dec), (x0, y0)


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


def simulate_galaxy(row, simple=False):
    flux = row.flux
    scale_length = row.size
    e1 = row.e1
    e2 = row.e2
    pos = galsim.PositionI(row.pix_x, row.pix_y)

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

    return stamp, sersic_index


def simulate_wide_field(patch, bottom_left, **kwargs):
    verbosity = kwargs.get("verbosity", 0)
    simple = kwargs.get("simple", False)
    NPIX_SKY = kwargs.get("NPIX_SKY", 25200)

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

    # Step 3: Loop over galaxies
    sersic_index_all = []
    isolated_stamps_all = []

    with tqdm(
        total=len(patch_out),
        unit="gal",
        desc="Simulating gal stamps",
        colour="green",
    ) as pbar:
        for row in patch_out.itertuples(index=False):
            pbar.update(1)
            stamp, sersic_index = simulate_galaxy(row, simple=simple)

            bounds = stamp.bounds & field.bounds
            field[bounds] += stamp[bounds]

            isolated_stamps_all.append(stamp.array.copy())
            sersic_index_all.append(sersic_index)

    sky_array = field.array.copy()
    patch_out["sersic_index"] = sersic_index_all

    return sky_array, patch_out, isolated_stamps_all


# %% Test the wide-field simulation
if __name__ == "__main__":
    patch, centre, bottom_left = random_patch()
    patch = filter_patch_by_size(filter_patch_by_flux(patch))

    patch_sample = patch.sample(frac=0.05, random_state=1)

    sky_array, patch_out, isolated_stamps = simulate_wide_field(
        patch_sample, bottom_left, simple=True, verbosity=1
    )

    plot([np.log10(sky_array + 1e-9)], size_fac=3)
    sky_patch = extract_image(sky_array, 8192)
    plot([np.log10(sky_patch + 1e-9)], size_fac=3)
