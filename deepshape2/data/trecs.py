# %%
import copy

import numpy as np
import pandas as pd
from astropy.io import fits
from deepshape2.utils import load_config

# Directory containing the fits files
cfg = load_config()
FITS_DIR = cfg["TRECS_dir"]

"""
To download the files:
wget -O SFG_deep.fits "https://cdsarc.u-strasbg.fr/ftp/VII/282/fits/catalogue_SFGs_complete_deep.fits"
"""


# %% Functions
def e1e2_to_g1g2(e1, e2):
    """Function to convert ellipticity into correct format"""
    e = np.sqrt(e1**2 + e2**2)

    cos_2t = e1 / e
    sin_2t = e2 / e

    r = np.sqrt((1 - e) / (1 + e))

    g_new = (1 - r) / (1 + r)

    g1 = g_new * cos_2t
    g2 = g_new * sin_2t

    return np.nan_to_num(g1), np.nan_to_num(g2)


def ra_min_max(ra_deg):
    ra_deg = np.mod(ra_deg, 360)  # ensure RA in [0,360)
    ra_sorted = np.sort(ra_deg)
    # differences between consecutive RAs (including wrap)
    diffs = np.diff(np.concatenate([ra_sorted, [ra_sorted[0] + 360]]))
    max_gap_idx = np.argmax(diffs)

    # minimal interval is complement of largest gap
    ra_min = ra_sorted[(max_gap_idx + 1) % len(ra_sorted)]
    ra_max = ra_sorted[max_gap_idx]

    # ensure ra_min < ra_max (if crossing 0, adjust)
    if ra_max < ra_min:
        ra_max += 360

    return ra_min, ra_max


# %% Convert the fits file to a pandas DataFrame for easy access


names = ["medium", "deep"]

for name in names:
    catalog = fits.open(FITS_DIR + f"SFG_{name}.fits")

    cat1 = catalog[1]  # Selecting the first slice of the FITS ("Catalogue")
    data = cat1.data  # Extract data from slice

    flux = copy.deepcopy(data["I1400"]) * 1e-3  # Flux density at 1400 MHz (in Janskys)
    size = copy.deepcopy(data["size"])  # angular size on the sky (in arcsec)
    ra = copy.deepcopy(data["longitude"])
    dec = copy.deepcopy(data["latitude"])

    x = copy.deepcopy(data["x_coord"])
    y = copy.deepcopy(data["y_coord"])

    e1 = copy.deepcopy(data["e1"])
    e2 = copy.deepcopy(data["e2"])

    g1, g2 = e1e2_to_g1g2(e1, e2)

    catalog = {
        "flux": np.array(flux),
        "size": np.array(size),
        "ra": np.array(ra),
        "dec": np.array(dec),
        "e1": copy.deepcopy(np.array(g1)),
        "e2": copy.deepcopy(np.array(g2)),
        "x": np.array(x),
        "y": np.array(y),
    }

    df = pd.DataFrame.from_dict(catalog)
    df.to_pickle(FITS_DIR + f"catalog_{name}.pkl")


# %% Do the same for wide field
all_data = []

for i in range(1, 11):
    catalog = fits.open(FITS_DIR + f"SFG_wide_{i}.fits")
    cat1 = catalog[1]  # first extension
    data = cat1.data

    flux = copy.deepcopy(data["I1400"]) * 1e-3  # Jy
    size = copy.deepcopy(data["size"])
    ra = copy.deepcopy(data["longitude"])
    dec = copy.deepcopy(data["latitude"])
    x = copy.deepcopy(data["x_coord"])
    y = copy.deepcopy(data["y_coord"])
    e1 = copy.deepcopy(data["e1"])
    e2 = copy.deepcopy(data["e2"])

    g1, g2 = e1e2_to_g1g2(e1, e2)

    catalog_dict = {
        "flux": np.array(flux),
        "size": np.array(size),
        "ra": np.array(ra),
        "dec": np.array(dec),
        "e1": np.array(g1),
        "e2": np.array(g2),
        "x": np.array(x),
        "y": np.array(y),
    }
    all_data.append(pd.DataFrame.from_dict(catalog_dict))

# concatenate all DataFrames vertically
combined_df = pd.concat(all_data, ignore_index=True)

# save as one pickle
combined_df.to_pickle(FITS_DIR + "catalog_wide.pkl")
