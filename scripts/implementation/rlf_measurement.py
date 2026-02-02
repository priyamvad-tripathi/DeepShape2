# %% Imports
import subprocess
from time import time

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_func_python.util.coordinate_support import skycoord_to_lmn

from deepshape2.utils import load_config, load_h5, time_string

# %% Constants / Configuration
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]
RLF_DIR = cfg["RLF_DIR"]
SCALE_RADIANS = cfg["SCALE_RADIANS"]
NPIX_SKY = cfg["NPIX_SKY"]

# %% Save catalog for RLF Measurements
data = load_h5(DATA_DIR + "deep_set.h5")
patch_df = pd.DataFrame.from_records(data["patch_000"]["patch_df"][()])
flux_mask = patch_df["flux_mask"].values


# Extract relevant columns for catalog
flux = patch_df["flux"][flux_mask].values
pix_x = patch_df["pix_x"][flux_mask].values
pix_y = patch_df["pix_y"][flux_mask].values
size = patch_df["size"][flux_mask].values

e1 = patch_df["e1"][flux_mask].values
e2 = patch_df["e2"][flux_mask].values


peak = data["patch_000/isolated_stamps"][:].max(axis=(1, 2))
# %% Convert pixel coordinates to l,m
phase_centre = data["patch_000"].attrs["centre"]
phase_centre_skycoord = SkyCoord(
    ra=phase_centre[0] * u.deg,
    dec=phase_centre[1] * u.deg,
    frame="icrs",
    equinox="J2000",
)

dx = pix_x - NPIX_SKY // 2.0
dy = pix_y - NPIX_SKY // 2.0


offset_ra = -dx * SCALE_RADIANS * u.rad
offset_dec = dy * SCALE_RADIANS * u.rad

galaxy_centres = phase_centre_skycoord.spherical_offsets_by(offset_ra, offset_dec)

gal_l, gal_m, _ = skycoord_to_lmn(galaxy_centres, phase_centre_skycoord)

# %% Write into catalog file

# subset_indices = np.where(flux > 50e-6)[0]

# # Randomly choose 10 indices from that subset
# idx1 = np.random.choice(subset_indices, size=100, replace=False)
# idx = np.argsort(flux[idx1])[::-1]

idx = np.argsort(flux)[::-1]

size_sorted = size[idx]
l_sorted = gal_l[idx]
m_sorted = gal_m[idx]
flux_sorted = flux[idx]
e1_sorted = e1[idx]
e2_sorted = e2[idx]

with open(DATA_DIR + "catalog.txt", "w") as f:
    for la, ma, fl, sz, ee1, ee2 in zip(
        l_sorted, m_sorted, flux_sorted, size_sorted, e1_sorted, e2_sorted
    ):
        line = f"{fl * 1e8:.02f} {la} {ma} {fl * 1e6:.06f} {sz:.04f} {ee1:0.04f} {ee2:0.04f}\n"
        f.write(line)

# %% Run RadioLensfit2 externally
start = time()
subprocess.run(
    [
        RLF_DIR + "RadioLensfit2",
        DATA_DIR + "catalog.txt",
        f"{len(idx)}",
        DATA_DIR + "MS/vis_deep_set_patch_000.ms",
    ],
    cwd=DATA_DIR,
)

cols = ["flux", "e1", "m_e1", "err1", "e2", "m_e2", "err2", "1D var", "SNR", "l", "m"]

df = pd.read_csv(
    DATA_DIR + "results.txt",
    sep="|",
    engine="python",
    skipinitialspace=True,
    names=cols,
    header=0,
    na_values=["nan", "-nan"],
)

df = df.apply(pd.to_numeric, errors="coerce")

end = time()

df["m_e2"] *= -1
df["flag"] = (df["m_e1"] == 0) & (df["m_e2"] == 0)

print(f"RadioLensfit2 run completed in {time_string(end - start)}")

df
