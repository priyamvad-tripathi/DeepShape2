# %% Import Libraries
import os
import time

from deepshape2.simulation import simulate_visibilities
from deepshape2.utils import load_config, load_h5, post_step

# %% Set default parameters
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]

patches = [f"patch_{nl + 1:03d}" for nl in range(71, 100)]

dirty_wide = {}
start = time.time()

# %% Loop over patches and simulate visibilities
h5_path = DATA_DIR + "wide_set.h5"
with load_h5(h5_path, "r") as data:
    for patch in patches:
        # Construct HDF5 path and visibility filename

        vis_filename = DATA_DIR + f"MS/vis_wide_set_{patch}.ms"

        if os.path.exists(vis_filename):
            print(f"Visibility file for {patch} already exists. Skipping...")
            continue

        post_step(f"Loading {patch} from wide_set", start)

        # Load HDF5 data

        sky = data[patch]["sky"][()]
        patch_ra, patch_dec = data[patch].attrs["centre"]

        # Simulate visibilities and create dirty image
        vt = simulate_visibilities(
            field=sky,
            ra_pointing=patch_ra,
            dec_pointing=patch_dec,
            filename=vis_filename,
            # create_dirty=True,
            threads=60,
        )

        post_step(f"Simulating visibilities for {patch}", start)
