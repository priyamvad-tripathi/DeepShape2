# %% Import Libraries
import time

from deepshape2.simulation import simulate_visibilities
from deepshape2.utils import load_config, load_h5, post_step

import os

# %% Set default parameters
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]

INCLUDE_DEEP_SET = True  # Set to False to skip deep set simulation

start = time.time()


# %% Deep set: single patch_000
if INCLUDE_DEEP_SET:
    h5_path = DATA_DIR + "deep_set_new.h5"
    patch = "patch_000"
    vis_filename = DATA_DIR + f"MS/vis_deep_set_{patch}.ms"
    

    if os.path.exists(vis_filename):
        print(f"Visibilities for {patch} already exist. Skipping simulation.")
    else:
        with load_h5(h5_path, "r") as data:
        
            sky = data[patch]["sky"][()]
            patch_ra, patch_dec = data[patch].attrs["centre"]
            
            post_step(f"loading deep {patch}", start)
            
            simulate_visibilities(
                field=sky,
                ra_pointing=patch_ra,
                dec_pointing=patch_dec,
                filename=vis_filename,
                threads=60,
            )
            post_step(f"simulating deep {patch}", start)


# %% Wide set: patches 51-100
wide_patches = [f"patch_{nl + 1:03d}" for nl in range(50, 100)]
h5_path = DATA_DIR + "wide_set_new.h5"

with load_h5(h5_path, "r") as data:
    for i, patch in enumerate(wide_patches):

        sky = data[patch]["sky"][()]
        patch_ra, patch_dec = data[patch].attrs["centre"]
        vis_filename = DATA_DIR + f"MS/vis_wide_set_{patch}.ms"

        if os.path.exists(vis_filename):
            print(f"Visibilities for {patch} already exist. Skipping simulation.")
            continue
        
        post_step(f"loading wide {i+1}/{len(wide_patches)}: {patch}", start)
        simulate_visibilities(
            field=sky,
            ra_pointing=patch_ra,
            dec_pointing=patch_dec,
            filename=vis_filename,
            threads=60,
        )
        post_step(f"simulating wide {i+1}/{len(wide_patches)}: {patch}", start)


post_step("All done", start)