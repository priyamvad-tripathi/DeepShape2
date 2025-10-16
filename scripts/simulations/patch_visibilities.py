# %% Import Libraries
import time

from deepshape2.simulation import simulate_visibilities
from deepshape2.utils import load_config, load_h5, post_step, save

# %% Set default parameters
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]

patches = {"wide_set": "patch_051", "deep_set": "patch_000"}

dirty_wide = {}
start = time.time()

# %% Loop over patches and simulate visibilities
for dataset_type, patch in patches.items():
    # Construct HDF5 path and visibility filename
    h5_path = DATA_DIR + f"{dataset_type}.h5"
    vis_filename = DATA_DIR + f"vis_{dataset_type}_{patch}.ms"

    post_step(f"Loading {patch} from {dataset_type}", start)

    # Load HDF5 data
    with load_h5(h5_path, "r") as data:
        sky = data[patch]["sky"][()]
        patch_ra, patch_dec = data[patch].attrs["centre"]

        # Simulate visibilities and create dirty image
        vt, dirty = simulate_visibilities(
            field=sky,
            ra_pointing=patch_ra,
            dec_pointing=patch_dec,
            filename=vis_filename,
            create_dirty=True,
        )

        post_step(f"Simulating visibilities for {patch} ({dataset_type})", start)

        # Store dirty image with clear key
        dirty_wide[f"{dataset_type}_{patch}"] = dirty

# %% Save all dirty images
save(dirty_wide, DATA_DIR + "dirty_image_patches.h5")
post_step("Saving all dirty images", start)
