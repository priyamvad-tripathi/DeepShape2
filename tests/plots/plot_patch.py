# %% imports
import h5py
import numpy as np

from deepshape2.utils import extract_image, load_config
from deepshape2.visualization import plot_log_image

# %%
cfg = load_config()

DATA_DIR = cfg["DATA_DIR"]
RESULTS_DIR = cfg["RESULTS_DIR"]

# %%

with h5py.File(DATA_DIR + "wide_set.h5", mode="r") as f:
    wide_patch = f["patch_010"]["sky"][()]


wide = np.array(extract_image(wide_patch, 8192), dtype=np.float32)

# %%
plot_log_image(wide, fname=RESULTS_DIR + "wide_patch.pdf", remove_bg=False)
