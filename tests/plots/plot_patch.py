# %% imports
import h5py
import numpy as np

from deepshape2.utils import extract_image, load_config
from deepshape2.visualization import plot

# %%
cfg = load_config()

DATA_DIR = cfg["DATA_DIR"]
RESULTS_DIR = cfg["RESULTS_DIR"]

# %%

with h5py.File(DATA_DIR + "sky_50.h5", mode="r") as f:
    wide_patch = f["patch_001"]["sky"][()]

with h5py.File(DATA_DIR + "deep_set.h5", mode="r") as f:
    deep_patch = f["patch_000"]["sky"][()]

wide = np.array(extract_image(wide_patch, 8192), dtype=np.float32)
deep = np.array(extract_image(deep_patch, 8192), dtype=np.float32)

# %%
log_wide = np.log10(wide + 1e-9)
log_deep = np.log10(deep + 1e-9)

plot(
    [log_wide], size_fac=3, cbar=True, fname=RESULTS_DIR + "patch_image/wide_patch.pdf"
)
plot(
    [log_deep], size_fac=3, cbar=True, fname=RESULTS_DIR + "patch_image/deep_patch.pdf"
)
