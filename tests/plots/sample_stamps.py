# %%
import h5py
import numpy as np
import pandas as pd

from deepshape2.utils import blendedness, extract_image, load_config
from deepshape2.visualization import plot

# %%
cfg = load_config()

DATA_DIR = cfg["DATA_DIR"]
RESULTS_DIR = cfg["RESULTS_DIR"]

# %%
with h5py.File(DATA_DIR + "deep_set.h5", mode="r") as f:
    blends = extract_image(f["patch_000"]["blended_stamps"][:], 128)
    isolated = extract_image(f["patch_000"]["isolated_stamps"][:], 128)

    patch_df = pd.DataFrame.from_records(f["patch_000"]["patch_df"][()])
    fluxes = patch_df["flux"].values[patch_df["flux_mask"]]


# %% Compute blendedness
blendedness = blendedness(isolated, blends)

# %%
percentiles = [10, 80, 90, 99]
values = np.percentile(blendedness, percentiles)
indices = [np.argmin(np.abs(blendedness - v)) for v in values]

# %%


tit1 = [rf"{fl * 1e6:.2f} $\mu$Jy" for fl in fluxes[indices]]
tit2 = [f"{bl:.3f}" for bl in blendedness[indices]]


plot(
    [isolated[indices], blends[indices]],
    cbar=True,
    same_scale=[0, 1],
    scale_row=0,
    caption=["Isolated", "Blended"],
    subtitles=[tit1, tit2],
    fname=RESULTS_DIR + "stamps.pdf",
)
