# %%
import numpy as np
import pandas as pd

from deepshape2.utils import extract_image, load_config, load_h5
from deepshape2.visualization import plot

# %% Configuration

cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]
RESULTS_DIR = cfg["RESULTS_DIR"]

DATASETS = {
    "iso": "deep_set.h5",
    "blend": "deep_set2.h5",
}

N_SHOW = 10


# %% Load reference dataset for selection
ref_data = load_h5(DATA_DIR + DATASETS["iso"], "r")
ref_patch = ref_data["patch_000"]
patch_df = pd.DataFrame.from_records(ref_patch["patch_df"][()])

flux_mask = patch_df["flux_mask"].values
flux = patch_df["flux"].values[flux_mask]

blendedness = ref_patch["blend"][:]

mask = (blendedness > 0.05) & (flux > 30e-6)
valid_inds = np.where(mask)[0]
# inds = np.sort(np.random.choice(valid_inds, size=min(N_SHOW, len(valid_inds)), replace=False))
inds = np.sort([11816, 1301, 5684, 14023, 663, 2411, 7566, 182, 2586, 3967])
print("Selected indices:", inds)


# %% Collect images and print metrics
images = []
subtitles = []
captions = []

iso_stamps = extract_image(ref_patch["isolated_stamps"][inds])
blend_stamps = extract_image(ref_patch["blended_stamps"][inds])
dirty_stmaps = ref_patch["dirty"][inds]

flux_str = [f"{f * 1e6:.2f} µJy" for f in flux[inds]]
blendedness_str = [f"{b:.3f}" for b in blendedness[inds]]

images.extend([iso_stamps, blend_stamps, dirty_stmaps])
captions.extend(["Iso", "Blend", "Dirty"])
subtitles.extend([flux_str, blendedness_str, [None] * len(inds)])

for name, fname in DATASETS.items():
    data = load_h5(DATA_DIR + fname, "r")
    patch = data["patch_000"]

    decon = patch["decon"][inds]
    recon = patch["recon"][inds]

    psnr_vals = patch["psnr"][:]
    shape_diff = patch["shape_diff"][:]

    # print(
    #     f"{name} Shape: Max {np.nanmax(shape_diff):.3f} | "
    #     f"Min {np.nanmin(shape_diff):.3f} | "
    #     f"Mean {np.nanmean(shape_diff):.3f}"
    # )
    # print(
    #     f"{name} PSNR: Max {psnr_vals.max():.3f} dB | "
    #     f"Min {psnr_vals.min():.3f} dB | "
    #     f"Mean {psnr_vals.mean():.3f} dB"
    # )
    #     shape_p25, shape_med, shape_p75 = np.nanpercentile(shape_diff, [25, 50, 75])
    #     psnr_p25, psnr_med, psnr_p75 = np.nanpercentile(psnr_vals, [25, 50, 75])
    #     print(
    #     f"{name} Shape: {shape_med:.3f} ± "
    #     f"{(shape_p75 - shape_p25)/2:.3f}"
    # )
    #     print(
    #     f"{name} PSNR: {psnr_med:.3f} ± "
    #     f"{(psnr_p75 - psnr_p25)/2:.3f}"
    # )

    metric_str = [
        f"{p:.02f} dB / {sd:.03f}" for p, sd in zip(psnr_vals[inds], shape_diff[inds])
    ]

    images.extend([decon, recon])
    captions.extend([f"{name} Decon", f"{name} Recon"])
    subtitles.extend([[None] * len(inds), metric_str])


# --------------------------------------------------
# Plot
# --------------------------------------------------
plot(
    images=images,
    caption=captions,
    subtitles=subtitles,
    cbar=True,
    scale_row=0,
    same_scale=[0, 1, 3, 4, 5, 6],
)
