# %%
import time

import h5py
import numpy as np
from numpy.fft import ifftshift, irfftn, rfftn

from deepshape2.utils import blendedness, extract_image, load_config, time_string

# %% Config Parameters


cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]

INP_PATH = DATA_DIR + "wide_set.h5"
OUT_PATH = DATA_DIR + "trainset.h5"

all_groups = group_names = [f"patch_{nl + 1:03d}" for nl in range(71, 100)]
# %% Main Function

start = time.time()

images = []
shape_all = []
peaks = []
blend_all = []


with (
    h5py.File(INP_PATH, "r") as hf_in,
    h5py.File(OUT_PATH, "a") as hf_out,
):
    # Create empty datasets with maxshape to allow resizing
    img_ds = hf_out.create_dataset(
        "images",
        shape=(0, 2, 128, 128),
        maxshape=(None, 2, 128, 128),
        dtype=np.float32,
        compression="gzip",
        chunks=(1000, 2, 128, 128),
    )

    shape_ds = hf_out.create_dataset(
        "shapes",
        shape=(0, 2),
        maxshape=(None, 2),
        dtype=np.float32,
        compression="gzip",
        chunks=(1000, 2),
    )

    blend_ds = hf_out.create_dataset(
        "blendedness",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float32,
        compression="gzip",
    )

    peak_ds = hf_out.create_dataset(
        "peaks",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float32,
        compression="gzip",
    )

    total = 0

    for group_name in all_groups:
        group = hf_in[group_name]

        print(
            f"Processing group: {group_name} | Time elapsed: {time_string(time.time() - start)}"
        )

        # 1) Shapes
        df = group["patch_df"][()]
        flux_mask = df["flux_mask"]
        shape = np.stack([df["e1"][flux_mask], df["e2"][flux_mask]], axis=1).astype(
            np.float32
        )

        # 2) Images
        recon = group["recon"][:]
        psf = group["psf"][:]
        image = np.stack([recon, psf], axis=1).astype(np.float32)

        img_min = image.min(axis=(2, 3), keepdims=True)
        img_max = image.max(axis=(2, 3), keepdims=True)
        image = (image - img_min) / (img_max - img_min)

        assert image.shape[0] == shape.shape[0]

        # 3) Blendedness
        isolated_stamps = extract_image(group["isolated_stamps"][:])
        blended_stamps = extract_image(group["blended_stamps"][:])
        blend = blendedness(isolated_stamps, blended_stamps).astype(np.float32)

        assert blend.shape[0] == shape.shape[0]

        # 4) Peaks
        psfs = group["psf"][:]
        img_f = rfftn(isolated_stamps, axes=(1, 2))
        psf_f = rfftn(ifftshift(psfs, axes=(1, 2)), axes=(1, 2))
        dirty = irfftn(img_f * psf_f, s=isolated_stamps.shape[1:], axes=(1, 2))
        peak_vals = dirty.max(axis=(1, 2)).astype(np.float32)

        print(f"Max peak value: {peak_vals.max() * 1e6: .02f}")
        print(f"Min peak value: {peak_vals.min() * 1e6: .02f}")
        print(f"Mean peak value: {peak_vals.mean() * 1e6: .02f}")
        print(
            "Finished getting peak values | Time elapsed:",
            time_string(time.time() - start),
        )
        print("--------------------------------------------------")

        # Append to output datasets
        n_new = image.shape[0]
        new_total = total + n_new

        img_ds.resize((new_total, 2, 128, 128))
        img_ds[total:new_total] = image

        shape_ds.resize((new_total, 2))
        shape_ds[total:new_total] = shape

        blend_ds.resize((new_total,))
        blend_ds[total:new_total] = blend

        peak_ds.resize((new_total,))
        peak_ds[total:new_total] = peak_vals

        total = new_total

        # Flush periodically for speed and safety
        hf_out.flush()

print(f"Saved training dataset to {OUT_PATH}")
print(f"Total samples: {total}")
print(f"Total time: {time_string(time.time() - start)}")
