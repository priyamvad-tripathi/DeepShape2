# %%
import os
import time

import h5py
import numpy as np

from deepshape2.utils import load_config, time_string

# %% Config Parameters
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]

INP_PATH = DATA_DIR + "training_set_3.h5"
OUT_PATH = DATA_DIR + "trainset.h5"


if os.path.exists(OUT_PATH):
    print(f"Removing existing file at {OUT_PATH}")
    os.remove(OUT_PATH)
# %% Main Function

start = time.time()

chunk_size = 512

with (
    h5py.File(INP_PATH, "r") as hf_in,
    h5py.File(OUT_PATH, "w") as hf_out,
):
    # --- flux mask
    flux = hf_in["flux"][:]
    flux_mask = flux > 50 * 1e-6
    flux_idxs = np.where(flux_mask)[0]
    n_total = len(flux_idxs)

    # --- create output datasets (extendable)
    hf_out.create_dataset(
        "peak_factor",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float32,
        chunks=(chunk_size,),
    )

    hf_out.create_dataset(
        "shapes",
        shape=(0, 2),
        maxshape=(None, 2),
        dtype=np.float32,
        chunks=(chunk_size, 2),
    )

    hf_out.create_dataset(
        "images",
        shape=(0, 2, 128, 128),
        maxshape=(None, 2, 128, 128),
        dtype=np.float32,
        chunks=(64, 2, 128, 128),
    )

    write_ptr = 0

    # --- process in chunks
    for i0 in range(0, n_total, chunk_size):
        i1 = min(i0 + chunk_size, n_total)
        idx = flux_idxs[i0:i1]

        # --- peak factor
        dirty = hf_in["dirty"][idx]
        peak_blend = dirty.max(axis=(1, 2))
        peak_iso = hf_in["peak"][idx]
        peak_factor = (peak_blend - 0.71e-6) / peak_iso

        # --- shapes
        shapes = hf_in["shapes"][idx]

        # --- images
        psf = hf_in["psf"][idx]
        recon = hf_in["recon"][idx]

        image = np.stack([recon, psf], axis=1).astype(np.float32)

        img_min = image.min(axis=(2, 3), keepdims=True)
        img_max = image.max(axis=(2, 3), keepdims=True)
        image = (image - img_min) / (img_max - img_min)

        n_chunk = len(idx)

        # --- extend datasets
        hf_out["peak_factor"].resize(write_ptr + n_chunk, axis=0)
        hf_out["shapes"].resize(write_ptr + n_chunk, axis=0)
        hf_out["images"].resize(write_ptr + n_chunk, axis=0)

        # --- write
        hf_out["peak_factor"][write_ptr : write_ptr + n_chunk] = peak_factor.astype(
            np.float32
        )
        hf_out["shapes"][write_ptr : write_ptr + n_chunk] = shapes.astype(np.float32)
        hf_out["images"][write_ptr : write_ptr + n_chunk] = image

        write_ptr += n_chunk

        print(
            f"Processed {write_ptr}/{n_total} samples | "
            f"time: {time_string(time.time() - start)}"
        )

print(f"Saved training dataset to {OUT_PATH}")
print(f"Total samples: {write_ptr}")
print(f"Total time: {time_string(time.time() - start)}")
