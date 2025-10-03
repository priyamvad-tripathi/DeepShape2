# %%
from functools import partial
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
import pandas as pd

from deepshape2.utils import extract_image, load_config

# %% Default parameters
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]
NPIX_STAMP = 256


# %%
def _worker(loc, h5_path, group_name, NPIX, relative, switch_xy):
    with h5py.File(h5_path, "r") as f:
        sky = f[group_name]["sky"]
        return extract_image(
            sky, NPIX=NPIX, center=[loc], relative=relative, switch_xy=switch_xy
        )


def get_blended_stamps(
    h5_path,
    group_name,
    NPIX=128,
    relative=True,
    switch_xy=True,
    n_jobs=None,
    with_progress=True,
    dataset_name="blended_stamps",
):
    # Step 1: Get locations (mask applied)
    with h5py.File(h5_path, "r") as f:
        patch_rec = f[group_name]["patch_df"][()]
        patch_out = pd.DataFrame.from_records(patch_rec)

        locs = patch_out[["pix_x", "pix_y"]].values
        mask = patch_out["flux_mask"].values.astype(bool)
        locs_masked = locs[mask]

    if n_jobs is None:
        n_jobs = cpu_count()

    # Step 2: Parallel crop extraction
    worker_fn = partial(
        _worker,
        h5_path=h5_path,
        group_name=group_name,
        NPIX=NPIX,
        relative=relative,
        switch_xy=switch_xy,
    )

    crops = []
    with Pool(processes=n_jobs) as pool:
        if with_progress:
            from tqdm import tqdm

            for crop in tqdm(
                pool.imap_unordered(worker_fn, locs_masked, chunksize=10),
                total=len(locs_masked),
                desc=f"Extracting {group_name}",
            ):
                crops.append(crop)
        else:
            crops = pool.map(worker_fn, locs_masked)

    crops = np.stack(crops, axis=0)  # (num_cutouts, NPIX, NPIX)

    # Step 3: Save results back to HDF5 group
    with h5py.File(h5_path, "a") as f:
        grp = f[group_name]

        # delete old dataset if it exists
        if dataset_name in grp:
            del grp[dataset_name]

        grp.create_dataset(
            dataset_name,
            data=crops,
            compression="gzip",
            chunks=(1, NPIX, NPIX),
        )

    return crops


# %%  Run for all patches

h5_path = DATA_DIR + "sky.h5"

with h5py.File(h5_path, "r") as f:
    groups = list(f.keys())

for g in groups:
    get_blended_stamps(h5_path, g, n_jobs=20, NPIX=NPIX_STAMP)
