# %% Load Modules
import time
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from dask import compute, delayed
from dask.distributed import Client, LocalCluster

from deepshape2.simulation import make_dirty_image_and_psf, simulate_visibilities
from deepshape2.utils import load_config, load_h5, post_step

warnings.filterwarnings("ignore", category=UserWarning)

# %% Defaults / Configuration
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]

hf_path = DATA_DIR + "deep_set_new.h5"

NPIX_SKY = cfg["NPIX_SKY"]
SCALE_RADIANS = cfg["SCALE_RADIANS"]


# %%

if __name__ == "__main__":
    start = time.time()

    # Open HDF5 file in append mode
    hf = load_h5(hf_path, "a", delete_if_exists=False)
    patch_group = hf["patch_000"]

    isolated_stamps = patch_group["isolated_stamps"][:]
    patch_df = patch_group["patch_df"][()]
    patch_ra, patch_dec = patch_group.attrs["centre"]

    mask = patch_df["flux_mask"]
    pix_x = patch_df["pix_x"][mask]
    pix_y = patch_df["pix_y"][mask]

    phase_centre = SkyCoord(
        ra=patch_ra * u.deg,
        dec=patch_dec * u.deg,
        frame="icrs",
        equinox="J2000",
    )

    dx = pix_x - NPIX_SKY // 2.0
    dy = pix_y - NPIX_SKY // 2.0

    offset_ra = -dx * SCALE_RADIANS * u.rad
    offset_dec = dy * SCALE_RADIANS * u.rad

    galaxy_centres = phase_centre.spherical_offsets_by(offset_ra, offset_dec)

    n_total = len(isolated_stamps)
    stamp_shape = isolated_stamps.shape[1:]

    # Output group
    group_name = "isolated_dirty_psf"
    if group_name in patch_group:
        del patch_group[group_name]

    out_group = patch_group.create_group(group_name)

    # Preallocate datasets
    batch_size = 500
    dirty_ds = out_group.create_dataset(
        "dirty",
        shape=(n_total, *stamp_shape),
        dtype=isolated_stamps.dtype,
        chunks=(batch_size, *stamp_shape),
        maxshape=(n_total, *stamp_shape),
    )
    psf_ds = out_group.create_dataset(
        "psf",
        shape=(n_total, *stamp_shape),
        dtype=isolated_stamps.dtype,
        chunks=(batch_size, *stamp_shape),
        maxshape=(n_total, *stamp_shape),
    )

    # Start Dask cluster
    with (
        LocalCluster(
            n_workers=64,
            processes=True,
            threads_per_worker=1,
            scheduler_port=8786,
            memory_limit=0,
        ) as cluster,
        Client(cluster) as client,
    ):
        print("Dask dashboard:", client.dashboard_link)

        for start_idx in range(0, n_total, batch_size):
            inds = np.arange(start_idx, min(start_idx + batch_size, n_total))

            # create chained delayed tasks
            delayed_dirty_psf = [
                delayed(make_dirty_image_and_psf)(
                    delayed(simulate_visibilities)(
                        field=isolated_stamps[i],
                        ra_pointing=galaxy_centres[i].ra.deg,
                        dec_pointing=galaxy_centres[i].dec.deg,
                        create_dirty=False,
                    )
                )
                for i in inds
            ]

            # Step 3: compute batch
            results = compute(*delayed_dirty_psf)

            # Step 4: write to HDF5
            for idx, (dirty, psf) in zip(inds, results):
                dirty_ds[idx] = dirty
                psf_ds[idx] = psf

            post_step(
                f"Processed stamps {start_idx}–{start_idx + len(inds)} / {n_total}",
                start,
                data=hf,
                client=client,
            )

    # Final checkpoint
    post_step("Simulated dirty images and PSFs", start, data=hf)
    hf.close()
