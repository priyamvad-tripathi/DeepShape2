# %% Import Libraries
import logging
import time
import warnings

import numpy as np
from dask.distributed import Client, LocalCluster
from ska_sdp_datamodels.visibility import create_visibility_from_ms

from deepshape2.reconstruction import get_facets, reconstruct_facets
from deepshape2.utils import get_freest_gpu, load_config, load_h5, set_seed

warnings.warn = lambda *args, **kwargs: None
logging.getLogger().addHandler(logging.NullHandler())


def log_time(msg, t0):
    now = time.time()
    print(f"[{now - t0:8.2f} s] {msg}")
    return now


# %% Full timing script

if __name__ == "__main__":
    t_start = time.time()

    cfg = load_config()
    DATA_DIR = cfg["DATA_DIR"]
    MODEL_DIR = cfg["MODEL_DIR"]

    device = get_freest_gpu(set_device=True)
    set_seed()

    log_time("Loaded config and selected device", t_start)

    h5_path = DATA_DIR + "deep_set.h5"
    h5 = load_h5(h5_path, "r")

    vis_filename = DATA_DIR + "MS/vis_deep_set_patch_000.ms"

    patch = h5["patch_000"]
    sky = patch["sky"][()]
    patch_ra, patch_dec = patch.attrs["centre"]

    t_vis = time.time()

    vis = create_visibility_from_ms(vis_filename)[0]

    t_vis = log_time("Simulated visibilities", t_vis)

    patch_df = patch["patch_df"][()]

    mask0 = patch_df["flux_mask"]
    true_indices = np.where(mask0)[0]
    mask = np.random.choice(true_indices, size=32, replace=False)

    galaxy_locations = patch_df[["pix_x", "pix_y"]][mask]

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

        t_facets = time.time()

        dirty_all, psf_all = get_facets(
            vis=vis,
            galaxy_locations=galaxy_locations,
            NPIX_facet=128,
            client=client,
        )
        log_time("Created dirty images and PSFs", t_facets)

        t_recon = time.time()

        recon_result = reconstruct_facets(
            dirty_all,
            psf_all,
            device,
            num_workers=4,
        )

        log_time("Reconstructed facets", t_recon)

        t_shape = time.time()

        log_time("Full pipeline excluding data loading", t_facets)
        log_time("Full pipeline finished", t_start)
