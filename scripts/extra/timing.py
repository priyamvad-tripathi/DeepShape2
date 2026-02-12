# %% Import Libraries
import logging
import time
import warnings

import numpy as np
import torch
from dask.distributed import Client, LocalCluster
from ska_sdp_datamodels.visibility import create_visibility_from_ms
from torch.utils.data import DataLoader, TensorDataset

from deepshape2.models import VAE, shapenet_full
from deepshape2.reconstruction import get_facets, reconstruct_facets
from deepshape2.shape_measurement.main import predict
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

    ckpt_path = MODEL_DIR + "vae_mha.pt"
    deblender = VAE().to(device)
    deblender.load_state_dict(
        torch.load(ckpt_path, map_location=device)["best_weights"]
    )
    deblender.eval()

    shape_network = shapenet_full().to(device)
    checkpoint = torch.load(
        MODEL_DIR + "shape_network_full.pt",
        map_location=device,
        weights_only=False,
    )
    best_weights = checkpoint["best_weights"]
    shape_network.eval()

    t_models = log_time("Loaded models", t_start)

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
    mask = np.random.choice(true_indices, size=1024, replace=False)

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
            deblender=deblender,
        )

        log_time("Reconstructed facets", t_recon)

        t_shape = time.time()

        images = np.stack(
            [recon_result["recon"], psf_all],
            axis=1,
        ).astype(np.float32)

        img_min = images.min(axis=(2, 3), keepdims=True)
        img_max = images.max(axis=(2, 3), keepdims=True)
        images = (images - img_min) / (img_max - img_min + 1e-8)

        shapes = np.zeros((images.shape[0], 2), dtype=np.float32)

        images_T = torch.from_numpy(images)
        shapes_T = torch.from_numpy(shapes)

        dataset = TensorDataset(images_T, shapes_T)
        testloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        ypred, ytest, _ = predict(
            shape_network,
            weights=best_weights,
            data_loader=testloader,
            device=device,
            tqdm_enabled=True,
        )

        log_time("Shape measurement inference", t_shape)

        log_time("Full pipeline excluding data loading", t_facets)
        log_time("Full pipeline finished", t_start)
