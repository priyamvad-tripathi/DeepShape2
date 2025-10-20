# %% Imports
import logging
import time
import warnings

import numpy as np
import pandas as pd
import torch
import xarray
from dask import compute, delayed
from dask.distributed import Client, LocalCluster
from ska_sdp_datamodels.visibility import create_visibility_from_ms
from ska_sdp_func_python.visibility import subtract_visibility
from torch.utils.data import DataLoader, TensorDataset

from deepshape2.models import create_model
from deepshape2.simulation import (
    make_dirty_image_and_psf,
    predict_visibilities_from_array,
    rephase_visibility,
)
from deepshape2.utils import get_progress_bar, get_tqdm, load_config, load_h5, post_step

# Disable warnings and logging from external libraries
warnings.warn = lambda *args, **kwargs: None
logging.getLogger().addHandler(logging.NullHandler())
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# TQDM settings
tqdm_kwargs = get_tqdm()


# %% Functions
def _process_facet(vis: xarray.Dataset, gal_center, NPIX_facet=256):
    """
    Process a single galaxy location:
    recenter visibilities, invert to get dirty and PSF facets.
    """
    vis_recentered = rephase_visibility(vis, gal_center)
    return make_dirty_image_and_psf(vis_recentered, NPIX=NPIX_facet)


def process_batch(vis, loc_batch, NPIX_facet):
    dirty_batch, psf_batch = [], []
    for loc in loc_batch:
        dirty, psf = _process_facet(vis, loc, NPIX_facet)
        dirty_batch.append(dirty)
        psf_batch.append(psf)
    return np.stack(dirty_batch), np.stack(psf_batch)


def chunked_iterable(seq, size):
    """Yield successive chunks from a sequence."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def get_facets(vis, galaxy_locations, NPIX_facet=256, batch_size=100, client=None):
    dirty_all, psf_all = [], []
    n = len(galaxy_locations)
    start = time.time()
    for i in range(0, n, batch_size):
        loc_batch = galaxy_locations[i : i + batch_size]

        delayed_tasks = [
            delayed(_process_facet)(vis, loc, NPIX_facet) for loc in loc_batch
        ]
        results = compute(*delayed_tasks)

        dirty, psf = zip(*results)
        dirty_all.append(np.stack(dirty))
        psf_all.append(np.stack(psf))

        post_step(f"processing batch {i // batch_size + 1}", start, client=client)
        # client.restart()

    return np.concatenate(dirty_all), np.concatenate(psf_all)


def reconstruct_facets(
    dirty_all, psf_all, device, hqs_params={}, bsize=64, num_workers=4
):
    """
    Reconstruct all facets using HQS-PnP model in batches.

    Args:
        dirty_all: np.ndarray, shape (N, H, W)
        psf_all: np.ndarray, shape (N, H, W)
        device: torch.device
        hqs_params: dict, arguments for create_model
        bsize: batch size
        num_workers: number of DataLoader workers
    Returns:
        recon_all: np.ndarray, shape (N, H, W)
    """
    # --- Create model once
    model = create_model(device=device, **hqs_params)
    model.eval()

    # --- Stack dirty + PSF as channels (C=2)
    im_all = np.stack([dirty_all, psf_all], axis=1)  # shape: (N, 2, H, W)
    im_tensor = torch.tensor(im_all, dtype=torch.float32, pin_memory=True)

    # --- DataLoader for batch processing
    dataset = TensorDataset(im_tensor)
    loader = DataLoader(
        dataset,
        batch_size=bsize,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    # --- Preallocate output array
    N, _, H, W = im_all.shape
    recon_all = np.empty((N, H, W), dtype=np.float32)

    # --- Batch reconstruction with progress bar
    idx_start = 0
    with torch.inference_mode():
        pbar = get_progress_bar(True, total=len(loader), **tqdm_kwargs)
        pbar.set_description("Reconstructing facets")
        for batch in loader:
            im = batch[0].to(device, non_blocking=True)
            recon = model(im).cpu().numpy().squeeze()
            B = recon.shape[0]

            recon_all[idx_start : idx_start + B] = recon
            idx_start += B
            pbar.update(1)
        pbar.close()
    return recon_all


def residual_facet_image(
    vis_original: xarray.Dataset, reconstructed_facet: np.ndarray, gal_center
):
    if reconstructed_facet.ndim != 2:
        raise ValueError("Reconstructed_facet must be a 2D array")

    # Residual visibilities & image
    vis_model = predict_visibilities_from_array(
        image_array=reconstructed_facet,
        ra_deg=gal_center[0],
        dec_deg=gal_center[1],
    )

    vis_facet = rephase_visibility(vis_original, gal_center)

    vis_residual = subtract_visibility(vis_facet, vis_model)

    image_residual = make_dirty_image_and_psf(
        vis_residual, NPIX=reconstructed_facet.shape[0], do_psf=False
    )

    return image_residual


# %%
if __name__ == "__main__":
    start = time.time()
    DATA_DIR = load_config()["DATA_DIR"]

    PATCHES = {
        "wide": {
            "h5_file": DATA_DIR + "wide_set.h5",
            "patch_key": "patch_051",
            "ms_file": DATA_DIR + "MS/vis_wide_set_patch_051.ms",
        },
        "deep": {
            "h5_file": DATA_DIR + "deep_set.h5",
            "patch_key": "patch_000",
            "ms_file": DATA_DIR + "MS/vis_deep_set_patch_000.ms",
        },
    }

    # -----------------------------
    # --- Create output HDF5
    # -----------------------------
    facets_h5_path = DATA_DIR + "facets.h5"
    data = load_h5(facets_h5_path, "a", delete_if_exists=True)

    # -----------------------------
    # --- Dask cluster
    # -----------------------------
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

        # -----------------------------
        # --- Process each patch
        # -----------------------------
        for patch_name, info in PATCHES.items():
            post_step(f"Start processing {patch_name} patch", start, data=data)

            patch_group = data.create_group(patch_name)

            # --- Load HDF5 patch data
            patch_data = load_h5(info["h5_file"], mode="r")
            patch_df = pd.DataFrame.from_records(
                patch_data[info["patch_key"]]["patch_df"][()]
            )

            # --- Galaxy locations and flux
            mask = patch_df["flux_mask"].values
            patch_group.create_dataset(
                "flux",
                data=patch_df["flux"][mask].values,
                compression="gzip",
            )
            galaxy_locations = patch_df[["RA_pix", "Dec_pix"]].to_numpy()[mask]

            # --- Stamp images
            peak = np.max(
                patch_data[info["patch_key"]]["isolated_stamps"][()], axis=(1, 2)
            )

            patch_group.create_dataset(
                "peak",
                data=peak,
                compression="gzip",
            )

            del patch_df, mask, patch_data

            # --- Load visibility
            vis = create_visibility_from_ms(info["ms_file"])[0]
            post_step(
                f"loading visibility MS for {patch_name}",
                start,
                data=data,
                client=client,
            )

            # --- Extract facets at different sizes
            for NPIX_facet in [128, 256, 512]:
                dirty_all, psf_all = get_facets(
                    vis=vis,
                    galaxy_locations=galaxy_locations,
                    NPIX_facet=NPIX_facet,
                    client=client,
                )

                facet_group = patch_group.create_group(f"facets_{NPIX_facet}")
                facet_group.create_dataset(
                    "dirty",
                    data=dirty_all,
                    compression="gzip",
                    chunks=(1, NPIX_facet, NPIX_facet),
                )
                facet_group.create_dataset(
                    "psf",
                    data=psf_all,
                    compression="gzip",
                    chunks=(1, NPIX_facet, NPIX_facet),
                )

                post_step(
                    f"extracting facet for {patch_name}, shape {dirty_all.shape}",
                    start,
                    data=data,
                )

    # -----------------------------
    # --- Close HDF5
    # -----------------------------
    data.close()
    post_step("Saved all facet results", start)
