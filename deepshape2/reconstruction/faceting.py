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
from deepshape2.utils import (
    get_progress_bar,
    get_tqdm,
    load_config,
    load_h5,
    post_step,
    save,
)

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


def _process_facet_batch(vis, loc_batch, NPIX_facet):
    """Processes a small batch of galaxy locations."""
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


def get_facets(vis, galaxy_locations, NPIX_facet=256, chunk_size=20):
    """
    Parallel facet extraction using Dask Delayed (no client needed).

    Args:
        vis: xarray.Dataset or preloaded visibility data.
        galaxy_locations: array/list of coordinates [(RA, Dec), ...].
        NPIX_facet: pixel size of each facet.
        chunk_size: number of galaxies processed per Dask task.

    Returns:
        (dirty_all, psf_all): np.ndarray aligned with galaxy_locations.
    """

    delayed_batches = []

    for loc_batch in chunked_iterable(galaxy_locations, chunk_size):
        delayed_result = delayed(_process_facet_batch)(vis, loc_batch, NPIX_facet)
        delayed_batches.append(delayed_result)

    # Compute in parallel
    results = compute(*delayed_batches)

    # Concatenate results in correct order
    dirty_all = np.concatenate([r[0] for r in results], axis=0)
    psf_all = np.concatenate([r[1] for r in results], axis=0)

    return dirty_all, psf_all


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
    pkl = {}

    DATA_DIR = load_config()["DATA_DIR"]

    # --- Define patches and corresponding HDF5 / MS paths
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

    with (
        LocalCluster(
            n_workers=60,
            processes=True,
            threads_per_worker=1,
            scheduler_port=8786,
            memory_limit=0,
        ) as cluster,
        Client(cluster) as client,
    ):
        print(client.dashboard_link)

        # --- Iterate over patches
        for patch_name, info in PATCHES.items():
            pkl[patch_name] = {}
            post_step(f"Start processing {patch_name} patch", start)

            # --- Load HDF5 data
            data = load_h5(info["h5_file"], mode="r")
            patch_df = pd.DataFrame.from_records(
                data[info["patch_key"]]["patch_df"][()]
            )
            post_step(f"Loaded HDF5 for {patch_name}", start)

            # --- Extract galaxy locations and flux info
            galaxy_locations = patch_df[["RA_pix", "Dec_pix"]].to_numpy()[
                :100
            ]  # limit for testing
            pkl[patch_name]["galaxy_locations"] = galaxy_locations.copy()

            mask = patch_df["flux_mask"].values
            pkl[patch_name]["flux"] = patch_df["flux"][mask].values.copy()
            pkl[patch_name]["isolated_stamps"] = data[info["patch_key"]][
                "isolated_stamps"
            ][()].copy()
            pkl[patch_name]["blended_stamps"] = data[info["patch_key"]][
                "blended_stamps"
            ][()].copy()

            # --- Load visibility from MeasurementSet
            vis = create_visibility_from_ms(info["ms_file"])[0]
            post_step(f"Loaded visibility MS for {patch_name}", start, client=client)

            # --- Extract facets of different sizes
            for NPIX_facet in [128, 256, 512]:
                dirty_all, psf_all = get_facets(
                    vis=vis,
                    galaxy_locations=galaxy_locations,
                    NPIX_facet=NPIX_facet,
                )
                pkl[patch_name][f"{NPIX_facet}"] = {
                    "dirty": dirty_all.copy(),
                    "psf": psf_all.copy(),
                }
                post_step(
                    f"Extracted {NPIX_facet}x{NPIX_facet} facets for {patch_name}",
                    start,
                    client=client,
                )

        # --- Save results
        save(pkl, DATA_DIR + "facets.pkl")
        post_step("Saved all facet results", start)
