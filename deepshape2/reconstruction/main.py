# %%
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from deepshape2.data.loaders import CenterCrop, ReconDataset
from deepshape2.models import create_model
from deepshape2.utils import chi2_dirty, get_progress_bar, get_tqdm, load_config

tqdm_kwargs = get_tqdm()


cfg = load_config()
SCALE_FACTOR = cfg["SCALE_FACTOR"]

__all__ = ["reconstruct_facets", "reconstruct_facets_h5"]

# %%


def reconstruct_facets(
    dirty_all,
    psf_all,
    device="cpu",
    hqs_params={},
    bsize=128,
    num_workers=4,
    deblender=None,
    do_chi2=False,
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
    im_tensor = torch.tensor(im_all, dtype=torch.float32)

    # --- DataLoader for batch processing
    if len(im_tensor) > 32:
        dataset = TensorDataset(im_tensor)
        loader = DataLoader(
            dataset,
            batch_size=bsize,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
    else:
        # For small datasets, avoid DataLoader overhead
        loader = [(im_tensor,)]

    crop_fn = CenterCrop(128)

    # --- Save reconstructions
    recon_all = []
    decon_all = []
    chi2_all = []

    # --- Batch reconstruction with progress bar
    with torch.inference_mode():
        pbar = get_progress_bar(True, total=len(loader), **tqdm_kwargs)
        pbar.set_description("Reconstructing facets")
        for batch in loader:
            im = batch[0].to(device, non_blocking=True)
            decon = model(im)

            if do_chi2:
                chi2, res = chi2_dirty(im[:, :1], decon, im[:, 1:])
                chi2_all.append(chi2.cpu().numpy())

            if deblender is not None:
                decon_crop = crop_fn(decon)
                decon_all.append(decon_crop.cpu().numpy().squeeze())

                # Deblend
                decon_scaled = torch.arcsinh_(decon_crop.mul_(SCALE_FACTOR))
                decon_deblended = deblender(decon_scaled)[0]
                recon = torch.sinh_(decon_deblended).div_(SCALE_FACTOR)
            else:
                recon = decon
            recon = recon.cpu().numpy().squeeze()

            recon_all.append(recon)
            pbar.update(1)
        pbar.close()

    result = {}

    result["recon"] = np.concatenate(recon_all, axis=0)

    # include chi2 if requested
    if do_chi2:
        result["chi2"] = np.concatenate(chi2_all, axis=0)

    if deblender is not None:
        result["decon"] = np.concatenate(decon_all, axis=0)

    return result


# %%
def reconstruct_facets_h5(
    h5_path,
    device,
    thresh=50,
    hqs_params={},
    bsize=128,
    num_workers=4,
    dirty_key="dirty",
    psf_key="psf",
    recon_key="recon",
    decon_key="decon",
    deblender=None,
    dump_to_file=True,
):
    """
    Batched reconstruction directly from HDF5 with masking and incremental saving.
    """

    # --- open once to get mask and shapes
    with h5py.File(h5_path, "r+") as hf:
        flux = hf["flux"][:]
        mask = flux > thresh * 1e-6
        n_total = len(flux)
        _, H, W = hf[dirty_key].shape

        # create output datasets if needed
        if dump_to_file:
            if recon_key not in hf:
                hf.create_dataset(
                    recon_key,
                    shape=(n_total, H, W),
                    dtype=np.float32,
                    chunks=(1, H, W),
                )
            else:
                del hf[recon_key]

            if deblender is not None:
                if decon_key in hf:
                    del hf[decon_key]

                hf.create_dataset(
                    decon_key,
                    shape=(n_total, 128, 128),
                    dtype=np.float32,
                    chunks=(1, 128, 128),
                )

    # --- dataset + loader
    dataset = ReconDataset(
        h5_path,
        mask,
        dirty_key=dirty_key,
        psf_key=psf_key,
    )

    loader = DataLoader(
        dataset,
        batch_size=bsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- model
    model = create_model(device=device, **hqs_params)
    model.eval()

    crop_fn = CenterCrop(128)

    # --- containers if not dumping to file
    if not dump_to_file:
        recon_all = []
        decon_all = [] if deblender is not None else None
        index_all = []

    with torch.inference_mode():
        pbar = get_progress_bar(True, total=len(loader), **tqdm_kwargs)
        pbar.set_description("Reconstructing facets")

        for im, indices in loader:
            im = im.to(device, non_blocking=True)

            decon = model(im)

            # --- deblending
            if deblender is not None:
                decon_crop = crop_fn(decon)
                decon_np = decon_crop.cpu().numpy().squeeze(1)

                decon_scaled = torch.arcsinh_(decon_crop.mul_(SCALE_FACTOR))
                decon_deblended = deblender(decon_scaled)[0]
                recon = torch.sinh_(decon_deblended).div_(SCALE_FACTOR)
            else:
                recon = decon

            recon_np = recon.cpu().numpy().squeeze(1)

            if dump_to_file:
                # --- write back
                with h5py.File(h5_path, "r+") as hf:
                    hf[recon_key][indices.numpy()] = recon_np

                    if deblender is not None:
                        hf[decon_key][indices.numpy()] = decon_np
            else:
                # --- accumulate in memory
                recon_all.append(recon_np)
                index_all.append(indices.numpy())

                if deblender is not None:
                    decon_all.append(decon_np)

            pbar.update(1)

        pbar.close()

    # --- return results if not dumping to file
    if not dump_to_file:
        recon_all = np.concatenate(recon_all, axis=0)
        index_all = np.concatenate(index_all, axis=0)

        order = np.argsort(index_all)
        recon_all = recon_all[order]

        if deblender is not None:
            decon_all = np.concatenate(decon_all, axis=0)
            decon_all = decon_all[order]
            return recon_all, decon_all

        return recon_all
