# %%
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from deepshape2.data.loaders import CenterCrop
from deepshape2.models import create_model
from deepshape2.utils import chi2_dirty, get_progress_bar, get_tqdm, load_config

tqdm_kwargs = get_tqdm()


cfg = load_config()
SCALE_FACTOR = cfg["SCALE_FACTOR"]

__all__ = ["reconstruct_facets"]

# %%


def reconstruct_facets(
    dirty_all,
    psf_all,
    device="cpu",
    hqs_params={},
    bsize=64,
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
