# %% Imports
import os

import numpy as np
import pandas as pd
import torch
from ska_sdp_datamodels.visibility import create_visibility_from_ms

from deepshape2.models import VAE
from deepshape2.reconstruction import reconstruct_facets, residual_facet_image
from deepshape2.utils import (
    blendedness,
    extract_image,
    get_freest_gpu,
    load_config,
    load_h5,
    psnr_batch,
    set_seed,
    shape_galsim,
)
from deepshape2.visualization import plot

# %% Constants / Configuration
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]
# %% Load Data
facet_data = load_h5(os.path.join(DATA_DIR, "facets.h5"))
data = load_h5(os.path.join(DATA_DIR, "deep_set.h5"))

# Metadata for patch_000
patch_df = pd.DataFrame.from_records(data["patch_000"]["patch_df"][()])

# Masks and relevant fields
mask_flux = patch_df["flux_mask"].values
flux = patch_df["flux"].values[mask_flux]
galaxy_locations = patch_df[["pix_x", "pix_y"]].to_numpy()[mask_flux]

# %% Torch Setup
device = get_freest_gpu(set_device=True)
set_seed()

# Load pre-trained deblender
ckpt_path = os.path.join(cfg["MODEL_DIR"], "vae_mha.pt")
deblender = VAE().to(device)
deblender.load_state_dict(torch.load(ckpt_path, map_location=device)["best_weights"])
deblender.eval()

# Load visibility data
vis = create_visibility_from_ms(os.path.join(DATA_DIR, "MS/vis_deep_set_patch_000.ms"))[
    0
]

# %% Select Sources
mask = flux > 50e-6  # Select bright enough sources

isolated_stamps = data["patch_000/isolated_stamps"][:]
blended_stamps = data["patch_000/blended_stamps"][:]
peak = isolated_stamps.max(axis=(1, 2))

# Random subset of sources to visualize
inds = np.random.choice(np.where(mask)[0], size=5, replace=False)

# %% Reconstruction and Evaluation
GRID_SIZE = 256
dirty = facet_data[f"deep/facets_{GRID_SIZE}/dirty"][:][inds]
psf = facet_data[f"deep/facets_{GRID_SIZE}/psf"][:][inds]
blend = blended_stamps[inds]
iso = isolated_stamps[inds]

recon, decon = reconstruct_facets(
    dirty, psf, device, num_workers=4, deblender=deblender
)

residuals = [
    residual_facet_image(vis, np.zeros_like(dec), galaxy_locations[inds[i]])
    for i, dec in enumerate(decon)
]

# Metrics
psnr_vals = psnr_batch(extract_image(iso), extract_image(recon))
blendedness_vals = blendedness(extract_image(iso), extract_image(blend))
shape_true = shape_galsim(extract_image(iso))[0]
shape_recon = shape_galsim(extract_image(recon))[0]
shape_diff = np.linalg.norm(shape_recon - shape_true, axis=1)

psnr_vals_2 = psnr_batch(extract_image(iso, 100), extract_image(decon, 100))
metrics_str_2 = [f"{p:.02f} dB" for p in psnr_vals_2]

metrics_str = [f"{p:.02f} dB / {sd:.03f}" for p, sd in zip(psnr_vals, shape_diff)]
flux_labels = [f"{p * 1e6:.3f} µJy " for p in peak[inds]]
blendedness_labels = [f"{b:.3f}" for b in blendedness_vals]

none_title = [None] * len(inds)


# %% Plotting
plot(
    images=[
        extract_image(iso),
        extract_image(blend),
        extract_image(dirty),
        extract_image(decon),
        extract_image(residuals),
        extract_image(recon),
    ],
    caption=[
        "Isolated",
        "Blended",
        "Dirty",
        "Deconvolved",
        "Residual",
        "Reconstructed",
    ],
    cbar=True,
    subtitles=[
        flux_labels,
        blendedness_labels,
        none_title,
        metrics_str_2,
        none_title,
        metrics_str,
    ],
    same_scale=[2, 4],
    scale_row=2,
)
