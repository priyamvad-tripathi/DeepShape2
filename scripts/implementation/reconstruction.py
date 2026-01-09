# %% Imports
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
from deepshape2.visualization import metric_dependence, plot

# %% Constants / Configuration
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]
RESULTS_DIR = cfg["RESULTS_DIR"]

# %% Load Data
facet_data = load_h5(DATA_DIR + "facets.h5")
data = load_h5(DATA_DIR + "deep_set.h5")

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
ckpt_path = cfg["MODEL_DIR"] + "vae_mha.pt"
deblender = VAE().to(device)
deblender.load_state_dict(torch.load(ckpt_path, map_location=device)["best_weights"])
deblender.eval()

# Load visibility data
vis = create_visibility_from_ms(DATA_DIR + "MS/vis_deep_set_patch_000.ms")[0]

# %% Select Sources


isolated_stamps = data["patch_000/isolated_stamps"][:]
blended_stamps = data["patch_000/blended_stamps"][:]
peak = isolated_stamps.max(axis=(1, 2))


# mask = flux > 50e-6
# mask2 = np.where(peak > 0.71e-06 / 3)[0]
# mask = np.intersect1d(mask1, mask2)

# %% Reconstruction and Evaluation
dirty = facet_data["deep/facets_128/dirty"]
psf = facet_data["deep/facets_128/psf"]
blend = extract_image(blended_stamps)
iso = extract_image(isolated_stamps)


result = reconstruct_facets(dirty, psf, device, num_workers=4, deblender=deblender)
recon = result["recon"].copy()
decon = result["decon"].copy()

blendedness_vals = blendedness(iso, blend)

# %% Metrics Calculation
# Metrics
psnr_vals = psnr_batch(iso, recon)
blendedness_vals = blendedness(iso, blend)
shape_true = shape_galsim(iso)[0]
shape_recon = shape_galsim(recon)[0]
shape_diff = np.linalg.norm(shape_recon - shape_true, axis=1)


# --- Report metrics ---
print(
    f"Shape: Max {shape_diff.max():.3f} | Min {shape_diff.min():.3f} | Mean {shape_diff.mean():.3f}"
)
print(
    f"PSNR: Max {psnr_vals.max():.3f} dB | Min {psnr_vals.min():.3f} dB | Mean {psnr_vals.mean():.3f} dB"
)

# %% Plot selected sources

np.random.seed(10)
mask = flux > 10e-6
inds = [11816, 12782, 509, 14023]
# inds = np.random.choice(len(recon[mask]), size=10, replace=False)
print(inds)

residuals = np.array(
    [residual_facet_image(vis, decon[mask][i], galaxy_locations[mask][i]) for i in inds]
)


metrics_str = [
    f"{p:.02f} dB / {sd:.03f}"
    for p, sd in zip(psnr_vals[mask][inds], shape_diff[mask][inds])
]
flux_labels = [f"{p * 1e6:.3f} µJy " for p in flux[mask][inds]]
blendedness_labels = [f"{b:.3f}" for b in blendedness_vals[mask][inds]]

none_title = [None] * len(inds)


# Plotting
images = [
    iso[mask][inds],
    blend[mask][inds],
    dirty[mask][inds],
    decon[mask][inds],
    residuals,
    recon[mask][inds],
]
subtitles = [
    flux_labels,
    blendedness_labels,
    none_title,
    none_title,
    none_title,
    metrics_str,
]
plot(
    images=images,
    caption=[
        "Isolated",
        "Blended",
        "Dirty",
        "Deconvolved",
        "Residual",
        "Reconstructed",
    ],
    cbar=True,
    subtitles=subtitles,
    same_scale=[0, 1, 3, 5],
    scale_row=0,
    fname=RESULTS_DIR + "reconstruction/stamps.pdf",
)
# %%

peak_edges = np.linspace(0, 1.8, 7)
size_edges = np.linspace(0, 5, 7)

size = (
    patch_df["size"].values[mask_flux] * 1.6783469900166605
)  # Connvert from scale length to half-light radius
metric_dependence(
    [psnr_vals, shape_diff],
    [peak * 1e6, size],
    bin_edges_list=[peak_edges, size_edges],
    metric_lims_list=[(10, 60), (0, 0.6)],
    fname=RESULTS_DIR + "reconstruction/metric_dependence.pdf",
)
