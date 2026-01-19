# %% Imports
import numpy as np
import pandas as pd
import torch
from ska_sdp_datamodels.visibility import create_visibility_from_ms

from deepshape2.models import VAE
from deepshape2.reconstruction import reconstruct_facets
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

# %% Constants / Configuration
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]
RESULTS_DIR = cfg["RESULTS_DIR"]

# %% Load Data
facet_data = load_h5(DATA_DIR + "facets.h5")
data = load_h5(DATA_DIR + "deep_set.h5", "a")

patch_group = data["patch_000"]
patch_df = pd.DataFrame.from_records(patch_group["patch_df"][()])

mask_flux = patch_df["flux_mask"].values
flux = patch_df["flux"].values[mask_flux]
galaxy_locations = patch_df[["pix_x", "pix_y"]].to_numpy()[mask_flux]

# %% Torch Setup
device = get_freest_gpu(set_device=True)
set_seed()

ckpt_path = cfg["MODEL_DIR"] + "vae_mha.pt"
deblender = VAE().to(device)
deblender.load_state_dict(torch.load(ckpt_path, map_location=device)["best_weights"])
deblender.eval()

# Load visibility data
vis = create_visibility_from_ms(DATA_DIR + "MS/vis_deep_set_patch_000.ms")[0]

# %% Extract stamps
isolated_stamps = patch_group["isolated_stamps"][:]
blended_stamps = patch_group["blended_stamps"][:]

blend_images = extract_image(blended_stamps)
iso_images = extract_image(isolated_stamps)

blendedness_vals = blendedness(iso_images, blend_images)
shape_true, flags_true = shape_galsim(iso_images)


# %% Helper
def compute_metrics(
    recon_images, iso_images, shapes_true=shape_true, flags_true=flags_true, label=""
):
    psnr_vals = psnr_batch(iso_images, recon_images)
    shape_recon, flags_recon = shape_galsim(recon_images)

    valid_mask = (flags_true == 0) & (flags_recon == 0)
    shape_diff = np.full(len(recon_images), np.nan)
    shape_diff[valid_mask] = np.linalg.norm(
        shape_recon[valid_mask] - shape_true[valid_mask], axis=1
    )

    print(f"{label} N valid shapes: {np.sum(valid_mask)} / {len(recon_images)}")
    print(
        f"{label} Shape: Max {np.nanmax(shape_diff):.3f} | Min {np.nanmin(shape_diff):.3f} | Mean {np.nanmean(shape_diff):.3f}"
    )
    print(
        f"{label} PSNR: Max {psnr_vals.max():.3f} dB | Min {psnr_vals.min():.3f} dB | Mean {psnr_vals.mean():.3f} dB"
    )

    return psnr_vals, shape_diff, shape_recon, flags_recon


# %% Reconstruct blended facets
dirty_facet = facet_data["deep/facets_128/dirty"]
psf_facet = facet_data["deep/facets_128/psf"]

recon_result = reconstruct_facets(
    dirty_facet, psf_facet, device, num_workers=4, deblender=deblender
)
recon_blend = recon_result["recon"].copy()
decon_blend = recon_result["decon"].copy()

psnr_blend, shape_diff_blend, _, flags_recon_blend = compute_metrics(
    recon_blend, iso_images, label="Blended"
)

# Save blended reconstruction
for name in ["recon", "decon", "dirty", "psf", "psnr", "shape_diff"]:
    if name in patch_group:
        del patch_group[name]

patch_group.create_dataset("recon", data=recon_blend, compression="gzip")
patch_group.create_dataset("decon", data=decon_blend, compression="gzip")
patch_group.create_dataset("dirty", data=dirty_facet, compression="gzip")
patch_group.create_dataset("psf", data=psf_facet, compression="gzip")
patch_group.create_dataset("psnr", data=psnr_blend, compression="gzip")
patch_group.create_dataset("shape_diff", data=shape_diff_blend, compression="gzip")
data.flush()

# %% Reconstruct isolated stamps
iso_group = patch_group["isolated_dirty_psf"]

dirty_iso = iso_group["dirty"]
psf_iso = iso_group["psf"]

# HQS hyperparameters from old config for isolated reconstruction
recon_iso_result = reconstruct_facets(
    dirty_iso,
    psf_iso,
    device,
    num_workers=4,
    hqs_params={
        "f1": 0.2076320305146005,
        "f2": 8.855988804161978,
        "falpha": 5.052044590041307,
    },
)
recon_iso = recon_iso_result["recon"]

psnr_iso, shape_diff_iso, _, flags_recon_iso = compute_metrics(
    recon_iso, iso_images, label="Isolated"
)

for name in ["recon", "psnr", "shape_diff"]:
    if name in iso_group:
        del iso_group[name]


iso_group.create_dataset("recon", data=recon_iso, compression="gzip")
iso_group.create_dataset("psnr", data=psnr_iso, compression="gzip")
iso_group.create_dataset("shape_diff", data=shape_diff_iso, compression="gzip")
data.flush()
# %% Close data
data.close()
