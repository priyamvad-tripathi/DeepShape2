# %% Imports
import numpy as np
import pandas as pd
from ska_sdp_datamodels.visibility import create_visibility_from_ms

from deepshape2.reconstruction import residual_facet_image
from deepshape2.utils import extract_image, load_config, load_h5
from deepshape2.visualization import metric_dependence, plot

# %% Load Data and Metrics
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]
RESULTS_DIR = cfg["RESULTS_DIR"]

data = load_h5(DATA_DIR + "deep_set.h5", "r")
patch_group = data["patch_000"]
patch_df = pd.DataFrame.from_records(patch_group["patch_df"][()])

mask_flux = patch_df["flux_mask"].values
flux = patch_df["flux"].values[mask_flux]
galaxy_locations = patch_df[["pix_x", "pix_y"]].to_numpy()[mask_flux]

iso_images = extract_image(patch_group["isolated_stamps"][:])
blend_images = extract_image(patch_group["blended_stamps"][:])

recon_blend = patch_group["recon"][:]
decon_blend = patch_group["decon"][:]
dirty_facet = patch_group["dirty"][:]
psf_facet = patch_group["psf"][:]

psnr_blend = patch_group["psnr"][:]
shape_diff_blend = patch_group["shape_diff"][:]

dirty_iso = patch_group["isolated_dirty_psf/dirty"][:]
recon_iso = patch_group["isolated_dirty_psf/recon"][:]
psnr_iso = patch_group["isolated_dirty_psf/psnr"][:]
shape_diff_iso = patch_group["isolated_dirty_psf/shape_diff"][:]

# %% Example indices to plot
inds = [11816, 1301, 5684, 14023]

vis = create_visibility_from_ms(DATA_DIR + "MS/vis_deep_set_patch_000.ms")[0]
residuals = np.array(
    [residual_facet_image(vis, decon_blend[i], galaxy_locations[i]) for i in inds]
)

# Prepare metric labels for display
metrics_labels = [
    f"{p:.02f} dB / {sd:.03f}"
    for p, sd in zip(psnr_blend[inds], shape_diff_blend[inds])
]
flux_labels = [f"{f * 1e6:.3f} µJy" for f in flux[inds]]
blendedness_vals = patch_group["blend"][:]
blendedness_labels = [f"{b:.3f}" for b in blendedness_vals[inds]]
none_titles = [None] * len(inds)

images_to_plot = [
    iso_images[inds],
    blend_images[inds],
    dirty_facet[inds],
    decon_blend[inds],
    residuals,
    recon_blend[inds],
]

subtitles_list = [
    flux_labels,
    blendedness_labels,
    none_titles,
    none_titles,
    none_titles,
    metrics_labels,
]

scales = plot(
    images=images_to_plot,
    caption=[
        "Isolated",
        "Blended",
        "Dirty",
        "Deconvolved",
        "Residual",
        "Reconstructed",
    ],
    cbar=True,
    subtitles=subtitles_list,
    same_scale=[0, 1, 3, 5],
    scale_row=0,
    return_scales=True,
    fname=RESULTS_DIR + "reconstruction/stamps.pdf",
)
# %%
# mask = (blendedness_vals > 0.1) & (flux > 30 * 1e-6)

# # Indices that satisfy the condition
# valid_indices = np.where(mask)[0]

# # Randomly choose 10 (or fewer if not enough)
# num_to_select = min(10, len(valid_indices))
# inds = np.random.choice(valid_indices, size=num_to_select, replace=False)

# flux_labels = [f"{f * 1e6:.3f} µJy" for f in flux[inds]]
# blendedness_labels = [f"{b:.3f}" for b in blendedness_vals[inds]]
# metrics1 = [
#     f"{p:.02f} dB / {sd:.03f}"
#     for p, sd in zip(psnr_blend[inds], shape_diff_blend[inds])
# ]
# metrics_2 = [
#     f"{p:.02f} dB / {sd:.03f}" for p, sd in zip(psnr_iso[inds], shape_diff_iso[inds])
# ]


# subs = [flux_labels, blendedness_labels, [None] * len(inds), metrics1, metrics_2]
# print("Selected indices:", inds)
# plot(
#     [
#         iso_images[inds],
#         blend_images[inds],
#         decon_blend[inds],
#         recon_blend[inds],
#         recon_iso[inds],
#     ],
#     cbar=True,
#     same_scale=[0, 1, 2, 3, 4],
#     scale_row=0,
#     subtitles=subs,
# )

# %% Plot isolated reconstruction for comparison
images2 = [
    dirty_iso[inds],
    recon_iso[inds],
]
metrics_2 = [
    f"{p:.02f} dB / {sd:.03f}" for p, sd in zip(psnr_iso[inds], shape_diff_iso[inds])
]
subtitles2 = [
    none_titles,
    metrics_2,
]

plot(
    images=images2,
    caption=[
        "Dirty (Iso)",
        "Reconstructed (Iso)",
    ],
    cbar=True,
    subtitles=subtitles2,
    same_scale=[1],
    scale_ranges=scales,
    fname=RESULTS_DIR + "reconstruction/iso_stamps.pdf",
)

# %% Metric dependence plots
peak = iso_images.max(axis=(1, 2))
size = patch_df["size"].values[mask_flux] * 1.6783469900166605
peak_edges = np.linspace(0, 1.8, 7)
size_edges = np.linspace(0, 5, 7)

metric_dependence(
    [[psnr_iso, psnr_blend], [shape_diff_iso, shape_diff_blend]],
    [peak * 1e6, size],
    bin_edges_list=[peak_edges, size_edges],
    metric_lims_list=[(10, 60), (0, 0.6)],
    fname=RESULTS_DIR + "reconstruction/metric_dependence.pdf",
)
