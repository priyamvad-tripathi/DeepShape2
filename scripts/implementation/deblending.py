# %%Import Libraries
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from deepshape2.data import loaders
from deepshape2.deblending import predict_multiple
from deepshape2.models import VAE, CAT_Unet
from deepshape2.utils import get_freest_gpu, load, load_config, load_h5, save, set_seed
from deepshape2.visualization import (
    binned_boxplot,
    plot,
    probability_distribution_metric,
)

# %% Set default parameters
cfg = load_config()

DATA_DIR = cfg["DATA_DIR"]
loc_weights = cfg["MODEL_DIR"]
RESULTS_DIR = cfg["RESULTS_DIR"]

# Torch Parameters
device = get_freest_gpu(set_device=True)
set_seed()


# %% Load the models
model1 = VAE()
model2 = VAE(attention=False)
model3 = CAT_Unet(
    img_size=128,
    in_chans=1,
    depth=[4, 6, 6, 8],
    split_size_0=[4, 4, 4, 4],
    dim=48,
    num_heads=[2, 2, 4, 8],
    mlp_ratio=4,
)

# %% Load Data and Model

test_dataset = loaders.BlendDataset(
    path=DATA_DIR + "deep_set.h5",
    x_key="blended_stamps",
    y_key="isolated_stamps",
    groups=["patch_000"],
)


# Initialize DataLoaders
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

models = [model1, model2, model3]
ckps = [
    loc_weights + "vae_deblender_MHA_3_low_beta.pt",
    loc_weights + "vae_cnn.pt",
    loc_weights + "cat_deblender.pt",
]
results = predict_multiple(models, ckps, test_loader, device, do_SSIM=True)

# %% Load fluxes and blends
try:
    results
except NameError:
    results = load(DATA_DIR + "deblending_results.pkl")

blend = results["blend"]

with load_h5(DATA_DIR + "deep_set.h5", mode="r") as f:
    patch_df = pd.DataFrame.from_records(f["patch_000"]["patch_df"][()])
    fluxes = patch_df["flux"].values[patch_df["flux_mask"]]
    del patch_df

shape_true = results["shape_true"]
for model in ["model_1", "model_2", "model_3"]:
    shape_recon = results[model]["shape"]
    shape_diff = np.linalg.norm(shape_recon - shape_true, axis=1)
    results[model]["shape_diff"] = shape_diff


# %% Plot Deblended Stamps

percentiles = [75, 95, 98.8, 99.92]
values = np.percentile(blend, percentiles)
indices = [np.argmin(np.abs(blend - v)) for v in values]

tit1 = [rf"{fl * 1e6:.2f} $\mu$Jy" for fl in fluxes[indices]]
tit2 = [f"{bl:.3f}" for bl in blend[indices]]

isolated = results["targets"][indices]
blends = results["inputs"][indices]

recon = []
tit_recons = []
for model in ["model_1", "model_2", "model_3"]:
    recon.append(results[model]["output"][indices])
    psnr = results[model]["psnr"][indices]

    shape_diff = results[model]["shape_diff"][indices]

    tit_recons.append([f"{p:.2f} dB/ {sh:.3f}" for p, sh in zip(psnr, shape_diff)])

plot(
    [isolated, blends, recon[0], recon[1], recon[2]],
    cbar=True,
    same_scale=[0, 1, 2, 3, 4],
    scale_row=0,
    caption=["Isolated", "Blended", "VAE-MHA", "VAE-CNN", "CAT"],
    subtitles=[tit1, tit2, tit_recons[0], tit_recons[1], tit_recons[2]],
    fname=RESULTS_DIR + "deblending/stamps.pdf",
)


# %% Add fluxes and save
results["flux"] = fluxes.copy() * 1e6
save(results, DATA_DIR + "deblending_results.pkl")

# %% Plot metrics distribution
psnr_all = [
    results["model_1"]["psnr"],
    results["model_2"]["psnr"],
    results["model_3"]["psnr"],
]
probability_distribution_metric(
    psnr_all,
    clip_left=20,
    fname=RESULTS_DIR + "deblending/psnr.pdf",
)

shape_diff = [
    results["model_1"]["shape_diff"] * 100,
    results["model_2"]["shape_diff"] * 100,
    results["model_3"]["shape_diff"] * 100,
]

# Filter out bad cases: status!=0 or true ellipticty too large
flags = results["status_true"] * (
    np.sqrt(np.sum(results["shape_true"] ** 2, axis=1)) < 0.6
)
for model in ["model_1", "model_2", "model_3"]:
    flags = flags * results[model]["status"]

for i in range(len(shape_diff)):
    shape_diff[i][flags != 0] = np.NaN


probability_distribution_metric(
    shape_diff,
    metric_name=r"$\Delta \epsilon \,[\times 100]$",
    clip_left=0,
    clip_right=10,
    fname=RESULTS_DIR + "deblending/ellipticity.pdf",
)

# %% Binned boxplots
flux = results["flux"]
blend = results["blend"]
metrics = np.array([shape_diff, psnr_all])

# binned_boxplot(
#     flux,
#     metrics,
#     bin_edges=[10, 40, 70, 100, 130, 160, 200],
#     fname=RESULTS_DIR + "deblending/box_plot_flux.pdf",
# )

log_edges = np.logspace(-2, 0, 7)

binned_boxplot(
    results["blend"],
    metrics,
    bin_edges=log_edges,
    logx=True,
    stat_name="Blendedness",
    legend=True,
    fname=RESULTS_DIR + "deblending/box_plot_blend.pdf",
)
