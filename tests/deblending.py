# %%Import Libraries
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from deepshape2.data import loaders
from deepshape2.deblending import predict_multiple
from deepshape2.models import VAE, CAT_Unet
from deepshape2.utils import get_freest_gpu, load, load_config, load_h5, save, set_seed
from deepshape2.visualization import plot, probability_distribution_metric

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


results = predict_multiple(
    [model1, model2, model3],
    [
        loc_weights + "vae_deblender.pt",
        loc_weights + "simple_deblender.pt",
        loc_weights + "cat_deblender.pt",
    ],
    test_loader,
    device,
)

# %% Load fluxes and blends
try:
    results
except NameError:
    results = load(DATA_DIR + "deblending_results.pkl")

blend = results["blend"]

percentiles = [75, 91, 99, 99.99]
values = np.percentile(blend, percentiles)
indices = [np.argmin(np.abs(blend - v)) for v in values]

with load_h5(DATA_DIR + "deep_set.h5", mode="r") as f:
    patch_df = pd.DataFrame.from_records(f["patch_000"]["patch_df"][()])
    fluxes = patch_df["flux"].values[patch_df["flux_mask"]]
    del patch_df

# %% Plot Deblended Stamps
tit1 = [rf"{fl * 1e6:.2f} $\mu$Jy" for fl in fluxes[indices]]
tit2 = [f"{bl:.3f}" for bl in blend[indices]]

isolated = results["targets"][indices]
blends = results["inputs"][indices]
shape_true = results["shape_true"][indices]

recon = []
tit_recons = []
for model in ["model_1", "model_2", "model_3"]:
    recon.append(results[model]["output"][indices])
    psnr = results[model]["psnr"][indices]

    shape_recon = results[model]["shape"][indices]
    shape_diff = np.linalg.norm(shape_recon - shape_true, axis=1)

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
results["flux"] = fluxes.copy()
save(results, DATA_DIR + "deblending_results.pkl")

# %% Plot metrics distribution
probability_distribution_metric(
    [
        results["model_1"]["psnr"],
        results["model_2"]["psnr"],
        results["model_3"]["psnr"],
    ],
    clip_left=20,
    fname=RESULTS_DIR + "deblending/psnr.pdf",
)


shape_diff = []
flags = results["status_true"] * (
    np.sqrt(np.sum(results["shape_true"] ** 2, axis=1)) < 0.6
)
for model in ["model_1", "model_2", "model_3"]:
    shape_recon = results[model]["shape"]
    flags = flags * results[model]["status"]
    shape_diff.append(np.linalg.norm(shape_recon - results["shape_true"], axis=1))

for i in range(len(shape_diff)):
    shape_diff[i] = shape_diff[i][flags == 0]
    shape_diff[i] *= 1e2

probability_distribution_metric(
    shape_diff,
    metric_name=r"$100 \, \Delta \epsilon$",
    clip_left=0,
    clip_right=10,
    fname=RESULTS_DIR + "deblending/ellipticity.pdf",
)
