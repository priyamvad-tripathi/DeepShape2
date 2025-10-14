# %%Import Libraries
import torch
from torch.utils.data import DataLoader

from deepshape2.data import loaders
from deepshape2.deblending import plot_bad_cases, predict
from deepshape2.models import VAE
from deepshape2.utils import get_freest_gpu, load_config, set_seed

scale_fac = 1e7
# %% Set default parameters
cfg = load_config()

DATA_DIR = cfg["DATA_DIR"]
beta = cfg["VAE_beta"]
lr_init = cfg["VAE_lr_init"]

loc_weights = cfg["MODEL_DIR"] + f"vae_deblender_{scale_fac:.0e}.pt"

# Torch Parameters
device = get_freest_gpu(set_device=True)
set_seed()

# %% Load Data and Model

test_dataset = loaders.BlendDataset(
    path=DATA_DIR + "deep_set.h5",
    x_key="blended_stamps",
    y_key="isolated_stamps",
    groups=["patch_000"],
    scale_fac=scale_fac,
)


# Initialize DataLoaders
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# Load the VAE model
model = VAE()
model = model.to(device)

# %% Load Model Weights
checkpoint = torch.load(loc_weights, map_location=device, weights_only=False)
metrics = predict(
    model, checkpoint["best_weights"], test_loader, device, scale_fac=scale_fac
)

# %% Plot the worst cases
plot_bad_cases(metrics, category="input", scale_fac=scale_fac)
plot_bad_cases(metrics, category="output", scale_fac=scale_fac)
