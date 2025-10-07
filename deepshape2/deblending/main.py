# %%Import Libraries
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepshape2.data import loaders
from deepshape2.deblending.training import predict, train
from deepshape2.models.vae import VAE
from deepshape2.utils import get_freest_gpu, load_config
from deepshape2.visualization import plot, plot_losses

# %% Set default parameters
cfg = load_config()

DATA_DIR = cfg["DATA_DIR"]
beta = cfg["VAE_beta"]
lr_init = cfg["VAE_lr_init"]

loc_weights = cfg["MODEL_DIR"] + "vae_deblender_30_200.pt"


# Torch Parameters
ndev = get_freest_gpu()

torch.cuda.set_device(ndev)
torch.cuda.empty_cache()
device = torch.device(f"cuda:{ndev}")

# Seed for reproducibility
torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2024)


# %% Load Data into loaders

group_names = [f"patch_{nl + 1:03d}" for nl in range(30)]


group_names_train, group_names_val = group_names[:25], group_names[25:]

# Split into train and validation sets
train_dataset = loaders.BlendDataset(
    path=DATA_DIR + "sky2.h5",
    x_key="blended_stamps",
    y_key="isolated_stamps",
    groups=group_names_train,
    min_flux=30e-6,
)

val_dataset = loaders.BlendDataset(
    path=DATA_DIR + "sky2.h5",
    x_key="blended_stamps",
    y_key="isolated_stamps",
    groups=group_names_val,
    min_flux=30e-6,
)

# Initialize DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# Load the VAE model
model = VAE().to(device)

# %% Train the model and plot results
n_epochs = 101

scheduler_params = {"factor": 0.5, "patience": 15, "min_lr": lr_init / (2**5)}


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr_init,
    weight_decay=lr_init,
)


best_weights, train_loss_list, val_loss_list = train(
    model,
    train_loader,
    val_loader,
    beta=beta,
    epochs=n_epochs,
    device=device,
    filename=loc_weights,
    plot=True,
    optimizer=optimizer,
    scheduler_params=scheduler_params,
    save_freq=50,
)


# %% Test
checkpoint = torch.load(loc_weights, map_location=device, weights_only=False)
best_weights = checkpoint["best_weights"]
beta = checkpoint["beta"]
recon_loss = checkpoint["recon_loss_list"]
kl_loss = checkpoint["kl_loss_list"]
val_loss = checkpoint["val_loss_list"]
train_loss = checkpoint["train_loss_list"]

plot_losses(
    [train_loss, recon_loss, beta * np.array(kl_loss)],
    labels=["Train", "Recon", "KL"],
    skip=0,
    logscale=True,
)

plot_losses([checkpoint["lr_list"]], labels=["Learning Rate"], skip=0, logscale=True)

plot_losses(
    [val_loss],
    labels=["Val"],
    skip=0,
)

metrics = predict(model, best_weights, val_loader, device)

# %% Plot the worst cases

inputs = metrics["input"]
outputs = metrics["output"]
targets = metrics["targets"]

psnr_out = metrics["psnr_out"]
ssim_out = metrics["ssim_out"]

psnr_in = metrics["psnr_in"]
ssim_in = metrics["ssim_in"]


names = ["PSNR", "SSIM"]

for stat, name in zip([psnr_out, ssim_out], names):
    ind = np.argsort(stat)[:5]

    tit_in = [f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_in[ind], psnr_in[ind])]
    tit_out = [f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_out[ind], psnr_out[ind])]
    blank_titles = [None] * len(tit_in)

    plot(
        images=[targets[ind], inputs[ind], outputs[ind], targets[ind] - outputs[ind]],
        caption=["Target", "Input", "Recon", "Residual"],
        cbar=True,
        scale_row=0,
        same_scale=[0, 1, 2],
        subtitles=[blank_titles, tit_in, tit_out, blank_titles],
        suptitle=f"Worst Reconstructions ({name})",
    )


for stat, name in zip([psnr_in, ssim_in], names):
    ind = np.argsort(stat)[:5]

    tit_in = [f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_in[ind], psnr_in[ind])]
    tit_out = [f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_out[ind], psnr_out[ind])]
    blank_titles = [None] * len(tit_in)

    plot(
        images=[targets[ind], inputs[ind], outputs[ind], targets[ind] - outputs[ind]],
        caption=["Target", "Input", "Recon", "Residual"],
        cbar=True,
        scale_row=0,
        same_scale=[0, 1, 2],
        subtitles=[blank_titles, tit_in, tit_out, blank_titles],
        suptitle=f"Worst Inputs ({name})",
    )
