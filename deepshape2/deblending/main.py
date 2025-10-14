# %%Import Libraries
import numpy as np
import torch
from torch.utils.data import DataLoader

from deepshape2.data import loaders
from deepshape2.deblending import plot_bad_cases, predict, train
from deepshape2.models import VAE
from deepshape2.utils import get_freest_gpu, load_config, set_seed
from deepshape2.visualization import plot_losses

# %% Set default parameters
cfg = load_config()

DATA_DIR = cfg["DATA_DIR"]
beta = cfg["VAE_beta"]
lr_init = cfg["VAE_lr_init"]

loc_weights = cfg["MODEL_DIR"] + "vae_deblender.pt"

scale_fac = cfg["scale_fac"]

# Torch Parameters
device = get_freest_gpu(set_device=True)
set_seed()

# %% Load Data into loaders

group_names = [f"patch_{nl + 1:03d}" for nl in range(50)]


group_names_train, group_names_val = group_names[:45], group_names[45:]

# Split into train and validation sets
train_dataset = loaders.BlendDataset(
    path=DATA_DIR + "sky_50.h5",
    x_key="blended_stamps",
    y_key="isolated_stamps",
    groups=group_names_train,
    scale_fac=scale_fac,
)

val_dataset = loaders.BlendDataset(
    path=DATA_DIR + "sky_50.h5",
    x_key="blended_stamps",
    y_key="isolated_stamps",
    groups=group_names_val,
    scale_fac=scale_fac,
)

# Initialize DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

# Load the VAE model
model = VAE()
model = model.to(device)
model = torch.compile(model)

# %% Train the model and plot results
n_epochs = 151

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

metrics = predict(model, best_weights, val_loader, device, scale_fac=scale_fac)

# %% Plot the worst cases
plot_bad_cases(metrics, category="input", scale_fac=scale_fac)
plot_bad_cases(metrics, category="output", scale_fac=scale_fac)
