# %%Import Libraries
import numpy as np
import torch
from torch.utils.data import DataLoader

from deepshape2.data.loaders import ShapeDataset
from deepshape2.models import shapenet_full
from deepshape2.shape_measurement.main import TupleSmoothL1WithBias, predict, train
from deepshape2.utils import get_freest_gpu, load_config, set_seed
from deepshape2.visualization import plot_bias, plot_losses

STAGE = 1
# %% Load Config and Set Parameters
cfg = load_config()
TQDM_FLAG = cfg["TQDM"]

print("TQDM enabled:", TQDM_FLAG)

DATA_DIR = cfg["DATA_DIR"]
MODEL_DIR = cfg["MODEL_DIR"]

loc_data = DATA_DIR + "wide_set.h5"


loc_weights = MODEL_DIR + f"shape_full_stg_{STAGE}.pt"
keys = ["recon", "psf"]
model = shapenet_full()

group_names = [f"patch_{nl + 1:03d}" for nl in range(71, 100)]
group_names_train, group_names_val = group_names[:22], group_names[22:]


# Torch Parameters
device = get_freest_gpu(set_device=True)
set_seed()

BSIZE = 32

# %% Load Data and Model

# Split into train and validation sets
train_dataset = ShapeDataset(
    path=loc_data,
    keys=keys,
    groups=group_names_train,
    metric_name="psnr",
    metric_threshold=20,
)

val_dataset = ShapeDataset(
    path=loc_data,
    keys=keys,
    groups=group_names_val,
    metric_name="psnr",
    metric_threshold=20,
)

# Initialize DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BSIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    prefetch_factor=2,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BSIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    prefetch_factor=2,
    persistent_workers=True,
)

#  Load model to device
model = model.to(device)
# print(model(torch.randn(10, 2, 128, 128).to(device)).size())

# %% Train the model and use it to make predictions
#! Test with different paramters for best results

# PSF correction only
if STAGE == 1:
    # Freeze pretrained equivariant features
    for p in model.eq.parameters():
        p.requires_grad = False

    # Train both heads
    for p in model.base_head.parameters():
        p.requires_grad = True

    for p in model.psf_head.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {"params": model.base_head.parameters(), "lr": 1e-3},
            {"params": model.psf_head.parameters(), "lr": 2e-3},
        ],
        weight_decay=0.0,
    )

    loss_fn = TupleSmoothL1WithBias(
        beta=0.05,
        lambda_bias=0.0,
    )

    n_epochs = 40

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=1000,
        min_lr=1e-6,
    )

# Partial Unfreezing
if STAGE == 2:
    # Load best weights from stage 1
    ckpt = torch.load(
        MODEL_DIR + "shape_psf_stg_1.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(ckpt["best_weights"])

    # Unfreeze equivariant backbone
    for p in model.eq.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {"params": model.psf_head.parameters(), "lr": 1e-3},
            {"params": model.base_head.parameters(), "lr": 5e-4},
            {"params": model.eq.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-6,
    )

    loss_fn = TupleSmoothL1WithBias(
        beta=0.05,
        lambda_bias=0.1,
    )

    n_epochs = 80

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=20,
        min_lr=5e-6,
    )

# Full fine-tuning
if STAGE == 3:
    # Load best weights from stage 2
    ckpt = torch.load(
        MODEL_DIR + "shape_psf_stg_2.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(ckpt["best_weights"])

    # Everything already unfrozen, but make it explicit
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {"params": model.psf_head.parameters(), "lr": 5e-4},
            {"params": model.base_head.parameters(), "lr": 5e-5},
            {"params": model.eq.parameters(), "lr": 5e-5},
        ],
        weight_decay=1e-6,
    )

    loss_fn = TupleSmoothL1WithBias(
        beta=0.02,
        lambda_bias=0.05,
    )

    n_epochs = 120

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=15,
        min_lr=1e-6,
    )

# optimizer = torch.optim.Adam(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=5e-4,
#     weight_decay=1e-6,
# )

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     factor=0.5,
#     patience=20,
#     min_lr=1e-6,
# )

best_weights, train_loss_list, val_loss_list = train(
    model,
    train_loader,
    val_loader,
    epochs=n_epochs,
    device=device,
    filename=loc_weights,
    optimizer=optimizer,
    scheduler=scheduler,
    save_freq=10,
    tqdm_enabled=TQDM_FLAG,
    loss_fn=loss_fn,
)

# %%  Calculate bias on validation set
checkpoint = torch.load(loc_weights, map_location=device, weights_only=False)
best_weights = checkpoint["best_weights"]
val_loss = checkpoint["val_loss_list"]
train_loss = checkpoint["train_loss_list"]
lr = checkpoint["lr_list"]

plot_losses([train_loss, val_loss], skip=0, logscale=True)
plot_losses([lr], labels=["Learning Rate"], skip=0, logscale=False)

ypred, ytest, images = predict(
    model,
    weights=best_weights,
    data_loader=val_loader,
    device=device,
    tqdm_enabled=TQDM_FLAG,
)

# Calculate Bias
plot_bias(ypred, ytest, power=1e3, ellipticity_cutoff=1, lim=0.4)
print(
    f"The pearson coefficients are: {1 - np.corrcoef(ytest[:, 0], ypred[:, 0])[0, 1]:.2e}/{1 - np.corrcoef(ytest[:, 1], ypred[:, 1])[0, 1]:.2e}"
)
