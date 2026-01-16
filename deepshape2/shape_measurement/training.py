# %%Import Libraries
import numpy as np
import torch
from torch.utils.data import DataLoader

from deepshape2.data.loaders import ShapeDataset
from deepshape2.models import shapenet, shapenet_full
from deepshape2.shape_measurement.main import predict, train
from deepshape2.utils import get_freest_gpu, load_config, set_seed
from deepshape2.visualization import plot_bias, plot_losses

FINAL = False

# %% Load Config and Set Parameters
cfg = load_config()
TQDM_FLAG = cfg["TQDM"]

print("TQDM enabled:", TQDM_FLAG)

DATA_DIR = cfg["DATA_DIR"]
MODEL_DIR = cfg["MODEL_DIR"]

loc_data = DATA_DIR + "wide_set.h5"

if FINAL:
    loc_weights = MODEL_DIR + "shape_network_full.pt"
    keys = ["recon", "psf"]
    model = shapenet_full()
else:
    loc_weights = MODEL_DIR + "shape_true_l1.pt"
    keys = ["isolated_stamps"]
    model = shapenet()

# Torch Parameters
device = get_freest_gpu(set_device=True)
set_seed()

BSIZE = 32

# %% Load Data and Model
group_names = [f"patch_{nl + 1:03d}" for nl in range(51, 71)]


group_names_train, group_names_val = group_names[:15], group_names[15:]

# Split into train and validation sets
train_dataset = ShapeDataset(
    path=loc_data,
    keys=keys,
    groups=group_names_train,
)

val_dataset = ShapeDataset(
    path=loc_data,
    keys=keys,
    groups=group_names_val,
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
# print(model(torch.randn(10, 1, 128, 128).to(device)).size())

# %% Train the model and use it to make predictions
#! Test with different paramters for best results
n_epochs = 301

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-5,
)

scheduler_params = {
    "factor": 0.5,
    "patience": 40,
    "min_lr": 1e-7,
}

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)

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
    loss_fn=torch.nn.SmoothL1Loss(beta=0.01),
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
plot_bias(ypred, ytest, power=1e4, ellipticity_cutoff=1, lim=0.05)
print(
    f"The pearson coefficients are: {1 - np.corrcoef(ytest[:, 0], ypred[:, 0])[0, 1]:.2e}/{1 - np.corrcoef(ytest[:, 1], ypred[:, 1])[0, 1]:.2e}"
)

# %% Save eq weights
if not FINAL:
    model.eval()
    model.load_state_dict(best_weights)
    torch.save(model.eq.state_dict(), MODEL_DIR + "eq_block.pt")
