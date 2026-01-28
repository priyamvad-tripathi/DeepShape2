# %%Import Libraries
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from deepshape2.data.loaders import ShapeDatasetLight, dataloader
from deepshape2.models import shapenet_full
from deepshape2.shape_measurement.main import predict, train2
from deepshape2.utils import get_freest_gpu, load_config, load_h5, set_seed
from deepshape2.visualization import plot_bias, plot_losses

# %% Load Config and Set Parameters
cfg = load_config()
TQDM_FLAG = cfg["TQDM"]

DATA_DIR = cfg["DATA_DIR"]
MODEL_DIR = cfg["MODEL_DIR"]


BSIZE = 64
peak_thresh = 3


loc_weights = MODEL_DIR + f"shape_full_new_thresh_{peak_thresh}.pt"
print("Weights location:", loc_weights)


# Torch Parameters
device = get_freest_gpu(set_device=True)
set_seed()


# %% Load Data and Model
dataset = ShapeDatasetLight(
    path=DATA_DIR + "trainset.h5",
    peak_thresh=peak_thresh,
    flux_thresh=50,
)


train_loader, val_loader = dataloader(
    dataset=dataset,
    batch_size=[BSIZE, BSIZE],
    split=(0.8, 0.2),
)

#  Load model
model = shapenet_full()
model = model.to(device)
# print(model(torch.randn(10, 2, 128, 128).to(device)).size())

# %% Train the model and use it to make predictions
#! Test with different paramters for best results

for p in model.encode.parameters():
    p.requires_grad = False

for name, p in model.encode.named_parameters():
    if "14" in name:
        p.requires_grad = True

psf_params = [p for p in model.encode.parameters() if p.requires_grad]

psf_param_ids = {id(p) for p in psf_params}

main_params = [
    p for p in model.parameters() if p.requires_grad and id(p) not in psf_param_ids
]

optimizer = torch.optim.Adam(
    [
        {"params": main_params, "lr": 1e-4},
        # {"params": psf_params, "lr": 1e-5},
    ],
    weight_decay=1e-5,
)

loss_fn = torch.nn.SmoothL1Loss(beta=0.1)
# loss_fn = torch.nn.MSELoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=40,
    min_lr=1e-6,
)


n_epochs = 300


best_weights, train_loss_list, val_loss_list = train2(
    model,
    train_loader,
    val_loader,
    epochs=n_epochs,
    device=device,
    filename=loc_weights,
    optimizer=optimizer,
    scheduler=scheduler,
    save_freq=20,
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

# %% Calculate results on test set
hf_test = load_h5(DATA_DIR + "deep_set.h5")["patch_000"]

recon = hf_test["isolated_dirty_psf"]["recon"][:]
psf = hf_test["isolated_dirty_psf"]["psf"][:]
images = np.stack([recon, psf], axis=1).astype(np.float32)

img_min = images.min(axis=(2, 3), keepdims=True)
img_max = images.max(axis=(2, 3), keepdims=True)
images = (images - img_min) / (img_max - img_min)


df = hf_test["patch_df"][()]
flux_mask = df["flux_mask"]
shapes = np.stack([df["e1"][flux_mask], df["e2"][flux_mask]], axis=1).astype(np.float32)
flux = df["flux"][flux_mask].astype(np.float32)

peaks = hf_test["peaks"][:]

mask = (flux > 50 * 1e-6) & (peaks > peak_thresh * 0.71e-6)


images_T = torch.tensor(images[mask], dtype=torch.float32)
shapes_T = torch.tensor(shapes[mask], dtype=torch.float32)
dataset = TensorDataset(images_T, shapes_T)

testloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
)

ypred, ytest, images = predict(
    model,
    weights=best_weights,
    data_loader=testloader,
    device=device,
    tqdm_enabled=TQDM_FLAG,
)

# Calculate Bias
plot_bias(ypred, ytest, power=1e3, ellipticity_cutoff=1, lim=0.4)
print(
    f"The pearson coefficients are: {1 - np.corrcoef(ytest[:, 0], ypred[:, 0])[0, 1]:.2e}/{1 - np.corrcoef(ytest[:, 1], ypred[:, 1])[0, 1]:.2e}"
)
