# %%Import Libraries
import numpy as np
import torch

from deepshape2.data.loaders import ShapeDatasetLight, dataloader
from deepshape2.models import shapenet_full
from deepshape2.shape_measurement.main import predict, train
from deepshape2.utils import get_freest_gpu, load_config, set_seed
from deepshape2.visualization import plot_bias, plot_losses

# %% Load Config and Set Parameters
cfg = load_config()
TQDM_FLAG = cfg["TQDM"]

DATA_DIR = cfg["DATA_DIR"]
MODEL_DIR = cfg["MODEL_DIR"]


BSIZE = 64
thresh = 6

loc_weights = MODEL_DIR + f"shape_full_thresh_{thresh}.pt"
print("Weights location:", loc_weights)


# Torch Parameters
device = get_freest_gpu(set_device=True)
set_seed()


# %% Load Data and Model
dataset = ShapeDatasetLight(
    path=DATA_DIR + "trainset.h5",
    thresh=thresh * 0.71e-6,
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


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-5,
)

loss_fn = torch.nn.MSELoss()

n_epochs = 300

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=40,
    min_lr=1e-6,
)

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
