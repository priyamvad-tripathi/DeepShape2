# %%Import Libraries
import torch
from deepshape2.data import loaders
from deepshape2.deblending import predict, train
from deepshape2.models import CAT_Unet
from deepshape2.utils import get_freest_gpu, load_config, set_seed
from torch.utils.data import DataLoader

# %% Set default parameters
cfg = load_config()


beta = cfg["VAE_beta"]
lr_init = cfg["VAE_lr_init"]

DATA_DIR = cfg["DATA_DIR"]
MODEL_DIR = cfg["MODEL_DIR"]

loc_weights = MODEL_DIR + "cat.pt"
SCALE_FACTOR = cfg["SCALE_FACTOR"]

TQDM_FLAG = cfg["TQDM"]

# Torch Parameters
device = get_freest_gpu(set_device=True)
set_seed()

# %% Load Data into loaders

group_names = [f"patch_{nl + 1:03d}" for nl in range(50)]

group_names_train, group_names_val = group_names[:45], group_names[45:50]

BATCH_SIZE = 6

# Split into train and validation sets
train_dataset = loaders.BlendDataset(
    path=DATA_DIR + "wide_set.h5",
    x_key="blended_stamps",
    y_key="isolated_stamps",
    groups=group_names_train,
)

val_dataset = loaders.BlendDataset(
    path=DATA_DIR + "wide_set.h5",
    x_key="blended_stamps",
    y_key="isolated_stamps",
    groups=group_names_val,
)

# Initialize DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

# Load model
model = CAT_Unet(
    img_size=128,
    in_chans=1,
    depth=[4, 6, 6, 8],
    split_size_0=[4, 4, 4, 4],
    dim=48,
    num_heads=[2, 2, 4, 8],
    mlp_ratio=4,
).to(device)

# model = torch.compile(model)

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
    plot=False,
    optimizer=optimizer,
    scheduler_params=scheduler_params,
    save_freq=1,
    tqdm_enabled=TQDM_FLAG,
)


# %% Test
checkpoint = torch.load(loc_weights, map_location=device, weights_only=False)
best_weights = checkpoint["best_weights"]

metrics = predict(
    model,
    best_weights,
    val_loader,
    device,
    n=0,
    tqdm_enabled=TQDM_FLAG,
)
