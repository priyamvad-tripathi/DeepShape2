# %%Import Libraries
from torch.utils.data import DataLoader

from deepshape2.data import loaders
from deepshape2.deblending import predict_multiple
from deepshape2.models import VAE
from deepshape2.utils import get_freest_gpu, load_config, set_seed

# %% Set default parameters
cfg = load_config()

DATA_DIR = cfg["DATA_DIR"]
beta = cfg["VAE_beta"]
lr_init = cfg["VAE_lr_init"]

loc_weights = cfg["MODEL_DIR"]

# Torch Parameters
device = get_freest_gpu(set_device=True)
set_seed()


# Load the models
model1 = VAE()
model2 = VAE(attention=False)


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
    [model1, model2],
    [loc_weights + "vae_deblender.pt", loc_weights + "simple_deblender.pt"],
    test_loader,
    device,
)
