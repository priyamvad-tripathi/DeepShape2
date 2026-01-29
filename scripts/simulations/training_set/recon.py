# %%
import torch

from deepshape2.models import VAE
from deepshape2.reconstruction import reconstruct_facets_h5
from deepshape2.utils import get_freest_gpu, load_config, set_seed

# %%
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]
device = get_freest_gpu(set_device=True)
set_seed()

ckpt_path = cfg["MODEL_DIR"] + "vae_mha.pt"
deblender = VAE().to(device)
deblender.load_state_dict(torch.load(ckpt_path, map_location=device)["best_weights"])
deblender.eval()

reconstruct_facets_h5(
    h5_path=DATA_DIR + "training_set_52_100.h5", device=device, deblender=deblender
)
