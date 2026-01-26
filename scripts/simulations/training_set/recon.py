# %%
import numpy as np
import torch

from deepshape2.models import VAE
from deepshape2.reconstruction import reconstruct_facets
from deepshape2.utils import get_freest_gpu, load_config, load_h5, set_seed

# %%
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]
device = get_freest_gpu(set_device=True)
set_seed()

ckpt_path = cfg["MODEL_DIR"] + "vae_mha.pt"
deblender = VAE().to(device)
deblender.load_state_dict(torch.load(ckpt_path, map_location=device)["best_weights"])
deblender.eval()

hf = load_h5(DATA_DIR + "training_set.h5", "a", delete_if_exists=False)

dirty = hf["dirty"][:]
psf = hf["psf"][:]


result = reconstruct_facets(
    dirty,
    psf,
    device=device,
    num_workers=4,
    deblender=deblender,
)

recon = result["recon"].astype(np.float32)

hf.create_dataset(
    "recon",
    data=recon,
    dtype=np.float32,
    chunks=(1, 128, 128),
    shape=(0, 128, 128),
    maxshape=(None, 128, 128),
)
hf.close()
