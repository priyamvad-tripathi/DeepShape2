# %%
import numpy as np
import torch
from deepinv.models import DRUNet

from deepshape2.utils import get_freest_gpu, load_config, set_seed
from deepshape2.visualization import plot

device = get_freest_gpu(set_device=True)


set_seed()

# %%
MODEL_DIR = load_config()["MODEL_DIR"]

denoiser = DRUNet(in_channels=1, out_channels=1, pretrained=None, device=device)
denoiser = denoiser.eval()


ckpt = torch.load(
    MODEL_DIR / "drunet_blended.pt",
    map_location=device,
    weights_only=False,
)
denoiser.load_state_dict(ckpt["best_weights"])

# %%

# SIGMA = 0.71e-6
# sigmas = np.geomspace(SIGMA, 0.001*SIGMA, 10)
sigmas = np.geomspace(0.7, 0.002, 10)

N_SAMPLES = 10


def denoiser_to_score(denoiser, x_noisy, sigma):
    with torch.inference_mode():
        denoised = denoiser(x_noisy, sigma)
        score = (denoised - x_noisy) / (sigma**2)
    return score


samples0 = torch.randn(N_SAMPLES, 1, 128, 128).to(device)


T = 100
epsilon = 1e-6
sigma_L = sigmas[-1]
all_steps = []
all_steps.append(samples0.clone())

samples = samples0.clone()

for ns, sigma in enumerate(sigmas, start=1):
    alpha = epsilon * (sigma**2) / (sigma_L**2)

    for t in range(T):
        score = denoiser_to_score(denoiser, samples, sigma)
        z = torch.randn_like(samples)
        samples = samples + 0.5 * alpha * score + np.sqrt(alpha) * z

    # store the sample after this step
    all_steps.append(samples.clone())

# convert to numpy images
imgs = [s.detach().cpu().numpy().squeeze() for s in all_steps]

# captions: Step 0, Step 1, ..., Step 10
caps = [f"Step {i}" for i in range(len(imgs))]

plot(imgs, cbar=True, caption=caps, max_imgs=11)
