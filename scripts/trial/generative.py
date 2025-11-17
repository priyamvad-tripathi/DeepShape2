# %%
import numpy as np
import torch
from deepinv.models import DRUNet

from deepshape2.utils import get_freest_gpu
from deepshape2.visualization import plot

device = get_freest_gpu(set_device=True)

# %%
denoiser = DRUNet(in_channels=1, out_channels=1, pretrained="download", device=device)
denoiser.eval()

# %%

SIGMA = 50 / 255.0
# sigmas = np.geomspace(SIGMA, 0.001*SIGMA, 10)
sigmas = np.geomspace(SIGMA, 1 / 255, 10)

N_SAMPLES = 10


def denoiser_to_score(denoiser, x_noisy, sigma):
    with torch.inference_mode():
        denoised = denoiser(x_noisy, sigma)
        score = (denoised - x_noisy) / (sigma**2)
    return score


samples = torch.rand(N_SAMPLES, 1, 128, 128).to(device)
# samples = samples*SIGMA


T = 200
epsilon = 1e-06
sigma_L = sigmas[-1]
for ns, sigma in enumerate(sigmas):
    print(f"Sampling step {ns + 1}/{len(sigmas)} with sigma={sigma:.2e}")
    plot(samples.detach().cpu().numpy().squeeze(), cbar=True)
    alpha = epsilon * (sigma**2) / (sigma_L**2)
    for t in range(T):
        score = denoiser_to_score(denoiser, samples, sigma)
        z = torch.randn_like(samples)
        samples = samples + 0.5 * alpha * score + np.sqrt(alpha) * z


plot(samples.detach().cpu().numpy().squeeze(), cbar=True)
