# %%
import numpy as np
import torch
from deepinv.models import DRUNet

from deepshape2.utils import get_freest_gpu, load_h5, set_seed
from deepshape2.visualization import plot

device = get_freest_gpu(set_device=True)

data = load_h5("/scratch/tripathi/Data_DS_1/dataset_dirty_train_sers.h5")

set_seed()

# %%
denoiser = DRUNet(in_channels=1, out_channels=1, pretrained="download", device=device)
denoiser = denoiser.eval()


ckpt = torch.load(
    "/scratch/tripathi/DS2/Model_weights/drunet_128.pt",
    map_location=device,
    weights_only=False,
)
denoiser.load_state_dict(ckpt["best_weights"])

# %%

# SIGMA = 0.71e-6
# sigmas = np.geomspace(SIGMA, 0.001*SIGMA, 10)
sigmas = np.geomspace(50 / 255, 1 / 255, 10)

N_SAMPLES = 10


def denoiser_to_score(denoiser, x_noisy, sigma):
    with torch.inference_mode():
        denoised = denoiser(x_noisy, sigma)
        score = (denoised - x_noisy) / (sigma**2)
    return score


true_images = torch.from_numpy(data["true image"][:N_SAMPLES]).unsqueeze(1).to(device)
true_images = true_images / true_images.amax(dim=(1, 2, 3)).view(-1, 1, 1, 1)
samples0 = true_images * 1 + torch.randn_like(true_images) * 0.2

# samples = torch.rand(N_SAMPLES, 1, 128, 128).to(device)
# samples = samples*SIGMA


T = 1000
epsilon = 1e-6
sigma_L = sigmas[-1]
for ns, sigma in enumerate(sigmas):
    # print(f"Sampling step {ns + 1}/{len(sigmas)} with sigma={sigma:.2e}")
    alpha = epsilon * (sigma**2) / (sigma_L**2)
    for t in range(T):
        if ns == 0 and t == 0:
            samples = samples0.clone()
        score = denoiser_to_score(denoiser, samples, sigma)
        # print(f" Score: {torch.norm(score):.2e}")
        z = torch.randn_like(samples)
        samples = samples + 0.5 * alpha * score + np.sqrt(alpha) * z


plot(
    [
        true_images.detach().cpu().numpy().squeeze(),
        samples0.detach().cpu().numpy().squeeze(),
        samples.detach().cpu().numpy().squeeze(),
    ],
    cbar=True,
    caption=["True", "Step 0", "Final Samples"],
)
