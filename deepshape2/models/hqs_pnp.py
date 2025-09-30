# %%Import Libraries
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepinv.models import DRUNet
from torch.fft import fft2, ifft2, ifftshift

# %% Load Model


class HQS_PnP(nn.Module):
    def __init__(
        self,
        niter,
        f1,
        f2,
        falpha,
        denoiser,
        # padding=True,
        SIGMA=0.71e-06,
    ):
        super().__init__()
        # self.steps = nn.ModuleList([iteration for _ in range(niter)])
        self.denoiser = denoiser
        self.sigma_k = np.geomspace(f1 * SIGMA, f2 * SIGMA, niter)[::-1]
        self.alpha_k = falpha * (SIGMA**2) / (self.sigma_k**2)
        # self.padding = padding

    def pad_batch(self, im_batch):
        _, _, N, N2 = im_batch.shape
        assert N == N2, "Images must be square"

        pad = (N // 2, N // 2, N // 2, N // 2)
        return F.pad(im_batch, pad, mode="constant", value=0)

    def unpad_image(self, im_batch_padded):
        _, _, H, W = im_batch_padded.shape
        assert H == W, "Images must be square"

        N = H // 2
        return im_batch_padded[:, :, N // 2 : N + N // 2, N // 2 : N + N // 2]

    def iteration_step(self, z, dirty, fpsf, alpha):
        """
        One HQS iteration step using FFT-domain inversion.
        """
        numerator = fpsf.conj() * fft2(dirty) + alpha * fft2(z)
        denominator = (fpsf.conj() * fpsf).real + alpha
        x_fft = numerator / denominator
        x = ifft2(x_fft).real
        return x

    def forward(self, im):
        dirty = im[:, 0, :, :].unsqueeze(1)
        psf = im[:, 1, :, :].unsqueeze(1)

        dirty = self.pad_batch(dirty)
        psf = self.pad_batch(psf)

        fpsf = fft2(ifftshift(psf, dim=(-2, -1)))

        z = dirty.clone().detach()

        for alpha, sigma in zip(self.alpha_k, self.sigma_k):
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x = self.iteration_step(z, dirty, fpsf, alpha).clone().detach()

                x = self.unpad_image(x)

                if self.denoiser is not None:
                    denoised = self.pad_batch(self.denoiser(x, sigma).clone().detach())
                    z.copy_(denoised)
                else:
                    z.copy_(x)

        return self.unpad_image(z)


# %%

try:
    path0 = os.environ["PATH_DEN"]
except KeyError:
    path0 = "download"


# %%
# Best Hyperparameters found by PnP_tuning.py
niter = 30
SIGMA = 0.71e-06
f1 = 0.1414
f2 = 1.9979
falpha = 3.9949


def create_model(
    device,
    niter=niter,
    f1=f1,
    f2=f2,
    falpha=falpha,
    path=None,
    SIGMA=SIGMA,
):
    denoiser = DRUNet(in_channels=1, out_channels=1, pretrained=path0, device=device)

    if path is not None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        denoiser.load_state_dict(ckpt["best_weights"])

    model = HQS_PnP(
        niter=niter,
        f1=f1,
        f2=f2,
        falpha=falpha,
        denoiser=denoiser,
        SIGMA=SIGMA,
    ).to(device)

    model.eval()

    return model
    return model
    return model
    return model
