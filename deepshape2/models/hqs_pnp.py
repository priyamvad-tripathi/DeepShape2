# %%Import Libraries
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepinv.models import DRUNet
from torch.fft import fft2, ifft2, ifftshift

from deepshape2.utils import load_config

cfg = load_config()


# %% Load Model
class HQS_PnP(nn.Module):
    def __init__(
        self,
        niter,
        f1,
        f2,
        falpha,
        denoiser,
        SIGMA=0.71e-06,
    ):
        super().__init__()
        self.denoiser = denoiser
        sigma_np = np.geomspace(f1 * SIGMA, f2 * SIGMA, niter)[::-1].copy()
        self.sigma_k = torch.tensor(sigma_np, dtype=torch.float32)
        self.alpha_k = falpha * (SIGMA**2) / (self.sigma_k**2)

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
        denominator = (fpsf.conj() * fpsf).real
        denominator = denominator + alpha
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

        with torch.inference_mode():
            for alpha, sigma in zip(self.alpha_k, self.sigma_k):
                x = self.iteration_step(z, dirty, fpsf, alpha)
                x = self.unpad_image(x)

                denoised = self.denoiser(x, sigma)
                z.copy_(self.pad_batch(denoised))

        return self.unpad_image(z)


# %%

# Default hyperparameters
DEF_niter = 30
DEF_SIGMA = 0.71e-6
DEF_f1 = 0.1414
DEF_f2 = 1.9979
DEF_falpha = 3.9949

# Predefined locations to check for weights
WEIGHT_PATHS = [
    cfg["GENCI_DIR"] + "drunet_deepinv_gray_finetune_26k.pth",
    cfg["MODEL_DIR"] + "drunet_deepinv_gray_finetune_26k.pth",
]


def resolve_pretrained_path():
    """
    Returns the first existing weight file path from predefined locations.
    If none exists, returns 'download' to automatically download weights.
    """
    for path in WEIGHT_PATHS:
        if path and os.path.exists(path):
            return path
    return "download"


def create_model(device, path=None, **hqs_params):
    """
    Creates HQS_PnP model with DRUNet denoiser.

    If `path` is provided, loads checkpoint weights from there.
    Otherwise, uses pretrained weights from predefined locations or downloads them.
    """
    pretrained_path = resolve_pretrained_path()

    denoiser = DRUNet(
        in_channels=1, out_channels=1, pretrained=pretrained_path, device=device
    )

    if path is not None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        denoiser.load_state_dict(ckpt["best_weights"])

    niter = hqs_params.get("niter", DEF_niter)
    f1 = hqs_params.get("f1", DEF_f1)
    f2 = hqs_params.get("f2", DEF_f2)
    falpha = hqs_params.get("falpha", DEF_falpha)
    SIGMA = hqs_params.get("SIGMA", DEF_SIGMA)

    model = HQS_PnP(
        niter=niter, f1=f1, f2=f2, falpha=falpha, denoiser=denoiser, SIGMA=SIGMA
    ).to(device)

    model.eval()
    return model
