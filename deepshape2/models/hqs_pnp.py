import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorist import Color
from deepinv.models import DRUNet
from torch.fft import fft2, ifft2, ifftshift

from ..utils.io import load_config

cfg = load_config()

__all__ = ["HQS_PnP"]

_HP_KEYS = ("niter", "f1", "f2", "falpha", "SIGMA")


class HQS_PnP(nn.Module):
    def __init__(
        self,
        niter=None,
        f1=None,
        f2=None,
        falpha=None,
        SIGMA=None,
        denoiser=None,
        defaults=None,
        verbose_hp=True,
    ):
        super().__init__()

        given = dict(niter=niter, f1=f1, f2=f2, falpha=falpha, SIGMA=SIGMA)
        default = defaults if defaults is not None else cfg["hqs_hyperparams"]

        hp = {k: (default[k] if given[k] is None else given[k]) for k in _HP_KEYS}

        fallback = [k for k in _HP_KEYS if given[k] is None]
        if fallback:
            shown = ", ".join(f"{k}={hp[k]:g}" for k in fallback)
            print(
                f"  {Color.YELLOW}default hqs_hyperparams (SKA-MID tuning){Color.OFF}"
                f" -> {shown}"
            )

        if verbose_hp:
            shown = ", ".join(
                f"{k}={hp[k]:g}" + ("*" if given[k] is None else "") for k in _HP_KEYS
            )
            tag = (
                f" {Color.YELLOW}(* = default hqs_hyperparams){Color.OFF}"
                if fallback
                else ""
            )
            print(f"  {shown}{tag}")

        self.hparams = hp
        self.denoiser = (
            denoiser
            if denoiser is not None
            else DRUNet(in_channels=1, out_channels=1, pretrained=None)
        )

        sigma_np = np.geomspace(
            hp["f1"] * hp["SIGMA"], hp["f2"] * hp["SIGMA"], hp["niter"]
        )[::-1].copy()
        sigma_k = torch.tensor(sigma_np, dtype=torch.float32)
        self.register_buffer("sigma_k", sigma_k, persistent=False)
        self.register_buffer(
            "alpha_k", hp["falpha"] * (hp["SIGMA"] ** 2) / sigma_k**2, persistent=False
        )

    def set_hyperparams(self, **kw):
        """Rebuild the sigma/alpha schedules in place. Returns self."""
        hp = {**self.hparams, **kw}
        sigma_np = np.geomspace(
            hp["f1"] * hp["SIGMA"], hp["f2"] * hp["SIGMA"], hp["niter"]
        )[::-1].copy()
        dev = next(self.denoiser.parameters()).device
        self.sigma_k = torch.tensor(sigma_np, dtype=torch.float32, device=dev)
        self.alpha_k = hp["falpha"] * (hp["SIGMA"] ** 2) / self.sigma_k**2
        self.hparams = hp
        return self

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


# def create_model(device, path=cfg["MODEL_DIR"] / "drunet_blended.pt", **params):
#     """
#     Creates HQS_PnP model with DRUNet denoiser.

#     If `path` is provided, loads checkpoint weights from there.
#     Otherwise, download pretrained weights from DeepInv.
#     """
#     if path is None:
#         print("Downloading pretrained weights from DeepInv.")
#         denoiser = DRUNet(
#             in_channels=1, out_channels=1, pretrained="download", device=device
#         )
#     else:
#         denoiser = DRUNet(in_channels=1, out_channels=1, pretrained=None, device=device)
#         ckpt = torch.load(path, map_location=device, weights_only=False)
#         denoiser.load_state_dict(ckpt["best_weights"])

#     # ! Load hyperparameters from config file if not provided by user
#     default = cfg["hqs_hyperparams"]
#     hyperparams = {**default, **params}

#     # unknown = hyperparams.keys() - default.keys()
#     # if unknown:
#     #     raise ValueError(f"Unknown hyperparameters: {unknown}")

#     model = HQS_PnP(
#         niter=hyperparams["niter"],
#         f1=hyperparams["f1"],
#         f2=hyperparams["f2"],
#         falpha=hyperparams["falpha"],
#         denoiser=denoiser,
#         SIGMA=hyperparams["SIGMA"],
#     ).to(device)

#     model.eval()
#     return model
