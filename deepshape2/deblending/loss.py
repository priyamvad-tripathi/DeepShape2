# %%
import numpy as np
import torch
import torch.nn.functional as F

from deepshape2.utils import psnr_torch

__all__ = ["vae_loss", "validation_loss"]


# %% Loss Functions
def circ_mask(device, height=128, width=128, radius=64):
    y, x = np.ogrid[:height, :width]
    center = (height // 2, width // 2)
    dist_from_center = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    mask = (dist_from_center <= radius).astype(float)
    mask = (
        torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    )
    return mask


def vae_loss(target, recon, mu, logvar, beta, device, alpha=0.7):
    mask = circ_mask(device)
    mask = mask.repeat(target.shape[0], 1, 1, 1)

    # Reconstruction Loss (MSE or BCE depending on your data)
    recon_loss = F.mse_loss(target, recon, reduction="sum")
    central_loss = F.mse_loss(target * mask, recon * mask, reduction="sum")

    # KL Divergence Loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss * (1 - alpha) + central_loss * alpha + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def validation_loss(model, val_loader, device, scale_fac):
    model.eval()
    val_loss_all = []

    with torch.inference_mode():
        for inp, target in val_loader:
            inp = inp.to(device)
            target = target.to(device)

            out = model(inp)
            out = out[0]

            out = torch.sinh(out) / scale_fac
            target = torch.sinh(target) / scale_fac

            batch_psnr = psnr_torch(target, out)
            val_loss_all.append(-batch_psnr.cpu())

    # Concatenate and average
    all_psnrs = torch.cat(val_loss_all)
    return all_psnrs.mean().item()
