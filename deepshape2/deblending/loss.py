# %%
import torch
import torch.nn.functional as F

from deepshape2.utils import psnr_torch

__all__ = ["vae_loss", "validation_loss"]


# %% Loss Functions


def circ_mask(device, height=128, width=128, radius=64):
    y = torch.arange(height, device=device).view(-1, 1)
    x = torch.arange(width, device=device).view(1, -1)
    center_y, center_x = height // 2, width // 2
    dist_from_center = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = (dist_from_center <= radius).float()
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    return mask


def vae_loss(target, recon, mu, logvar, beta, device, alpha=0.7, mask=None):
    # Generate mask once and broadcast if not provided
    if mask is None or mask.shape[0] != target.shape[0]:
        mask_single = circ_mask(device, target.shape[2], target.shape[3])
        mask = mask_single.expand(target.shape[0], -1, -1, -1)

    # Reconstruction Loss
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
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            out = model(inp)

            if isinstance(out, (tuple, list)):
                out = out[0]  # Get reconstructed output

            # Scale images
            target_sc = torch.sinh(target) / scale_fac
            out_sc = torch.sinh(out) / scale_fac

            # Compute batch PSNR on GPU
            batch_psnr = psnr_torch(target_sc, out_sc)
            val_loss_all.append(-batch_psnr)

    # Concatenate all batches and move to CPU once
    all_psnrs = torch.cat(val_loss_all)
    return all_psnrs.mean().item()
