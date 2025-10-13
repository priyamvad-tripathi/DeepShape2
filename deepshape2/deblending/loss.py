# %%
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["vae_loss", "psnr_torch", "ssim_torch", "validation_loss"]


# %% Quality Metrics for Single Images
def psnr_torch(true_images: torch.Tensor, recon_images: torch.Tensor):
    if true_images.shape != recon_images.shape:
        raise ValueError("Shapes of true and reconstructed images must match.")

    # Ensure float32 on same device
    true_images = true_images.float()
    recon_images = recon_images.to(true_images.device, dtype=torch.float32)

    # Compute MSE per image (already parallelized on GPU)
    mse = F.mse_loss(recon_images, true_images, reduction="none")
    mse = mse.mean(dim=(1, 2, 3))  # mean over pixels per image

    # Compute max value per image
    max_vals = true_images.amax(dim=(1, 2, 3))

    psnr = 10 * torch.log10((max_vals**2) / (mse))
    return psnr


def ssim_torch(targets, recons, data_range=None, window_size=11, K1=0.01, K2=0.03):
    """Compute SSIM for a batch of grayscale images on GPU."""
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(recons, np.ndarray):
        recons = torch.from_numpy(recons)

    targets = targets.float()
    recons = recons.float()

    device = (
        targets.device
        if targets.is_cuda
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    targets, recons = targets.to(device), recons.to(device)

    if data_range is None:
        data_range = recons.max() - recons.min()

    def _gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32, device=device)
        gauss = torch.exp(-((coords - size // 2) ** 2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        window = (gauss[:, None] @ gauss[None, :]).unsqueeze(0).unsqueeze(0)
        return window

    window = _gaussian_window(window_size)

    mu1 = F.conv2d(targets, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(recons, window, padding=window_size // 2, groups=1)

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = (
        F.conv2d(targets * targets, window, padding=window_size // 2, groups=1) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(recons * recons, window, padding=window_size // 2, groups=1) - mu2_sq
    )
    sigma12 = (
        F.conv2d(targets * recons, window, padding=window_size // 2, groups=1) - mu1_mu2
    )

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    all_ssim = ssim_map.mean(dim=[1, 2, 3])
    mean_ssim = all_ssim.mean()

    return mean_ssim.item(), all_ssim


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
