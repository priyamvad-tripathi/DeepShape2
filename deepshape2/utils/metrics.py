# %% Import modules
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

__all__ = [
    "psnr_batch",
    "ssim_batch",
    "blendedness",
    "contamination",
    "psnr_torch",
]


# %% Reconstruction Metrics
def psnr_batch(true_images, recon_images):
    if true_images.shape != recon_images.shape:
        raise ValueError("Input arrays must have the same shape.")

    if true_images.ndim != 3:
        true_images = true_images[None, ...]
        recon_images = recon_images[None, ...]

    # Compute mean squared error per image
    mse = np.mean((true_images - recon_images) ** 2, axis=(-2, -1))
    # mse = np.clip(mse, 1e-10, None)  # prevent division by zero

    # Compute max value per image
    max_vals = np.max(true_images, axis=(-2, -1))

    psnr = 10 * np.log10((max_vals**2) / mse)
    return psnr


def ssim_batch(targets, recons):
    """
    Compute average SSIM for a batch of grayscale images using skimage.

    Args:
        targets: numpy array of shape [N, H, W]
        recons: numpy array of shape [N, H, W]
        kwargs: additional arguments for skimage's ssim (e.g., data_range)

    Returns:
        mean_ssim: scalar float
        all_ssim: list of SSIM values per image
    """
    assert targets.shape == recons.shape
    N = targets.shape[0]
    all_ssim = np.zeros(N)

    for i in range(N):
        ssim_val = ssim(
            targets[i], recons[i], data_range=recons[i].max() - recons[i].min()
        )
        all_ssim[i] = ssim_val

    return all_ssim


# %% Torch versions for GPU acceleration
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


# %% Quality Metrics for Blended Images
def blendedness(true_images, blended_images):
    true_images = np.asarray(true_images, dtype=float)
    blended_images = np.asarray(blended_images, dtype=float)

    if true_images.shape != blended_images.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Ensure 3D: (batch, H, W)
    if true_images.ndim != 3:
        true_images = true_images[None, ...]
        blended_images = blended_images[None, ...]

    batch_size = true_images.shape[0]
    result = np.zeros(batch_size, dtype=float)

    for i in range(batch_size):
        t = true_images[i]
        b = blended_images[i]

        # Mask of valid pixels (both arrays not NaN)
        mask = np.isfinite(t) & np.isfinite(b)

        num = np.sum(t[mask] * t[mask])
        denom = np.sum(b[mask] * t[mask])

        result[i] = 1 - num / denom if denom != 0 else np.nan

    return result


def contamination(true_images, blended_images):
    true_images = np.asarray(true_images, dtype=float)
    blended_images = np.asarray(blended_images, dtype=float)

    if true_images.shape != blended_images.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Ensure 3D: (batch, H, W)
    if true_images.ndim != 3:
        true_images = true_images[None, ...]
        blended_images = blended_images[None, ...]

    batch_size = true_images.shape[0]
    result = np.zeros(batch_size, dtype=float)

    for i in range(batch_size):
        t = true_images[i]
        b = blended_images[i]

        mask = np.isfinite(t) & np.isfinite(b)

        denom = np.sum(t[mask])
        num = np.sum(b[mask])

        result[i] = (num / denom - 1) if denom != 0 else np.nan

    return result
