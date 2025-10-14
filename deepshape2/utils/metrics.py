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
    "ssim_torch",
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

    psnr = 10 * np.log10((max_vals**2) / (mse + 1e-10))
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
