# %%Import Libraries
import copy
import os
import time

import numpy as np
import torch
from colorist import Color
from deepinv.models import DRUNet
from torch.utils.data import DataLoader, SubsetRandomSampler

from deepshape2.data.loaders import DenoiseDataset
from deepshape2.models import RefineNet
from deepshape2.utils import (
    get_freest_gpu,
    get_progress_bar,
    get_tqdm,
    load_ckp,
    load_config,
    psnr_torch,
    save_ckp,
    set_seed,
    ssim_batch,
    time_string,
)
from deepshape2.visualization import plot, plot_losses

# %% Defaults and Configurations
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]
MODEL_DIR = cfg["MODEL_DIR"]
TQDM_FLAG = cfg["TQDM"]

run_env = os.getenv("RUN_ENV", "local")
if run_env == "genci":
    BATCH_SIZE = 24
else:
    BATCH_SIZE = 16


SIGMA = 0.71e-06
NITER = 10
SIGMA_VALS = np.geomspace(2 * SIGMA, 0.1 * SIGMA, NITER)
SIGMA_DICT = {idx: sig for idx, sig in enumerate(SIGMA_VALS)}
CROP_SIZE = 96


MSE_FACTOR = 1

loc_data = DATA_DIR + "wide_set.h5"
loc_weights = MODEL_DIR + f"denoiser_isolated_{CROP_SIZE}.pt"

device = get_freest_gpu(set_device=True)
set_seed()

lr_init = 1e-4

tqdm_kwargs = get_tqdm()
# %% Denoiser Model Setup and data
group_names = [f"patch_{nl + 1:03d}" for nl in range(50)]


group_names_train, group_names_val = group_names[:40], group_names[40:41]

# Split into train and validation sets
train_dataset = DenoiseDataset(
    path=loc_data,
    key="isolated_stamps",
    # key="blended_stamps",
    groups=group_names_train,
    crop=CROP_SIZE,
)


val_dataset = DenoiseDataset(
    path=loc_data,
    key="isolated_stamps",
    # key="blended_stamps",
    groups=group_names_val,
    crop=CROP_SIZE,
)

# Initialize DataLoaders
total_size = len(train_dataset)
subset_size = 100_000

indices = np.random.choice(total_size, subset_size, replace=False)
sampler = SubsetRandomSampler(indices)

indices_val = np.random.choice(len(val_dataset), 10_000, replace=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset[indices_val],
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)


model = RefineNet(n_noise_scale=len(SIGMA_DICT))
model = model.to(device)
# model = torch.compile(model)
# %% Define Training and Testing Function


def process_batch(clean_batch, device):
    # Move to device; ensure float
    clean = clean_batch.to(device, non_blocking=True).float()
    N = clean.size(0)

    # optional random rotation / flip
    if torch.rand(1) < 0.5:
        clean = clean.flip(-1)
    if torch.rand(1) < 0.5:
        clean = clean.flip(-2)

    # Random PSNR target per sample
    target_psnr = torch.rand(N, device=device) * 49.5 + 0.5  # range [0.5, 50]
    peak = clean.abs().amax(dim=(1, 2, 3))  # shape (N,)

    # Choose sigma values from dictionary (indices sampled uniformly)
    sigma_idx = torch.randint(
        0, len(SIGMA_DICT), size=(N,), device=device, dtype=torch.long
    )

    # Convert selected sigma values to a tensor
    sigma_vals = torch.tensor(
        [SIGMA_DICT[int(i)] for i in sigma_idx.cpu()], device=device, dtype=clean.dtype
    ).view(N, 1, 1, 1)

    # Correct scaling factor
    scale = (target_psnr / peak).view(N, 1, 1, 1) * sigma_vals
    clean_scaled = clean * scale

    # Add noise
    noisy = clean_scaled + torch.randn_like(clean_scaled) * sigma_vals

    return noisy.float(), clean_scaled.float(), sigma_idx, target_psnr, sigma_vals


def train_denoiser(
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    device,
    sigma_dict=SIGMA_DICT,
    **kwargs,
):
    start_time = time.time()
    best_val_loss = np.inf
    best_weights = None
    current_epoch = 0
    best_epoch = 0

    train_loss_list, val_loss_list, lr_list = [], [], [np.inf]

    # --- Config ---
    filename = kwargs.get("filename")
    scheduler = kwargs.get("scheduler", None)
    save_freq = kwargs.get("save_freq", 50)
    precision = kwargs.get("precision", 4)

    print(f"Running on device: {device}")

    # --- Load checkpoint ---
    try:
        model, optimizer, checkpoint = load_ckp(filename, model, optimizer, device)
        current_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", np.inf)
        best_weights = checkpoint.get("best_weights")
        val_loss_list = checkpoint.get("val_loss_list", [])
        train_loss_list = checkpoint.get("train_loss_list", [])
        lr_list = checkpoint.get("lr_list", [])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {current_epoch}")
    except (AttributeError, FileNotFoundError, TypeError):
        print("No saved checkpoints found. Starting from scratch.")

    # --- Training loop ---
    mse = torch.nn.MSELoss()

    for epoch in range(epochs):
        if epoch < current_epoch:
            continue

        model.train()
        batch_losses = []

        if scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            new_lr = current_lr < lr_list[-1]
            lr_list.append(current_lr)
        else:
            current_lr = None
            new_lr = False

        pbar = get_progress_bar(TQDM_FLAG, total=len(train_loader), **tqdm_kwargs)
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

        with pbar:
            for clean_batch in train_loader:
                # Prepare batch
                noisy, clean, sigma_idx, _, _ = process_batch(clean_batch, device)

                # ---------------------------
                # Model forward + loss
                # ---------------------------
                optimizer.zero_grad(set_to_none=True)
                noise = model(noisy, sigma_idx)
                denoise = noisy - noise
                loss = mse(denoise, clean) * MSE_FACTOR

                loss.backward()

                # Compute gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm**0.5

                optimizer.step()
                if scheduler:
                    scheduler.step()

                # Logging
                batch_losses.append(loss.detach().cpu())

                postfix = {
                    "Train Loss": f"{torch.stack(batch_losses).mean():.{precision}e}",
                    "Grad Norm": f"{total_norm:.2e}",
                }
                if current_lr is not None:
                    postfix["LR"] = (
                        f"{Color.RED}{current_lr:.2e}{Color.OFF}"
                        if new_lr
                        else f"{current_lr:.2e}"
                    )

                pbar.update(1)
                pbar.set_postfix(postfix)

            # --- End of Epoch ---
            epoch_loss = torch.stack(batch_losses).mean().item()
            train_loss_list.append(epoch_loss)

            if not TQDM_FLAG:
                line0 = f"Epoch {epoch + 1}/{epochs}"
                if current_lr is not None:
                    line0 += f" | LR: {current_lr:.2e}" + (" NEW" if new_lr else "")
                print(line0)
                line = f"Train Loss: {epoch_loss:.{precision}e} | Grad Norm: {total_norm:.2e} | "

            # --- Validation ---
            if val_loader:
                val_loss = validation_loss_denoiser(
                    model,
                    val_loader,
                    device=device,
                )
                val_loss_list.append(val_loss)

                is_best = val_loss < best_val_loss
                if is_best:
                    best_epoch = epoch
                    best_val_loss = val_loss
                    best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

                pfix = {
                    "Train Loss": f"{epoch_loss:.{precision}e}",
                    "Grad Norm": f"{total_norm:.2e}",
                    "Val Loss": (
                        f"{Color.RED}{val_loss:.{precision}e}{Color.OFF}"
                        if is_best
                        else f"{val_loss:.{precision}e}"
                    ),
                }
                pbar.set_postfix(pfix)

                if not TQDM_FLAG:
                    marker = "BEST" if is_best else ""
                    line += f" | Val Loss: {val_loss:.{precision}e} {marker}"

            else:
                best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

            if not TQDM_FLAG:
                print(line)
                print("-" * 50)

        # --- Save checkpoint ---
        if filename:
            is_final_epoch = (epoch + 1) == epochs
            is_save_epoch = (epoch + 1) % save_freq == 0
            if is_final_epoch or is_save_epoch or is_best:
                checkpoint_data = {
                    "epoch": epoch + 1,
                    "model": model,
                    "optimizer": optimizer,
                    "best_weights": best_weights,
                    "filename": filename,
                    "best_val_loss": best_val_loss,
                    "val_loss_list": val_loss_list,
                    "train_loss_list": train_loss_list,
                    "lr_list": lr_list[1:],
                }
                if scheduler:
                    checkpoint_data["scheduler_state_dict"] = copy.deepcopy(
                        scheduler.state_dict()
                    )
                time_elapsed = time_string(time.time() - start_time)
                print(
                    f"Saving {'final' if is_final_epoch else 'intermediate'} checkpoint at Epoch {epoch + 1} at {time_elapsed}"
                )
                save_ckp(**checkpoint_data)

    # Summary
    total_time = time.time() - start_time
    print("-" * 50)
    if val_loader:
        best_idx = val_loss_list.index(min(val_loss_list))
        print(
            f"Training completed in {time_string(total_time)}\n"
            f"Best Val Epoch: {best_idx + 1}\n"
            f"MSE: Train={train_loss_list[best_epoch]:.{precision}f}, "
            f"Val={val_loss_list[best_epoch]:.{precision}f}\n"
        )
    else:
        best_idx = train_loss_list.index(min(train_loss_list))
        print(
            f"Training completed in {time_string(total_time)}\n"
            f"Best Training Loss Epoch {best_idx + 1}: {min(train_loss_list):.{precision}f}"
        )

    print(f"Save path: {filename}")
    print("-" * 50)

    return best_weights, train_loss_list, val_loss_list


@torch.no_grad()
def validation_loss_denoiser(model, val_loader, device, sigma_dict=SIGMA_DICT):
    model.eval()
    psnr_all = []

    for clean_batch in val_loader:
        # Prepare batch
        noisy, clean, sigma_idx, _, _ = process_batch(clean_batch, device)

        noise = model(noisy, sigma_idx)
        denoise = noisy - noise

        psnr_batch = psnr_torch(clean, denoise)
        psnr_all.append(-psnr_batch)

    return torch.cat(psnr_all).mean().item()


def predict_denoiser(
    model,
    weights,
    val_loader,
    device,
    print_stats=True,
    n=5,
    sigma_dict=SIGMA_DICT,
):
    """
    Predict + evaluate for the noise-conditioned denoiser.

    Dataset must return:
        clean, noisy, sigma_idx
    """

    if weights is not None:
        model.load_state_dict(weights)

    model.eval()

    clean_all = []
    noisy_all = []
    out_all = []
    psnr_all = []
    sigma_all = []
    out_all_2 = []
    psnr_all_2 = []

    model2 = DRUNet(
        in_channels=1,
        out_channels=1,
        pretrained=MODEL_DIR + "drunet_deepinv_gray_finetune_26k.pth",
        device=device,
    )
    model2 = model2.eval()

    with torch.inference_mode():
        pbar = get_progress_bar(TQDM_FLAG, total=len(val_loader), **tqdm_kwargs)

        with pbar:
            for nc, clean_batch in enumerate(val_loader):
                if nc > 5:
                    continue
                pbar.update(1)

                noisy, clean, sigma_idx, target_psnr, sigma_vals = process_batch(
                    clean_batch, device
                )
                sigma_all.append(target_psnr.cpu().numpy())

                # Forward pass
                noise = model(noisy, sigma_idx)
                out = noisy - noise

                out2 = model2(noisy, sigma_vals.squeeze().float())

                # Accumulate tensors
                clean_all.append(clean.cpu().numpy().squeeze())
                noisy_all.append(noisy.cpu().numpy().squeeze())
                out_all.append(out.cpu().numpy().squeeze())
                out_all_2.append(out2.cpu().numpy().squeeze())

                # Metrics on GPU
                psnr_batch = psnr_torch(clean, out).cpu().numpy()
                psnr_batch_2 = psnr_torch(clean, out2).cpu().numpy()

                psnr_all.append(psnr_batch)
                psnr_all_2.append(psnr_batch_2)

    # ---- Move to CPU once ---- #
    clean_all = np.concatenate(clean_all)
    noisy_all = np.concatenate(noisy_all)
    out_all = np.concatenate(out_all)
    out_all_2 = np.concatenate(out_all_2)
    psnr_all = np.concatenate(psnr_all)
    psnr_all_2 = np.concatenate(psnr_all_2)
    sigma_all = np.concatenate(sigma_all)

    # ssim_all = ssim_batch(clean_all, out_all)

    # ---- Print stats ---- #
    if print_stats:
        print("Refinenet:")
        print(
            f"PSNR  Mean {psnr_all.mean():.03f} | "
            f"Min {psnr_all.min():.03f} | "
            f"Max {psnr_all.max():.03f}"
        )
        print("DRUNet:")
        print(
            f"PSNR  Mean {psnr_all_2.mean():.03f} | "
            f"Min {psnr_all_2.min():.03f} | "
            f"Max {psnr_all_2.max():.03f}"
        )
        # print(
        #     f"SSIM  Mean {ssim_all.mean():.03f} | "
        #     f"Min {ssim_all.min():.03f} | "
        #     f"Max {ssim_all.max():.03f}"
        # )

    # ---- Build metrics dict ---- #
    metrics = {
        "clean": clean_all,
        "noisy": noisy_all,
        "denoised": out_all,
        "psnr": psnr_all,
        "denoised_2": out_all_2,
        "psnr_2": psnr_all_2,
        # "ssim": ssim_all,
    }

    # ---- Plotting ---- #
    if n > 0:
        np.random.seed(40)
        inds = np.random.choice(range(len(clean_all)), size=n, replace=False)
        clean_inds = clean_all[inds]
        noisy_inds = noisy_all[inds]
        out_inds = out_all[inds]
        out_inds_2 = out_all_2[inds]

        ssim_1 = ssim_batch(clean_inds, out_inds)
        ssim_2 = ssim_batch(clean_inds, out_inds_2)

        plot(
            images=[
                clean_inds,
                noisy_inds,
                out_inds,
                out_inds_2,
            ],
            caption=["True", "Noisy", "Refinenet", "DRUNet"],
            cbar=True,
            # scale_row=0,
            # same_scale=[0, 1, 2],
            subtitles=[
                [None] * n,
                [f"{s:.02f}" for s in sigma_all[inds]],
                [f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_1, psnr_all[inds])],
                [f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_2, psnr_all_2[inds])],
            ],
        )

    return metrics


# %% Train the model and plot results

n_epochs = 201

scheduler_params = {"factor": 0.5, "patience": 25, "min_lr": lr_init / (2**5)}


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr_init,
    weight_decay=lr_init,
)


def lr_lambda(step):
    # step=0 => factor=1, step=100k => factor=0.5, etc.
    factor = 0.5 ** (step // 100_000)
    # enforce minimum LR
    min_lr_factor = 5e-7 / 1e-4
    return max(factor, min_lr_factor)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


best_weights, train_loss_list, val_loss_list = train_denoiser(
    model,
    train_loader,
    val_loader,
    epochs=n_epochs,
    device=device,
    filename=loc_weights,
    optimizer=optimizer,
    scheduler=scheduler,
    save_freq=1,
    tqdm_enabled=False,
)


# %% Test
checkpoint = torch.load(loc_weights, map_location=device, weights_only=False)
best_weights = checkpoint["best_weights"]
val_loss = checkpoint["val_loss_list"]
train_loss = checkpoint["train_loss_list"]

plot_losses(
    [train_loss, val_loss],
    labels=["Train", "Validation"],
    skip=0,
    logscale=True,
)

plot_losses([checkpoint["lr_list"]], labels=["Learning Rate"], skip=0, logscale=True)

metrics = predict_denoiser(
    model,
    best_weights,
    val_loader,
    device=device,
)
