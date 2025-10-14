# %%Import Libraries
import copy
import time

import numpy as np
import torch
from colorist import Color

from deepshape2.data.loaders import dataloader
from deepshape2.models import Autoender
from deepshape2.utils import (
    get_freest_gpu,
    get_progress_bar,
    get_tqdm,
    load_ckp,
    load_config,
    psnr_torch,
    save_ckp,
    set_seed,
    ssim_torch,
    time_string,
)
from deepshape2.visualization import plot, plot_losses

tqdm_kwargs = get_tqdm()

# %% Load Device and Model
device = get_freest_gpu(set_device=True)
set_seed(2024)

model = Autoender().to(device)

model = torch.compile(model)
# %% Load Config and Data
cfg = load_config()

DATA_DIR = cfg["DATA_DIR"]

loc_weights = cfg["MODEL_DIR"] + "autoencoder.pt"

train_loader, val_loader = dataloader(
    path=DATA_DIR + "PSF_set.h5",
    x_key=["psf"],
    y_key=None,
    split=[0.8, 0.2],
    batch_size=[64, 64],
)


# %%
def validation_loss(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device, non_blocking=True)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train(
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    device,
    **kwargs,
):
    """Train an autoencoder model with checkpointing and optional validation."""
    start_time = time.time()
    best_val_loss = np.inf
    best_weights = None
    current_epoch = 0
    best_epoch = 0

    train_loss_list, val_loss_list, lr_list = [], [], [np.inf]

    # --- Configuration ---
    filename = kwargs.get("filename")
    criterion = kwargs.get("criterion", torch.nn.MSELoss())
    scheduler_params = kwargs.get("scheduler_params", None)
    save_freq = kwargs.get("save_freq", 50)
    precision = kwargs.get("precision", 4)
    tqdm_enabled = kwargs.get("tqdm_enabled", True)
    tqdm_kwargs = kwargs.get("tqdm_kwargs", dict(colour="green", unit="batch"))

    # Optional scheduler
    scheduler = None
    if scheduler_params:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_params
        )

    print(f"Running on device: {device}")

    # --- Checkpoint loading ---
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

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler()

    # --- Training loop ---
    for epoch in range(epochs):
        if epoch < current_epoch:
            continue

        model.train()
        total_loss = []

        if scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            new_lr = current_lr < lr_list[-1]
            lr_list.append(current_lr)

        pbar = get_progress_bar(tqdm_enabled, total=len(train_loader), **tqdm_kwargs)
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

        with pbar:
            for x in train_loader:
                x = x.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    x_hat = model(x)
                    loss = criterion(x, x_hat)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss.append(loss.detach())

                postfix = {"Train Loss": f"{np.mean(total_loss):.{precision}e}"}
                if current_lr is not None:
                    postfix["LR"] = (
                        f"{Color.RED}{current_lr:.2e}{Color.OFF}"
                        if new_lr
                        else f"{current_lr:.2e}"
                    )
                pbar.update(1)
                pbar.set_postfix(postfix)

            epoch_loss = torch.stack(total_loss).mean().item()
            train_loss_list.append(epoch_loss)

            # Print updates if tqdm is disabled
            if not tqdm_enabled:
                line0 = f"Epoch {epoch + 1}/{epochs}"
                if current_lr is not None:
                    line0 += f" | LR: {current_lr:.2e}"
                    if new_lr:
                        line0 += " NEW"
                print(line0)
                line = f"Train Loss: {epoch_loss:.{precision}e}"

            # --- Validation ---
            if val_loader:
                val_loss = validation_loss(model, val_loader, criterion, device=device)
                val_loss_list.append(val_loss)

                if scheduler:
                    scheduler.step(val_loss)

                is_best = val_loss < best_val_loss
                if is_best and epoch >= 5:
                    best_epoch = epoch
                    best_val_loss = val_loss
                    # store CPU copy of weights
                    best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

                pfix = {
                    "Train Loss": f"{epoch_loss:.{precision}e}",
                    "Val Loss": (
                        f"{Color.RED}{-val_loss:.3f} dB{Color.OFF}"
                        if is_best
                        else f"{-val_loss:.3f} dB"
                    ),
                }
                pbar.set_postfix(pfix)

                if not tqdm_enabled:
                    marker = "BEST" if is_best else ""
                    line += f" | Val Loss: {-val_loss:.3f} dB {marker}"

            else:
                best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

            if not tqdm_enabled:
                print(line)
                print("-" * 50)

        # --- Save checkpoints ---
        if filename:
            is_final_epoch = (epoch + 1) == epochs
            is_save_epoch = (epoch + 1) % save_freq == 0
            is_best_epoch = val_loader and (val_loss < best_val_loss)
            if is_final_epoch or is_save_epoch or is_best_epoch:
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
                elapsed = time_string(time.time() - start_time)
                print(
                    f"Saving {'final' if is_final_epoch else 'intermediate'} checkpoint at Epoch {epoch + 1} ({elapsed})"
                )
                save_ckp(**checkpoint_data)

    total_time = time.time() - start_time
    print("-" * 50)

    # --- Summary ---
    if val_loader:
        best_idx = val_loss_list.index(min(val_loss_list))
        print(
            f"Training completed in {time_string(total_time)}\n"
            f"Best Val Loss at Epoch {best_idx + 1}\n"
            f"Saved Model from Epoch {best_epoch + 1}\n"
            f"MSE: Train={train_loss_list[best_epoch]:.{precision}f}, "
            f"Val={val_loss_list[best_epoch]:.{precision}f}"
        )
    else:
        best_idx = train_loss_list.index(min(train_loss_list))
        print(
            f"Training completed in {time_string(total_time)}\n"
            f"Best Train Loss at Epoch {best_idx + 1}: {min(train_loss_list):.{precision}f}"
        )

    print("-" * 50)
    print(f"Save path: {filename}")

    return best_weights, train_loss_list, val_loss_list


# Prediction function
def predict(model, val_loader, n=5, weights=None, tqdm_enabled=True):
    if weights is not None:
        model.load_state_dict(weights)
    model.eval()

    inputs, outputs = [], []
    psnr_all, ssim_all = [], []

    with torch.inference_mode():
        pbar = get_progress_bar(tqdm_enabled, total=len(val_loader), **tqdm_kwargs)

        with pbar:
            for x in val_loader:
                pbar.update(1)

                x = x.to(device, non_blocking=True)
                xhat = model(x)

                inputs.append(x.cpu().numpy().squeeze())
                outputs.append(xhat.cpu().numpy().squeeze())

                psnr_batch = psnr_torch(x, xhat).cpu().numpy()
                _, ssim_batch = ssim_torch(x, xhat)

                psnr_all.extend(psnr_batch)
                ssim_all.extend(ssim_batch)

    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)

    psnr_out = torch.cat(psnr_all).numpy()
    ssim_out = torch.cat(ssim_all).numpy()

    print(
        f"SSIM: Max {np.max(ssim_out):.03f} | Min {np.min(ssim_out):.03f} | Mean {np.mean(ssim_out):.03f}"
    )
    print(
        f"PSNR: Max {np.max(psnr_out):.03f} dB | Min {np.min(psnr_out):.03f} dB | Mean {np.mean(psnr_out):.03f} dB"
    )

    # Optional: Visualization
    if n > 0:
        tit_out = [f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_out[:n], psnr_out[:n])]
        blank_titles = [None] * n

        plot(
            images=[inputs[:n], outputs[:n], inputs[:n] - outputs[:n]],
            caption=["Input", "Recon", "Residual"],
            cbar=True,
            scale_row=1,
            same_scale=[0, 1],
            subtitles=[blank_titles, tit_out, blank_titles],
        )


# %% Train the model and plot results
n_epochs = 1201

scheduler_params = {"factor": 0.5, "patience": 40, "min_lr": 1e-06}

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5
)

best_weights, train_loss_list, val_loss_list = train(
    model,
    train_loader,
    val_loader,
    epochs=n_epochs,
    device=device,
    filename=loc_weights,
    plot=True,
    optimizer=optimizer,
    scheduler_params=scheduler_params,
    save_freq=25,
)


# %% Test
checkpoint = torch.load(loc_weights, map_location=device, weights_only=False)
best_weights = checkpoint["best_weights"]
beta = checkpoint["beta"]
recon_loss = checkpoint["recon_loss_list"]
kl_loss = checkpoint["kl_loss_list"]
val_loss = checkpoint["val_loss_list"]
train_loss = checkpoint["train_loss_list"]

plot_losses(
    [train_loss, val_loss],
    labels=["Train", "Val"],
    skip=0,
    logscale=True,
)

plot_losses([checkpoint["lr_list"]], labels=["Learning Rate"], skip=0, logscale=True)


predict(model, best_weights, val_loader)

# %% Save Model

# model.load_state_dict(best_weights)
# model_scripted = torch.jit.script(model)
# torch.jit.save(
#     m=model_scripted,
#     f=loc_weights[:-3] + "_jit.pt",
# )
