# %%Import Libraries
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
from colorist import Color
from torch.utils.data import DataLoader, TensorDataset

# from deepshape2.models.drunet import DRUNet
from ..utils import (
    get_progress_bar,
    get_tqdm,
    load_ckp,
    load_config,
    save_ckp,
    set_seed,
    time_string,
)

__all__ = ["predict_shape"]
# %% Load Config and Set Parameters
cfg = load_config()
tqdm_kwargs = get_tqdm()
set_seed()


# %%
def validation_loss(model, val_loader, loss_fn, device):
    model.eval()
    val_loss_all = []

    with torch.inference_mode():
        for image, target in val_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(image)

            val_loss = loss_fn(pred, target)
            val_loss_all.append(val_loss)

    # Concatenate all batches and move to CPU once
    val_loss_all = torch.stack(val_loss_all)
    return val_loss_all.mean().item()


def train(
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    device,
    **kwargs,
):
    start_time = time.time()
    best_val_loss = np.inf
    best_weights = None
    current_epoch = 0

    train_loss_list, val_loss_list, lr_list = [], [], [np.inf]
    val_loss_ema = None

    # --- Config ---
    filename = kwargs.get("filename")
    scheduler = kwargs.get("scheduler", None)
    save_freq = kwargs.get("save_freq", 50)
    precision = kwargs.get("precision", 4)
    tqdm_enabled = kwargs.get("tqdm_enabled", False)
    loss_fn = kwargs.get("loss_fn", torch.nn.MSELoss())
    ema_alpha = kwargs.get("ema_alpha", 0.1)
    log_freq = kwargs.get("log_freq", 10)  # batches between pbar refreshes
    save_best_separately = kwargs.get("save_best_separately", True)

    # Separate lightweight file for best weights: avoids writing a full
    # optimizer-state checkpoint every time val loss improves.
    best_filename = None
    if filename and save_best_separately:
        p = Path(filename)
        best_filename = str(p.with_name(p.stem + "_best" + p.suffix))

    print(f"Running on device: {device}")

    # --- Load checkpoint ---
    try:
        model, optimizer, checkpoint = load_ckp(filename, model, optimizer, device)
        current_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", np.inf)
        best_weights = checkpoint.get("best_weights")
        val_loss_list = checkpoint.get("val_loss_list", [])
        train_loss_list = checkpoint.get("train_loss_list", [])
        val_loss_ema = checkpoint.get("val_loss_ema", None)
        lr_list = checkpoint.get("lr_list") or [np.inf]
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {current_epoch}")
    except (AttributeError, FileNotFoundError, TypeError) as err:
        print(
            f"No usable checkpoint ({type(err).__name__}: {err}). "
            "Starting from scratch."
        )

    # --- Training loop ---
    for epoch in range(epochs):
        if epoch < current_epoch:
            continue

        model.train()
        is_best = False

        # Running totals kept on-device: no host sync inside the batch loop.
        loss_sum = torch.zeros((), device=device)
        n_seen = 0

        current_lr = optimizer.param_groups[0]["lr"]
        new_lr = current_lr < lr_list[-1]
        lr_list.append(current_lr)

        if not tqdm_enabled:
            print("-" * 50)
            line0 = f"Epoch {epoch + 1}/{epochs}"
            line0 += f" | LR: {current_lr:.2e}" + (" NEW" if new_lr else "")
            print(line0, flush=True)

        pbar = get_progress_bar(tqdm_enabled, total=len(train_loader), **tqdm_kwargs)
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

        with pbar:
            for i, (image, target) in enumerate(train_loader):
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # ---------------------------
                # Model forward + loss
                # ---------------------------
                optimizer.zero_grad(set_to_none=True)
                pred = model(image)
                loss = loss_fn(pred, target)

                loss.backward()
                optimizer.step()

                # Accumulate on-device, weighted by batch size so the final
                # partial batch is not over-counted.
                bs = image.shape[0]
                loss_sum += loss.detach() * bs
                n_seen += bs

                pbar.update(1)

                # .item() forces a sync, so only do it every log_freq batches.
                if tqdm_enabled and (i % log_freq == 0):
                    running = (loss_sum / n_seen).item()
                    pbar.set_postfix(
                        {
                            "Train Loss": f"{running:.{precision}e}",
                            "LR": (
                                f"{Color.RED}{current_lr:.4e}{Color.OFF}"
                                if new_lr
                                else f"{current_lr:.4e}"
                            ),
                        }
                    )

            # --- End of Epoch --- (single host sync)
            epoch_loss = (loss_sum / n_seen).item()
            train_loss_list.append(epoch_loss)

            line = f"Train Loss: {epoch_loss:.{precision}e} "

            # --- Validation ---
            val_loss = validation_loss(
                model,
                val_loader,
                loss_fn=loss_fn,
                device=device,
            )

            # EMA is used for the scheduler only; selection uses raw val loss.
            if val_loss_ema is None:
                val_loss_ema = val_loss
            else:
                val_loss_ema = (1 - ema_alpha) * val_loss_ema + ema_alpha * val_loss

            if scheduler is not None:
                scheduler.step(val_loss_ema)

            val_loss_list.append(val_loss)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_weights = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }

                if best_filename:
                    _atomic_save(
                        {
                            "epoch": epoch + 1,
                            "best_val_loss": best_val_loss,
                            "state_dict": best_weights,
                        },
                        best_filename,
                    )

            if tqdm_enabled:
                pbar.set_postfix(
                    {
                        "Train Loss": f"{epoch_loss:.{precision}e}",
                        "Val Loss": (
                            f"{Color.RED}{val_loss:.4e}{Color.OFF}"
                            if is_best
                            else f"{val_loss:.4e}"
                        ),
                    }
                )

            line += f" | Val Loss: {val_loss:.4e}" + (" BEST" if is_best else "")
            line += f" | Time Elapsed: {time_string(time.time() - start_time)}"

            if not tqdm_enabled:
                print(line, flush=True)

        # --- Save checkpoint ---
        if filename:
            is_final_epoch = (epoch + 1) == epochs
            is_save_epoch = (epoch + 1) % save_freq == 0
            # If best weights go to their own file, a new best no longer forces
            # a full checkpoint write.
            force = is_best and not save_best_separately

            if is_final_epoch or is_save_epoch or force:
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
                    "scheduler_state_dict": (
                        copy.deepcopy(scheduler.state_dict())
                        if scheduler is not None
                        else None
                    ),
                    "val_loss_ema": val_loss_ema,
                }

                time_elapsed = time_string(time.time() - start_time)
                print(
                    f"Saving {'final' if is_final_epoch else 'intermediate'} "
                    f"checkpoint at Epoch {epoch + 1} at {time_elapsed}"
                )
                save_ckp(**checkpoint_data)

    # --- Summary ---
    total_time = time.time() - start_time
    print("-" * 50)

    best_idx = int(np.argmin(val_loss_list))
    print(
        f"Training completed in {time_string(total_time)}\n"
        f"Best Val Epoch: {best_idx + 1}\n"
        f"Loss: Train={train_loss_list[best_idx]:.{precision}e}, "
        f"Val={val_loss_list[best_idx]:.{precision}e}\n"
    )

    print(f"Save path: {filename}")
    if best_filename:
        print(f"Best weights: {best_filename}")
    print("-" * 50)

    return best_weights, train_loss_list, val_loss_list


def _atomic_save(obj, path):
    """Write to a temp file then rename, so a preempted job cannot corrupt it."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def predict(
    model,
    data_loader,
    device,
    weights=None,
    tqdm_enabled=True,
):
    if weights is not None:
        model.load_state_dict(weights)
    model.eval()

    # Accumulate results as tensors on GPU
    targets, images, preds = [], [], []

    with torch.inference_mode():
        pbar = get_progress_bar(tqdm_enabled, total=len(data_loader), **tqdm_kwargs)

        with pbar:
            for image, target in data_loader:
                pbar.update(1)

                image_gpu = image.to(device, non_blocking=True)
                target_gpu = target.to(device, non_blocking=True)

                pred = model(image_gpu)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]

                # Append GPU tensors
                targets.append(target_gpu.detach().cpu().numpy())
                images.append(image_gpu.detach().cpu().numpy())
                preds.append(pred.detach().cpu().numpy())

    return (
        np.concatenate(preds),
        np.concatenate(targets),
        np.concatenate(images).squeeze(),
    )


# %%
def train2(
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    device,
    **kwargs,
):
    start_time = time.time()
    best_val_loss = np.inf
    best_weights = None
    current_epoch = 0
    best_epoch = 0

    train_loss_list, val_loss_list, lr_list = [], [], [np.inf]
    val_loss_ema = None

    # --- Debug stats ---
    val_bias_list = []
    val_scatter_list = []

    # --- Config ---
    filename = kwargs.get("filename")
    scheduler = kwargs.get("scheduler", None)
    save_freq = kwargs.get("save_freq", 50)
    precision = kwargs.get("precision", 4)
    tqdm_enabled = kwargs.get("tqdm_enabled", False)
    loss_fn = kwargs.get("loss_fn", torch.nn.MSELoss())
    ema_alpha = kwargs.get("ema_alpha", 0.1)

    print(f"Running on device: {device}")

    # --- Load checkpoint ---
    try:
        model, optimizer, checkpoint = load_ckp(filename, model, optimizer, device)
        current_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", np.inf)
        best_weights = checkpoint.get("best_weights")
        val_loss_list = checkpoint.get("val_loss_list", [])
        train_loss_list = checkpoint.get("train_loss_list", [])
        val_loss_ema = checkpoint.get("val_loss_ema", None)
        lr_list = checkpoint.get("lr_list", [])
        val_bias_list = checkpoint.get("val_bias_list", [])
        val_scatter_list = checkpoint.get("val_scatter_list", [])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {current_epoch}")
    except (AttributeError, FileNotFoundError, TypeError):
        print("No saved checkpoints found. Starting from scratch.")

    # --- Training loop ---
    for epoch in range(epochs):
        if epoch < current_epoch:
            continue

        model.train()
        batch_losses = []

        current_lr = optimizer.param_groups[0]["lr"]
        new_lr = current_lr < lr_list[-1]
        lr_list.append(current_lr)

        if not tqdm_enabled:
            print("-" * 50)
            print(
                f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.2e}"
                + (" NEW" if new_lr else ""),
                flush=True,
            )

        pbar = get_progress_bar(tqdm_enabled, total=len(train_loader), **tqdm_kwargs)
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

        with pbar:
            for image, target in train_loader:
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = model(image)
                loss = loss_fn(pred, target)

                loss.backward()
                optimizer.step()

                # with torch.no_grad():
                #     err = (pred - target).abs()

                # grad_norm = (
                #     list(model.encode.named_parameters())[12][1].grad.norm().item()
                # )

                batch_losses.append(loss.detach().cpu())

                postfix = {
                    "Train": f"{torch.stack(batch_losses).mean():.{precision}e}",
                    "LR": (
                        f"{Color.RED}{current_lr:.3e}{Color.OFF}"
                        if new_lr
                        else f"{current_lr:.3e}"
                    ),
                }

                pbar.update(1)
                pbar.set_postfix(postfix)

        epoch_loss = torch.stack(batch_losses).mean().item()
        train_loss_list.append(epoch_loss)

        line = f"Train Loss: {epoch_loss:.{precision}e}"
        # line += f" | Grad Norm: {grad_norm:.3e}"
        # line += f" | Frac < beta: {(err < 0.05).float().mean().item():.3e}"

        # --- Validation ---
        if val_loader:
            model.eval()
            val_losses = []
            residuals = []

            with torch.no_grad():
                for image, target in val_loader:
                    image = image.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    pred = model(image)
                    val_losses.append(loss_fn(pred, target).detach().cpu())
                    residuals.append((pred - target).detach().cpu())

            val_loss_raw = torch.stack(val_losses).mean().item()
            residuals = torch.cat(residuals, dim=0)

            val_bias = residuals.mean(dim=0).norm().item()
            val_scatter = residuals.std(dim=0).mean().item()

            val_bias_list.append(val_bias)
            val_scatter_list.append(val_scatter)

            # --- Scheduler uses RAW val loss ---
            if scheduler is not None:
                scheduler.step(val_loss_raw)

            # --- EMA only for logging / best model ---
            if val_loss_ema is None:
                val_loss_ema = val_loss_raw
            else:
                val_loss_ema = (1 - ema_alpha) * val_loss_ema + ema_alpha * val_loss_raw

            val_loss_list.append(val_loss_ema)

            is_best = val_loss_ema < best_val_loss
            if is_best:
                best_epoch = epoch
                best_val_loss = val_loss_ema
                best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

            postfix = {
                "Train": f"{epoch_loss:.{precision}e}",
                "Val": (
                    f"{Color.RED}{val_loss_ema:.3e}{Color.OFF}"
                    if is_best
                    else f"{val_loss_ema:.3e}"
                ),
                "Bias": f"{val_bias:.2e}",
                "Scat": f"{val_scatter:.2e}",
            }
            pbar.set_postfix(postfix)

            line += (
                f" | Val: {val_loss_ema:.3e}"
                + (" BEST" if is_best else "")
                + f" | Bias: {val_bias:.2e}"
                + f" | Scat: {val_scatter:.2e}"
                + f" | Time: {time_string(time.time() - start_time)}"
            )

        else:
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

        if not tqdm_enabled:
            print(line, flush=True)

        # --- Save checkpoint ---
        if filename:
            is_final_epoch = (epoch + 1) == epochs
            is_save_epoch = (epoch + 1) % save_freq == 0
            if is_final_epoch or is_save_epoch or (val_loader and is_best):
                checkpoint_data = {
                    "epoch": epoch + 1,
                    "model": model,
                    "optimizer": optimizer,
                    "best_weights": best_weights,
                    "filename": filename,
                    "best_val_loss": best_val_loss,
                    "val_loss_list": val_loss_list,
                    "train_loss_list": train_loss_list,
                    "val_bias_list": val_bias_list,
                    "val_scatter_list": val_scatter_list,
                    "lr_list": lr_list[1:],
                    "scheduler_state_dict": (
                        copy.deepcopy(scheduler.state_dict())
                        if scheduler is not None
                        else None
                    ),
                    "val_loss_ema": val_loss_ema,
                }

                print(
                    f"Saving checkpoint at Epoch {epoch + 1} | "
                    f"{time_string(time.time() - start_time)}"
                )
                save_ckp(**checkpoint_data)

    print("-" * 50)
    print(f"Training completed in {time_string(time.time() - start_time)}")
    if val_loader:
        print(
            f"Best Epoch: {best_epoch + 1}\n"
            f"Val Loss EMA: {best_val_loss:.{precision}e}\n"
            f"Bias: {val_bias_list[best_epoch]:.2e}\n"
            f"Scatter: {val_scatter_list[best_epoch]:.2e}"
        )
    print(f"Save path: {filename}")
    print("-" * 50)

    return best_weights, train_loss_list, val_loss_list


# %%
def _to_tensor(x, dtype=torch.float32):
    """Convert numpy array, list, or tensor to a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(np.asarray(x), dtype=dtype)


def predict_shape(
    recon,
    psf,
    model,
    device,
    batch_size=32,
    description=None,
    tqdm_enabled=True,
    weight_path=None,
    weight_key="best_weights",
):
    recon = _to_tensor(recon)  # (N, H, W)
    psf = _to_tensor(psf)  # (H, W) or (N, H, W)

    if psf.ndim == 2:
        print("Using same PSF for all reconstructions")
        psf = psf.unsqueeze(0).expand(recon.shape[0], -1, -1)

    images = torch.stack([recon, psf], dim=1)  # (N, 2, H, W)

    img_min = images.amin(dim=(2, 3), keepdim=True)
    img_max = images.amax(dim=(2, 3), keepdim=True)
    images = (images - img_min) / (img_max - img_min)

    # images is already a tensor, skip the np -> tensor conversion
    dataset = TensorDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if weight_path is not None:
        ckpt = torch.load(weight_path, map_location=device, weights_only=False)
        if weight_key in ckpt:
            model.load_state_dict(ckpt[weight_key])
        else:
            print(
                f"Warning: weight_key '{weight_key}' not found in checkpoint. Available keys: {list(ckpt.keys())}"
            )
            print("Proceeding without loading weights.")
        model = model.eval()

    ypred_all = []

    with torch.inference_mode():
        pbar = get_progress_bar(tqdm_enabled, total=len(dataloader), **tqdm_kwargs)
        pbar.set_description(description if description else "Predicting shapes")

        with pbar:
            for (im,) in dataloader:
                pbar.update(1)
                im = im.to(device)  # moves to device only if not already there
                ypred = model(im)
                ypred_all.append(ypred.detach().cpu().numpy())

    return np.concatenate(ypred_all)
