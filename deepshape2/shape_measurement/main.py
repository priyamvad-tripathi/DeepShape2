# %%Import Libraries
import copy
import time

import numpy as np
import torch
from colorist import Color

# from deepshape2.models.drunet import DRUNet
from deepshape2.utils import (
    get_progress_bar,
    get_tqdm,
    load_ckp,
    load_config,
    save_ckp,
    set_seed,
    time_string,
)

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

            val_loss = loss_fn(target, pred)
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
    best_epoch = 0

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

        # Epoch header
        if not tqdm_enabled:
            print("-" * 50)
            line0 = f"Epoch {epoch + 1}/{epochs}"
            line0 += f" | LR: {current_lr:.2e}" + (" NEW" if new_lr else "")
            print(line0, flush=True)

        pbar = get_progress_bar(tqdm_enabled, total=len(train_loader), **tqdm_kwargs)
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

        # params_before = [p.detach().clone() for p in model.parameters() if p.requires_grad]

        with pbar:
            for image, target in train_loader:
                image, target = (
                    image.to(device, non_blocking=True),
                    target.to(device, non_blocking=True),
                )

                # ---------------------------
                # Model forward + loss
                # ---------------------------
                optimizer.zero_grad(set_to_none=True)
                pred = model(image)
                loss = loss_fn(pred, target)

                loss.backward()

                optimizer.step()

                # Logging
                batch_losses.append(loss.detach().cpu())

                postfix = {
                    "Train Loss": f"{torch.stack(batch_losses).mean():.{precision}e}",
                }

                postfix["LR"] = (
                    f"{Color.RED}{current_lr:.4e}{Color.OFF}"
                    if new_lr
                    else f"{current_lr:.4e}"
                )

                pbar.update(1)
                pbar.set_postfix(postfix)

            # --- End of Epoch ---
            epoch_loss = torch.stack(batch_losses).mean().item()
            train_loss_list.append(epoch_loss)

            line = f"Train Loss: {epoch_loss:.{precision}e} "

            # --- Validation ---
            if val_loader:
                val_loss = validation_loss(
                    model,
                    val_loader,
                    loss_fn=loss_fn,
                    device=device,
                )
                scheduler.step(val_loss)

                # EMA update
                if val_loss_ema is None:
                    val_loss_ema = val_loss
                else:
                    val_loss_ema = (1 - ema_alpha) * val_loss_ema + ema_alpha * val_loss

                # Step scheduler on smoothed value
                scheduler.step(val_loss_ema)

                val_loss_list.append(val_loss_ema)

                is_best = val_loss_ema < best_val_loss
                if is_best:
                    best_epoch = epoch
                    best_val_loss = val_loss_ema
                    best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

                pfix = {
                    "Train Loss": f"{epoch_loss:.{precision}e}",
                    "Val Loss": (
                        f"{Color.RED}{val_loss_ema:.4e}{Color.OFF}"
                        if is_best
                        else f"{val_loss_ema:.4e}"
                    ),
                }
                pbar.set_postfix(pfix)
                line += f" | Val Loss: {val_loss_ema:.4e}" + (
                    " BEST" if is_best else ""
                )
                line += f"\n Time Elapsed: {time_string(time.time() - start_time)}"

            else:
                best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

            if not tqdm_enabled:
                print(line, flush=True)

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
                    "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
                    "val_loss_ema": val_loss_ema,
                }

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
class TupleSmoothL1WithBias(torch.nn.Module):
    def __init__(self, beta=0.05, lambda_bias=0.0):
        super().__init__()
        self.beta = beta
        self.lambda_bias = lambda_bias

    def forward(self, output, target):
        if isinstance(output, (tuple, list)):
            e_pred = output[0]
        else:
            e_pred = output

        err = e_pred - target

        scatter = torch.nn.functional.smooth_l1_loss(e_pred, target, beta=self.beta)

        if self.lambda_bias > 0:
            bias = err.mean(dim=0).pow(2).sum()
            return scatter + self.lambda_bias * bias
        else:
            return scatter
