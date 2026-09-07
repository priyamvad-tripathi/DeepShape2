import copy
import time

import numpy as np
import torch
from colorist import Color

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

cfg = load_config()
tqdm_kwargs = get_tqdm()
set_seed()


def shear_bias(pred, true):
    """Per-component least squares pred = (1 + m) * true + c.

    This is an object-by-object regression, not the shear-averaged m you will
    quote in the paper, but it tracks it and costs nothing.
    """
    m, c = [], []
    for i in range(2):
        t, p = true[:, i], pred[:, i]
        tc = t - t.mean()
        slope = (tc * (p - p.mean())).sum() / (tc * tc).sum()
        m.append((slope - 1.0).item())
        c.append((p.mean() - slope * t.mean()).item())
    return m, c


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
NEW_PREFIXES = ("cond.", "eq.film", "head.film_inv", "head.coef")


def param_groups(model, lr_pre=1e-4, lr_new=1e-3, wd_pre=1e-5):
    """Split transferred parameters from newly initialised conditioning ones."""
    pre, new = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (new if name.startswith(NEW_PREFIXES) else pre).append(p)
    print(f"param groups: {len(pre)} pretrained, {len(new)} new")
    return [
        {"params": pre, "lr": lr_pre, "weight_decay": wd_pre},
        {"params": new, "lr": lr_new, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
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

    train_loss_list, val_loss_list, val_ema_list = [], [], []
    lr_list = [np.inf]
    val_loss_ema = None

    val_m_list, val_c_list, val_scatter_list = [], [], []

    filename = kwargs.get("filename")
    scheduler = kwargs.get("scheduler", None)
    save_freq = kwargs.get("save_freq", 50)
    precision = kwargs.get("precision", 4)
    tqdm_enabled = kwargs.get("tqdm_enabled", False)
    loss_fn = kwargs.get("loss_fn", torch.nn.MSELoss())
    ema_alpha = kwargs.get("ema_alpha", 0.1)

    print(f"Running on device: {device}")

    try:
        model, optimizer, checkpoint = load_ckp(filename, model, optimizer, device)
        current_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", np.inf)
        best_weights = checkpoint.get("best_weights")
        val_loss_list = checkpoint.get("val_loss_list", [])
        val_ema_list = checkpoint.get("val_ema_list", [])
        train_loss_list = checkpoint.get("train_loss_list", [])
        val_loss_ema = checkpoint.get("val_loss_ema", None)
        lr_list = [np.inf] + checkpoint.get("lr_list", [])
        val_m_list = checkpoint.get("val_m_list", [])
        val_c_list = checkpoint.get("val_c_list", [])
        val_scatter_list = checkpoint.get("val_scatter_list", [])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {current_epoch}")
    except (AttributeError, FileNotFoundError, TypeError):
        print("No saved checkpoints found. Starting from scratch.")

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
            for image, cond, target in train_loader:
                image = image.to(device, non_blocking=True)
                cond = cond.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = model(image, cond)
                loss = loss_fn(pred, target)

                loss.backward()
                optimizer.step()

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

        # --- Validation ---
        if val_loader:
            model.eval()
            val_losses, preds, targets = [], [], []

            with torch.no_grad():
                for image, cond, target in val_loader:
                    image = image.to(device, non_blocking=True)
                    cond = cond.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    pred = model(image, cond)
                    val_losses.append(loss_fn(pred, target).detach().cpu())
                    preds.append(pred.detach().cpu())
                    targets.append(target.detach().cpu())

            val_loss_raw = torch.stack(val_losses).mean().item()
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)

            m, c = shear_bias(preds, targets)
            val_scatter = (preds - targets).std(dim=0).mean().item()

            val_m_list.append(m)
            val_c_list.append(c)
            val_scatter_list.append(val_scatter)
            val_loss_list.append(val_loss_raw)

            if val_loss_ema is None:
                val_loss_ema = val_loss_raw
            else:
                val_loss_ema = (1 - ema_alpha) * val_loss_ema + ema_alpha * val_loss_raw
            val_ema_list.append(val_loss_ema)

            # scheduler on the smoothed signal, selection on the raw one
            if scheduler is not None:
                scheduler.step(val_loss_ema)

            is_best = val_loss_raw < best_val_loss
            if is_best:
                best_epoch = epoch
                best_val_loss = val_loss_raw
                best_weights = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

            postfix = {
                "Train": f"{epoch_loss:.{precision}e}",
                "Val": (
                    f"{Color.RED}{val_loss_raw:.3e}{Color.OFF}"
                    if is_best
                    else f"{val_loss_raw:.3e}"
                ),
                "m": f"{m[0]:+.1e},{m[1]:+.1e}",
                "c": f"{c[0]:+.1e},{c[1]:+.1e}",
            }
            pbar.set_postfix(postfix)

            line += (
                f" | Val: {val_loss_raw:.3e}"
                + (" BEST" if is_best else "")
                + f" | m: {m[0]:+.2e},{m[1]:+.2e}"
                + f" | c: {c[0]:+.2e},{c[1]:+.2e}"
                + f" | Scat: {val_scatter:.2e}"
                + f" | Time: {time_string(time.time() - start_time)}"
            )

        else:
            best_weights = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        if not tqdm_enabled:
            print(line, flush=True)

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
                    "val_ema_list": val_ema_list,
                    "train_loss_list": train_loss_list,
                    "val_m_list": val_m_list,
                    "val_c_list": val_c_list,
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
        m, c = val_m_list[best_epoch], val_c_list[best_epoch]
        print(
            f"Best Epoch: {best_epoch + 1}\n"
            f"Val Loss: {best_val_loss:.{precision}e}\n"
            f"m: {m[0]:+.3e}, {m[1]:+.3e}\n"
            f"c: {c[0]:+.3e}, {c[1]:+.3e}\n"
            f"Scatter: {val_scatter_list[best_epoch]:.3e}"
        )
    print(f"Save path: {filename}")
    print("-" * 50)

    return best_weights, train_loss_list, val_loss_list


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict(
    model,
    data_loader,
    device,
    weights=None,
    tqdm_enabled=True,
    return_images=False,
):
    """Returns (preds, targets, conds) and optionally the images.

    conds comes back because m and c are calibrated as a surface in (snr, size)
    -- the calibration needs the conditioning columns aligned row-for-row with
    the predictions, and this is the only place that alignment is guaranteed.
    Images are 20000 x 128 x 128 f32 = 1.3 GB, hence off by default.
    """
    if weights is not None:
        model.load_state_dict(weights)
    model.eval()

    targets, preds, conds, images = [], [], [], []

    with torch.inference_mode():
        pbar = get_progress_bar(tqdm_enabled, total=len(data_loader), **tqdm_kwargs)

        with pbar:
            for image, cond, target in data_loader:
                pbar.update(1)

                image_gpu = image.to(device, non_blocking=True)
                cond_gpu = cond.to(device, non_blocking=True)

                pred = model(image_gpu, cond_gpu)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]

                preds.append(pred.detach().cpu().numpy())
                targets.append(target.numpy())
                conds.append(cond.numpy())
                if return_images:
                    images.append(image.numpy())

    out = (np.concatenate(preds), np.concatenate(targets), np.concatenate(conds))
    if return_images:
        out = out + (np.concatenate(images).squeeze(),)
    return out
