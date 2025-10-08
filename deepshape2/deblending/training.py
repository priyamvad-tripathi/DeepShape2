# %% Import Libraries
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from colorist import Color
from tqdm import tqdm

from deepshape2.utils import (
    get_tqdm,
    load_ckp,
    psnr_batch,
    save_ckp,
    ssim_batch,
    time_string,
)
from deepshape2.visualization import plot

tqdm_kwargs = get_tqdm()


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


def psnr_batch_torch(true_images, recon_images, eps=1e-10):
    if true_images.shape != recon_images.shape:
        raise ValueError("Shapes of true and reconstructed images must match.")

    # Mean squared error per image
    mse = F.mse_loss(recon_images, true_images, reduction="none")
    mse = mse.flatten(1).mean(dim=1)  # mean over pixels per image

    # Max value per image
    max_vals = true_images.flatten(1).max(dim=1).values

    psnr = 10 * torch.log10((max_vals**2) / (mse + eps))
    return -psnr  # returns PSNR for each image in the batch


def validation_loss(model, val_loader, device, inv_scale=False):
    model.eval()
    val_loss_all = []

    with torch.inference_mode():
        for inp, target in val_loader:
            inp = inp.to(device)
            target = target.to(device)

            out = model(inp)
            out = out[0]

            if inv_scale:
                out = torch.sinh(out) / 1e7
                target = torch.sinh(target) / 1e7

            batch_psnr = psnr_batch_torch(target, out)
            val_loss_all.append(batch_psnr)

    # Concatenate and average
    all_psnrs = torch.cat(val_loss_all)
    return all_psnrs.mean().item()


# %% Scaling function
def inverse_scale(
    arr: np.ndarray, scale_fac: float = 1e7, arcsin: bool = True
) -> np.ndarray:
    if arcsin:
        arr = np.sinh(arr)
    arr = arr / scale_fac
    return arr


# %% Training Function
def train(
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    device,
    **kwargs,
):
    # Initialize tracking variables
    start_time = time.time()
    best_val_mse = np.inf
    best_weights = None
    current_epoch = 0
    best_epoch = 0

    train_loss_list, recon_loss_list, kl_loss_list, val_loss_list, lr_list = (
        [],
        [],
        [],
        [],
        [],
    )

    # Configuration from kwargs
    filename = kwargs.get("filename")
    scheduler_params = kwargs.get("scheduler_params", None)
    save_freq = kwargs.get("save_freq", 50)
    beta = kwargs.get("beta", 1.0)
    precision = kwargs.get("precision", 4)
    inv_scale = kwargs.get("inv_scale", False)

    # Optional scheduler
    scheduler = None
    if scheduler_params:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_params
        )
    lr_init = np.inf

    print(f"Running on device: {device}")

    try:
        model, optimizer, checkpoint = load_ckp(filename, model, optimizer, device)

        current_epoch = checkpoint["epoch"]
        best_val_mse = checkpoint.get("best_val_mse", np.inf)
        best_weights = checkpoint.get("best_weights")
        val_loss_list = checkpoint.get("val_loss_list", [])
        recon_loss_list = checkpoint.get("recon_loss_list", [])
        kl_loss_list = checkpoint.get("kl_loss_list", [])
        train_loss_list = checkpoint.get("train_loss_list", [])
        lr_list = checkpoint.get("lr_list", [])

        if scheduler_params and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {current_epoch}")

    except (AttributeError, FileNotFoundError, TypeError):
        print("No saved checkpoints found. Starting from scratch.")

    # Training loop
    for epoch in range(epochs):
        if epoch < current_epoch:
            continue

        model.train()
        total_loss, recon_losses, kl_losses = [], [], []

        with tqdm(total=len(train_loader), **tqdm_kwargs) as pbar:
            pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

            for inp, target in train_loader:
                inp, target = inp.to(device), target.to(device)

                out = model(inp)
                # if epoch < 10:
                #     loss, recon, kl = vae_loss(target, *out, beta=0)
                # else:
                loss, recon, kl = vae_loss(target, *out, beta=beta, device=device)

                total_loss.append(loss.item())
                recon_losses.append(recon.item())
                kl_losses.append(kl.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Progress bar updates
                postfix = {
                    "Train Loss": f"{np.mean(total_loss):.{precision}e}",
                    "Recon Loss": f"{np.mean(recon_losses):.{precision}e}",
                    "KL Loss": f"{np.mean(kl_losses):.{precision}e}",
                }

                if scheduler:
                    current_lr = optimizer.param_groups[0]["lr"]
                    postfix["LR"] = f"{current_lr:.2e}"
                    if current_lr < lr_init:
                        print(f"Changing learning rate to {current_lr:.2e}")
                        lr_init = current_lr

                pbar.update(1)
                pbar.set_postfix(postfix)

            # Logging epoch results
            epoch_loss = np.mean(total_loss)
            epoch_recon = np.mean(recon_losses)
            epoch_kl = np.mean(kl_losses)

            train_loss_list.append(epoch_loss)
            recon_loss_list.append(epoch_recon)
            kl_loss_list.append(epoch_kl)
            lr_list.append(optimizer.param_groups[0]["lr"])

            # Validation
            if val_loader:
                val_loss = validation_loss(
                    model, val_loader, device=device, inv_scale=inv_scale
                )
                val_loss_list.append(val_loss)

                if scheduler:
                    scheduler.step(val_loss)

                # Update best model
                is_best = val_loss < best_val_mse
                if is_best and epoch >= 5:  # int(0.02 * epochs):
                    best_epoch = epoch
                    best_val_mse = val_loss
                    best_weights = copy.deepcopy(model.state_dict())

                # Prepare metrics for display
                pfix = {
                    "Train Loss": f"{epoch_loss:.{precision}e}",
                    "Recon Loss": f"{epoch_recon:.{precision}e}",
                    "KL Loss": f"{epoch_kl:.{precision}e}",
                    "Val Loss": (
                        f"{Color.RED}{-val_loss:.3f} dB{Color.OFF}"
                        if is_best
                        else f"{-val_loss:.3f} dB"
                    ),
                }
                pbar.set_postfix(pfix)
            else:
                best_weights = copy.deepcopy(model.state_dict())

        # Save final checkpoint
        if filename:
            # scripted_model = torch.jit.script(model)
            # scripted_model.save(filename[:-3] + ".jit")
            is_final_epoch = (epoch + 1) == epochs
            is_save_epoch = (epoch + 1) % save_freq == 0
            if is_final_epoch or is_save_epoch or is_best:
                checkpoint_data = {
                    "epoch": epoch + 1,
                    "model": model,
                    "optimizer": optimizer,
                    "best_weights": best_weights,
                    "filename": filename,
                    "best_val_mse": best_val_mse,
                    "val_loss_list": val_loss_list,
                    "recon_loss_list": recon_loss_list,
                    "kl_loss_list": kl_loss_list,
                    "beta": beta,
                    "train_loss_list": train_loss_list,
                    "lr_list": lr_list,
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

    total_time = time.time() - start_time
    print("-" * 50)

    if val_loader:
        best_idx = val_loss_list.index(min(val_loss_list))
        print(
            f"Training completed in {time_string(total_time)}\n"
            f"Best Val Loss at Epoch {best_idx + 1}\n"
            f"Saved Model from Epoch {best_epoch + 1}\n"
            f"MSE: Train={train_loss_list[best_epoch]:.{precision}f}, "
            f"Val={val_loss_list[best_epoch]:.{precision}f}\n"
            f"Recon={recon_loss_list[best_epoch]:.{precision}f}, "
            f"KL={kl_loss_list[best_epoch]:.{precision}f}"
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


# %% Prediction Function
def predict(
    model,
    weights,
    val_loader,
    device,
    print_stats=True,
    n=5,
    inv_scale={"arcsin": True, "scale_fac": 1e7},
):
    model.load_state_dict(weights)
    model.eval()

    # Accumulate results
    targets, inputs, outputs = [], [], []

    with torch.inference_mode():
        with tqdm(total=len(val_loader), **tqdm_kwargs) as pbar:
            for inp, target in val_loader:
                pbar.update(1)

                # Save original (before to(device))
                targets.append(target.numpy().squeeze())
                inputs.append(inp.numpy().squeeze())

                inp = inp.to(device)
                out = model(inp)

                outputs.append(out[0].cpu().numpy().squeeze())

    # Concatenate batch-wise results
    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    inputs = np.concatenate(inputs)

    if inv_scale:
        targets = inverse_scale(targets, **inv_scale)
        outputs = inverse_scale(outputs, **inv_scale)
        inputs = inverse_scale(inputs, **inv_scale)

    # Compute image quality metrics
    psnr_out = psnr_batch(targets, outputs)
    ssim_out = ssim_batch(targets, outputs)

    psnr_in = psnr_batch(targets, inputs)
    ssim_in = ssim_batch(targets, inputs)

    if print_stats:
        print(
            f"SSIM: Max {np.max(ssim_out):.03f} | Min {np.min(ssim_out):.03f} | Mean {np.mean(ssim_out):.03f}"
        )
        print(
            f"PSNR: Max {np.max(psnr_out):.03f} dB | Min {np.min(psnr_out):.03f} dB | Mean {np.mean(psnr_out):.03f} dB"
        )

    # Store metrics for external use
    metrics = {
        "psnr_out": psnr_out,
        "ssim_out": ssim_out,
        "psnr_in": psnr_in,
        "ssim_in": ssim_in,
        "targets": targets,
        "output": outputs,
        "input": inputs,
    }

    # Optional: Visualization
    if n > 0:
        tit_out = [f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_out[:n], psnr_out[:n])]
        blank_titles = [None] * n
        tit_in = [f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_in[:n], psnr_in[:n])]

        plot(
            images=[targets[:n], inputs[:n], outputs[:n], targets[:n] - outputs[:n]],
            caption=["Target", "Input", "Recon", "Residual"],
            cbar=True,
            scale_row=0,
            same_scale=[0, 1, 2],
            subtitles=[blank_titles, tit_in, tit_out, blank_titles],
        )

    return metrics


def plot_bad_cases(
    metrics,
    names=["PSNR", "SSIM"],
    category="input",
    n=5,
    inv_scale={"arcsin": True, "scale_fac": 1e7},
):
    psnr_out = metrics["psnr_out"]
    ssim_out = metrics["ssim_out"]
    psnr_in = metrics["psnr_in"]
    ssim_in = metrics["ssim_in"]
    targets = metrics["targets"]
    outputs = metrics["output"]
    inputs = metrics["input"]

    if category == "input":
        stats = [psnr_in, ssim_in]
        key = "inputs"
    elif category == "output":
        stats = [psnr_out, ssim_out]
        key = "reconstructions"
    else:
        print("Unknown category")

    for stat, name in zip(stats, names):
        ind = np.argsort(stat)[:n]

        tit_in = [f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_in[ind], psnr_in[ind])]
        tit_out = [
            f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_out[ind], psnr_out[ind])
        ]
        blank_titles = [None] * len(tit_in)

        target_ind = targets[ind]
        input_ind = inputs[ind]
        output_ind = outputs[ind]

        if inv_scale:
            target_ind = inverse_scale(target_ind, **inv_scale)
            input_ind = inverse_scale(input_ind, **inv_scale)
            output_ind = inverse_scale(output_ind, **inv_scale)

        residual_ind = target_ind - output_ind

        plot(
            images=[target_ind, input_ind, output_ind, residual_ind],
            caption=["Target", "Input", "Recon", "Residual"],
            cbar=True,
            scale_row=0,
            same_scale=[0, 1, 2],
            subtitles=[blank_titles, tit_in, tit_out, blank_titles],
            suptitle=f"Worst {key}: ({name})",
        )
