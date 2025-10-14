# %% Import Libraries
import copy
import time

import numpy as np
import torch
from colorist import Color

from deepshape2.deblending import vae_loss, validation_loss
from deepshape2.utils import (
    get_progress_bar,
    get_tqdm,
    load_ckp,
    psnr_torch,
    save_ckp,
    ssim_torch,
    time_string,
)
from deepshape2.visualization import plot

tqdm_kwargs = get_tqdm()


__all__ = ["train", "predict", "plot_bad_cases"]


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
    best_val_loss = np.inf
    best_weights = None
    current_epoch = 0
    best_epoch = 0

    train_loss_list, recon_loss_list, kl_loss_list, val_loss_list, lr_list = (
        [],
        [],
        [],
        [],
        [np.inf],
    )

    # Configuration from kwargs
    filename = kwargs.get("filename")
    scheduler_params = kwargs.get("scheduler_params", None)
    save_freq = kwargs.get("save_freq", 50)
    beta = kwargs.get("beta", 1.0)
    precision = kwargs.get("precision", 4)
    tqdm_enabled = kwargs.get("tqdm_enabled", True)

    # Optional scheduler
    scheduler = None
    if scheduler_params:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_params
        )

    print(f"Running on device: {device}")

    try:
        model, optimizer, checkpoint = load_ckp(filename, model, optimizer, device)

        current_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", np.inf)
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

    scale_fac = train_loader.dataset.scale_fac

    # Training loop
    for epoch in range(epochs):
        if epoch < current_epoch:
            continue

        model.train()
        total_loss, recon_losses, kl_losses = [], [], []

        if scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            new_lr = current_lr < lr_list[-1]
            lr_list.append(current_lr)

        pbar = get_progress_bar(tqdm_enabled, total=len(train_loader), **tqdm_kwargs)
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

        with pbar:
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

                if current_lr is not None:
                    postfix["LR"] = (
                        f"{Color.RED}{current_lr:.2e}{Color.OFF}"
                        if new_lr
                        else f"{current_lr:.2e}"
                    )

                pbar.update(1)
                pbar.set_postfix(postfix)

            # Logging epoch results
            epoch_loss = np.mean(total_loss)
            epoch_recon = np.mean(recon_losses)
            epoch_kl = np.mean(kl_losses)

            train_loss_list.append(epoch_loss)
            recon_loss_list.append(epoch_recon)
            kl_loss_list.append(epoch_kl)

            # Print updates if tqdm is disabled
            if not tqdm_enabled:
                line0 = f"Epoch {epoch + 1}/{epochs}"
                if current_lr is not None:
                    line0 += f" | LR: {current_lr:.2e}"
                    if new_lr:
                        line0 += " NEW"
                print(line0)
                line = (
                    f"Train Loss: {epoch_loss:.{precision}e} | "
                    f"Recon: {epoch_recon:.{precision}e} | "
                    f"KL: {epoch_kl:.{precision}e}"
                )

            # Validation
            if val_loader:
                val_loss = validation_loss(
                    model, val_loader, device=device, scale_fac=scale_fac
                )
                val_loss_list.append(val_loss)

                if scheduler:
                    scheduler.step(val_loss)

                # Update best model
                is_best = val_loss < best_val_loss
                if is_best and epoch >= 5:  # int(0.02 * epochs):
                    best_epoch = epoch
                    best_val_loss = val_loss
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

                if not tqdm_enabled:
                    marker = "BEST" if is_best else ""
                    line += f" | Val Loss: {-val_loss:.3f} dB {marker}"
            else:
                best_weights = copy.deepcopy(model.state_dict())

            if not tqdm_enabled:
                print(line)
                print("-" * 50)

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
                    "best_val_loss": best_val_loss,
                    "val_loss_list": val_loss_list,
                    "recon_loss_list": recon_loss_list,
                    "kl_loss_list": kl_loss_list,
                    "beta": beta,
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
    scale_fac,
    print_stats=True,
    n=5,
    tqdm_enabled=True,
):
    if weights is not None:
        model.load_state_dict(weights)
    model.eval()

    # Accumulate results
    targets, inputs, outputs = [], [], []

    psnr_out_all, ssim_out_all = [], []
    psnr_in_all, ssim_in_all = [], []

    with torch.inference_mode():
        pbar = get_progress_bar(tqdm_enabled, total=len(val_loader), **tqdm_kwargs)

        with pbar:
            for inp, target in val_loader:
                pbar.update(1)

                # Save original (before to(device))
                targets.append(target.numpy().squeeze())
                inputs.append(inp.numpy().squeeze())

                inp = inp.to(device)
                out = model(inp)
                out = out[0]

                outputs.append(out.cpu().numpy().squeeze())

                target_sc = torch.sinh(target) / scale_fac
                out_sc = torch.sinh(out) / scale_fac
                inp_sc = torch.sinh(inp) / scale_fac

                psnr_out = psnr_torch(target_sc, out_sc)
                _, ssim_out = ssim_torch(target_sc, out_sc)

                psnr_in = psnr_torch(target_sc, inp_sc)
                _, ssim_in = ssim_torch(target_sc, inp_sc)

                psnr_out_all.append(psnr_out.cpu())
                ssim_out_all.append(ssim_out.cpu())

                psnr_in_all.append(psnr_in.cpu())
                ssim_in_all.append(ssim_in.cpu())

    # Concatenate batch-wise results
    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    inputs = np.concatenate(inputs)

    # Compute image quality metrics
    psnr_out = torch.cat(psnr_out_all).numpy()
    ssim_out = torch.cat(ssim_out_all).numpy()

    psnr_in = torch.cat(psnr_in_all).numpy()
    ssim_in = torch.cat(ssim_in_all).numpy()

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


# %% Plotting Function
def plot_bad_cases(
    metrics,
    scale_fac,
    names=["PSNR", "SSIM"],
    category="input",
    n=5,
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

        target_ind = np.sinh(target_ind) / scale_fac
        input_ind = np.sinh(input_ind) / scale_fac
        output_ind = np.sinh(output_ind) / scale_fac

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
