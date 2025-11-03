# %% Import Libraries
import copy
import time

import numpy as np
import torch
from colorist import Color

from deepshape2.deblending import vae_loss, validation_loss
from deepshape2.utils import (
    blendedness,
    get_progress_bar,
    get_tqdm,
    load_ckp,
    load_config,
    psnr_torch,
    save_ckp,
    ssim_batch,
    time_string,
)
from deepshape2.visualization import plot

tqdm_kwargs = get_tqdm()


__all__ = ["train", "predict", "plot_bad_cases", "predict_multiple"]

scale_fac = load_config().get("scale_fac", 5e7)


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

    # --- Config ---
    filename = kwargs.get("filename")
    scheduler_params = kwargs.get("scheduler_params", None)
    save_freq = kwargs.get("save_freq", 50)
    beta = kwargs.get("beta", 1.0)
    precision = kwargs.get("precision", 4)
    tqdm_enabled = kwargs.get("tqdm_enabled", True)
    tqdm_kwargs = kwargs.get("tqdm_kwargs", dict(colour="green", unit="batch"))
    variational = kwargs.get("variational", True)

    scheduler = None
    if scheduler_params:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_params
        )

    print(f"Running on device: {device}")

    # --- Load checkpoint ---
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
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {current_epoch}")
    except (AttributeError, FileNotFoundError, TypeError):
        print("No saved checkpoints found. Starting from scratch.")

    scale_fac = getattr(train_loader.dataset, "scale_fac", 1.0)

    # --- Training loop ---
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
                inp, target = (
                    inp.to(device, non_blocking=True),
                    target.to(device, non_blocking=True),
                )

                optimizer.zero_grad(set_to_none=True)
                out = model(inp)
                if variational:
                    loss, recon, kl = vae_loss(target, *out, beta=beta, device=device)
                else:
                    recon = torch.nn.functional.mse_loss(out, target)
                    kl = torch.tensor(0.0)
                    loss = recon

                loss.backward()
                optimizer.step()

                total_loss.append(loss.detach().cpu())
                recon_losses.append(recon.detach().cpu())
                kl_losses.append(kl.detach().cpu())

                # Progress bar updates
                postfix = {
                    "Train Loss": f"{(torch.stack(total_loss).mean()):.{precision}e}",
                    "Recon Loss": f"{(torch.stack(recon_losses).mean()):.{precision}e}",
                    "KL Loss": f"{(torch.stack(kl_losses).mean()):.{precision}e}",
                }
                if current_lr is not None:
                    postfix["LR"] = (
                        f"{Color.RED}{current_lr:.2e}{Color.OFF}"
                        if new_lr
                        else f"{current_lr:.2e}"
                    )

                pbar.update(1)
                pbar.set_postfix(postfix)

            # Epoch metrics
            epoch_loss = torch.stack(total_loss).mean().item()
            epoch_recon = torch.stack(recon_losses).mean().item()
            epoch_kl = torch.stack(kl_losses).mean().item()

            train_loss_list.append(epoch_loss)
            recon_loss_list.append(epoch_recon)
            kl_loss_list.append(epoch_kl)

            # Print updates if tqdm disabled
            if not tqdm_enabled:
                line0 = f"Epoch {epoch + 1}/{epochs}"
                if current_lr is not None:
                    line0 += f" | LR: {current_lr:.2e}" + (" NEW" if new_lr else "")
                print(line0)
                line = f"Train Loss: {epoch_loss:.{precision}e} | Recon: {epoch_recon:.{precision}e} | KL: {epoch_kl:.{precision}e}"

            # --- Validation ---
            if val_loader:
                val_loss = validation_loss(
                    model,
                    val_loader,
                    device=device,
                    scale_fac=scale_fac,
                    variational=variational,
                )
                val_loss_list.append(val_loss)
                if scheduler:
                    scheduler.step(val_loss)

                is_best = val_loss < best_val_loss
                if is_best:
                    best_epoch = epoch
                    best_val_loss = val_loss
                    best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

                pfix = {
                    "Train Loss": f"{epoch_loss:.{precision}e}",
                    "Recon Loss": f"{epoch_recon:.{precision}e}",
                    "KL Loss": f"{epoch_kl:.{precision}e}",
                    "Val Loss": f"{Color.RED}{-val_loss:.3f} dB{Color.OFF}"
                    if is_best
                    else f"{-val_loss:.3f} dB",
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
    scale_fac=scale_fac,
    print_stats=True,
    n=5,
    tqdm_enabled=True,
):
    if weights is not None:
        model.load_state_dict(weights)
    model.eval()

    # Accumulate results as tensors on GPU
    targets_all, inputs_all, outputs_all = [], [], []
    psnr_out_all, psnr_in_all = [], []

    with torch.inference_mode():
        pbar = get_progress_bar(tqdm_enabled, total=len(val_loader), **tqdm_kwargs)

        with pbar:
            for inp, target in val_loader:
                pbar.update(1)

                inp_gpu = inp.to(device, non_blocking=True)
                target_gpu = target.to(device, non_blocking=True)

                out = model(inp_gpu)
                if isinstance(out, (tuple, list)):
                    out = out[0]

                # Append GPU tensors
                targets_all.append(target_gpu)
                inputs_all.append(inp_gpu)
                outputs_all.append(out)

                # Scale images
                target_sc = torch.sinh(target_gpu) / scale_fac
                out_sc = torch.sinh(out) / scale_fac
                inp_sc = torch.sinh(inp_gpu) / scale_fac

                # Compute metrics on GPU
                psnr_batch_out = psnr_torch(target_sc, out_sc)
                psnr_batch_in = psnr_torch(target_sc, inp_sc)

                psnr_out_all.append(psnr_batch_out)
                psnr_in_all.append(psnr_batch_in)

    # Concatenate batches and move to CPU only once
    targets_all = np.sinh(torch.cat(targets_all).cpu().numpy().squeeze()) / scale_fac
    outputs_all = np.sinh(torch.cat(outputs_all).cpu().numpy().squeeze()) / scale_fac
    inputs_all = np.sinh(torch.cat(inputs_all).cpu().numpy().squeeze()) / scale_fac

    psnr_out = torch.cat(psnr_out_all).cpu().numpy()

    ssim_out = ssim_batch(targets_all, outputs_all)

    blend = blendedness(targets_all, inputs_all)

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
        "targets": targets_all,
        "output": outputs_all,
        "input": inputs_all,
        "blend": blend,
    }

    # Optional: Visualization
    if n > 0:
        tit_out = [f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_out[:n], psnr_out[:n])]
        tit2 = [f"{bl:.3f}" for bl in blend[:n]]
        blank_titles = [None] * n

        plot(
            images=[
                targets_all[:n],
                inputs_all[:n],
                outputs_all[:n],
                targets_all[:n] - outputs_all[:n],
            ],
            caption=["Target", "Input", "Recon", "Residual"],
            cbar=True,
            scale_row=0,
            same_scale=[0, 1, 2],
            subtitles=[blank_titles, tit2, tit_out, blank_titles],
        )

    return metrics


# %% Plotting Function
def plot_bad_cases(
    metrics,
    scale_fac=scale_fac,
    names=["PSNR", "SSIM"],
    n=5,
):
    psnr_out = metrics["psnr_out"]
    ssim_out = metrics["ssim_out"]
    targets = metrics["targets"]
    outputs = metrics["output"]
    inputs = metrics["input"]
    blend = metrics["blend"]

    stats = [psnr_out, ssim_out]
    key = "reconstructions"

    for stat, name in zip(stats, names):
        ind = np.argsort(stat)[:n]

        tit_in = [f"{bl:.3f}" for bl in blend[:n]]
        tit_out = [
            f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_out[ind], psnr_out[ind])
        ]
        blank_titles = [None] * len(tit_in)

        target_ind = targets[ind]
        input_ind = inputs[ind]
        output_ind = outputs[ind]

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


# %%


def predict_multiple(
    models,
    ckpts,
    val_loader,
    device,
    scale_fac,
    model_names=None,
    print_stats=True,
):
    """
    Predict with multiple models on the same dataloader in one pass.

    Args:
        models (list): List of PyTorch models.
        ckpts (list): List of checkpoint file paths (same order as models).
        val_loader (DataLoader): DataLoader for test/validation set.
        device (torch.device): Device to run inference on.
        scale_fac (float): Scaling factor for sinh transform.
        model_names (list[str], optional): Names for models; if None -> model_1, model_2, ...
        print_stats (bool): Whether to print summary metrics.

    Returns:
        dict: {model_name: metrics_dict}
    """
    assert len(models) == len(ckpts), "Each model must have a corresponding checkpoint"

    # Assign default names
    if model_names is None:
        model_names = [f"model_{i + 1}" for i in range(len(models))]

    # Load weights & move to device
    for model, ckp in zip(models, ckpts):
        state = torch.load(ckp, map_location=device, weights_only=False)
        model.load_state_dict(state["best_weights"])
        model.to(device)
        model.eval()

    # Initialize accumulators per model
    results = {
        name: {
            "targets": [],
            "inputs": [],
            "outputs": [],
            "psnr": [],
        }
        for name in model_names
    }

    # Single dataloader loop
    with torch.inference_mode():
        pbar = get_progress_bar(True, total=len(val_loader), **tqdm_kwargs)
        with pbar:
            for inp, target in pbar:
                pbar.update(1)
                inp_gpu = inp.to(device, non_blocking=True)
                target_gpu = target.to(device, non_blocking=True)

                # Precompute scaled tensors for PSNR comparison
                target_sc = torch.sinh(target_gpu) / scale_fac

                # Run all models on this batch
                for model, name in zip(models, model_names):
                    out = model(inp_gpu)
                    if isinstance(out, (tuple, list)):
                        out = out[0]

                    # Store tensors
                    results[name]["targets"].append(target_gpu)
                    results[name]["inputs"].append(inp_gpu)
                    results[name]["outputs"].append(out)

                    # Scale for PSNR
                    out_sc = torch.sinh(out) / scale_fac
                    psnr_batch = psnr_torch(target_sc, out_sc)
                    results[name]["psnr"].append(psnr_batch)

    # Post-processing (CPU transfer, SSIM, etc.)
    for name in model_names:
        r = results[name]

        targets_all = (
            np.sinh(torch.cat(r["targets"]).cpu().numpy().squeeze()) / scale_fac
        )
        outputs_all = (
            np.sinh(torch.cat(r["outputs"]).cpu().numpy().squeeze()) / scale_fac
        )
        inputs_all = np.sinh(torch.cat(r["inputs"]).cpu().numpy().squeeze()) / scale_fac
        psnr_out = torch.cat(r["psnr"]).cpu().numpy()
        ssim_out = ssim_batch(targets_all, outputs_all)
        blend = blendedness(targets_all, inputs_all)

        if print_stats:
            print(f"\nModel: {name}")
            print(
                f"SSIM → Max: {np.max(ssim_out):.03f} | Min: {np.min(ssim_out):.03f} | Mean: {np.mean(ssim_out):.03f}"
            )
            print(
                f"PSNR → Max: {np.max(psnr_out):.03f} dB | Min: {np.min(psnr_out):.03f} dB | Mean: {np.mean(psnr_out):.03f} dB"
            )

        # Replace raw tensors with final arrays
        results[name] = {
            "psnr_out": psnr_out,
            "ssim_out": ssim_out,
            "targets": targets_all,
            "output": outputs_all,
            "input": inputs_all,
            "blend": blend,
        }

    return results
