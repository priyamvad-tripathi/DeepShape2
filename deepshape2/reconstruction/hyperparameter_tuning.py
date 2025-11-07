# %% Imports
import argparse
import os
import time

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset

from deepshape2.data.loaders import CenterCrop
from deepshape2.models import VAE, create_model
from deepshape2.utils import (
    get_freest_gpu,
    get_progress_bar,
    get_tqdm,
    load,
    load_config,
    load_h5,
    psnr_torch,
    save,
    set_seed,
    ssim_batch,
    time_string,
)
from deepshape2.visualization import plot

# %% Constants / Configuration

SPLITS = [5, 10, 20, 30]
SCALE_FAC = 5e7
N_PLOT = 5
N_TRIALS = 200


# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Sky simulation arguments")

    parser.add_argument(
        "-f",
        "--facet-size",
        type=int,
        default=256,
        help="Facet size (default: 256)",
    )

    parser.add_argument(
        "-s",
        "--split-index",
        type=int,
        default=0,
        help="Split index (default: 0)",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )

    return parser.parse_args()


# %% Helper Functions


def objective(trial, device, val_loader_subset, deblender, crop_fn):
    """Optuna objective: tune f1, f2, falpha."""
    f1 = trial.suggest_float("f1", 0, 0.5)
    f2 = trial.suggest_float("f2", f1, 4)
    falpha = trial.suggest_float("falpha", 1, 6)

    model = create_model(device=device, f1=f1, f2=f2, falpha=falpha)
    model.eval()
    torch.cuda.empty_cache()

    psnr_values = []
    with torch.inference_mode():
        for im, isolated_stamp in val_loader_subset:
            im, isolated_stamp = (
                im.to(device, non_blocking=True),
                isolated_stamp.to(device, non_blocking=True),
            )

            # Deconvolution
            deconvolved = model(im)
            decon_crop = crop_fn(deconvolved)

            # Deblend
            decon_scaled = torch.arcsinh_(decon_crop.mul_(SCALE_FAC))
            decon_deblended = deblender(decon_scaled)[0]
            recon = torch.sinh_(decon_deblended).div_(SCALE_FAC)

            psnr_values.append(
                psnr_torch(recon, isolated_stamp.unsqueeze(1)).detach().cpu()
            )

    return -torch.cat(psnr_values).mean()


def evaluate_best_params(
    val_loader, best_params, device, deblender, crop_fn=CenterCrop(128), n=N_PLOT
):
    """
    Evaluate PSNR and SSIM using the best Optuna parameters and visualize examples.
    """

    # Initialize model
    model = create_model(device=device, **best_params)
    model.eval()

    # Containers for metrics and outputs
    psnr_list, ssim_list = [], []
    targets, inputs, outputs, decon_all = [], [], [], []

    # Progress bar
    pbar = get_progress_bar(True, total=len(val_loader), **get_tqdm())

    with torch.inference_mode(), pbar:
        for im, iso in val_loader:
            im = im.to(device, non_blocking=True)
            iso = iso.to(device, non_blocking=True)

            # --- Model inference ---
            decon = crop_fn(model(im))

            decon_all.append(decon.squeeze(1).cpu())

            # --- Deblending ---
            decon_scaled = torch.arcsinh_(decon * SCALE_FAC)
            deblended = deblender(decon_scaled)[0]
            recon = torch.sinh_(deblended) / SCALE_FAC

            # --- Move to CPU once ---
            recon_cpu = recon.squeeze(1).cpu()
            iso_cpu = iso.cpu()
            inp_cpu = crop_fn(im[:, 0]).cpu()

            # --- Metrics ---
            psnr = psnr_torch(recon_cpu.unsqueeze(1), iso_cpu.unsqueeze(1))
            ssim = ssim_batch(recon_cpu.numpy(), iso_cpu.numpy())

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            targets.append(iso_cpu)
            inputs.append(inp_cpu)
            outputs.append(recon_cpu)

            pbar.update(1)

    # --- Aggregate ---
    psnr_all = torch.cat(psnr_list).numpy()
    ssim_all = np.concatenate(ssim_list)
    targets_all = torch.cat(targets).numpy()
    inputs_all = torch.cat(inputs).numpy()
    outputs_all = torch.cat(outputs).numpy()
    decon_all = torch.cat(decon_all).numpy()

    # --- Report ---
    print(
        f"SSIM: Max {ssim_all.max():.03f} | Min {ssim_all.min():.03f} | Mean {ssim_all.mean():.03f}"
    )
    print(
        f"PSNR: Max {psnr_all.max():.03f} dB | Min {psnr_all.min():.03f} dB | Mean {psnr_all.mean():.03f} dB"
    )

    # --- Visualization ---
    if n > 0:
        subtitles = [None] * n
        metrics_str = [
            f"{s:.02f}/{p:.02f} dB" for s, p in zip(ssim_all[:n], psnr_all[:n])
        ]

        plot(
            images=[
                targets_all[:n],
                inputs_all[:n],
                decon_all[:n],
                outputs_all[:n],
                targets_all[:n] - outputs_all[:n],
            ],
            caption=["Target", "Input", "Deconvolved", "Recon", "Residual"],
            cbar=True,
            subtitles=[subtitles, subtitles, subtitles, metrics_str, subtitles],
            same_scale=[2, 3],
            scale_row=3,
        )


# %% Main Execution
if __name__ == "__main__":
    args = parse_args()
    GRID_SIZE = args.facet_size
    USE_SPLIT = args.split_index
    BATCH_SIZE = args.batch_size

    # --- Load Config and Data
    cfg = load_config()
    DATA_DIR = cfg["DATA_DIR"]
    facet_data = load_h5(DATA_DIR + "facets.h5")

    # --- Torch setup
    device = get_freest_gpu(set_device=True)
    set_seed()

    # --- Load best params (if exist)
    optuna_best_params_path = os.path.join(DATA_DIR, "HQS_PNP_hyperaparamters.pkl")
    best_params_dict = (
        load(optuna_best_params_path) if os.path.exists(optuna_best_params_path) else {}
    )

    # --- Cropping transform
    crop_128 = CenterCrop(128)

    # --- Load pre-trained deblender
    deblender = VAE().to(device)
    ckpt_path = os.path.join(cfg["MODEL_DIR"], "vae_mha.pt")
    deblender.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)["best_weights"]
    )

    # Dataset setup
    dirty_all = facet_data[f"wide/facets_{GRID_SIZE}/dirty"][:]
    psf = facet_data[f"wide/facets_{GRID_SIZE}/psf"][:]
    im_all = np.stack([dirty_all, psf], axis=1)

    isolated_stamps = facet_data["wide/isolated_stamps"][:]

    im_all_t = torch.from_numpy(im_all)
    stamps_t = torch.from_numpy(isolated_stamps)
    dataset_all = TensorDataset(im_all_t, stamps_t)

    # Create subset for validation set for Optuna
    peak = facet_data["wide/peak"][:]
    threshold = np.percentile(peak, SPLITS[USE_SPLIT])
    mask = np.where(peak > threshold)[0]

    # Limit to at most 1000 samples
    max_samples = 1000
    num_samples = min(len(mask), max_samples)

    # Randomly sample indices without replacement
    rand_indices = np.random.choice(mask, size=num_samples, replace=False)

    # Create TensorDataset with selected samples
    dataset_subset = TensorDataset(im_all_t[rand_indices], stamps_t[rand_indices])

    val_loader_all = DataLoader(
        dataset_all,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    val_loader_subset = DataLoader(
        dataset_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- Optuna study
    study_name = f"facets_{GRID_SIZE}_split_{SPLITS[USE_SPLIT]}"
    optuna_trials_dir = f"{DATA_DIR}/optuna_trials/"
    os.makedirs(optuna_trials_dir, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=f"sqlite:///{optuna_trials_dir}/{study_name}.db",
        load_if_exists=True,
    )

    start = time.time()
    study.optimize(
        lambda trial: objective(trial, device, val_loader_subset, deblender, crop_128),
        n_trials=N_TRIALS,
    )

    # --- Save best parameters
    best_params_dict[study_name] = study.best_params.copy()
    save(best_params_dict, optuna_best_params_path)

    print("Optuna Hyperparameter Tuning Complete.")
    print("Time Taken:", time_string(time.time() - start))
    print("Best Parameters:", study.best_params)

    # --- Evaluate and plot
    evaluate_best_params(
        val_loader_all, study.best_params, device, deblender, crop_fn=crop_128
    )

    # --- Optuna visualization
    fig1 = optuna.visualization.plot_parallel_coordinate(study)
    fig1.write_html(optuna_trials_dir + f"Figs/parallel_coordinate_{study_name}.html")
    fig2 = optuna.visualization.plot_optimization_history(study)
    fig2.write_html(optuna_trials_dir + f"Figs/optimization_history_{study_name}.html")
    fig3 = optuna.visualization.plot_param_importances(study)
    fig3.write_html(optuna_trials_dir + f"Figs/param_importances_{study_name}.html")
