# %% Imports
import argparse
import os
import time

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset

from deepshape2.data.loaders import CenterCrop
from deepshape2.models import create_model
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
N_PLOT = 5
N_TRIALS = 100


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


def objective(trial, device, val_loader_subset, crop_fn=CenterCrop(64)):
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
            iso_crop = crop_fn(isolated_stamp)

            psnr_values.append(
                psnr_torch(decon_crop, iso_crop.unsqueeze(1)).detach().cpu()
            )

    return -torch.cat(psnr_values).mean()


def evaluate_best_params(
    val_loader,
    best_params,
    device,
    crop_fn=CenterCrop(128),
    n=N_PLOT,
):
    """
    Evaluate the model using the best Optuna parameters on a validation dataloader.

    Computes PSNR and SSIM, reports their statistics, and optionally visualizes
    a few representative examples (targets, inputs, reconstructions, and residuals).
    """

    # --- Initialize model ---
    model = create_model(device=device, **best_params)
    model.eval()

    # --- Metric and image containers ---
    psnr_all = []
    targets, inputs, recon_all = [], [], []

    # --- Progress bar setup ---
    pbar = get_progress_bar(True, total=len(val_loader), **get_tqdm())

    with torch.inference_mode(), pbar:
        for im, iso in val_loader:
            # Move to device
            im = im.to(device, non_blocking=True)
            iso = iso.to(device, non_blocking=True)

            # Store target (ground truth)
            targets.append(iso.cpu().numpy())

            # --- Model inference ---
            recon = crop_fn(model(im))

            # --- Metrics ---
            psnr_val = psnr_torch(recon, iso.unsqueeze(1))
            psnr_all.append(psnr_val.cpu().numpy())

            # --- Store reconstructions and inputs ---
            recon_all.append(recon.cpu().numpy().squeeze())
            inputs.append(crop_fn(im[:, 0]).cpu().numpy())

            pbar.update(1)

    # --- Aggregate arrays ---
    targets = np.concatenate(targets)
    inputs = np.concatenate(inputs)
    recon_all = np.concatenate(recon_all)
    psnr_all = np.concatenate(psnr_all)

    # --- Compute SSIM ---
    ssim_all = ssim_batch(targets, recon_all)

    # --- Report metrics ---
    print(
        f"SSIM: Max {ssim_all.max():.3f} | Min {ssim_all.min():.3f} | Mean {ssim_all.mean():.3f}"
    )
    print(
        f"PSNR: Max {psnr_all.max():.3f} dB | Min {psnr_all.min():.3f} dB | Mean {psnr_all.mean():.3f} dB"
    )

    # --- Visualization ---
    if n > 0:
        subtitles = [None] * n
        flux_labels = [f"{np.max(fl) * 1e9:.2f} nJy" for fl in inputs[:n]]
        metric_labels = [
            f"{s:.2f}/{p:.2f} dB" for s, p in zip(ssim_all[:n], psnr_all[:n])
        ]

        plot(
            images=[
                targets[:n],
                inputs[:n],
                recon_all[:n],
                targets[:n] - recon_all[:n],
            ],
            caption=["Target", "Input", "Reconstructed", "Residual"],
            cbar=True,
            subtitles=[flux_labels, subtitles, metric_labels, subtitles],
            # same_scale=[2, 3],
            # scale_row=3,
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
        lambda trial: objective(trial, device, val_loader_subset),
        n_trials=N_TRIALS,
    )

    # --- Save best parameters
    best_params_dict[study_name] = study.best_params.copy()
    save(best_params_dict, optuna_best_params_path)

    print("Optuna Hyperparameter Tuning Complete.")
    print("Time Taken:", time_string(time.time() - start))
    print("Best Parameters:", study.best_params)

    # --- Evaluate and plot
    evaluate_best_params(val_loader_all, study.best_params, device)

    # --- Optuna visualization
    fig1 = optuna.visualization.plot_parallel_coordinate(study)
    fig1.write_html(optuna_trials_dir + f"Figs/parallel_coordinate_{study_name}.html")
    fig2 = optuna.visualization.plot_optimization_history(study)
    fig2.write_html(optuna_trials_dir + f"Figs/optimization_history_{study_name}.html")
    fig3 = optuna.visualization.plot_param_importances(study)
    fig3.write_html(optuna_trials_dir + f"Figs/param_importances_{study_name}.html")
