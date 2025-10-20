# %% Imports
import os
import time

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset

from deepshape2.data.loaders import CenterCrop
from deepshape2.models import VAE, create_model
from deepshape2.utils import (
    extract_image,
    get_freest_gpu,
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
GRID_SIZE = 128
SPLITS = [5, 10, 20, 30]
USE_SPLIT = 0
BATCH_SIZE = 64
SCALE_FAC = 5e7
N_PLOT = 5
N_TRIALS = 200

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
            iso_crop = crop_fn(isolated_stamp)

            # Deblend
            decon_scaled = torch.arcsinh_(decon_crop.mul_(SCALE_FAC))
            decon_deblended = deblender(decon_scaled)[0]
            recon = torch.sinh_(decon_deblended).div_(SCALE_FAC)

            psnr_values.append(psnr_torch(recon, iso_crop.unsqueeze(1)).detach().cpu())

    return -torch.cat(psnr_values).mean()


def evaluate_best_params(
    val_loader, best_params, device, deblender, crop_fn=CenterCrop(128), n=N_PLOT
):
    """Evaluate PSNR and SSIM using best Optuna parameters and plot first n examples."""
    model = create_model(device=device, **best_params)
    model.eval()

    psnr_all, ssim_all = [], []
    targets_all, inputs_all, outputs_all = [], [], []

    with torch.inference_mode():
        for im, isolated_stamp in val_loader:
            im, isolated_stamp = (
                im.to(device, non_blocking=True),
                isolated_stamp.to(device, non_blocking=True),
            )

            # Deconvolution and cropping
            deconvolved = model(im)
            decon_crop = crop_fn(deconvolved)
            iso_crop = crop_fn(isolated_stamp)

            # Deblend & inverse scale
            decon_scaled = torch.arcsinh_(decon_crop * SCALE_FAC)
            decon_deblended = deblender(decon_scaled)[0]
            recon = torch.sinh_(decon_deblended) / SCALE_FAC

            # Metrics
            psnr_all.append(psnr_torch(recon, iso_crop.unsqueeze(1)))
            ssim_all.append(
                ssim_batch(recon.squeeze(1).cpu().numpy(), iso_crop.cpu().numpy())
            )

            targets_all.append(iso_crop.cpu())
            inputs_all.append(im[:, 0].cpu())  # first channel of input
            outputs_all.append(recon.squeeze(1).cpu())

    psnr_mean = torch.cat(psnr_all).mean().item()
    ssim_mean = np.concatenate(ssim_all).mean()
    print(f"Average PSNR: {psnr_mean:.3f} dB | Average SSIM: {ssim_mean:.3f}")

    if n > 0:
        # Stack first n examples
        targets_all = torch.cat(targets_all)[:n].numpy()
        inputs_all = torch.cat(inputs_all)[:n].numpy()
        outputs_all = torch.cat(outputs_all)[:n].numpy()
        tit_out = [
            f"{s:.02f}/{p:.02f} dB"
            for s, p in zip(
                np.concatenate(ssim_all)[:n], torch.cat(psnr_all)[:n].numpy()
            )
        ]
        blank_titles = [None] * n

        plot(
            images=[targets_all, inputs_all, outputs_all, targets_all - outputs_all],
            caption=["Target", "Input", "Recon", "Residual"],
            cbar=True,
            scale_row=0,
            same_scale=[0, 1, 2],
            subtitles=[blank_titles, blank_titles, tit_out, blank_titles],
        )


# %% Main Execution
if __name__ == "__main__":
    # --- Load Config and Data
    cfg = load_config()
    DATA_DIR = cfg["DATA_DIR"]
    facet_data = load_h5(DATA_DIR + "facets.h5")
    with load_h5(DATA_DIR + "wide_set.h5") as wide_set:
        isolated_stamps = extract_image(wide_set["patch_051/blended_stamps"][:], 128)

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
    ckpt_path = os.path.join(cfg["MODEL_DIR"], "vae_deblender.pt")
    deblender.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)["best_weights"]
    )

    # Dataset setup
    dirty_all = facet_data[f"wide/facets_{GRID_SIZE}/dirty"][:]
    psf = facet_data[f"wide/facets_{GRID_SIZE}/psf"][:]
    im_all = np.stack([dirty_all, psf], axis=1)

    im_all_t = torch.from_numpy(im_all)
    stamps_t = torch.from_numpy(isolated_stamps)
    dataset_all = TensorDataset(im_all_t, stamps_t)

    # Create subset for validation set for Optuna
    peak = facet_data["wide/peak"][:]
    threshold = np.percentile(peak, SPLITS[USE_SPLIT])
    mask = np.where(peak > threshold)[0]

    # Limit to at most 2000 samples
    max_samples = 2000
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
    study_name = f"facets_{GRID_SIZE}_split_{SPLITS[USE_SPLIT]}_trials_{N_TRIALS}"
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
    best_params_dict[f"facets_{GRID_SIZE}"] = {
        f"split_{SPLITS[USE_SPLIT]}": study.best_params
    }
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
