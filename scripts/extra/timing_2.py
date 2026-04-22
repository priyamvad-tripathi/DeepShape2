# %% Import Libraries
import argparse
import logging
import os
import time
import warnings

import numpy as np

from deepshape2.utils import (
    extract_image,
    get_freest_gpu,
    load,
    load_config,
    load_h5,
    psnr_batch,
    save,
    set_seed,
)

warnings.warn = lambda *args, **kwargs: None
logging.getLogger().addHandler(logging.NullHandler())


def log_time(msg, t0):
    now = time.time()
    print(f"[{now - t0:8.2f} s] {msg}")
    return now


# %% Full timing script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["facets", "recons", "plot"],
        required=True,
        help="'facets' = compute dirty/PSF, 'recons' = reconstruct+PSNR, 'plot' = plotting",
    )
    args = parser.parse_args()

    t_start = time.time()

    cfg = load_config()
    DATA_DIR = cfg["DATA_DIR"]
    MODEL_DIR = cfg["MODEL_DIR"]

    RESULTS_PATH = DATA_DIR + "timing_results_diff_grid_size.pkl"

    # Configs to sweep
    npix_batch_pairs = [
        # (512, 16),
        # (256, 32),
        # (128, 64),
        (64, 128),
    ]

    # =========================================================================
    # MODE: facets=on  →  compute dirty images + PSFs, save results
    # =========================================================================
    if args.mode == "facets":
        from dask.distributed import Client, LocalCluster
        from ska_sdp_datamodels.visibility import create_visibility_from_ms

        from deepshape2.reconstruction import get_facets

        # Load or create results dict
        if os.path.exists(RESULTS_PATH):
            print(f"Results file found at {RESULTS_PATH}, loading...")
            results = load(RESULTS_PATH)
        else:
            print(f"No results file found at {RESULTS_PATH}, creating new one...")
            results = {}

        set_seed()
        log_time("Loaded config and selected device", t_start)

        h5_path = DATA_DIR + "deep_set.h5"
        h5 = load_h5(h5_path, "r")

        vis_filename = DATA_DIR + "MS/vis_deep_set_patch_000.ms"

        patch = h5["patch_000"]
        sky = patch["sky"][()]
        patch_ra, patch_dec = patch.attrs["centre"]

        t_vis = time.time()
        vis = create_visibility_from_ms(vis_filename)[0]
        log_time("Loaded visibilities", t_vis)

        patch_df = patch["patch_df"][()]
        flux_mask = patch_df["flux_mask"]
        flux = patch_df["flux"][flux_mask]
        mask = (patch["peaks"][:] > 3 * 0.71e-6) & (flux > 50e-6)

        galaxy_locations = patch_df[["pix_x", "pix_y"]][flux_mask][mask]
        inds = np.random.choice(range(len(galaxy_locations)), size=128, replace=False)
        galaxy_locations_sub = galaxy_locations[inds]
        ground_truth = extract_image(patch["blended_stamps"][mask][inds])

        results["ground_truth"] = ground_truth
        results["galaxy_locations_sub"] = galaxy_locations_sub
        results["inds"] = inds

        with (
            LocalCluster(
                n_workers=64,
                processes=True,
                threads_per_worker=1,
                scheduler_port=8786,
                memory_limit=0,
            ) as cluster,
            Client(cluster) as client,
        ):
            print("Dask dashboard:", client.dashboard_link)

            for NPIX_facet, _ in npix_batch_pairs:
                key = f"npix_{NPIX_facet}"
                results.setdefault(key, {})

                print(f"\n{'=' * 50}")
                print(f"  NPIX_facet={NPIX_facet}  [get_facets]")
                print(f"{'=' * 50}")

                t_facets = time.time()

                dirty_all, psf_all = get_facets(
                    vis=vis,
                    galaxy_locations=galaxy_locations_sub,
                    NPIX_facet=NPIX_facet,
                    client=client,
                )

                t_facets_elapsed = time.time() - t_facets
                log_time(f"[NPIX={NPIX_facet}] Created dirty images and PSFs", t_facets)

                results[key]["dirty_all"] = dirty_all
                results[key]["psf_all"] = psf_all
                results[key]["t_get_facets"] = t_facets_elapsed

                save(results, RESULTS_PATH)
                log_time("facets construction complete", t_start)

    # =========================================================================
    # MODE: facets=off  →  load results, reconstruct + PSNR
    # =========================================================================
    elif args.mode == "recons":
        from deepshape2.reconstruction import reconstruct_facets

        if not os.path.exists(RESULTS_PATH):
            raise FileNotFoundError(
                f"Results file not found at {RESULTS_PATH}. Run with --mode facets first."
            )

        results = load(RESULTS_PATH)
        print(f"Results loaded from {RESULTS_PATH}")

        device = get_freest_gpu(set_device=True)
        set_seed()
        log_time("Loaded config and selected device", t_start)

        ground_truth = results["ground_truth"]

        for NPIX_facet, batch_size in npix_batch_pairs:
            key = f"npix_{NPIX_facet}"

            if key not in results:
                raise KeyError(
                    f"Key '{key}' not found in results. "
                    "Make sure --facets on was run for this NPIX size."
                )

            print(f"\n{'=' * 50}")
            print(f"  NPIX_facet={NPIX_facet}, batch_size={batch_size}  [reconstruct]")
            print(f"{'=' * 50}")

            dirty_all = results[key]["dirty_all"]
            psf_all = results[key]["psf_all"]

            t_recon = time.time()

            recon_result = reconstruct_facets(
                dirty_all,
                psf_all,
                device,
                num_workers=4,
                bsize=batch_size,
            )

            recon = recon_result["recon"]

            if NPIX_facet > 128:
                recon = extract_image(recon)

            t_recon_elapsed = time.time() - t_recon
            log_time(f"[NPIX={NPIX_facet}] Reconstructed facets", t_recon)

            psnr_values = psnr_batch(
                recon, extract_image(ground_truth, recon.shape[-1])
            )

            psnr_mean = float(np.mean(psnr_values))
            psnr_std = float(np.std(psnr_values))

            results[key]["recon"] = recon
            results[key]["t_reconstruct_facets"] = t_recon_elapsed
            results[key]["psnr_values"] = psnr_values

            print(f"  PSNR  mean={psnr_mean:.4f} dB  |  std={psnr_std:.4f} dB")

        # Summary table
        print(f"\n{'=' * 50}")
        print("  Summary")
        print(f"{'=' * 50}")
        print(
            f"  {'NPIX':<8} {'batch':>6}  {'t_facets':>10}  {'t_recon':>10}  {'PSNR mean':>10}  {'PSNR std':>9}"
        )
        for NPIX_facet, batch_size in npix_batch_pairs:
            key = f"npix_{NPIX_facet}"
            r = results[key]
            print(
                f"  {NPIX_facet:<8} {batch_size:>6}  "
                f"{r.get('t_get_facets', float('nan')):>9.2f}s  "
                f"{r['t_reconstruct_facets']:>9.2f}s  "
            )

        save(results, RESULTS_PATH)
        log_time("recons complete", t_start)

    # =========================================================================
    # MODE: plot  →  placeholder
    # =========================================================================
    elif args.facets == "plot":
        import matplotlib.pyplot as plt

        from deepshape2.visualization.base import savefig, set_style

        if not os.path.exists(RESULTS_PATH):
            raise FileNotFoundError(
                f"Results file not found at {RESULTS_PATH}. "
                "Run with --facets on and --facets off first."
            )

        results = load(RESULTS_PATH)
        set_style()
        print(f"Results loaded from {RESULTS_PATH}")

        n_galaxies = len(results["ground_truth"])

        npix_sizes = [64, 128, 256, 512]
        x_ticks = [f"{n}×{n}" for n in npix_sizes]
        x_pos = np.arange(len(npix_sizes))

        t_total = []
        psnr_median = []

        for npix in npix_sizes:
            key = f"npix_{npix}"
            r = results[key]
            t_total.append((r["t_get_facets"] + r["t_reconstruct_facets"]) / n_galaxies)
            psnr_median.append(float(np.median(r["psnr_values"])))

        t_total = np.array(t_total)
        psnr_median = np.array(psnr_median)

        fig, ax_time = plt.subplots(figsize=(6, 4.2))
        ax_psnr = ax_time.twinx()

        COLOR_TIME = "C0"
        COLOR_PSNR = "C3"

        # --- Total time (left axis) ---
        (l_time,) = ax_time.plot(
            x_pos,
            t_total,
            marker="o",
            lw=1.5,
            ls="-",
            color=COLOR_TIME,
            label="Total time",
        )
        ax_time.set_ylabel("Time per galaxy [s]", color=COLOR_TIME)
        ax_time.tick_params(axis="y", colors=COLOR_TIME)
        ax_time.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_time.yaxis.get_offset_text().set_color(COLOR_TIME)

        # --- PSNR (right axis) ---
        (l_psnr,) = ax_psnr.plot(
            x_pos,
            psnr_median,
            marker="s",
            lw=1.5,
            ls="--",
            color=COLOR_PSNR,
            label="PSNR (median)",
        )
        ax_psnr.set_ylabel("PSNR [dB]", color=COLOR_PSNR)
        ax_psnr.tick_params(axis="y", colors=COLOR_PSNR)

        ax_time.set_xticks(x_pos)
        ax_time.set_xticklabels(x_ticks)
        ax_time.set_xlabel("Facet size")

        ax_time.legend([l_time, l_psnr], ["Total time", "PSNR"], loc="center left")

        fig.tight_layout()
        # savefig(None)
        savefig(cfg["RESULTS_DIR"] + "timing_summary_plot.pdf")
