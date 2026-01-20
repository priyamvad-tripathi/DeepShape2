import argparse
import time

import numpy as np
import torch
from dask.distributed import Client, LocalCluster

from deepshape2.models import VAE
from deepshape2.reconstruction import get_facets, reconstruct_facets
from deepshape2.simulation import simulate_visibilities
from deepshape2.utils import (
    extract_image,
    get_freest_gpu,
    load_config,
    load_h5,
    post_step,
    psnr_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Facet generation and reconstruction")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["facets", "reconstruct"],
        required=True,
        default="reconstruct",
        help="Run only facet extraction or only reconstruction",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start = time.time()

    cfg = load_config()
    DATA_DIR = cfg["DATA_DIR"]
    hf_path = DATA_DIR + "wide_set.h5"

    patches = [f"patch_{nl + 1:03d}" for nl in range(71, 100)]
    data = load_h5(hf_path, "a", delete_if_exists=False)

    # -----------------------------
    # --- GPU setup (only if reconstructing)
    # -----------------------------
    if args.mode == "reconstruct":
        device = get_freest_gpu(set_device=True)

        ckpt_path = cfg["MODEL_DIR"] + "vae_mha.pt"
        deblender = VAE().to(device)
        deblender.load_state_dict(
            torch.load(ckpt_path, map_location=device)["best_weights"]
        )
        deblender.eval()

    # -----------------------------
    # --- Dask cluster (only for facets)
    # -----------------------------
    dask_ctx = (
        LocalCluster(
            n_workers=64,
            processes=True,
            threads_per_worker=1,
            scheduler_port=8786,
            memory_limit=0,
        )
        if args.mode == "facets"
        else None
    )

    client_ctx = Client(dask_ctx) if dask_ctx else None

    if client_ctx:
        print("Dask dashboard:", client_ctx.dashboard_link)

    # -----------------------------
    # --- Process patches
    # -----------------------------
    for patch_name in patches:
        patch_group = data[patch_name]
        post_step(f"opening {patch_name}", start, data=data)

        if args.mode == "facets":
            patch_df = patch_group["patch_df"][()]
            patch_sky = patch_group["sky"][()]
            patch_ra, patch_dec = patch_group.attrs["centre"]

            vis = simulate_visibilities(
                field=patch_sky,
                ra_pointing=patch_ra,
                dec_pointing=patch_dec,
                create_dirty=False,
                threads=60,
            )

            mask = patch_df["flux_mask"]
            galaxy_locations = patch_df[["pix_x", "pix_y"]][mask]

            dirty_all, psf_all = get_facets(
                vis=vis,
                galaxy_locations=galaxy_locations,
                NPIX_facet=128,
                client=client_ctx,
            )

            patch_group.create_dataset(
                "dirty", data=dirty_all, compression="gzip", chunks=(1, 128, 128)
            )
            patch_group.create_dataset(
                "psf", data=psf_all, compression="gzip", chunks=(1, 128, 128)
            )

            post_step(f"saved dirty/psf facets {dirty_all.shape}", start, data=data)

        elif args.mode == "reconstruct":
            dirty_all = patch_group["dirty"][:]
            psf_all = patch_group["psf"][:]

            result = reconstruct_facets(
                dirty_all,
                psf_all,
                device=device,
                num_workers=4,
                deblender=deblender,
            )

            recon = result["recon"]
            decon = result["decon"]

            patch_group.create_dataset(
                "recon", data=recon, compression="gzip", chunks=(1, 128, 128)
            )
            patch_group.create_dataset(
                "decon", data=decon, compression="gzip", chunks=(1, 128, 128)
            )

            iso = extract_image(patch_group["isolated_stamps"][:], NPIX=128)
            psnr_vals = psnr_batch(iso, recon)

            patch_group.create_dataset("psnr", data=psnr_vals, compression="gzip")

            print(
                f"{patch_name} PSNR "
                f"min {np.nanmin(psnr_vals):.2f} | "
                f"median {np.nanmedian(psnr_vals):.2f} | "
                f"max {np.nanmax(psnr_vals):.2f}"
            )

            post_step(f"reconstructed facets {recon.shape}", start, data=data)

    # -----------------------------
    # --- Cleanup
    # -----------------------------
    if client_ctx:
        client_ctx.close()
        dask_ctx.close()

    data.close()
    post_step(f"finished {len(patches)} patches", start)
