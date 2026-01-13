# %%
import time

# import torch
from dask.distributed import Client, LocalCluster

# from deepshape2.models import VAE
from deepshape2.reconstruction import get_facets  # , reconstruct_facets
from deepshape2.simulation import simulate_visibilities
from deepshape2.utils import (  # extract_image,; get_freest_gpu,; psnr_batch,
    load_config,
    load_h5,
    post_step,
)

# %%
if __name__ == "__main__":
    start = time.time()
    cfg = load_config()
    DATA_DIR = cfg["DATA_DIR"]
    hf_path = DATA_DIR + "wide_set.h5"

    # device = get_freest_gpu(set_device=True)

    # # Load pre-trained deblender
    # ckpt_path = cfg["MODEL_DIR"] + "vae_mha.pt"
    # deblender = VAE().to(device)
    # deblender.load_state_dict(
    #     torch.load(ckpt_path, map_location=device)["best_weights"]
    # )
    # deblender.eval()

    patches = [f"patch_{nl + 1:03d}" for nl in range(71, 100)]

    data = load_h5(hf_path, "a", delete_if_exists=False)

    # -----------------------------
    # --- Dask cluster
    # -----------------------------
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

        # -----------------------------
        # --- Process each patch
        # -----------------------------
        for patch_name in patches:
            patch_group = data[patch_name]
            post_step(f"opening {patch_name} patch", start, data=data)

            # --- Load HDF5 patch data
            patch_df = patch_group["patch_df"][()]
            patch_sky = patch_group["sky"][()]

            patch_ra, patch_dec = patch_group.attrs["centre"]

            # Simulate visibilities and create dirty image
            vis = simulate_visibilities(
                field=patch_sky,
                ra_pointing=patch_ra,
                dec_pointing=patch_dec,
                create_dirty=False,
                threads=60,
            )
            post_step(f"simulating visibilities for {patch_name}", start, data=data)

            # --- Extract facets at different galaxy locations
            mask = patch_df["flux_mask"]
            galaxy_locations = patch_df[["pix_x", "pix_y"]][mask]

            dirty_all, psf_all = get_facets(
                vis=vis,
                galaxy_locations=galaxy_locations,
                NPIX_facet=128,
                client=client,
            )
            # --- Save facets to HDF5
            patch_group.create_dataset(
                "dirty", data=dirty_all, compression="gzip", chunks=(1, 128, 128)
            )
            patch_group.create_dataset(
                "psf", data=psf_all, compression="gzip", chunks=(1, 128, 128)
            )

            post_step(
                f"extracting dirty/psf facets for {patch_name}, shape {dirty_all.shape}",
                start,
                data=data,
            )

            # -- Reconstruct facets and save results
            # result = reconstruct_facets(
            #     dirty_all, psf_all, device, num_workers=4, deblender=deblender
            # )
            # recon = result["recon"].copy()
            # decon = result["decon"].copy()

            # patch_group.create_dataset(
            #     "recon", data=recon, compression="gzip", chunks=(1, 128, 128)
            # )
            # patch_group.create_dataset(
            #     "decon", data=decon, compression="gzip", chunks=(1, 128, 128)
            # )
            # post_step(
            #     f"reconstructing facets for {patch_name}, shape {recon.shape}",
            #     start,
            #     data=data,
            # )

            # # Save metrics for data filtering
            # iso = extract_image(patch_group["isolated_stamps"][:], NPIX=128)
            # psnr = psnr_batch(iso, recon)
            # patch_group.create_dataset("psnr", data=psnr, compression="gzip")
            # post_step(
            #     f"calculating PSNR for {patch_name}, shape {psnr.shape}",
            #     start,
            #     data=data,
            # )

    # -----------------------------
    # --- Close HDF5
    # -----------------------------
    data.close()
    post_step(f"Saved facet results for {len(patches)} patches", start)
