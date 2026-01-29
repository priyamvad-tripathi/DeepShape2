import argparse
import logging
import time
import warnings

import numpy as np
import torch
from dask.distributed import Client, LocalCluster
from numpy.fft import ifftshift, irfftn, rfftn

from deepshape2.models import VAE
from deepshape2.reconstruction import get_facets, reconstruct_facets
from deepshape2.simulation import simulate_visibilities
from deepshape2.utils import (
    blendedness,
    get_freest_gpu,
    load_config,
    load_h5,
    post_step,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
# --------------------------------------------------
# Config
# --------------------------------------------------
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]

TRAIN_PATH = DATA_DIR + "trainingset2.h5"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def get_or_create(hf, name, shape, maxshape, dtype, chunks):
    if name in hf:
        return hf[name]
    return hf.create_dataset(
        name,
        shape=shape,
        maxshape=maxshape,
        dtype=dtype,
        compression="gzip",
        chunks=chunks,
    )


def compute_blend_and_peak(iso, blend, psf):
    blend_val = blendedness(iso[None], blend[None])[0]
    img_f = rfftn(iso)
    psf_f = rfftn(ifftshift(psf))
    dirty = irfftn(img_f * psf_f, s=iso.shape)
    return np.float32(blend_val), np.float32(dirty.max())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        choices=["facets", "reconstruct"],
        required=True,
    )
    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    start = time.time()

    patches = [f"patch_{nl + 1:03d}" for nl in range(71, 100)]
    hf_inp = load_h5(DATA_DIR + "wide_set.h5", "r", delete_if_exists=False)
    hf_out = load_h5(TRAIN_PATH, "a", delete_if_exists=False)

    # --------------------------------------------------
    # Create / load datasets
    # --------------------------------------------------
    dirty_ds = get_or_create(
        hf_out, "dirty", (0, 128, 128), (None, 128, 128), np.float32, (1000, 128, 128)
    )
    psf_ds = get_or_create(
        hf_out, "psf", (0, 128, 128), (None, 128, 128), np.float32, (1000, 128, 128)
    )
    shape_ds = get_or_create(hf_out, "shapes", (0, 2), (None, 2), np.float32, (1000, 2))
    blend_ds = get_or_create(hf_out, "blendedness", (0,), (None,), np.float32, (1000,))
    peak_ds = get_or_create(hf_out, "peaks", (0,), (None,), np.float32, (1000,))
    recon_ds = get_or_create(
        hf_out, "recon", (0, 128, 128), (None, 128, 128), np.float32, (1000, 128, 128)
    )

    # --------------------------------------------------
    # FACETS MODE
    # --------------------------------------------------
    if args.mode == "facets":
        dask_ctx = LocalCluster(
            n_workers=64,
            processes=True,
            threads_per_worker=1,
            memory_limit=0,
        )
        client = Client(dask_ctx)
        print("Dask dashboard:", client.dashboard_link)

        total = dirty_ds.shape[0]

        for patch_name in patches:
            patch_group = hf_inp[patch_name]
            post_step(f"processing {patch_name}", start)

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
            post_step("simulating visibilities", start)

            mask = patch_df["flux_mask"]
            galaxy_locations = patch_df[["pix_x", "pix_y"]][mask]

            dirty_all, psf_all = get_facets(
                vis=vis,
                galaxy_locations=galaxy_locations,
                NPIX_facet=128,
                client=client,
            )

            # isolated = extract_image(patch_group["isolated_stamps"][:], NPIX=128)
            # blended = extract_image(patch_group["blended_stamps"][:], NPIX=128)

            # shapes = np.stack(
            #     [patch_df["e1"][mask], patch_df["e2"][mask]],
            #     axis=1,
            # ).astype(np.float32)

            # futures = [
            #     client.submit(
            #         compute_blend_and_peak, isolated[i], blended[i], psf_all[i]
            #     )
            #     for i in range(len(isolated))
            # ]
            # results = client.gather(futures)
            # blend_vals, peak_vals = map(np.array, zip(*results))

            n_new = dirty_all.shape[0]
            new_total = total + n_new

            dirty_ds.resize((new_total, 128, 128))
            psf_ds.resize((new_total, 128, 128))
            shape_ds.resize((new_total, 2))
            blend_ds.resize((new_total,))
            peak_ds.resize((new_total,))

            dirty_ds[total:new_total] = dirty_all.astype(np.float32)
            psf_ds[total:new_total] = psf_all.astype(np.float32)
            # shape_ds[total:new_total] = shapes
            # blend_ds[total:new_total] = blend_vals
            # peak_ds[total:new_total] = peak_vals

            total = new_total
            hf_out.flush()

            post_step(f"saved {n_new} facets", start)

        client.close()
        dask_ctx.close()

    # --------------------------------------------------
    # RECONSTRUCT MODE
    # --------------------------------------------------
    elif args.mode == "reconstruct":
        device = get_freest_gpu(set_device=True)

        ckpt_path = cfg["MODEL_DIR"] + "vae_mha.pt"
        deblender = VAE().to(device)
        deblender.load_state_dict(
            torch.load(ckpt_path, map_location=device)["best_weights"]
        )
        deblender.eval()

        dirty_all = dirty_ds[:]
        psf_all = psf_ds[:]

        post_step(f"reconstructing {dirty_all.shape[0]} facets", start)

        result = reconstruct_facets(
            dirty_all,
            psf_all,
            device=device,
            num_workers=4,
            deblender=deblender,
        )

        recon = result["recon"].astype(np.float32)

        n_new = recon.shape[0]
        recon_ds.resize((n_new, 128, 128))
        recon_ds[:] = recon

        hf_out.flush()
        post_step("finished reconstruction", start)

    hf_inp.close()
    hf_out.close()
    post_step("all done", start)
