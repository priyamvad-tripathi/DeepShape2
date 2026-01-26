# %% Imports
import time

import dask
import numpy as np
from dask.distributed import Client, LocalCluster
from numpy.fft import ifftshift, irfftn, rfftn
from ska_sdp_datamodels.visibility import create_visibility_from_ms

from deepshape2.reconstruction import get_facets
from deepshape2.utils import blendedness, extract_image, load_config, load_h5, post_step


def compute_peak_chunk(isolated_chunk, psf_chunk):
    img_f = rfftn(isolated_chunk, axes=(1, 2))
    psf_f = rfftn(ifftshift(psf_chunk, axes=(1, 2)), axes=(1, 2))
    dirty = irfftn(
        img_f * psf_f,
        s=isolated_chunk.shape[1:],
        axes=(1, 2),
    )
    return dirty.max(axis=(1, 2)).astype(np.float32)


def compute_peaks_dask(isolated_stamps, psf_all, chunk_size=512):
    n = isolated_stamps.shape[0]
    delayed_chunks = []

    for i in range(0, n, chunk_size):
        isolated_chunk = isolated_stamps[i : i + chunk_size]
        psf_chunk = psf_all[i : i + chunk_size]

        delayed_chunks.append(
            dask.delayed(compute_peak_chunk)(isolated_chunk, psf_chunk)
        )

    # This returns a list of numpy arrays, one per chunk
    peak_chunks = dask.compute(*delayed_chunks)

    # Concatenate into final shape
    return np.concatenate(peak_chunks, axis=0)


# %%
if __name__ == "__main__":
    start = time.time()
    DATA_DIR = load_config()["DATA_DIR"]

    patches = [f"patch_{nl + 1:03d}" for nl in range(71, 100)]
    catalog_path = DATA_DIR + "wide_set.h5"

    # -----------------------------
    # --- Create output HDF5
    # -----------------------------
    hf_out = load_h5(DATA_DIR + "training_set.h5", "a", delete_if_exists=True)

    # --- running index
    idx = 0

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
        with load_h5(catalog_path, "r") as data:
            for patch in patches:
                post_step(f"opening {patch} patch", start, data=data)

                patch_group = data[patch]

                # --- Load HDF5 patch data
                patch_data = patch_group["patch_df"][()]

                # --- Galaxy locations and flux
                mask = patch_data["flux_mask"]
                galaxy_locations = patch_data[["pix_x", "pix_y"]][mask]

                # -- Get galaxy shapes
                shapes = np.stack(
                    [patch_data["e1"][mask], patch_data["e2"][mask]],
                    axis=1,
                ).astype(np.float32)

                # --- Stamp images
                isolated_stamps = extract_image(
                    patch_group["isolated_stamps"][:], NPIX=128
                )
                blended_stamps = extract_image(
                    patch_group["blended_stamps"][:], NPIX=128
                )

                post_step(
                    f"processing stamps for {patch}",
                    start,
                    data=data,
                    client=client,
                )

                # --- Load visibility
                vis_name = DATA_DIR + f"MS/vis_wide_set_{patch}.ms"
                vis = create_visibility_from_ms(vis_name)[0]
                post_step(
                    f"loading visibility MS for {patch}",
                    start,
                    data=data,
                    client=client,
                )

                # --- Extract facets
                dirty_all, psf_all = get_facets(
                    vis=vis,
                    galaxy_locations=galaxy_locations,
                    NPIX_facet=128,
                    client=client,
                )

                n = dirty_all.shape[0]

                post_step(
                    f"extracting facet for {patch}, shape {dirty_all.shape}",
                    start,
                    data=hf_out,
                    client=client,
                )

                # --- Create datasets once
                if "dirty" not in hf_out:
                    hf_out.create_dataset(
                        "dirty",
                        shape=(0, 128, 128),
                        maxshape=(None, 128, 128),
                        dtype=dirty_all.dtype,
                        chunks=(1, 128, 128),
                    )
                    hf_out.create_dataset(
                        "psf",
                        shape=(0, 128, 128),
                        maxshape=(None, 128, 128),
                        dtype=psf_all.dtype,
                        chunks=(1, 128, 128),
                    )
                    hf_out.create_dataset(
                        "shapes",
                        shape=(0, 2),
                        maxshape=(None, 2),
                        dtype=np.float32,
                        chunks=(1, 2),
                    )
                    hf_out.create_dataset(
                        "peak",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.float32,
                        chunks=(1,),
                    )
                    hf_out.create_dataset(
                        "blendedness",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.float32,
                        chunks=(1,),
                    )

                # --- Resize and append dirty/psf
                hf_out["dirty"].resize(idx + n, axis=0)
                hf_out["psf"].resize(idx + n, axis=0)
                hf_out["dirty"][idx : idx + n] = dirty_all.copy()
                hf_out["psf"][idx : idx + n] = psf_all.copy()
                hf_out["shapes"].resize(idx + n, axis=0)
                hf_out["shapes"][idx : idx + n] = shapes.astype(np.float32)

                post_step(
                    "saving dirty/psf/shapes",
                    start,
                    data=hf_out,
                    client=client,
                )

                # --- Calculate peaks
                peak_vals = compute_peaks_dask(isolated_stamps, psf_all, chunk_size=512)

                hf_out["peak"].resize(idx + n, axis=0)
                hf_out["peak"][idx : idx + n] = peak_vals

                post_step(
                    f"calculating peaks for {patch}, above thresh {(peak_vals > 2.13e-07).sum()}",
                    start,
                    data=hf_out,
                    client=client,
                )

                # --- Calculate blendedness
                blend = blendedness(isolated_stamps, blended_stamps).astype(np.float32)

                hf_out["blendedness"].resize(idx + n, axis=0)
                hf_out["blendedness"][idx : idx + n] = blend

                post_step(
                    f"calculating blendedness for {patch}, above thresh ",
                    start,
                    data=hf_out,
                    client=client,
                )

                idx += n

    # -----------------------------
    # --- Close HDF5
    # -----------------------------
    data.close()
    hf_out.close()
    post_step("Saved all facet results", start)
