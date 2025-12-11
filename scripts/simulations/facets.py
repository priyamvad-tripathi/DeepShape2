# %% Imports
import time

import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
from ska_sdp_datamodels.visibility import create_visibility_from_ms

from deepshape2.reconstruction import get_facets
from deepshape2.utils import extract_image, load_config, load_h5, post_step

# %%
if __name__ == "__main__":
    start = time.time()
    DATA_DIR = load_config()["DATA_DIR"]

    PATCHES = {
        "wide": {
            "h5_file": DATA_DIR + "wide_set.h5",
            "patch_key": "patch_051",
            "ms_file": DATA_DIR + "MS/vis_wide_set_patch_051.ms",
        },
        "deep": {
            "h5_file": DATA_DIR + "deep_set.h5",
            "patch_key": "patch_000",
            "ms_file": DATA_DIR + "MS/vis_deep_set_patch_000.ms",
        },
    }

    # -----------------------------
    # --- Create output HDF5
    # -----------------------------
    facets_h5_path = DATA_DIR + "facets.h5"
    data = load_h5(facets_h5_path, "a", delete_if_exists=True)

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
        for patch_name, info in PATCHES.items():
            post_step(f"Start processing {patch_name} patch", start, data=data)

            patch_group = data.create_group(patch_name)

            # --- Load HDF5 patch data
            patch_data = load_h5(info["h5_file"], mode="r")
            patch_df = pd.DataFrame.from_records(
                patch_data[info["patch_key"]]["patch_df"][()]
            )

            # --- Galaxy locations and flux
            mask = patch_df["flux_mask"].values
            patch_group.create_dataset(
                "flux",
                data=patch_df["flux"][mask].values,
                compression="gzip",
            )
            galaxy_locations = patch_df[["pix_x", "pix_y"]].to_numpy()[mask]

            # --- Stamp images
            isolated_stamps = patch_data[info["patch_key"]]["isolated_stamps"][:]
            isolated_stamps_crop = extract_image(isolated_stamps, NPIX=128)

            peak = np.max(isolated_stamps, axis=(1, 2))

            patch_group.create_dataset(
                "peak",
                data=peak,
                compression="gzip",
            )

            # Save cropped isolated stamps for hyperparameter tuning
            patch_group.create_dataset(
                "isolated_stamps",
                data=isolated_stamps_crop,
                compression="gzip",
            )

            del patch_df, mask, patch_data, peak, isolated_stamps, isolated_stamps_crop
            post_step(
                f"processing isolated stamps for {patch_name}",
                start,
                data=data,
                client=client,
            )

            # --- Load visibility
            vis = create_visibility_from_ms(info["ms_file"])[0]
            post_step(
                f"loading visibility MS for {patch_name}",
                start,
                data=data,
                client=client,
            )

            # --- Extract facets at different sizes
            for NPIX_facet in [128, 256]:
                dirty_all, psf_all = get_facets(
                    vis=vis,
                    galaxy_locations=galaxy_locations,
                    NPIX_facet=NPIX_facet,
                    client=client,
                )

                facet_group = patch_group.create_group(f"facets_{NPIX_facet}")
                facet_group.create_dataset(
                    "dirty",
                    data=dirty_all,
                    compression="gzip",
                    chunks=(1, NPIX_facet, NPIX_facet),
                )
                facet_group.create_dataset(
                    "psf",
                    data=psf_all,
                    compression="gzip",
                    chunks=(1, NPIX_facet, NPIX_facet),
                )

                post_step(
                    f"extracting facet for {patch_name}, shape {dirty_all.shape}",
                    start,
                    data=data,
                )

    # -----------------------------
    # --- Close HDF5
    # -----------------------------
    data.close()
    post_step("Saved all facet results", start)
