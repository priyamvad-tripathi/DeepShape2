# %% Load Modules
import argparse
import time
import warnings

import dask.array as da
import numpy as np
from colorist import Color
from dask.distributed import Client, LocalCluster

from deepshape2.simulation import (
    filter_patch_by_flux,
    filter_patch_by_size,
    generate_patch_locations,
    random_patch,
    simulate_wide_field,
)
from deepshape2.utils import centers_to_limits, load_config, load_h5, post_step

warnings.filterwarnings("ignore", category=UserWarning)


# %% Set up argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Sky simulation arguments")

    parser.add_argument(
        "-w",
        "--n-workers",
        type=int,
        default=50,
        help="Number of Dask workers to use (default: 50)",
    )

    parser.add_argument(
        "-nf",
        "--n-patches",
        type=int,
        default=50,
        help="Number of simulated patches (each 1 deg2 wide) (default: 50)",
    )

    return parser.parse_args()


# %% Default Parameters

cfg = load_config()

DATA_DIR = cfg["DATA_DIR"]
NPIX_STAMP = 256

"""
Other default parameters are set in simulation/wide_field.py
"""

# %% Test the wide-field simulation
if __name__ == "__main__":
    args = parse_args()

    NUM_DASK_WORKERS = args.n_workers
    NUM_PATCHES = args.n_patches
    MIN_FLUX = 10e-6  # Min flux in Jy for extracting stamps

    data = load_h5(DATA_DIR + "sky.h5", mode="a", delete_if_exists=True)
    data.attrs["min_flux_for_stamps"] = MIN_FLUX

    start = time.time()
    with (
        LocalCluster(
            n_workers=NUM_DASK_WORKERS,
            processes=True,
            threads_per_worker=1,
            scheduler_port=8786,
            memory_limit=0,
        ) as cluster,
        Client(cluster) as client,
    ):
        print(client.dashboard_link)

        locations = generate_patch_locations()[:NUM_PATCHES]

        for nl, location in enumerate(locations):
            print(
                f"{Color.GREEN}Simulating patch {nl + 1}/{NUM_PATCHES} at location ({location[0]:.3f}, {location[1]:.3f}){Color.OFF}"
            )

            group = data.create_group(f"patch_{nl + 1:03d}")

            # Extract galaxies at patch location
            patch, centre = random_patch(location)
            patch = filter_patch_by_size(filter_patch_by_flux(patch))
            print(f"Number of galaxies in patch: {len(patch)}")

            # Simulate wide-field image of the patch
            sky_array, patch_out, isolated_stamps = simulate_wide_field(
                patch, location, min_flux=MIN_FLUX
            )

            locs_pix = patch_out[["pix_x", "pix_y"]].values
            mask = patch_out["flux_mask"].values
            centers = locs_pix[mask]

            lims = centers_to_limits(centers, stamp_size=NPIX_STAMP)

            print(f"Number of bright galaxies (flux>=10uJy): {len(centers)}")

            # Save results
            patch_rec = patch_out.to_records(index=False)
            group.create_dataset("patch_df", data=patch_rec, compression="gzip")

            group.create_dataset(
                "sky", data=sky_array, compression="gzip", chunks=(1024, 1024)
            )
            group.attrs["centre"] = centre

            group.create_dataset(
                "isolated_stamps",
                data=isolated_stamps,
                compression="gzip",
                chunks=(1, 256, 256),
            )

            post_step("field image simulation", start, client, data)

            darr = da.from_array(sky_array, chunks=(5000, 5000))

            def crop_dask(limits):
                x0, x1, y0, y1 = limits
                return darr[y0:y1, x0:x1]

            del (
                patch,
                patch_out,
                patch_rec,
                isolated_stamps,
                locs_pix,
                centers,
                sky_array,
            )

            crops = [crop_dask(lim) for lim in lims]

            # Trigger computation in parallel
            blended_stamps = da.compute(*crops)
            blended_stamps = np.stack(blended_stamps, axis=0)

            group.create_dataset(
                "blended_stamps",
                data=blended_stamps,
                compression="gzip",
                chunks=(1, 256, 256),
            )

            post_step("blended stamp extraction", start, client, data)

    data.close()
