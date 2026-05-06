# %% Imports
import os
import time

from colorist import Color
from dask.distributed import Client, LocalCluster
from ska_sdp_datamodels.visibility import create_visibility_from_ms

from deepshape2.reconstruction import get_facets
from deepshape2.utils import load_config, load_h5, post_step, calculate_dirty_peak

# %%
if __name__ == "__main__":
    start = time.time()
    DATA_DIR = load_config()["DATA_DIR"]

    WIDE_PATCHES = [f"patch_{i:03d}" for i in range(51, 101)]
    DEEP_PATCHES = ["patch_000"]

    PATCH_CONFIGS = [
        {
            "set_name": "deep",
            "patch_key": p,
            "h5_file": DATA_DIR + "deep_set_new.h5",
            "ms_file": DATA_DIR + f"MS/vis_deep_set_{p}.ms",
            "out_key": p,
        }
        for p in DEEP_PATCHES
    ] + [
        {
            "set_name": "wide",
            "patch_key": p,
            "h5_file": DATA_DIR + "wide_set_new.h5",
            "ms_file": DATA_DIR + f"MS/vis_wide_set_{p}.ms",
            "out_key": p,
        }
        for p in WIDE_PATCHES
    ]

    NPIX_facet = 128

    # -----------------------------
    # --- Open wide + deep HDF5 sources and output h5 files
    # -----------------------------
    wide_h5 = load_h5(DATA_DIR + "wide_set_new.h5", "a")
    deep_h5 = load_h5(DATA_DIR + "deep_set_new.h5", "a")

    def get_handles(set_name):
        if set_name == "wide":
            return wide_h5
        return deep_h5

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
        print(f"Dask dashboard: {client.dashboard_link}")

        total = len(PATCH_CONFIGS)
        for idx, cfg in enumerate(PATCH_CONFIGS, 1):
            set_name = cfg["set_name"]
            patch_key = cfg["patch_key"]
            ms_file = cfg["ms_file"]
            out_key = cfg["out_key"]
            src_h5 = get_handles(set_name)

            prefix = f"[{idx}/{total}] [{set_name}:{patch_key}]"

            # --- Check MS exists
            if not os.path.exists(ms_file):
                print(f"{Color.YELLOW}{prefix} MS file not found, skipping: {ms_file}{Color.OFF}")
                continue

            # --- Check if already computed
            if (
                out_key in src_h5
                and "dirty" in src_h5[out_key]
                and "psf" in src_h5[out_key]
            ):
                print(f"{Color.YELLOW}{prefix} dirty+psf already exist, skipping.{Color.OFF}")
                continue

            post_step(f"{prefix} start", start, data=src_h5)

            # --- Load HDF5 patch data
            patch_df = src_h5[patch_key]["patch_df"][()]
            mask = patch_df["flux_mask"]
            patch_df = patch_df[mask]
            galaxy_locations = patch_df[["pix_x", "pix_y"]]
            del patch_df, mask

            # --- Load visibility
            vis = create_visibility_from_ms(ms_file)[0]
            post_step(f"{prefix} MS loaded", start, data=src_h5, client=client)

            # --- Compute facets
            dirty_all, psf_all = get_facets(
                vis=vis,
                galaxy_locations=galaxy_locations,
                NPIX_facet=NPIX_facet,
                client=client,
            )

            # --- Save into source h5
            patch_group = src_h5.require_group(out_key)
            patch_group.create_dataset(
                "dirty",
                data=dirty_all,
                chunks=(1, NPIX_facet, NPIX_facet),
            )
            patch_group.create_dataset(
                "psf",
                data=psf_all,
                chunks=(1, NPIX_facet, NPIX_facet),
            )

            post_step(f"{prefix} dirty+psf calculation", start, data=src_h5)

            iso_stamps = src_h5[patch_key]["isolated_stamps"][:]
            peaks = calculate_dirty_peak(iso_stamps, psf_all)
            patch_group.create_dataset("peaks", data=peaks)
            post_step(f"{prefix} dirty peak calculation", start, data=src_h5)

    # -----------------------------
    # --- Close HDF5 files
    # -----------------------------
    for h5 in [wide_h5, deep_h5]:
        try:
            h5.close()
        except Exception:
            pass

    post_step("full script", start)
