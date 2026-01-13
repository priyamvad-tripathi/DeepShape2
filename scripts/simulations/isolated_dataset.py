# %% Load Modules
import time
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from dask import compute, delayed
from dask.distributed import Client, LocalCluster

from deepshape2.simulation import make_dirty_image_and_psf, simulate_visibilities
from deepshape2.utils import extract_image, load_config, load_h5, post_step, save

warnings.filterwarnings("ignore", category=UserWarning)

# %% Defaults / Configuration
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]

hf_path = DATA_DIR + "deep_set.h5"

NPIX_SKY = cfg["NPIX_SKY"]
SCALE_RADIANS = cfg["SCALE_RADIANS"]


# %% Functions
def simulate_batch(inds):
    isolated_stamps_batch = isolated_stamps[inds]
    centres = galaxy_centres[inds]

    dirty_batch = np.zeros_like(isolated_stamps_batch)
    psf_batch = np.zeros_like(isolated_stamps_batch)

    for i, (stamp, centre) in enumerate(zip(isolated_stamps_batch, centres)):
        vis = simulate_visibilities(
            field=stamp,
            ra_pointing=centre.ra.deg,
            dec_pointing=centre.dec.deg,
            create_dirty=False,
        )

        dirty, psf = make_dirty_image_and_psf(vis)
        dirty_batch[i] = dirty
        psf_batch[i] = psf

    return dirty_batch, psf_batch


# %%

if __name__ == "__main__":
    start = time.time()
    hf = load_h5(hf_path, "r", delete_if_exists=False)
    patch_group = hf["patch_000"]
    isolated_stamps = extract_image(patch_group["isolated_stamps"][:])

    patch_df = patch_group["patch_df"][()]
    patch_ra, patch_dec = patch_group.attrs["centre"]

    mask = patch_df["flux_mask"]
    pix_x = patch_df["pix_x"][mask]
    pix_y = patch_df["pix_y"][mask]

    phase_centre = SkyCoord(
        ra=patch_ra * u.deg,
        dec=patch_dec * u.deg,
        frame="icrs",
        equinox="J2000",
    )

    dx = pix_x - NPIX_SKY // 2.0
    dy = pix_y - NPIX_SKY // 2.0

    offset_ra = -dx * SCALE_RADIANS * u.rad
    offset_dec = dy * SCALE_RADIANS * u.rad

    galaxy_centres = phase_centre.spherical_offsets_by(offset_ra, offset_dec)

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

        chunk_size = 100
        n = len(isolated_stamps)

        index_chunks = [
            np.arange(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)
        ]

        tasks = [delayed(simulate_batch)(inds) for inds in index_chunks]

        results = compute(*tasks)

        dirty_all = np.concatenate([r[0] for r in results], axis=0)
        psf_all = np.concatenate([r[1] for r in results], axis=0)

        post_step("Simulated dirty images and PSFs", start, data=hf)

        data = {"dirty": dirty_all, "psf": psf_all}
        save(data, DATA_DIR + "isolated_dirty_psf_deep_set.pkl")
