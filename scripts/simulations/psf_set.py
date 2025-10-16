# %% Import Modules and set constants
import logging
import time
import warnings

import astropy.units as u
import dask
import numpy as np
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from dask.distributed import Client, LocalCluster

from deepshape2.simulation.visibilities import (
    dirty_psf_from_visibilities,
    generate_visibilities,
)
from deepshape2.utils import load_config, load_h5, post_step

warnings.warn = lambda *args, **kwargs: None
logging.getLogger().addHandler(logging.NullHandler())

# %% Load configuration
cfg = load_config()
SCALE_RADIANS = cfg["SCALE_RADIANS"]

# Sky centre
RA0 = cfg["RA0"]
DEC0 = cfg["DEC0"]


DATA_DIR = cfg["DATA_DIR"]

# %% Get sky limits

origin = SkyCoord(ra=RA0 * u.deg, dec=DEC0 * u.deg, frame="icrs")
offset_frame = SkyOffsetFrame(origin=origin)

offset_coords = SkyCoord(
    lon=[-25, 25] * u.deg, lat=[-25, 25] * u.deg, frame=offset_frame
)

# Convert back to ICRS RA,Dec for limits
icrs_coords = offset_coords.icrs
RA_lim = icrs_coords.ra.deg
Dec_lim = icrs_coords.dec.deg


# %% Choose random sky positions
np.random.seed(2345)
TRAIN_OBJS = 100000

ra_choice = np.random.uniform(*RA_lim, TRAIN_OBJS)
dec_choice = np.random.uniform(*Dec_lim, TRAIN_OBJS)
sky_positions = np.column_stack((ra_choice, dec_choice))

# %% Run the deconvolution for all sources

if __name__ == "__main__":
    start = time.time()

    data = load_h5(DATA_DIR + "PSF_set.h5", mode="a", delete_if_exists=True)
    psf = data.create_dataset(
        name="psf",
        shape=(0, 128, 128),
        maxshape=(None, 128, 128),
        chunks=(128, 128, 128),
        compression="gzip",
    )

    chunk_size = 1000
    N_total = len(sky_positions)

    with (
        LocalCluster(
            n_workers=60,
            processes=True,
            threads_per_worker=1,
            # scheduler_port=8786,
            memory_limit=0,
        ) as cluster,
        Client(cluster) as client,
    ):
        print(client.dashboard_link)
        post_step("Dask cluster setup", start, client=client)

        for start_idx in range(0, N_total, chunk_size):
            end_idx = min(start_idx + chunk_size, N_total)
            batch_positions = sky_positions[start_idx:end_idx]

            # build delayed tasks for this chunk only
            lazy_results = []
            for pos in batch_positions:
                ra_gal, dec_gal = pos

                phasecentre = dask.delayed(SkyCoord)(
                    ra=ra_gal * u.deg,
                    dec=dec_gal * u.deg,
                    frame="icrs",
                    equinox="J2000",
                )

                vt0 = dask.delayed(generate_visibilities)(phasecentre=phasecentre)
                block = dask.delayed(dirty_psf_from_visibilities)(vt0)
                lazy_results.append(block)

            # compute only for this chunk
            results = dask.compute(*lazy_results)
            results = np.array(results)

            # extract PSFs
            psf_batch = np.stack([r[1] for r in results])

            # append to HDF5
            old_size = psf.shape[0]
            new_size = old_size + psf_batch.shape[0]
            psf.resize((new_size, 128, 128))
            psf[old_size:new_size] = psf_batch

            # free memory
            del lazy_results, results, psf_batch
            post_step(
                f"Processed chunk {start_idx}:{end_idx}. Total={new_size}",
                start,
                client=client,
                data=data,
            )

    post_step("PSF simulation", start, client=client, data=data)
