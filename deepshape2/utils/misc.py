import ctypes
import gc
import time
from functools import reduce

from colorist import Color

__all__ = [
    "time_string",
    "trim_memory",
    "post_step",
]


def time_string(t):
    return "%02d:%02d:%02d.%03d" % reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(round(t * 1000),), 1000, 60, 60]
    )


def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


def post_step(step_name: str, start_time: float, client=None, data=None):
    elapsed = time.time() - start_time
    print(
        f"Finished {step_name}. Time elapsed: {Color.GREEN}{time_string(elapsed)}{Color.OFF}"
    )

    # Dask worker memory cleanup
    if client is not None:
        try:
            client.run(lambda: gc.collect())  # clean up memory on all workers
            client.run(trim_memory)
        except Exception as e:
            print(f"Warning: Dask client cleanup failed: {e}")

    # local garbage collection
    gc.collect()

    # flush h5py data to disk
    if data is not None:
        try:
            data.flush()
        except Exception as e:
            print(f"Warning: data.flush() failed: {e}")
