import ctypes
import gc
import time

from colorist import Color
from tqdm import tqdm

__all__ = [
    "time_string",
    "trim_memory",
    "post_step",
    "get_progress_bar",
]


def time_string(t):
    total_seconds = int(t)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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


class DummyTqdm:
    """A no-op tqdm replacement for non-interactive environments."""

    def __init__(self, total=None, **kwargs):
        self.total = total
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # Dummy methods that do nothing
    def update(self, n=1):
        self.n += n
        return self

    def set_postfix(self, postfix=None, **kwargs):
        return self

    def set_description(self, desc=None, **kwargs):
        return self


def get_progress_bar(enabled, total=None, **kwargs):
    """
    Return tqdm if enabled, else a no-op context manager.
    """
    if enabled:
        return tqdm(total=total, **kwargs)
    else:
        return DummyTqdm(total=total)
