import os
import pickle
from pathlib import Path

import h5py
from colorist import Color
import pandas as pd
from omegaconf import OmegaConf

_all__ = ["load", "save", "load_h5", "load_config"]


# Default config location (relative to project root)
DEFAULT_CONFIG_PATH = "/home/tripathi/DeepShape2/deepshape2/config/default.yaml"


def load_config(path: str = None):
    """
    Load YAML config. Falls back to default if no path is provided.
    """
    cfg_path = path or DEFAULT_CONFIG_PATH
    cfg = OmegaConf.load(cfg_path)
    return cfg


def load(path):
    "Convenient function to load pickled data"
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return data


def save(data, path):
    "Convenient function to dump data in pickled format"
    parent_dir = Path(path).parent

    try:
        parent_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        1
    else:
        print(f"New folder created {Color.GREEN}{parent_dir}{Color.OFF}")

    if isinstance(data, pd.DataFrame):
        data.to_pickle(path)
    else:
        with open(path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dumped data at {Color.GREEN}{path}{Color.OFF}")


def load_h5(path, mode="r"):
    if mode not in ["w", "r", "a"]:
        raise ValueError(f"Invalid mode '{mode}'. Allowed modes are 'w', 'r', 'a'.")

    if mode == "w":
        if os.path.exists(path):
            os.remove(path)

    return h5py.File(path, mode)
    return h5py.File(path, mode)
    return h5py.File(path, mode)
    return h5py.File(path, mode)
