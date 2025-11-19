import os
import pickle
from pathlib import Path

import h5py
import pandas as pd
from colorist import Color
from omegaconf import OmegaConf

_all__ = ["load", "save", "load_h5", "load_config", "get_tqdm"]


# Compute path relative to this file's directory
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # up one level from utils/
    "config",
    "default.yaml",
)


def load_config(path: str = None):
    """
    Load YAML config. Falls back to default if no path is provided.
    """
    cfg_path = path or DEFAULT_CONFIG_PATH
    cfg = OmegaConf.load(cfg_path)

    run_env = os.getenv("RUN_ENV", "local")

    if run_env == "genci":
        base = cfg["GENCI_DIR"]
        cfg["TQDM"] = False
    else:
        base = cfg["LOCAL_DIR"]
        cfg["TQDM"] = True

    cfg["DATA_DIR"] = base + "Data/"
    cfg["MODEL_DIR"] = base + "Model_weights/"

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


def load_h5(path, mode="r", delete_if_exists=False):
    if mode not in ["w", "r", "a"]:
        raise ValueError(f"Invalid mode '{mode}'. Allowed modes are 'w', 'r', 'a'.")

    if mode != "r" and (mode == "w" or delete_if_exists) and os.path.exists(path):
        os.remove(path)

    return h5py.File(path, mode)


def get_tqdm(path: str = None):
    cfg_path = path or DEFAULT_CONFIG_PATH
    cfg = OmegaConf.load(cfg_path)
    tqdm_kwargs = OmegaConf.to_container(cfg.tqdm, resolve=True)
    return tqdm_kwargs
