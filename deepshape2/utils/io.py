import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from colorist import Color
from omegaconf import OmegaConf

_all__ = [
    "load",
    "save",
    "load_h5",
    "load_config",
    "get_tqdm",
    "print_h5",
]


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


def load_h5(path, mode="r", delete_if_exists=False, pprint=False):
    if mode not in ["w", "r", "a"]:
        raise ValueError(f"Invalid mode '{mode}'. Allowed modes are 'w', 'r', 'a'.")

    if mode != "r" and (mode == "w" or delete_if_exists) and os.path.exists(path):
        os.remove(path)

    hf = h5py.File(path, mode)

    if pprint:
        print_h5(hf)

    return hf


def get_tqdm(path: str = None):
    cfg_path = path or DEFAULT_CONFIG_PATH
    cfg = OmegaConf.load(cfg_path)
    tqdm_kwargs = OmegaConf.to_container(cfg.tqdm, resolve=True)
    return tqdm_kwargs


def print_h5(obj, indent=0, show_attrs=True, max_attr_len=80, _is_root=True):
    """
    Pretty-print the structure of an h5py File, Group, or Dataset.

    Args:
        obj:          h5py.File, h5py.Group, or h5py.Dataset
        indent:       current indentation level (used internally)
        show_attrs:   whether to print attributes
        max_attr_len: truncate attribute value strings longer than this
    """
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    RESET = "\033[0m"

    PIPE = "│"
    TEE = "├── "
    ELBOW = "└── "
    BLANK = "    "
    ATTR_PFX = "    @"

    def _fmt_attr(val):
        """Format an attribute value for display."""
        if isinstance(val, (np.ndarray, list)):
            s = np.array2string(np.asarray(val), threshold=6, edgeitems=2)
        else:
            s = str(val)
        return s if len(s) <= max_attr_len else s[:max_attr_len] + "…"

    def _fmt_dtype(dtype):
        """Return a short, readable dtype string."""
        if np.issubdtype(dtype, np.floating):
            return f"f{dtype.itemsize * 8}"
        if np.issubdtype(dtype, np.integer):
            kind = "u" if np.issubdtype(dtype, np.unsignedinteger) else "i"
            return f"{kind}{dtype.itemsize * 8}"
        if dtype.kind == "S":
            return f"bytes{dtype.itemsize}"
        if dtype.kind == "U":
            return f"str{dtype.itemsize // 4}"
        return str(dtype)

    def _render(node, prefix, is_last):
        connector = ELBOW if is_last else TEE
        child_prefix = prefix + (BLANK if is_last else PIPE + "   ")

        if isinstance(node, h5py.Dataset):
            shape_str = "×".join(str(d) for d in node.shape) or "scalar"
            dtype_str = _fmt_dtype(node.dtype)
            size_str = ""
            nbytes = node.nbytes
            if nbytes >= 1 << 30:
                size_str = f"  {GRAY}{nbytes / (1 << 30):.1f} GB{RESET}"
            elif nbytes >= 1 << 20:
                size_str = f"  {GRAY}{nbytes / (1 << 20):.1f} MB{RESET}"
            elif nbytes >= 1 << 10:
                size_str = f"  {GRAY}{nbytes / (1 << 10):.1f} KB{RESET}"

            print(
                f"{prefix}{connector}{GREEN}{node.name.split('/')[-1]}{RESET}"
                f"  {BLUE}[{shape_str}  {dtype_str}]{RESET}{size_str}"
            )

        else:  # Group
            label = node.name.split("/")[-1] or "/"
            n_children = len(node)
            print(
                f"{prefix}{connector}{YELLOW}{label}{RESET}"
                f"  {GRAY}({n_children} item{'s' if n_children != 1 else ''}){RESET}"
            )

        if show_attrs and node.attrs:
            attr_items = list(node.attrs.items())
            for i, (k, v) in enumerate(attr_items):
                # is_last_attr = (i == len(attr_items) - 1) and (
                #     isinstance(node, h5py.Dataset) or not list(node.keys())
                # )
                print(f"{child_prefix}{ATTR_PFX}{GRAY}{k}{RESET} = {_fmt_attr(v)}")

        if isinstance(node, h5py.Group):
            children = list(node.keys())
            for i, key in enumerate(children):
                _render(node[key], child_prefix, is_last=(i == len(children) - 1))

    # ── entry point ──────────────────────────────────────────────────────────
    if _is_root:
        name = getattr(obj, "filename", obj.name)
        print(f"{YELLOW}{name}{RESET}")
        if show_attrs and obj.attrs:
            for k, v in obj.attrs.items():
                print(f"  {ATTR_PFX}{GRAY}{k}{RESET} = {_fmt_attr(v)}")
        if isinstance(obj, h5py.Group):
            children = list(obj.keys())
            for i, key in enumerate(children):
                _render(obj[key], prefix="", is_last=(i == len(children) - 1))
    else:
        _render(obj, prefix="", is_last=True)
