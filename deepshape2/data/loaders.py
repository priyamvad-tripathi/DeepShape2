# %% Import Libraries
import bisect
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

# Seeds to ensure reproducibility
torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2024)


# %% Deblending Loader


class CenterCrop:
    """Crop the central (h, w) region from a 2D or 3D array."""

    def __init__(self, crop_size):
        self.crop_size = (
            crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        )

    def __call__(self, arr):
        if isinstance(arr, torch.Tensor):
            h, w = arr.shape[-2], arr.shape[-1]
        else:
            h, w = arr.shape[-2], arr.shape[-1]

        ch, cw = self.crop_size
        top = (h - ch) // 2
        left = (w - cw) // 2
        return arr[..., top : top + ch, left : left + cw]


crop_128 = CenterCrop(128)


class BlendDataset(Dataset):
    def __init__(
        self,
        path: str,
        x_key: str,
        y_key: str,
        groups=None,
        scale_fac: float = 1e7,
        min_max: bool = False,
        tanh: bool = False,
        arcsin: bool = True,
        transform=crop_128,
        min_flux: float = None,
    ):
        """
        Multi-group HDF5 dataset for paired (x, y) samples.
        Optimized for PyTorch multi-worker loading, with optional flux filtering.

        Args:
            path (str): Path to HDF5 file.
            x_key (str): Dataset key for inputs.
            y_key (str): Dataset key for targets.
            groups (list[str], optional): Restrict to specific groups.
            scale_fac (float): Multiplicative scaling factor.
            min_max (bool): Apply min-max normalization.
            tanh (bool): Apply tanh normalization.
            arcsin (bool): Apply arcsinh normalization.
            transform (callable): Optional transform to apply to each image.
            min_flux (float): Keep only samples with stamp_flux > min_flux.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        self.path = path
        self.x_key = x_key
        self.y_key = y_key
        self.scale_fac = scale_fac
        self.min_max = min_max
        self.tanh = tanh
        self.arcsin = arcsin
        self.transform = transform
        self.min_flux = min_flux

        # ---------------- Initialize metadata ---------------- #
        self.groups, self.group_sizes, self.valid_indices = [], [], []

        with h5py.File(self.path, "r") as hf:
            available_groups = list(hf.keys())

            # Validate requested groups
            if groups is not None:
                invalid = [g for g in groups if g not in available_groups]
                if invalid:
                    raise ValueError(f"Invalid groups: {invalid}")
                self.groups = groups
            else:
                self.groups = available_groups

            # Compute valid indices per group
            for g in self.groups:
                group = hf[g]
                n = len(group[self.x_key])

                if self.min_flux is not None:
                    if "stamp_flux" not in group:
                        raise KeyError(f"Group '{g}' missing 'stamp_flux' dataset")
                    flux = np.array(group["stamp_flux"])
                    valid_idx = np.where(flux > self.min_flux)[0]
                else:
                    valid_idx = np.arange(n)

                self.valid_indices.append(valid_idx)
                self.group_sizes.append(len(valid_idx))

        # Build cumulative index map for global indexing
        self.cumulative_sizes = np.cumsum(self.group_sizes).tolist()

        # Worker-local file handles (lazy initialization)
        self._hf = None

    # ---------------- Scaling ---------------- #
    def _scale(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32) * self.scale_fac
        if self.arcsin:
            arr = np.arcsinh(arr)
        if self.min_max:
            a_min, a_max = np.min(arr), np.max(arr)
            arr = (arr - a_min) / (a_max - a_min + 1e-8)
        elif self.tanh:
            arr = np.tanh(arr)
        return arr

    # ---------------- Worker-safe handle ---------------- #
    def _get_h5_handle(self):
        """Each worker lazily opens its own HDF5 file handle."""
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process case
            if self._hf is None:
                self._hf = h5py.File(self.path, "r")
        else:
            # Multi-worker: separate file handle per worker
            worker_id = worker_info.id
            if not hasattr(self, "_worker_h5"):
                self._worker_h5 = {}
            if worker_id not in self._worker_h5:
                self._worker_h5[worker_id] = h5py.File(self.path, "r")
            self._hf = self._worker_h5[worker_id]
        return self._hf

    # ---------------- Index utilities ---------------- #
    def __len__(self):
        return self.cumulative_sizes[-1]

    def _locate_index(self, global_idx: int):
        """Convert a global index to (group_name, local_idx)."""
        if global_idx < 0 or global_idx >= self.cumulative_sizes[-1]:
            raise IndexError(f"Index {global_idx} out of range")

        group_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        prev_cum = 0 if group_idx == 0 else self.cumulative_sizes[group_idx - 1]
        local_pos = global_idx - prev_cum
        local_idx = self.valid_indices[group_idx][local_pos]
        return self.groups[group_idx], local_idx

    # ---------------- Get item ---------------- #
    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.item()

        hf = self._get_h5_handle()
        group_name, local_idx = self._locate_index(idx)
        group = hf[group_name]

        x = np.array(group[self.x_key][local_idx])
        y = np.array(group[self.y_key][local_idx])

        x = self._scale(x)
        y = self._scale(y)

        if x.ndim != 3:
            x = x[np.newaxis, :, :]
        if y.ndim != 3:
            y = y[np.newaxis, :, :]

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    # ---------------- Cleanup ---------------- #
    def __del__(self):
        """Ensure HDF5 handles are closed when dataset is destroyed."""
        if hasattr(self, "_hf") and self._hf is not None:
            try:
                self._hf.close()
            except Exception:
                pass
        if hasattr(self, "_worker_h5"):
            for f in self._worker_h5.values():
                try:
                    f.close()
                except Exception:
                    pass
