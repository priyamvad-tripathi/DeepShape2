# %% Import Libraries
import bisect
import os
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from deepshape2.utils import set_seed

# Seeds to ensure reproducibility
set_seed()


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


# %% Shape Measurement Loader
class ImageDataset(Dataset):
    """
    PyTorch Dataset for loading images and ellipticities from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    x_key : list of str
        Keys for the input images (e.g., ['dirty_image', 'PSF']).
    y_key : list of str
        Key(s) for the target (e.g., ellipticity).
    peak : float, optional
        If provided, applies a cutoff using the 'Peak' dataset.
    transform : callable, optional
        Transform function applied to input images.
    scale : bool, default=True
        Whether to normalize input images to [0, 1].
    """

    def __init__(
        self,
        path: str,
        x_key: List[str],
        y_key: Optional[List[str]] = None,
        transform: Optional[callable] = None,
        scale: bool = True,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        self.hf = h5py.File(path, "r")
        self.x_key = x_key
        self.y_key = y_key or []
        self.transform = transform
        self.scale = scale

    def __len__(self) -> int:
        return len(self.hf[self.x_key[0]])

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        img_min = img.min()
        img_range = np.ptp(img)
        return (img - img_min) / img_range if img_range > 0 else img

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Load all input channels and normalize if needed
        x = [self.hf[k][idx] for k in self.x_key]
        if self.scale:
            x = [self._normalize(im) for im in x]

        x = np.stack(x, axis=0)  # shape: (C, H, W)
        x = torch.from_numpy(x).float()

        if self.transform is not None:
            x = self.transform(x)

        if self.y_key:
            y = torch.from_numpy(self.hf[self.y_key[0]][idx]).float()
            return x, y
        return x

    def close(self):
        """Manually close the HDF5 file."""
        if self.hf:
            self.hf.close()


def dataloader(
    path: str,
    x_key: List[str],
    y_key: Optional[List[str]],
    split: Union[List[float], int],
    batch_size: Union[List[int], int],
    **kwargs,
):
    """
    Create PyTorch DataLoader(s) from an HDF5 dataset.

    Parameters
    ----------
    path : str
        Path to HDF5 dataset.
    x_key : list of str
        Keys for the input images.
    y_key : list of str
        Keys for the target values.
    split : list of floats or int
        - If list: [train_split, val_split] fractions (should sum to 1).
        - If int: dataset size (no split).
    batch_size : list of int or int
        Batch size(s) for training and validation sets.
    **kwargs : dict
        Additional arguments for `ImageDataset`.

    Returns
    -------
    DataLoader or (DataLoader, DataLoader)
        Returns one or two DataLoaders depending on split type.
    """

    dataset = ImageDataset(path=path, x_key=x_key, y_key=y_key, **kwargs)

    if isinstance(split, (list, tuple)):
        if not np.isclose(sum(split), 1.0):
            raise ValueError("Split ratios must sum to 1.")
        if not isinstance(batch_size, (list, tuple)) or len(batch_size) != len(split):
            raise ValueError("`batch_size` must match number of splits.")

        n_total = len(dataset)
        lengths = [int(n_total * s) for s in split]
        # Ensure rounding errors don't lose samples
        lengths[-1] = n_total - sum(lengths[:-1])

        train_ds, val_ds = torch.utils.data.random_split(dataset, lengths)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size[0],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size[1],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader, val_loader

    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        return loader
