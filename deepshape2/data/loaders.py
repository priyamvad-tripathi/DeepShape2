# %% Import Libraries
import bisect
import os

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset, Sampler, get_worker_info

from deepshape2.utils import load_config, set_seed

# Seeds to ensure reproducibility
set_seed()
cfg = load_config()
SCALE_FACTOR = cfg["SCALE_FACTOR"]
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
        scale_fac: float = SCALE_FACTOR,
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


# %% Denoiser dataloader
class DenoiseDataset(Dataset):
    def __init__(
        self,
        path: str,
        key: str,
        groups=None,
        transform=None,
        min_flux: float = None,
        crop: int = 128,
    ):
        """
        Minimal multi-group HDF5 dataset for clean images.

        Args:
            path (str): path to HDF5 file
            key (str): key inside each group, e.g. 'isolated_stamps'
            groups (list[str], optional): restrict to these groups (useful for train/val split)
            transform (callable): transforms applied to the tensor (default flip-only)
            min_flux (float, optional): optionally filter based on stamp_flux
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        self.path = path
        self.key = key
        self.min_flux = min_flux
        self.crop = crop

        # Default transforms (same structure as you specified)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    v2.RandomHorizontalFlip(),
                    v2.RandomVerticalFlip(),
                ]
            )
        else:
            self.transform = transform

        # Store group metadata
        self.groups = []
        self.group_sizes = []
        self.valid_indices = []

        with h5py.File(self.path, "r") as hf:
            available = list(hf.keys())

            # Validate group selection
            if groups is not None:
                missing = [g for g in groups if g not in available]
                if missing:
                    raise ValueError(f"These groups do not exist: {missing}")
                self.groups = groups
            else:
                self.groups = available

            # Precompute valid indices per group
            for g in self.groups:
                group = hf[g]
                total = len(group[self.key])

                if self.min_flux is not None:
                    flux = np.sum(group["isolated_stamps"], axis=(1, 2))
                    valid = np.where(flux > self.min_flux)[0]
                else:
                    valid = np.arange(total)

                self.valid_indices.append(valid)
                self.group_sizes.append(len(valid))

        # Cumulative sizes for global indexing
        self.cumulative_sizes = np.cumsum(self.group_sizes).tolist()

        # Worker-local handle
        self._hf = None

    # ---------------- HDF5 handle ---------------- #
    def _get_h5(self):
        """Ensure each worker opens its own HDF5 file handle."""
        worker_info = get_worker_info()

        # Single worker
        if worker_info is None:
            if self._hf is None:
                self._hf = h5py.File(self.path, "r")
            return self._hf

        # Multi-worker
        wid = worker_info.id
        if not hasattr(self, "_worker_files"):
            self._worker_files = {}

        if wid not in self._worker_files:
            self._worker_files[wid] = h5py.File(self.path, "r")

        return self._worker_files[wid]

    # ---------------- Length ---------------- #
    def __len__(self):
        return self.cumulative_sizes[-1]

    # ---------------- Locate index ---------------- #
    def _locate(self, global_idx):
        if global_idx < 0 or global_idx >= self.cumulative_sizes[-1]:
            raise IndexError(f"Index {global_idx} out of range")

        group_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        prev = 0 if group_idx == 0 else self.cumulative_sizes[group_idx - 1]
        offset = global_idx - prev
        local_idx = self.valid_indices[group_idx][offset]
        return self.groups[group_idx], local_idx

    # ---------------- Retrieve item ---------------- #
    def __getitem__(self, idx):
        hf = self._get_h5()

        # -----------------------
        # 1. Handle slice objects
        # -----------------------
        if isinstance(idx, slice):
            # Convert slice → list of indices
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        # -----------------------
        # 2. Handle list or array
        # -----------------------
        if isinstance(idx, (list, tuple, np.ndarray)):
            return [self[int(i)] for i in idx]

        # -----------------------
        # 3. Handle tensor index
        # -----------------------
        if torch.is_tensor(idx):
            if idx.dim() == 0:
                idx = idx.item()
            else:
                # Tensor of indices → list
                return [self[int(i)] for i in idx.tolist()]

        # -----------------------
        # 4. Standard single index
        # -----------------------
        group_name, local_idx = self._locate(idx)
        group = hf[group_name]

        img = np.array(group[self.key][local_idx])

        if img.ndim != 3:
            img = img[np.newaxis, :, :]

        img = torch.from_numpy(img.astype(np.float32))
        crop_fn = CenterCrop(self.crop)
        img = crop_fn(img)

        if self.transform:
            img = self.transform(img)

        return img

    # ---------------- Cleanup ---------------- #
    def __del__(self):
        try:
            if hasattr(self, "_hf") and self._hf is not None:
                self._hf.close()
        except Exception:
            pass

        if hasattr(self, "_worker_files"):
            for f in self._worker_files.values():
                try:
                    f.close()
                except Exception:
                    pass


# %% Shape Measurement Loader


class ShapeDataset(Dataset):
    def __init__(self, path: str, keys, groups=None):
        """
        Optimized HDF5 dataset returning (image, label) pairs.

        Args:
            path (str): Path to HDF5 file
            keys (list[str]): Keys inside each group to load as image channels
            groups (list[str], optional): Restrict to these groups
            crop (int): Center crop size if image larger
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        if isinstance(keys, str):
            keys = [keys]
        self.path = path
        self.keys = keys

        self.groups = []
        self.group_sizes = []
        self.valid_indices = []
        self.labels_per_group = []

        # ---------------- Preload group metadata and labels ---------------- #
        with h5py.File(self.path, "r") as hf:
            available = list(hf.keys())
            self.groups = groups if groups is not None else available

            for g in self.groups:
                group = hf[g]
                total = len(group[self.keys[0]])
                self.valid_indices.append(np.arange(total))
                self.group_sizes.append(total)

                # Preload labels once per group
                patch_df = group["patch_df"][()]
                mask = patch_df["flux_mask"]
                e1 = patch_df["e1"][mask]
                e2 = patch_df["e2"][mask]
                self.labels_per_group.append(np.stack([e1, e2], axis=1))  # shape (N,2)

        # Cumulative sizes for global indexing
        self.cumulative_sizes = np.cumsum(self.group_sizes).tolist()
        self._hf = None  # HDF5 file handle

    # ---------------- HDF5 handle ---------------- #
    def _get_h5(self):
        worker_info = get_worker_info()
        if worker_info is None:
            if self._hf is None:
                self._hf = h5py.File(self.path, "r")
            return self._hf

        wid = worker_info.id
        if not hasattr(self, "_worker_files"):
            self._worker_files = {}
        if wid not in self._worker_files:
            self._worker_files[wid] = h5py.File(self.path, "r")
        return self._worker_files[wid]

    # ---------------- Length ---------------- #
    def __len__(self):
        return self.cumulative_sizes[-1]

    # ---------------- Locate index ---------------- #
    def _locate(self, global_idx):
        if global_idx < 0 or global_idx >= self.cumulative_sizes[-1]:
            raise IndexError(f"Index {global_idx} out of range")
        group_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        prev = 0 if group_idx == 0 else self.cumulative_sizes[group_idx - 1]
        offset = global_idx - prev
        local_idx = self.valid_indices[group_idx][offset]
        return group_idx, local_idx

    # ---------------- Normalize ---------------- #
    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        """Normalize image to [0,1] range."""
        img_min = img.min()
        img_range = img.max() - img_min
        return (img - img_min) / img_range if img_range > 0 else img

    # ---------------- Get item ---------------- #
    def __getitem__(self, idx):
        hf = self._get_h5()

        # ---------------- Handle slices / lists / tensors ---------------- #
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        if isinstance(idx, (list, tuple, np.ndarray)):
            return [self[int(i)] for i in idx]
        if torch.is_tensor(idx):
            idx = idx.item() if idx.dim() == 0 else idx.tolist()
            return (
                [self[int(i)] for i in idx] if isinstance(idx, list) else self[int(idx)]
            )

        # ---------------- Single index ---------------- #
        group_idx, local_idx = self._locate(idx)
        group_name = self.groups[group_idx]
        group = hf[group_name]

        # ---------------- Load images from keys ---------------- #
        imgs = []
        for k in self.keys:
            img = np.array(group[k][local_idx], dtype=np.float32)
            img = self._normalize(img)
            imgs.append(img)

        # Stack along channel dimension if multiple keys
        img = np.stack(imgs, axis=0)  # shape (C,H,W)

        # Convert to tensor
        img = torch.from_numpy(img)

        # ---------------- Center crop if needed ---------------- #
        if img.shape[-2] > 128 or img.shape[-1] > 128:
            img = crop_128(img)

        # ---------------- Load labels ---------------- #
        y = torch.from_numpy(
            self.labels_per_group[group_idx][local_idx]
        ).float()  # shape (2,)

        return img, y

    # ---------------- Cleanup ---------------- #
    def __del__(self):
        try:
            if hasattr(self, "_hf") and self._hf is not None:
                self._hf.close()
        except Exception:
            pass
        if hasattr(self, "_worker_files"):
            for f in self._worker_files.values():
                try:
                    f.close()
                except Exception:
                    pass


# %%
class RandomSubsetSampler(Sampler):
    def __init__(self, dataset_size, subset_size):
        self.dataset_size = dataset_size
        self.subset_size = subset_size

    def __iter__(self):
        # generate new random subset each epoch
        idx = np.random.choice(self.dataset_size, self.subset_size, replace=False)
        return iter(idx.tolist())

    def __len__(self):
        return self.subset_size
