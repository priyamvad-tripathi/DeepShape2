# %% Import Libraries
import bisect
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

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


class HDF5WorkerMixin:
    """Shared worker-safe HDF5 handle logic."""

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

    def _cleanup_h5(self):
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


class MultiGroupIndexMixin:
    """Shared global → (group, local) index mapping."""

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _locate(self, global_idx, apply_valid_indices=True):
        if global_idx < 0 or global_idx >= self.cumulative_sizes[-1]:
            raise IndexError(f"Index {global_idx} out of range")

        group_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        prev = 0 if group_idx == 0 else self.cumulative_sizes[group_idx - 1]
        offset = global_idx - prev

        if apply_valid_indices:
            return group_idx, self.valid_indices[group_idx][offset]
        else:
            return group_idx, offset


# %% Blending dataloader
class BlendDataset(Dataset, HDF5WorkerMixin, MultiGroupIndexMixin):
    def __init__(
        self,
        path,
        x_key,
        y_key,
        groups=None,
        scale_fac=SCALE_FACTOR,
        min_max=False,
        tanh=False,
        arcsin=True,
        transform=crop_128,
        min_flux=None,
    ):
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

        self.groups = []
        self.group_sizes = []
        self.valid_indices = []

        with h5py.File(self.path, "r") as hf:
            available = list(hf.keys())
            self.groups = groups if groups is not None else available

            for g in self.groups:
                group = hf[g]
                n = len(group[self.x_key])

                if self.min_flux is not None:
                    flux = np.array(group["stamp_flux"])
                    valid = np.where(flux > self.min_flux)[0]
                else:
                    valid = np.arange(n)

                self.valid_indices.append(valid)
                self.group_sizes.append(len(valid))

        self.cumulative_sizes = np.cumsum(self.group_sizes).tolist()
        self._hf = None

    def _scale(self, arr):
        arr = arr.astype(np.float32) * self.scale_fac
        if self.arcsin:
            arr = np.arcsinh(arr)
        if self.min_max:
            a_min, a_max = arr.min(), arr.max()
            arr = (arr - a_min) / (a_max - a_min + 1e-8)
        elif self.tanh:
            arr = np.tanh(arr)
        return arr

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        hf = self._get_h5()
        group_idx, local_idx = self._locate(idx)
        group = hf[self.groups[group_idx]]

        x = self._scale(np.array(group[self.x_key][local_idx]))
        y = self._scale(np.array(group[self.y_key][local_idx]))

        if x.ndim != 3:
            x = x[np.newaxis]
        if y.ndim != 3:
            y = y[np.newaxis]

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    def __del__(self):
        self._cleanup_h5()


# %% Denoiser dataloader
class DenoiseDataset(Dataset, HDF5WorkerMixin, MultiGroupIndexMixin):
    def __init__(
        self,
        path,
        key,
        groups=None,
        transform=None,
        min_flux=None,
        crop=128,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        self.path = path
        self.key = key
        self.min_flux = min_flux
        self.crop = crop
        self.transform = transform
        self.groups = []
        self.group_sizes = []
        self.valid_indices = []

        with h5py.File(self.path, "r") as hf:
            available = list(hf.keys())
            self.groups = groups if groups is not None else available

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

        self.cumulative_sizes = np.cumsum(self.group_sizes).tolist()
        self._hf = None

    def __getitem__(self, idx):
        hf = self._get_h5()

        if isinstance(idx, (slice, list, tuple, np.ndarray)):
            return (
                [self[i] for i in range(*idx.indices(len(self)))]
                if isinstance(idx, slice)
                else [self[int(i)] for i in idx]
            )

        if torch.is_tensor(idx):
            idx = idx.item() if idx.dim() == 0 else idx.tolist()
            return (
                [self[int(i)] for i in idx] if isinstance(idx, list) else self[int(idx)]
            )

        group_idx, local_idx = self._locate(idx)
        group = hf[self.groups[group_idx]]

        img = np.array(group[self.key][local_idx])
        if img.ndim != 3:
            img = img[np.newaxis]

        img = torch.from_numpy(img.astype(np.float32))
        if self.crop and (img.shape[-1] > self.crop or img.shape[-2] > self.crop):
            img = CenterCrop(self.crop)(img)

        if self.transform:
            img = self.transform(img)

        return img

    def __del__(self):
        self._cleanup_h5()


class StampDatasetSingle(Dataset):
    def __init__(self, path, key="blended_stamps"):
        self.path = path
        self.key = key
        self._hf = None

        with h5py.File(self.path, "r") as hf:
            self.length = len(hf[self.key])

    def _get_h5(self):
        if self._hf is None:
            self._hf = h5py.File(self.path, "r")
        return self._hf

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        hf = self._get_h5()

        img = np.array(hf[self.key][idx], dtype=np.float32)

        if img.ndim == 2:
            img = img[np.newaxis]

        return torch.from_numpy(img)

    def __del__(self):
        if self._hf is not None:
            self._hf.close()


# %% Shape Measurement Loader
class ShapeDataset(Dataset, HDF5WorkerMixin, MultiGroupIndexMixin):
    def __init__(
        self,
        path,
        keys,
        groups=None,
        metric_name=None,
        metric_threshold=20,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        self.path = path
        self.keys = [keys] if isinstance(keys, str) else keys
        self.groups = []
        self.group_sizes = []
        self.valid_indices = []
        self.labels_per_group = []

        self.metric_name = metric_name
        self.metric_threshold = metric_threshold

        with h5py.File(self.path, "r") as hf:
            available = list(hf.keys())
            self.groups = groups if groups is not None else available

            for g in self.groups:
                group = hf[g]
                total = len(group[self.keys[0]])

                # -------------------------
                # Metric based filtering
                # -------------------------
                if metric_name is not None:
                    if metric_name not in group:
                        raise KeyError(
                            f"Metric '{metric_name}' not found in group '{g}'"
                        )

                    metric_vals = group[metric_name][:]
                    valid_idx = np.where(metric_vals > metric_threshold)[0]
                else:
                    valid_idx = np.arange(total)

                self.valid_indices.append(valid_idx)
                self.group_sizes.append(len(valid_idx))

                # -------------------------
                # Labels (apply flux mask first)
                # -------------------------
                df = group["patch_df"][()]
                flux_mask = df["flux_mask"]

                labels = np.stack(
                    [df["e1"][flux_mask], df["e2"][flux_mask]],
                    axis=1,
                )

                # metric + keys are full length, labels are flux-masked
                self.labels_per_group.append(labels[valid_idx])

        self.cumulative_sizes = np.cumsum(self.group_sizes).tolist()
        self._hf = None

    @staticmethod
    def _normalize(img):
        img_min = img.min()
        rng = img.max() - img_min
        return (img - img_min) / rng if rng > 0 else img

    def __getitem__(self, idx):
        hf = self._get_h5()

        if isinstance(idx, (slice, list, tuple, np.ndarray)):
            return (
                [self[int(i)] for i in range(*idx.indices(len(self)))]
                if isinstance(idx, slice)
                else [self[int(i)] for i in idx]
            )

        if torch.is_tensor(idx):
            idx = idx.item() if idx.dim() == 0 else idx.tolist()
            return (
                [self[int(i)] for i in idx] if isinstance(idx, list) else self[int(idx)]
            )

        # -------------------------
        # Correct index resolution
        # -------------------------
        group_idx, local_pos = self._locate(idx, apply_valid_indices=False)
        local_idx = self.valid_indices[group_idx][local_pos]
        group = hf[self.groups[group_idx]]

        # -------------------------
        # Load images
        # -------------------------
        imgs = [
            self._normalize(np.array(group[k][local_idx], dtype=np.float32))
            for k in self.keys
        ]
        img = torch.from_numpy(np.stack(imgs, axis=0))

        if img.shape[-1] > 128 or img.shape[-2] > 128:
            img = crop_128(img)

        # -------------------------
        # Labels
        # -------------------------
        y = torch.from_numpy(self.labels_per_group[group_idx][local_pos]).float()

        return img, y

    def __del__(self):
        self._cleanup_h5()


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


# %%


class ShapeDatasetLight(Dataset):
    def __init__(self, path, peak_thresh=4, flux_thresh=50, peak_factor_thresh=None):
        self.hf = h5py.File(path, "r")

        if peak_factor_thresh is None:
            peaks = self.hf["peaks"][:]
            fluxes = self.hf["fluxes"][:]

            self.idxs = np.where(
                (peaks > peak_thresh * 0.71e-6) & (fluxes > flux_thresh * 1e-6)
            )[0]
        else:
            peak_factors = self.hf["peak_factor"][:]
            self.idxs = np.where(peak_factors < peak_factor_thresh)[0]

        self.images = self.hf["images"]
        self.shapes = self.hf["shapes"]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i = self.idxs[idx]

        x = torch.from_numpy(self.images[i])
        y = torch.from_numpy(self.shapes[i])

        return x, y


# %%
class ReconDataset(Dataset):
    def __init__(
        self,
        h5_path,
        mask,
        dirty_key="dirty",
        psf_key="psf",
    ):
        self.h5_path = h5_path
        self.mask = np.asarray(mask)
        self.indices = np.where(self.mask)[0]
        self.dirty_key = dirty_key
        self.psf_key = psf_key

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        with h5py.File(self.h5_path, "r") as hf:
            dirty = hf[self.dirty_key][i]
            psf = hf[self.psf_key][i]

        im = np.stack([dirty, psf], axis=0)
        return torch.from_numpy(im).float(), i


# %% PSF Dataloader
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
        x_key,
        y_key=None,
        transform=None,
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

    def __getitem__(self, idx: int):
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
    split,
    batch_size,
    dataset=None,
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

    if dataset is None:
        dataset = ImageDataset(**kwargs)

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
            prefetch_factor=2,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size[1],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
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
