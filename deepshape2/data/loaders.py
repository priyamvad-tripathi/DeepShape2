# %% Import Libraries
import bisect
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, get_worker_info

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
        img = CenterCrop(self.crop)(img)

        if self.transform:
            img = self.transform(img)

        return img

    def __del__(self):
        self._cleanup_h5()


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
    def __init__(self, root_dir):
        self.imgs = np.load(
            os.path.join(root_dir, "imgs.npy"),
            mmap_mode="r+",
        )
        self.labels = np.load(
            os.path.join(root_dir, "labels.npy"),
            mmap_mode="r+",
        )

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = torch.as_tensor(self.imgs[idx])

        if img.shape[-1] > 128 or img.shape[-2] > 128:
            img = crop_128(img)

        label = torch.as_tensor(self.labels[idx])
        return img, label


def build_fast_loaders(
    dataset_dir,
    train_groups,
    val_groups,
    batch_size=32,
    num_workers=4,
):
    group_names = np.load(os.path.join(dataset_dir, "group_names.npy"))
    group_offsets = np.load(os.path.join(dataset_dir, "group_offsets.npy"))

    def groups_to_indices(groups):
        idx = []
        for g in groups:
            pos = np.where(group_names == g)[0]
            if len(pos) == 0:
                continue
            start, end = group_offsets[pos[0]]
            idx.extend(range(start, end))
        return idx

    train_idx = groups_to_indices(train_groups)
    val_idx = groups_to_indices(val_groups)

    dataset = ShapeDatasetLight(dataset_dir)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    return train_loader, val_loader
