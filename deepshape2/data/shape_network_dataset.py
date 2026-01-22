# %%
import argparse
import os
import time

import h5py
import numpy as np

from deepshape2.utils import load_config, time_string

cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]

# %% Load Config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--h5_path",
    type=str,
    default=DATA_DIR + "wide_set.h5",
    help="Path to the HDF5 dataset file.",
)

parser.add_argument(
    "--out_dir",
    "-o",
    type=str,
    default=None,
    help="Directory to save the curated dataset.",
)

parser.add_argument(
    "--groups",
    "-g",
    nargs="+",
    type=str,
    default=[f"patch_{nl + 1:03d}" for nl in range(71, 100)],
    help="List of group names to include",
)
parser.add_argument(
    "--keys",
    "-k",
    nargs="+",
    default=["recon", "psf"],
    help="List of dataset keys to include.",
)
parser.add_argument(
    "--metric_name",
    "-m",
    type=str,
    default="psnr",
    help="Name of the metric dataset to use for filtering.",
)


parser.add_argument(
    "--no_thresholding",
    "-ntr",
    action="store_true",
    help="Skip thresholding",
)

parser.add_argument(
    "--metric_threshold",
    "-t",
    type=float,
    default=25,
    help="Threshold value for the metric to filter images.",
)


def build_train_dataset(
    h5_path,
    keys,
    groups,
    metric_name,
    metric_threshold,
    out_dir,
    disable_thresholding=False,
):
    start = time.time()
    keys = [keys] if isinstance(keys, str) else keys

    if out_dir is None:
        out_dir = DATA_DIR + f"trainset_psnr_{metric_threshold}"
    else:
        out_dir = DATA_DIR + out_dir

    os.makedirs(out_dir, exist_ok=True)

    imgs_list = []
    labels_list = []
    group_names = []
    group_offsets = []

    offset = 0

    if disable_thresholding:
        print("Thresholding disabled. Including all images.")

    with h5py.File(h5_path, "r") as hf:
        all_groups = groups if groups is not None else list(hf.keys())

        for g in all_groups:
            group = hf[g]

            # --------------------------------------------------
            # 1) Flux mask (catalog space)
            # --------------------------------------------------
            df = group["patch_df"][()]
            flux_mask = df["flux_mask"]
            cat_after_flux = np.where(flux_mask)[0]

            # --------------------------------------------------
            # 2) PSNR mask (image space)
            # --------------------------------------------------
            if not disable_thresholding:
                psnr_vals = group[metric_name][:]
                valid_img_idx = np.where(psnr_vals > metric_threshold)[0]
            else:
                valid_img_idx = np.arange(len(cat_after_flux))

            if len(valid_img_idx) == 0:
                continue

            valid_cat_idx = cat_after_flux[valid_img_idx]

            # --------------------------------------------------
            # 3) Load images
            # --------------------------------------------------
            group_imgs = np.stack(
                [group[k][valid_img_idx].astype(np.float32) for k in keys],
                axis=1,
            )

            # --------------------------------------------------
            # 4) Load labels
            # --------------------------------------------------
            e1 = df["e1"][valid_cat_idx]
            e2 = df["e2"][valid_cat_idx]
            group_labels = np.stack([e1, e2], axis=1).astype(np.float32)

            imgs_list.append(group_imgs)
            labels_list.append(group_labels)

            group_names.append(g)
            group_offsets.append((offset, offset + len(valid_img_idx)))
            offset += len(valid_img_idx)

            print(
                f"Finished processing group: {g} | "
                f"Time elapsed: {time_string(time.time() - start)}"
            )

    # --------------------------------------------------
    # Concatenate once (offline cost)
    # --------------------------------------------------
    imgs = np.concatenate(imgs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # --------------------------------------------------
    # Save as .npy (memory-mappable)
    # --------------------------------------------------
    np.save(os.path.join(out_dir, "imgs.npy"), imgs)
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    np.save(
        os.path.join(out_dir, "group_offsets.npy"),
        np.asarray(group_offsets, dtype=np.int64),
    )
    np.save(os.path.join(out_dir, "group_names.npy"), np.asarray(group_names))

    print(f"Saved curated dataset to {out_dir}")
    print(f"Total samples: {len(imgs)}")
    print(f"Total time: {time_string(time.time() - start)}")


if __name__ == "__main__":
    args = parser.parse_args()

    build_train_dataset(
        h5_path=args.h5_path,
        keys=args.keys,
        groups=args.groups,
        metric_name=args.metric_name,
        metric_threshold=args.metric_threshold,
        out_dir=args.out_dir,
        disable_thresholding=args.no_thresholding,
    )
