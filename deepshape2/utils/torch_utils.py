# %%
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

# Seeds to ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2024)

__all__ = [
    "get_freest_gpu",
    "save_ckp",
    "load_ckp",
    "set_seed",
    "count_params",
    "count_params_by_module",
    "ema",
]


# %%
def get_freest_gpu(set_device=True):
    """Return the index of the GPU with the most free memory."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"]
        )
        memory_free = [int(x) for x in output.decode("utf-8").strip().split("\n")]
        ndev = int(torch.tensor(memory_free).argmax())
    except Exception as e:
        print("Could not query GPUs:", e)
        return -1

    if set_device:
        torch.cuda.set_device(ndev)
        torch.cuda.empty_cache()
        device = torch.device(f"cuda:{ndev}")
        print(f"Using device: {device}")
        return device

    return ndev


def save_ckp(model, optimizer, filename, **kwargs):
    """Function to save torch model and optimizer state_dict along with any other data in a dictionary."""

    data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    parent_dir = Path(filename).parent
    try:
        parent_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    else:
        print(f"New folder created {parent_dir}")

    if kwargs:
        data = {**data, **kwargs}
    torch.save(
        data,
        filename,
    )


def load_ckp(filename, model, optimizer, device):
    """Function to load torch model using a saved checkpoint"""

    checkpoint = torch.load(filename, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint


def set_seed(seed=2024, deterministic=True):
    """Function to set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_params(model, trainable_only=True, verbose=True):
    """Print and return the number of parameters in a network."""
    params = [p for p in model.parameters() if p.requires_grad or not trainable_only]
    total = sum(p.numel() for p in params)

    if verbose:
        print(f"{type(model).__name__}: {total:,} parameters ({total / 1e6:.2f}M)")
    return total


def count_params_by_module(model, depth=1):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'module':<30} {'params':>12}   {'%':>6}")
    print("-" * 52)
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"{name:<30} {n:>12,}   {100 * n / total:5.1f}%")
        if depth > 1:
            for sub, m in mod.named_children():
                k = sum(p.numel() for p in m.parameters() if p.requires_grad)
                print(f"  {sub:<28} {k:>12,}   {100 * k / total:5.1f}%")
    print("-" * 52)
    print(f"{'TOTAL':<30} {total:>12,}")
    return total


# %%
def ema(values, alpha=0.1):
    """Reproduce the val_loss_ema sequence from raw per-epoch val losses."""
    out = []
    s = None
    for v in values:
        s = v if s is None else (1 - alpha) * s + alpha * v
        out.append(s)
    return out
