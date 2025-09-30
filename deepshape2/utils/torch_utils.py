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

__all__ = ["get_freest_gpu", "save_ckp", "load_ckp"]


# %%
def get_freest_gpu():
    """Return the index of the GPU with the most free memory."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"]
        )
        memory_free = [int(x) for x in output.decode("utf-8").strip().split("\n")]
        return int(torch.tensor(memory_free).argmax())
    except Exception as e:
        print("Could not query GPUs:", e)
        return -1


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
        1
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
