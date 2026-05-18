import warnings
from pathlib import Path

import torch
from colorist import Color
from torch import nn

from ..models.shape_network import shapenet_full
from ..models.vae import VAE
from .io import load_config

MODEL_REGISTRY = {
    "VAE": VAE,
    "shapenet_full": shapenet_full,
}


def load_model(
    name: str,
    device: torch.device | str,
    verbose: bool = True,
    weights_key: str = "best_weights",
    path: str | Path | None = None,
) -> nn.Module:
    """
    Load a registered model by name from the config.

    Args:
        name:        Model name as defined in config MODELS section.
        device:      Target device (e.g. 'cuda', 'cpu', torch.device).
        verbose:     Print loading info.
        weights_key: Key to extract from checkpoint dict. Falls back to
                     the full checkpoint if the key is absent.
        path:        Optional custom path to weights file.

    Returns:
        Model loaded with weights, moved to device, in eval mode.

    Raises:
        ValueError: If model name or class is not found in config / registry.
        FileNotFoundError: If the weights file does not exist.
        KeyError: If the checkpoint does not contain the expected state dict.
    """
    cfg = load_config()

    available = list(cfg.get("MODELS", {}).keys())
    if name not in available:
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    model_cfg = cfg["MODELS"][name]
    class_name = model_cfg["class"]

    if class_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model class '{class_name}' is not registered. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    default_weights_path = Path(cfg["MODEL_DIR"]) / model_cfg["weights"]

    # NEW: handle optional custom path
    if path is not None:
        custom_path = Path(path)
        if custom_path.exists():
            weights_path = custom_path
            if verbose:
                print(
                    f"Using provided weights path: {Color.GREEN}{weights_path}{Color.OFF}"
                )
        else:
            weights_path = default_weights_path
            if verbose:
                print(
                    f"Provided path does not exist: {custom_path}. "
                    f"Falling back to default: {Color.GREEN}{weights_path}{Color.OFF}"
                )
    else:
        weights_path = default_weights_path
        if verbose:
            print(
                f"Loading {name} ({class_name}) from {Color.GREEN}{weights_path}{Color.OFF} (Default path)"
            )

        if not default_weights_path.exists():
            raise FileNotFoundError(f"Default weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

    if weights_key in checkpoint:
        state_dict = checkpoint[weights_key]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # assume raw state dict

    model = MODEL_REGISTRY[class_name]()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        warnings.warn(f"Missing keys when loading '{name}': {missing}")
    if unexpected:
        warnings.warn(f"Unexpected keys when loading '{name}': {unexpected}")

    return model.to(device).eval()
