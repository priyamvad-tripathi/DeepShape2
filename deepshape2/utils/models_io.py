from pathlib import Path

import torch
from colorist import Color
from omegaconf import DictConfig, OmegaConf
from torch import nn

from ..models.shape_network import shapenet_full
from ..models.vae import VAE, VAE_skip
from .io import load_config

MODEL_REGISTRY = {
    "VAE": VAE,
    "shapenet_full": shapenet_full,
    "VAE_skip": VAE_skip,
}

_FALLBACK_KEYS = ("state_dict", "model_state_dict", "model")


def _resolve_weights(cfg, model_cfg, path: str | Path | None) -> tuple[Path, str]:
    """An explicit path is used as-is; otherwise MODEL_DIR / weights."""
    if path is not None:
        weights_path, source = Path(path).expanduser(), "user path"
    else:
        model_dir = Path(str(cfg["MODEL_DIR"])).expanduser()
        weights_path, source = model_dir / str(model_cfg["weights"]), "MODEL_DIR"

    if not weights_path.exists():
        parent = weights_path.parent
        nearby = (
            sorted(p.name for p in parent.glob("*.pt*"))
            if parent.is_dir()
            else f"(directory does not exist: {parent})"
        )
        raise FileNotFoundError(
            f"Weights file not found ({source}): {weights_path}\nIn that directory: {nearby}"
        )
    return weights_path, source


def _extract_state_dict(ckpt, weights_key: str, weights_path: Path) -> dict:
    """Pull the tensor dict out of whatever shape the checkpoint happens to be."""
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unexpected checkpoint type in {weights_path}: {type(ckpt)}")

    for key in (weights_key, *_FALLBACK_KEYS):
        value = ckpt.get(key)
        if isinstance(value, nn.Module):
            return value.state_dict()
        if isinstance(value, dict):
            return value

    if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # already a bare state dict

    raise KeyError(
        f"No weights found in {weights_path}. Looked for "
        f"{[weights_key, *_FALLBACK_KEYS]}; keys present: {list(ckpt)}"
    )


def _clean_keys(state_dict: dict) -> dict:
    """Strip DataParallel/DDP and torch.compile prefixes."""
    out = {}
    for k, v in state_dict.items():
        k = k.removeprefix("module.").removeprefix("_orig_mod.")
        out[k] = v
    return out


def load_model(
    name: str,
    device: torch.device | str = "cpu",
    *,
    path: str | Path | None = None,
    weights_key: str = "best_weights",
    strict: bool = True,
    verbose: bool = True,
    cfg=None,
) -> nn.Module:
    """
    Load a model declared under MODELS in the config.

    Args:
        name:        Key under MODELS.
        device:      Target device.
        path:        Explicit weights file. Used as-is; MODEL_DIR is ignored.
        weights_key: Checkpoint key holding the state dict.
        strict:      Passed through to load_state_dict.
        verbose:     Print the resolved path and any key mismatches.
        cfg:         Preloaded config, to avoid re-reading the YAML in a loop.
    """
    cfg = cfg if cfg is not None else load_config()

    models = cfg.get("MODELS", {})
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Available: {list(models)}")

    model_cfg = models[name]
    class_name = str(model_cfg["class"])
    if class_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model class '{class_name}' is not registered. "
            f"Available: {list(MODEL_REGISTRY)}"
        )

    weights_path, source = _resolve_weights(cfg, model_cfg, path)
    if verbose:
        print(
            f"Loading {name} ({class_name}) from "
            f"{Color.GREEN}{weights_path}{Color.OFF} ({source})"
        )

    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    state_dict = _clean_keys(_extract_state_dict(ckpt, weights_key, weights_path))

    args = model_cfg.get("args", {})
    if isinstance(args, DictConfig):
        args = OmegaConf.to_container(args, resolve=True)
    model = MODEL_REGISTRY[class_name](**(args or {}))

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as err:
        raise RuntimeError(
            f"State dict from {weights_path} does not match {class_name}. "
            f"If the checkpoint was trained with different hyperparameters, "
            f"add an `args:` block for '{name}' in the config.\n{err}"
        ) from err

    if verbose and (missing or unexpected):
        print(f"  missing={list(missing)} unexpected={list(unexpected)}")

    return model.to(device).eval()
