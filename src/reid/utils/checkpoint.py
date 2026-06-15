"""Checkpoint and model (de)serialization helpers.

Two complementary layers are provided:

* :func:`save_checkpoint` / :func:`load_checkpoint` work with arbitrary state
  dictionaries (model weights, metrics, config, ...).
* :func:`save_model` / :func:`load_model` work directly with an
  :class:`torch.nn.Module`, optionally attaching arbitrary metadata, for
  shipping inference-ready weights.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger("reid")


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Save an arbitrary training-state dictionary to disk.

    Args:
        state: A mapping containing whatever should be persisted, for example
            ``{"model": ..., "optimizer": ..., "epoch": ..., "mAP": ...}``.
        path: Destination file path. Parent directories are created as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic on the same filesystem (POSIX + Windows)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
    *,
    weights_only: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint dictionary from disk.

    Args:
        path: Path to a checkpoint produced by :func:`save_checkpoint` (or any
            ``torch.save``-d mapping).
        map_location: Device mapping passed to :func:`torch.load`.
        weights_only: If ``True`` (default) use PyTorch's safe unpickler, which
            forbids arbitrary code execution. Set to ``False`` only for fully
            trusted checkpoints that contain non-tensor objects the safe loader
            rejects.

    Returns:
        The loaded state dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
    logger.info("Loaded checkpoint from %s", path)
    return checkpoint


def save_model(model: nn.Module, path: str | Path, **meta: Any) -> None:
    """Save a model's ``state_dict`` together with optional metadata.

    The on-disk format is a dictionary with a ``"state_dict"`` key plus any
    metadata passed as keyword arguments (for example ``mAP`` or ``epoch``).

    Args:
        model: The model whose parameters are saved.
        path: Destination file path. Parent directories are created as needed.
        **meta: Arbitrary JSON-/torch-serializable metadata to store alongside
            the weights.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"state_dict": model.state_dict()}
    if meta:
        state["meta"] = meta
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic on the same filesystem (POSIX + Windows)
    logger.info("Saved model to %s", path)


def load_model(
    model: nn.Module,
    path: str | Path,
    map_location: str = "cpu",
    *,
    weights_only: bool = True,
) -> nn.Module:
    """Load weights into ``model`` from a file saved by :func:`save_model`.

    The function is tolerant of both the ``{"state_dict": ...}`` format written
    by :func:`save_model` and a bare ``state_dict``. Keys prefixed with one or
    more ``"module."`` segments (from ``DataParallel``/DDP wrapping) are stripped
    automatically.

    Args:
        model: The model to load parameters into (modified in place).
        path: Path to the saved weights.
        map_location: Device mapping passed to :func:`torch.load`.
        weights_only: If ``True`` (default) use PyTorch's safe unpickler, which
            forbids arbitrary code execution. Set to ``False`` only for fully
            trusted checkpoints.

    Returns:
        The same ``model`` instance, with parameters loaded.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned = {re.sub(r"^(module\.)+", "", k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning("Missing keys when loading model: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading model: %s", unexpected)
    logger.info("Loaded model weights from %s", path)
    return model


def infer_num_classes_from_checkpoint(path: str | Path | None, fallback: int) -> int:
    """Infer the classifier size from a checkpoint's ``classifier.weight`` shape.

    The classifier head depends on the number of training identities, which the
    query/gallery split alone cannot reveal; reading it from the checkpoint lets
    the model be sized so weights load cleanly.

    Args:
        path: Path to the weights / checkpoint file, or ``None`` when no
            checkpoint is available.
        fallback: Number of classes to use if the count cannot be inferred.

    Returns:
        The inferred (or ``fallback``) number of identity classes.
    """
    if path is None:
        return fallback
    path = Path(path)
    if not path.exists():
        return fallback
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - best-effort inference; fall back on any failure
        return fallback

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        return fallback

    for key, value in state_dict.items():
        if key.endswith("classifier.weight") and hasattr(value, "shape"):
            return int(value.shape[0])
    return fallback


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "save_model",
    "load_model",
    "infer_num_classes_from_checkpoint",
]
