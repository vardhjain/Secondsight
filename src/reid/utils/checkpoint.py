"""Checkpoint and model (de)serialization helpers.

Two complementary layers are provided. :func:`save_checkpoint` and
:func:`load_checkpoint` work with arbitrary state dictionaries such as model
weights, metrics and the training config. :func:`save_model` and
:func:`load_model` work directly with an :class:`torch.nn.Module`, optionally
attaching metadata, for shipping inference-ready weights.
:func:`read_checkpoint_config` recovers the config embedded by either writer so
inference code can rebuild the exact training architecture.
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


def _extract_state_dict(checkpoint: Any) -> dict[str, Any] | None:
    """Return the weights mapping inside a loaded checkpoint.

    Args:
        checkpoint: The object returned by :func:`torch.load`.

    Returns:
        The ``"state_dict"`` entry (written by :func:`save_model`), the
        ``"model"`` entry (written by the trainer), the checkpoint itself when
        it is a bare mapping, or ``None`` when it is not a mapping at all.
    """
    if not isinstance(checkpoint, dict):
        return None
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_model(
    model: nn.Module,
    path: str | Path,
    map_location: str = "cpu",
    *,
    weights_only: bool = True,
    strict: bool = True,
) -> nn.Module:
    """Load weights into ``model`` from a saved checkpoint.

    The function accepts the ``{"state_dict": ...}`` format written by
    :func:`save_model`, the ``{"model": ...}`` format written by the trainer,
    and a bare ``state_dict``. Keys prefixed with one or more ``"module."``
    segments (from ``DataParallel`` or DDP wrapping) are stripped
    automatically.

    Args:
        model: The model to load parameters into (modified in place).
        path: Path to the saved weights.
        map_location: Device mapping passed to :func:`torch.load`.
        weights_only: If ``True`` (default) use PyTorch's safe unpickler, which
            forbids arbitrary code execution. Set to ``False`` only for fully
            trusted checkpoints.
        strict: If ``True`` (default) any missing or unexpected key raises. If
            ``False`` such keys are only logged as warnings.

    Returns:
        The same ``model`` instance, with parameters loaded.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If the checkpoint holds no weights mapping, if a tensor
            shape differs from the model, or, when ``strict`` is ``True``, if
            any key is missing or unexpected.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
    state_dict = _extract_state_dict(checkpoint)
    if state_dict is None:
        raise RuntimeError(f"Checkpoint {path} does not contain a state_dict mapping.")

    cleaned = {re.sub(r"^(module\.)+", "", k): v for k, v in state_dict.items()}
    try:
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(f"Weights in {path} do not fit the model: {exc}") from exc

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Weights in {path} do not match the model architecture. "
            f"Missing keys: {list(missing)}. Unexpected keys: {list(unexpected)}. "
            "Rebuild the model from the checkpoint's own config, for example with "
            "reid.models.reid_model.load_trained_model."
        )
    if missing:
        logger.warning("Missing keys when loading model: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading model: %s", unexpected)
    logger.info("Loaded model weights from %s", path)
    return model


def read_checkpoint_config(path: str | Path) -> dict[str, Any] | None:
    """Return the config dictionary embedded in a checkpoint, if any.

    The trainer stores the config at the top level (``{"config": ...}``) and
    :func:`save_model` stores it under its metadata (``{"meta": {"config":
    ...}}``). Both are read with the safe ``weights_only=True`` unpickler.

    Args:
        path: Path to the checkpoint file.

    Returns:
        The embedded config mapping, or ``None`` when the file is missing,
        unreadable, or carries no config.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - best-effort lookup; callers handle None
        logger.warning("Could not read an embedded config from %s", path)
        return None
    if not isinstance(checkpoint, dict):
        return None
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        meta = checkpoint.get("meta")
        config = meta.get("config") if isinstance(meta, dict) else None
    return config if isinstance(config, dict) else None


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

    state_dict = _extract_state_dict(checkpoint)
    if state_dict is None:
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
    "read_checkpoint_config",
    "infer_num_classes_from_checkpoint",
]
