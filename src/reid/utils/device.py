"""Compute-device resolution with a graceful CPU fallback."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("reid")


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device string with a CPU fallback.

    If a CUDA device is requested but CUDA is unavailable, a warning is logged
    and ``cpu`` is returned instead.

    Args:
        requested: The device string from the config / CLI (e.g. ``"cuda"`` or
            ``"cpu"``).

    Returns:
        A usable :class:`torch.device`.
    """
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


__all__ = ["resolve_device"]
