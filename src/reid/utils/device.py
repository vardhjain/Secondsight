"""Compute-device resolution with a graceful CPU fallback."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("reid")


def _mps_available() -> bool:
    """Report whether Apple's Metal (MPS) backend is usable.

    Returns:
        ``True`` when this PyTorch build has the MPS backend and it is available.
    """
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device string with a CPU fallback.

    ``"auto"`` picks CUDA when available, then MPS, then CPU. An explicit
    ``"cuda"`` or ``"mps"`` request that the machine cannot honor logs a warning
    and falls back to CPU instead of failing later at ``model.to(device)``.
    CUDA is never silently rerouted to MPS, since MPS op coverage differs.

    Args:
        requested: The device string from the config or CLI, for example
            ``"auto"``, ``"cuda"``, ``"cuda:0"``, ``"mps"`` or ``"cpu"``.

    Returns:
        A usable :class:`torch.device`.
    """
    requested = requested.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        hint = " Use --device mps or auto to run on Apple's GPU." if _mps_available() else ""
        logger.warning("CUDA requested but not available, falling back to CPU.%s", hint)
        return torch.device("cpu")
    if requested.startswith("mps") and not _mps_available():
        logger.warning("MPS requested but not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


__all__ = ["resolve_device"]
