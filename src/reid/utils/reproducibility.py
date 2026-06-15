"""Reproducibility helpers.

A single :func:`set_seed` entry point seeds Python's ``random`` module,
``numpy`` and ``torch`` (CPU and CUDA), and optionally enables PyTorch's
deterministic algorithms. Scripts call this once at startup so that runs are
reproducible.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger("reid")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all relevant RNGs and optionally enable deterministic mode.

    Args:
        seed: The integer seed applied to ``random``, ``numpy`` and ``torch``
            (including all CUDA devices).
        deterministic: When ``True``, enable deterministic behavior across cuDNN
            (``cudnn.deterministic = True``, ``cudnn.benchmark = False``) and
            other ops via ``torch.use_deterministic_algorithms(True,
            warn_only=True)``, and set ``PYTHONHASHSEED`` /
            ``CUBLAS_WORKSPACE_CONFIG``. This trades some throughput for
            run-to-run reproducibility. Note ``PYTHONHASHSEED`` only affects
            subprocesses spawned afterward, since CPython reads it once at
            startup.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Only affects subprocesses spawned later (e.g. DataLoader workers);
        # CPython reads this once at startup, so it is a no-op for the current
        # interpreter's own hashing.
        os.environ["PYTHONHASHSEED"] = str(seed)
        # Required for deterministic CUBLAS GEMMs; must be set before CUDA init.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Force deterministic kernels for ops outside cuDNN (scatter/gather,
        # index_add, some pooling/interpolation). warn_only avoids hard crashes
        # on ops that lack a deterministic implementation.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    logger.debug("Seed set to %d (deterministic=%s).", seed, deterministic)


__all__ = ["set_seed"]
