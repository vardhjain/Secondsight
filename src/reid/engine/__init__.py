"""Training engine: schedulers and the :class:`Trainer`.

This subpackage exposes the warmup-enabled learning-rate schedulers and the
training orchestrator for the Re-ID strong baseline:

* :class:`WarmupMultiStepLR` / :class:`WarmupCosineLR` and the
  :func:`build_scheduler` factory.
* :class:`Trainer`, which runs the AMP-aware training loop with periodic
  evaluation and best-by-mAP checkpointing.

Only :mod:`torch` is required at import time.
"""

from __future__ import annotations

from reid.engine.scheduler import (
    WarmupCosineLR,
    WarmupMultiStepLR,
    build_scheduler,
)
from reid.engine.trainer import Trainer

__all__ = [
    "Trainer",
    "WarmupCosineLR",
    "WarmupMultiStepLR",
    "build_scheduler",
]
