"""Learning-rate schedulers with linear warmup.

This module provides two warmup-enabled schedulers and a factory that selects
between them from configuration:

* :class:`WarmupMultiStepLR` -- linear (or constant) warmup followed by a
  decoupled multi-step decay at fixed milestone epochs. This is the schedule
  used by the BNNeck strong baseline and is *decoupled* in the sense that the
  warmup factor and the milestone decay are computed independently and then
  multiplied, so they compose cleanly.
* :class:`WarmupCosineLR` -- linear warmup followed by cosine annealing to a
  minimum learning rate.

Both schedulers are *epoch-stepped*: call ``scheduler.step()`` once per epoch.
The ``last_epoch`` argument follows the usual PyTorch convention.

Only :mod:`torch` is required.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from reid.config import Config


def _warmup_factor_at_epoch(
    method: str, epoch: int, warmup_epochs: int, warmup_factor: float
) -> float:
    """Compute the multiplicative warmup factor for a given epoch.

    Args:
        method: Either ``"linear"`` (ramp from ``warmup_factor`` to ``1.0``) or
            ``"constant"`` (hold ``warmup_factor`` for the whole warmup).
        epoch: Current epoch index (``last_epoch``).
        warmup_epochs: Number of warmup epochs.
        warmup_factor: Starting factor at epoch ``0``.

    Returns:
        A multiplier in ``[warmup_factor, 1.0]``.

    Raises:
        ValueError: If ``method`` is not a recognized warmup method.
    """
    if epoch >= warmup_epochs:
        return 1.0
    if method == "constant":
        return warmup_factor
    if method == "linear":
        alpha = epoch / max(warmup_epochs, 1)
        return warmup_factor * (1.0 - alpha) + alpha
    raise ValueError(f"Unknown warmup method: {method!r}")


class WarmupMultiStepLR(LRScheduler):
    """Multi-step LR decay with a linear/constant warmup prefix.

    During the first ``warmup_epochs`` the learning rate is scaled by a factor
    ramping from ``warmup_factor`` up to ``1.0`` (linear) or held constant. After
    warmup the rate is the base LR scaled by ``gamma ** k``, where ``k`` is the
    number of passed milestones.

    Args:
        optimizer: Wrapped optimizer.
        milestones: Epoch indices at which to decay by ``gamma``. Need not be
            sorted.
        gamma: Multiplicative decay factor per milestone. Defaults to ``0.1``.
        warmup_factor: Starting warmup multiplier. Defaults to ``0.01``.
        warmup_epochs: Number of warmup epochs. Defaults to ``10``.
        warmup_method: ``"linear"`` or ``"constant"``. Defaults to ``"linear"``.
        last_epoch: The index of the last epoch. Defaults to ``-1``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: list[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.01,
        warmup_epochs: int = 10,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ) -> None:
        if warmup_method not in ("linear", "constant"):
            raise ValueError(f"Unknown warmup method: {warmup_method!r}")
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current epoch.

        Returns:
            A list of learning rates, one per optimizer parameter group.
        """
        warmup = _warmup_factor_at_epoch(
            self.warmup_method,
            self.last_epoch,
            self.warmup_epochs,
            self.warmup_factor,
        )
        decay = self.gamma ** bisect_right(self.milestones, self.last_epoch)
        return [base_lr * warmup * decay for base_lr in self.base_lrs]


class WarmupCosineLR(LRScheduler):
    """Cosine-annealing LR with a linear warmup prefix.

    During warmup the LR ramps linearly from ``base_lr * warmup_factor`` to
    ``base_lr``. Afterwards it follows a cosine curve from ``base_lr`` down to
    ``eta_min`` over the remaining epochs.

    Args:
        optimizer: Wrapped optimizer.
        max_epochs: Total number of training epochs (warmup included).
        warmup_epochs: Number of warmup epochs. Defaults to ``10``.
        warmup_factor: Starting warmup multiplier. Defaults to ``0.01``.
        eta_min: Minimum learning rate at the end of annealing. Defaults to
            ``0``.
        last_epoch: The index of the last epoch. Defaults to ``-1``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: int,
        warmup_epochs: int = 10,
        warmup_factor: float = 0.01,
        eta_min: float = 0,
        last_epoch: int = -1,
    ) -> None:
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current epoch.

        Returns:
            A list of learning rates, one per optimizer parameter group.
        """
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            factor = _warmup_factor_at_epoch(
                "linear", epoch, self.warmup_epochs, self.warmup_factor
            )
            return [base_lr * factor for base_lr in self.base_lrs]

        # Cosine annealing over the post-warmup span.
        progress = (epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.eta_min + (base_lr - self.eta_min) * cosine for base_lr in self.base_lrs]


def build_scheduler(cfg: Config, optimizer: Optimizer) -> LRScheduler:
    """Build a warmup scheduler from configuration.

    Dispatches on ``cfg.optim.scheduler``:

    * ``"warmup_multistep"`` -> :class:`WarmupMultiStepLR`
    * ``"warmup_cosine"`` / ``"cosine"`` -> :class:`WarmupCosineLR`

    Args:
        cfg: The complete experiment configuration.
        optimizer: The optimizer to schedule.

    Returns:
        A configured :class:`~torch.optim.lr_scheduler.LRScheduler`.

    Raises:
        ValueError: If ``cfg.optim.scheduler`` is not recognized.
    """
    name = cfg.optim.scheduler.lower()
    if name == "warmup_multistep":
        return WarmupMultiStepLR(
            optimizer,
            milestones=list(cfg.optim.milestones),
            gamma=cfg.optim.gamma,
            warmup_factor=cfg.optim.warmup_factor,
            warmup_epochs=cfg.optim.warmup_epochs,
        )
    if name in ("warmup_cosine", "cosine"):
        return WarmupCosineLR(
            optimizer,
            max_epochs=cfg.train.max_epochs,
            warmup_epochs=cfg.optim.warmup_epochs,
            warmup_factor=cfg.optim.warmup_factor,
        )
    raise ValueError(f"Unknown scheduler: {cfg.optim.scheduler!r}")
