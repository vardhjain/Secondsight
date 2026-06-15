"""Lightweight metric accumulators.

:class:`AverageMeter` is the standard running-average helper used throughout
the training loop to track losses and timings.
"""

from __future__ import annotations


class AverageMeter:
    """Track the running sum, count and average of a streamed scalar.

    Attributes:
        val: The most recently observed value.
        avg: The running average over all observations.
        sum: The running sum of ``value * n`` over all observations.
        count: The total number of observations (weighted by ``n``).
    """

    def __init__(self) -> None:
        """Initialize all statistics to zero."""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def reset(self) -> None:
        """Reset all accumulated statistics to zero."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Incorporate a new observation into the running statistics.

        Args:
            val: The observed scalar value.
            n: The weight / multiplicity of this observation (e.g. batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


__all__ = ["AverageMeter"]
