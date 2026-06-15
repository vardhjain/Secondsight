"""Loss functions for Person Re-Identification.

This subpackage bundles the loss components of the BNNeck strong baseline:

* :class:`CrossEntropyLabelSmooth` -- label-smoothing identity classification.
* :class:`TripletLoss` -- batch-hard triplet loss (with optional soft margin).
* :class:`CenterLoss` -- learnable per-class center loss (optional).
* :class:`ReIDLoss` -- the weighted combination used during training, built via
  :func:`build_loss`.

All modules depend only on :mod:`torch` (and :mod:`reid.utils.distance`), so the
subpackage stays importable without torchvision.
"""

from __future__ import annotations

from reid.losses.build import ReIDLoss, build_loss
from reid.losses.center import CenterLoss
from reid.losses.cross_entropy import CrossEntropyLabelSmooth
from reid.losses.triplet import TripletLoss

__all__ = [
    "CenterLoss",
    "CrossEntropyLabelSmooth",
    "ReIDLoss",
    "TripletLoss",
    "build_loss",
]
