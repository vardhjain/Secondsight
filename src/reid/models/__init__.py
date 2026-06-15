"""Model components for the Re-ID toolkit.

This subpackage exposes the building blocks of the Re-ID network:

* :class:`ReIDModel` and :func:`build_model` -- the full ResNet-50 + BNNeck
  model and its config-driven factory.
* :func:`build_backbone` -- the ResNet-50 convolutional feature extractor with
  optional ``last_stride=1`` and IBN support.
* :class:`GeMPooling` and :func:`build_pooling` -- global pooling layers.

Importing this subpackage is lightweight: ``torchvision`` is only imported when
:func:`build_backbone` (directly or via :func:`build_model`) is actually
called, keeping ``import reid.models`` free of heavy dependencies.
"""

from __future__ import annotations

from reid.models.backbone import build_backbone
from reid.models.pooling import GeMPooling, build_pooling
from reid.models.reid_model import ReIDModel, build_model

__all__ = [
    "GeMPooling",
    "ReIDModel",
    "build_backbone",
    "build_model",
    "build_pooling",
]
