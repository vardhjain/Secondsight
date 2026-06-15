"""Data subpackage: dataset, transforms, sampler and loader construction.

Public API:
    - :class:`Market1501`: Market-1501 image dataset (no ``torchvision`` import).
    - :func:`build_transforms`: train/test transform pipelines.
    - :class:`RandomIdentitySampler`: identity-balanced PK sampler.
    - :func:`build_dataloaders`: assemble train/query/gallery loaders.
    - :data:`IMAGENET_MEAN`, :data:`IMAGENET_STD`: normalisation constants.

None of these imports pulls in ``torchvision`` at import time; heavy
dependencies are imported lazily inside the functions that require them.
"""

from __future__ import annotations

from reid.data.build import build_dataloaders
from reid.data.dataset import Market1501
from reid.data.sampler import RandomIdentitySampler
from reid.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transforms

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "Market1501",
    "RandomIdentitySampler",
    "build_dataloaders",
    "build_transforms",
]
