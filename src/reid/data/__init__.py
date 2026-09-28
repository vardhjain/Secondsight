"""Data subpackage: dataset, transforms, sampler and loader construction.

Public API:
    - :class:`Market1501`: Market-1501 image dataset (no ``torchvision`` import).
    - :func:`build_transforms`: train/test transform pipelines.
    - :class:`RandomIdentitySampler`: identity-balanced PK sampler.
    - :class:`ImageListDataset`: dataset over explicit image and identity lists.
    - :func:`build_dataloaders`: assemble train, query, gallery and validation loaders.
    - :func:`split_validation_ids`: hold out training identities for validation.
    - :data:`IMAGENET_MEAN`, :data:`IMAGENET_STD`: normalisation constants.

None of these imports pulls in ``torchvision`` at import time; heavy
dependencies are imported lazily inside the functions that require them.
"""

from __future__ import annotations

from reid.data.build import ValidationSplit, build_dataloaders, split_validation_ids
from reid.data.dataset import ImageListDataset, Market1501
from reid.data.sampler import RandomIdentitySampler
from reid.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transforms

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ImageListDataset",
    "Market1501",
    "RandomIdentitySampler",
    "ValidationSplit",
    "build_dataloaders",
    "build_transforms",
    "split_validation_ids",
]
