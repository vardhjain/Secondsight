"""Utility helpers for the Re-ID package.

This subpackage groups small, dependency-light helpers used across the rest of
the library: distance computations, reproducibility seeding, logging setup,
checkpoint (de)serialization and metric meters. Everything here depends only on
``numpy`` / ``torch`` (no ``torchvision``/``cv2``), so importing it is cheap.
"""

from __future__ import annotations

from reid.utils.checkpoint import (
    infer_num_classes_from_checkpoint,
    load_checkpoint,
    load_model,
    save_checkpoint,
    save_model,
)
from reid.utils.device import resolve_device
from reid.utils.distance import (
    compute_distance_matrix,
    cosine_distance,
    euclidean_distance,
)
from reid.utils.logging import setup_logger
from reid.utils.meters import AverageMeter
from reid.utils.reproducibility import set_seed

__all__ = [
    "euclidean_distance",
    "cosine_distance",
    "compute_distance_matrix",
    "resolve_device",
    "set_seed",
    "setup_logger",
    "save_checkpoint",
    "load_checkpoint",
    "save_model",
    "load_model",
    "infer_num_classes_from_checkpoint",
    "AverageMeter",
]
