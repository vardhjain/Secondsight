"""Data loader construction for Market-1501.

This module wires together the dataset, transforms and identity-balanced
sampler into ready-to-use :class:`torch.utils.data.DataLoader` objects for
training and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader

from reid.data.dataset import Market1501
from reid.data.sampler import RandomIdentitySampler
from reid.data.transforms import build_transforms

if TYPE_CHECKING:
    from reid.config import Config

__all__ = ["build_dataloaders"]

logger = logging.getLogger(__name__)


def build_dataloaders(cfg: Config, root: str | Path) -> dict[str, Any]:
    """Build train / query / gallery data loaders.

    The training loader uses a :class:`RandomIdentitySampler` (PK sampling) with
    ``drop_last=True`` so every batch contains exactly
    ``P * K = batch_size`` samples suitable for batch-hard triplet mining. The
    query and gallery loaders iterate deterministically (``shuffle=False``).

    Args:
        cfg: Full configuration. ``cfg.data`` supplies geometry, batch size,
            ``num_instances`` and worker count.
        root: Path to the Market-1501 dataset root.

    Returns:
        A dictionary with keys:
            ``train_loader``: PK-sampled training loader.
            ``query_loader``: Deterministic query loader.
            ``gallery_loader``: Deterministic gallery loader.
            ``num_classes``: Number of training identities.
    """
    data_cfg = cfg.data
    root = Path(root)

    train_transform = build_transforms(data_cfg, is_train=True)
    test_transform = build_transforms(data_cfg, is_train=False)

    train_set = Market1501(root, subset="train", transform=train_transform)
    query_set = Market1501(root, subset="query", transform=test_transform)
    gallery_set = Market1501(root, subset="gallery", transform=test_transform)

    logger.info(
        "Loaded Market-1501: train=%d imgs / %d ids, query=%d imgs, gallery=%d imgs.",
        len(train_set),
        train_set.num_classes,
        len(query_set),
        len(gallery_set),
    )

    train_sampler = RandomIdentitySampler(
        train_set,
        batch_size=data_cfg.batch_size,
        num_instances=data_cfg.num_instances,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=data_cfg.batch_size,
        sampler=train_sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    query_loader = DataLoader(
        query_set,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    gallery_loader = DataLoader(
        gallery_set,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return {
        "train_loader": train_loader,
        "query_loader": query_loader,
        "gallery_loader": gallery_loader,
        "num_classes": train_set.num_classes,
    }
