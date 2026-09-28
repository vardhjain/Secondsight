"""Data loader construction for Market-1501.

This module wires together the dataset, transforms and identity-balanced
sampler into ready-to-use :class:`torch.utils.data.DataLoader` objects for
training and evaluation.

It can also hold out a few training identities as a validation split, so model
selection never has to look at the test split. For every held-out identity one
image per camera becomes a validation query and the remaining images form the
validation gallery. The standard Market-1501 protocol, which ignores gallery
images that share both identity and camera with the query, then applies to the
validation split unchanged.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Dataset

from reid.data.dataset import ImageListDataset, Market1501
from reid.data.sampler import RandomIdentitySampler
from reid.data.transforms import build_transforms

if TYPE_CHECKING:
    from collections.abc import Callable

    from reid.config import Config, DataConfig

__all__ = ["ValidationSplit", "build_dataloaders", "split_validation_ids"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationSplit:
    """Index lists describing a train and validation partition of a dataset.

    Attributes:
        train: Indices kept for training.
        val_query: Indices of the validation queries (one per camera for each
            held-out identity).
        val_gallery: Indices of the remaining images of the held-out identities.
        val_pids: The held-out identities, sorted.
    """

    train: list[int]
    val_query: list[int]
    val_gallery: list[int]
    val_pids: list[int]


def split_validation_ids(
    pids: Sequence[int], camids: Sequence[int], num_ids: int, seed: int
) -> ValidationSplit:
    """Hold out whole training identities as a validation query and gallery split.

    Only identities seen by at least two cameras are eligible, so every
    validation query has a cross-camera match in the validation gallery
    whenever that identity has more than one image in some camera. The held-out
    identities are drawn with a private :class:`random.Random` seeded by
    ``seed``, so the split is deterministic and does not disturb the global RNG.
    For each held-out identity the first image (in dataset order) from each
    camera becomes a query and every other image goes to the gallery.

    Args:
        pids: Person identities, one per image.
        camids: Camera identities aligned with ``pids``.
        num_ids: Number of identities to hold out. ``0`` keeps every image for
            training.
        seed: Seed that selects the held-out identities.

    Returns:
        A :class:`ValidationSplit` with the train, query and gallery indices.

    Raises:
        ValueError: If ``num_ids`` is negative or larger than the number of
            eligible identities.
    """
    if num_ids < 0:
        raise ValueError(f"num_ids must be >= 0; got {num_ids}.")
    if num_ids == 0:
        return ValidationSplit(list(range(len(pids))), [], [], [])

    cams_per_pid: dict[int, set[int]] = defaultdict(set)
    for pid, camid in zip(pids, camids, strict=True):
        cams_per_pid[pid].add(camid)
    eligible = sorted(pid for pid, cams in cams_per_pid.items() if len(cams) >= 2)
    if num_ids > len(eligible):
        raise ValueError(
            f"Cannot hold out {num_ids} validation identities; only {len(eligible)} "
            f"training identities are seen by at least two cameras."
        )

    held_out = set(random.Random(seed).sample(eligible, num_ids))
    train: list[int] = []
    val_query: list[int] = []
    val_gallery: list[int] = []
    queried: set[tuple[int, int]] = set()
    for index, (pid, camid) in enumerate(zip(pids, camids, strict=True)):
        if pid not in held_out:
            train.append(index)
        elif (pid, camid) in queried:
            val_gallery.append(index)
        else:
            queried.add((pid, camid))
            val_query.append(index)
    return ValidationSplit(train, val_query, val_gallery, sorted(held_out))


def _select(dataset: Market1501, indices: list[int], *, relabel: bool) -> ImageListDataset:
    """Build an :class:`ImageListDataset` from a subset of a Market-1501 split.

    Args:
        dataset: Source dataset.
        indices: Positions to keep, in order.
        relabel: Whether the new dataset maps identities to contiguous labels.

    Returns:
        The selected images as a new dataset sharing the source transform.
    """
    return ImageListDataset(
        [dataset.img_paths[i] for i in indices],
        [dataset.pids[i] for i in indices],
        [dataset.camids[i] for i in indices],
        relabel=relabel,
        transform=dataset.transform,
    )


def _eval_loader(dataset: Dataset, data_cfg: DataConfig, pin: bool) -> DataLoader:
    """Build a deterministic, non-shuffled evaluation loader.

    Args:
        dataset: Query or gallery dataset.
        data_cfg: Data configuration supplying batch size and worker count.
        pin: Whether to pin host memory.

    Returns:
        A :class:`DataLoader` that visits ``dataset`` in order.
    """
    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=data_cfg.num_workers > 0,
    )


def _count_train_ids(root: Path, val_ids: int, seed: int) -> int | None:
    """Count training identities from filenames without building a loader.

    Args:
        root: Market-1501 dataset root.
        val_ids: Number of identities held out for validation.
        seed: Seed used to select the held-out identities.

    Returns:
        The number of training identities after the hold-out, or ``None`` when
        the train folder is missing or holds no images.
    """
    try:
        train_set = Market1501(root, subset="train")
    except (FileNotFoundError, ValueError):
        return None
    split = split_validation_ids(train_set.pids, train_set.camids, val_ids, seed)
    return len({train_set.pids[i] for i in split.train})


def build_dataloaders(
    cfg: Config, root: str | Path, *, include_train: bool = True
) -> dict[str, Any]:
    """Build the train, query, gallery and optional validation data loaders.

    The training loader uses a :class:`RandomIdentitySampler` (PK sampling) with
    ``drop_last=True`` so every batch contains exactly ``P * K = batch_size``
    samples, as batch-hard triplet mining needs. The evaluation loaders iterate
    in a fixed order (``shuffle=False``). Worker processes persist across epochs
    whenever ``cfg.data.num_workers > 0``, which avoids re-spawning them for
    every pass.

    When ``cfg.data.val_ids > 0`` that many training identities are held out
    (see :func:`split_validation_ids`, seeded by ``cfg.train.seed``) and served
    through the validation loaders. The remaining identities are relabelled
    contiguously for training.

    Args:
        cfg: Full configuration. ``cfg.data`` supplies geometry, batch size,
            ``num_instances``, worker count and ``val_ids``.
        root: Path to the Market-1501 dataset root.
        include_train: When ``False`` the training transform, dataset and sampler
            are skipped, so evaluation works on a root that only has the
            ``query`` and ``bounding_box_test`` folders.

    Returns:
        A dictionary with the following keys. ``train_loader`` is the PK-sampled
        training loader, or ``None`` when ``include_train`` is ``False``.
        ``query_loader`` and ``gallery_loader`` serve the test split.
        ``num_classes`` is the number of training identities after any
        validation hold-out; without ``include_train`` it is counted from the
        train folder filenames, or ``None`` when that folder is unusable.
        ``val_query_loader`` and ``val_gallery_loader`` serve the validation
        split, and are ``None`` when ``cfg.data.val_ids`` is ``0`` or
        ``include_train`` is ``False``.

    Raises:
        ValueError: If the validation hold-out cannot be satisfied, or if too
            few training identities remain for PK sampling.
    """
    data_cfg = cfg.data
    root = Path(root)
    val_ids = data_cfg.val_ids
    seed = cfg.train.seed

    # Pin memory only when a CUDA device is actually present. Derive it from the
    # real runtime (not cfg.train.device, which resolve_device may downgrade to
    # CPU), keeping host to GPU copies fast while silencing the CPU-only warning.
    pin = torch.cuda.is_available()

    test_transform: Callable = build_transforms(data_cfg, is_train=False)
    query_set = Market1501(root, subset="query", transform=test_transform)
    gallery_set = Market1501(root, subset="gallery", transform=test_transform)

    loaders: dict[str, Any] = {
        "train_loader": None,
        "query_loader": _eval_loader(query_set, data_cfg, pin),
        "gallery_loader": _eval_loader(gallery_set, data_cfg, pin),
        "num_classes": None,
        "val_query_loader": None,
        "val_gallery_loader": None,
    }

    if not include_train:
        loaders["num_classes"] = _count_train_ids(root, val_ids, seed)
        logger.info(
            "Loaded Market-1501 test split: query=%d imgs, gallery=%d imgs.",
            len(query_set),
            len(gallery_set),
        )
        return loaders

    full_train = Market1501(
        root, subset="train", transform=build_transforms(data_cfg, is_train=True)
    )
    split = split_validation_ids(full_train.pids, full_train.camids, val_ids, seed)
    train_set = _select(full_train, split.train, relabel=True)

    if split.val_pids:
        val_query = _select(full_train, split.val_query, relabel=False)
        val_gallery = _select(full_train, split.val_gallery, relabel=False)
        val_query.transform = test_transform
        val_gallery.transform = test_transform
        loaders["val_query_loader"] = _eval_loader(val_query, data_cfg, pin)
        loaders["val_gallery_loader"] = _eval_loader(val_gallery, data_cfg, pin)
        logger.info(
            "Held out %d training identities for validation: query=%d imgs, gallery=%d imgs.",
            len(split.val_pids),
            len(val_query),
            len(val_gallery),
        )

    sampler = RandomIdentitySampler(
        train_set, batch_size=data_cfg.batch_size, num_instances=data_cfg.num_instances
    )
    loaders["train_loader"] = DataLoader(
        train_set,
        batch_size=data_cfg.batch_size,
        sampler=sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=data_cfg.num_workers > 0,
    )
    loaders["num_classes"] = train_set.num_classes

    logger.info(
        "Loaded Market-1501: train=%d imgs / %d ids, query=%d imgs, gallery=%d imgs.",
        len(train_set),
        train_set.num_classes,
        len(query_set),
        len(gallery_set),
    )
    return loaders
