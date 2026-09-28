"""Tests for loader construction and the validation split (:mod:`reid.data.build`).

The loader tests need ``torchvision`` (for the transforms) and are skipped when
it is not installed. The validation split helper is pure Python and always runs.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from reid.config import Config
from reid.data.build import build_dataloaders, split_validation_ids

LOADER_KEYS = {
    "train_loader",
    "query_loader",
    "gallery_loader",
    "num_classes",
    "val_query_loader",
    "val_gallery_loader",
}


def _tiny_config(val_ids: int = 0, num_workers: int = 0) -> Config:
    """Return a configuration sized for the fake Market-1501 dataset.

    Args:
        val_ids: Number of training identities to hold out for validation.
        num_workers: DataLoader worker count.

    Returns:
        A :class:`Config` with ``32 x 16`` images and ``P = K = 2`` batches.
    """
    cfg = Config()
    cfg.data.height = 32
    cfg.data.width = 16
    cfg.data.pad = 2
    cfg.data.batch_size = 4
    cfg.data.num_instances = 2
    cfg.data.num_workers = num_workers
    cfg.data.val_ids = val_ids
    return cfg


def test_split_validation_zero_keeps_everything() -> None:
    """Holding out no identities keeps every index for training."""
    split = split_validation_ids([1, 1, 2, 2], [1, 2, 1, 2], 0, seed=0)
    assert split.train == [0, 1, 2, 3]
    assert split.val_query == split.val_gallery == split.val_pids == []


def test_split_validation_one_query_per_camera() -> None:
    """Each held-out identity yields one query per camera and a gallery of the rest."""
    pids = [1, 1, 1, 1, 2, 2, 2, 3, 3]
    camids = [1, 1, 2, 2, 1, 2, 3, 1, 1]
    split = split_validation_ids(pids, camids, 2, seed=0)

    # Identity 3 is seen by one camera only, so it is never held out.
    assert split.val_pids == [1, 2]
    assert split.train == [7, 8]
    assert split.val_query == [0, 2, 4, 5, 6]
    assert split.val_gallery == [1, 3]
    assert sorted(split.train + split.val_query + split.val_gallery) == list(range(len(pids)))


def test_split_validation_is_seeded_and_isolated() -> None:
    """The same seed picks the same identities without touching the global RNG."""
    pids = [p for p in range(20) for _ in range(2)]
    camids = [1, 2] * 20
    state = random.getstate()
    first = split_validation_ids(pids, camids, 5, seed=7)
    assert random.getstate() == state
    assert split_validation_ids(pids, camids, 5, seed=7) == first
    assert any(
        split_validation_ids(pids, camids, 5, seed=s).val_pids != first.val_pids
        for s in range(8, 16)
    )


def test_split_validation_rejects_impossible_requests() -> None:
    """Negative counts and more identities than are eligible raise ``ValueError``."""
    with pytest.raises(ValueError, match=">= 0"):
        split_validation_ids([1, 1], [1, 2], -1, seed=0)
    with pytest.raises(ValueError, match="only 1"):
        split_validation_ids([1, 1, 2, 2], [1, 2, 1, 1], 2, seed=0)


def test_build_dataloaders_contract(fake_market_root: Path) -> None:
    """The default loaders have the PK layout, fixed eval order and expected metadata."""
    pytest.importorskip("torchvision")
    out = build_dataloaders(_tiny_config(), fake_market_root)

    assert set(out) == LOADER_KEYS
    assert out["num_classes"] == 4
    assert out["val_query_loader"] is None
    assert out["val_gallery_loader"] is None

    train_loader = out["train_loader"]
    assert train_loader.drop_last is True
    assert len(train_loader) == len(train_loader.sampler) // 4
    images, labels, _ = next(iter(train_loader))
    assert images.shape == (4, 3, 32, 16)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]

    query_loader = out["query_loader"]
    first = [label for _, batch, _ in query_loader for label in batch.tolist()]
    second = [label for _, batch, _ in query_loader for label in batch.tolist()]
    assert first == second == [1, 2]
    assert query_loader.pin_memory == torch.cuda.is_available()
    assert len(out["gallery_loader"].dataset) == 5


def test_build_dataloaders_without_train(fake_market_root: Path) -> None:
    """``include_train=False`` skips the train loader but still counts identities."""
    pytest.importorskip("torchvision")
    out = build_dataloaders(_tiny_config(), fake_market_root, include_train=False)
    assert set(out) == LOADER_KEYS
    assert out["train_loader"] is None
    assert out["num_classes"] == 4
    assert len(out["query_loader"].dataset) == 2


def test_build_dataloaders_without_train_folder(fake_market_root: Path) -> None:
    """Evaluation works on a root that has only the query and gallery folders."""
    pytest.importorskip("torchvision")
    for path in (fake_market_root / "bounding_box_train").iterdir():
        path.unlink()
    (fake_market_root / "bounding_box_train").rmdir()

    out = build_dataloaders(_tiny_config(), fake_market_root, include_train=False)
    assert out["train_loader"] is None
    assert out["num_classes"] is None
    assert len(out["gallery_loader"].dataset) == 5


def test_build_dataloaders_with_validation_split(fake_market_root: Path) -> None:
    """Held-out identities leave training and feed the validation loaders."""
    pytest.importorskip("torchvision")
    out = build_dataloaders(_tiny_config(val_ids=1), fake_market_root)

    assert out["num_classes"] == 3
    train_set = out["train_loader"].dataset
    val_query = out["val_query_loader"].dataset
    val_gallery = out["val_gallery_loader"].dataset

    held_out = set(val_query.pids)
    assert len(held_out) == 1
    assert held_out == set(val_gallery.pids)
    assert held_out.isdisjoint(train_set.pids)
    # Two cameras with two images each give two queries and two gallery images.
    assert sorted(val_query.camids) == [1, 2]
    assert sorted(val_gallery.camids) == [1, 2]
    # Training labels are contiguous again and validation labels are raw pids.
    assert sorted(train_set.pid2label.values()) == [0, 1, 2]
    image, label, _ = val_query[0]
    assert label in held_out
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 16)
    assert torch.equal(val_query[0][0], val_query[0][0])

    evaluation_only = build_dataloaders(
        _tiny_config(val_ids=1), fake_market_root, include_train=False
    )
    assert evaluation_only["num_classes"] == 3


def test_build_dataloaders_persistent_workers(fake_market_root: Path) -> None:
    """Loaders keep their workers alive across epochs when workers are used."""
    pytest.importorskip("torchvision")
    with_workers = build_dataloaders(_tiny_config(val_ids=1, num_workers=2), fake_market_root)
    without = build_dataloaders(_tiny_config(val_ids=1), fake_market_root)
    for key in ("train_loader", "query_loader", "gallery_loader", "val_query_loader"):
        assert with_workers[key].persistent_workers is True
        assert without[key].persistent_workers is False
