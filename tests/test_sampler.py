"""Tests for the identity-balanced PK sampler (:mod:`reid.data.sampler`).

Light tests: only ``Pillow`` / ``torch`` are required -- the sampler operates on
:class:`reid.data.dataset.Market1501` metadata and the dataset itself does not
import ``torchvision``. A tiny on-disk fake dataset (the ``fake_market_root``
fixture) backs these tests.

The key invariants verified here are the ``PK`` batch layout: each batch of
``batch_size = P * K`` indices contains exactly ``P`` identities with ``K``
consecutive instances each, and the total number of yielded indices is a
multiple of ``batch_size``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reid.data.dataset import Market1501
from reid.data.sampler import RandomIdentitySampler

# Sampling geometry used throughout: P = batch_size // num_instances = 2 ids,
# K = num_instances = 2 instances each. The fake train set has 4 identities with
# 4 images each, so full PK batches are guaranteed.
BATCH_SIZE = 4
NUM_INSTANCES = 2
NUM_PIDS_PER_BATCH = BATCH_SIZE // NUM_INSTANCES


def _train_dataset(root: Path) -> Market1501:
    """Build the train-subset dataset from a fake Market-1501 root.

    Args:
        root: Filesystem root of the fake dataset.

    Returns:
        A train-subset :class:`Market1501` (no transform).
    """
    return Market1501(root, subset="train")


def test_sampler_length_is_multiple_of_batch_size(fake_market_root: Path) -> None:
    """The number of yielded indices is divisible by the batch size."""
    dataset = _train_dataset(fake_market_root)
    sampler = RandomIdentitySampler(dataset, batch_size=BATCH_SIZE, num_instances=NUM_INSTANCES)
    indices = list(iter(sampler))
    assert len(indices) > 0
    assert len(indices) % BATCH_SIZE == 0
    # __len__ stays consistent with the most recent iteration.
    assert len(sampler) == len(indices)


def test_sampler_batches_have_correct_pk_structure(fake_market_root: Path) -> None:
    """Every batch holds P identities, each contributing K consecutive items."""
    dataset = _train_dataset(fake_market_root)
    sampler = RandomIdentitySampler(dataset, batch_size=BATCH_SIZE, num_instances=NUM_INSTANCES)
    indices = list(iter(sampler))

    labels = [dataset.pid2label[dataset.pids[i]] for i in indices]
    for start in range(0, len(labels), BATCH_SIZE):
        batch = labels[start : start + BATCH_SIZE]
        # Exactly P distinct identities per batch.
        assert len(set(batch)) == NUM_PIDS_PER_BATCH
        # Each consecutive K-block is a single identity.
        for grp in range(0, BATCH_SIZE, NUM_INSTANCES):
            block = batch[grp : grp + NUM_INSTANCES]
            assert len(set(block)) == 1


def test_sampler_indices_are_in_range(fake_market_root: Path) -> None:
    """All yielded indices are valid positions into the dataset."""
    dataset = _train_dataset(fake_market_root)
    sampler = RandomIdentitySampler(dataset, batch_size=BATCH_SIZE, num_instances=NUM_INSTANCES)
    indices = list(iter(sampler))
    assert all(0 <= i < len(dataset) for i in indices)


def test_sampler_rejects_non_divisible_batch_size(fake_market_root: Path) -> None:
    """A batch size not divisible by ``num_instances`` raises ``ValueError``."""
    dataset = _train_dataset(fake_market_root)
    with pytest.raises(ValueError):
        RandomIdentitySampler(dataset, batch_size=5, num_instances=2)


def test_sampler_rejects_batch_smaller_than_num_instances(fake_market_root: Path) -> None:
    """A batch size below ``num_instances`` raises ``ValueError``."""
    dataset = _train_dataset(fake_market_root)
    with pytest.raises(ValueError):
        RandomIdentitySampler(dataset, batch_size=2, num_instances=4)


def test_sampler_is_reiterable(fake_market_root: Path) -> None:
    """The sampler can be iterated more than once (fresh batches each epoch)."""
    dataset = _train_dataset(fake_market_root)
    sampler = RandomIdentitySampler(dataset, batch_size=BATCH_SIZE, num_instances=NUM_INSTANCES)
    first = list(iter(sampler))
    second = list(iter(sampler))
    assert len(first) % BATCH_SIZE == 0
    assert len(second) % BATCH_SIZE == 0
