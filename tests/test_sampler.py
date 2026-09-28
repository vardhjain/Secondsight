"""Tests for the identity-balanced PK sampler (:mod:`reid.data.sampler`).

Light tests: only ``Pillow`` and ``torch`` are required, because the sampler operates on
:class:`reid.data.dataset.Market1501` metadata and the dataset itself does not
import ``torchvision``. A tiny on-disk fake dataset (the ``fake_market_root``
fixture) backs these tests.

The key invariants verified here are the ``PK`` batch layout: each batch of
``batch_size = P * K`` indices contains exactly ``P`` identities with ``K``
consecutive instances each, and the total number of yielded indices is a
multiple of ``batch_size``.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    # ``__len__`` is a stable, batch-aligned estimate fixed at construction time.
    # Under random identity selection the PK sampler can strand a few groups, so
    # the yielded count is at most ``len(sampler)`` (and never exceeds it).
    assert len(sampler) % BATCH_SIZE == 0
    assert len(indices) <= len(sampler)


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


def test_sampler_rejects_too_few_identities(fake_market_root: Path) -> None:
    """Asking for more identities per batch than exist raises ``ValueError``."""
    dataset = _train_dataset(fake_market_root)
    # P = 16 // 2 = 8 identities per batch, but the fake train set has only 4.
    with pytest.raises(ValueError, match="at least 8 identities"):
        RandomIdentitySampler(dataset, batch_size=16, num_instances=2)


def test_sampler_iteration_does_not_mutate_state(fake_market_root: Path) -> None:
    """Iterating twice leaves the index table and length unchanged."""
    dataset = _train_dataset(fake_market_root)
    sampler = RandomIdentitySampler(dataset, batch_size=BATCH_SIZE, num_instances=NUM_INSTANCES)
    index_dic = copy.deepcopy(dict(sampler.index_dic))
    length = len(sampler)

    for _ in range(2):
        indices = list(iter(sampler))
        assert len(indices) % BATCH_SIZE == 0

    assert dict(sampler.index_dic) == index_dic
    assert len(sampler) == length


def test_sampler_is_reproducible_under_seed(fake_market_root: Path) -> None:
    """Re-seeding the global RNG reproduces the exact index sequence."""
    dataset = _train_dataset(fake_market_root)
    sampler = RandomIdentitySampler(dataset, batch_size=BATCH_SIZE, num_instances=NUM_INSTANCES)
    random.seed(123)
    first = list(sampler)
    random.seed(123)
    second = list(sampler)
    assert first == second


def _stub_dataset(pids: list[int]) -> Any:
    """Build a metadata-only stand-in for a training dataset.

    Args:
        pids: Identity of every image, already contiguous from ``0``.

    Returns:
        An object exposing ``pids`` and ``pid2label`` like a training dataset.
    """
    return SimpleNamespace(pids=pids, pid2label={p: p for p in sorted(set(pids))})


def test_sampler_handles_uneven_and_short_identities() -> None:
    """Short identities are oversampled to a full group and uneven sizes keep PK layout."""
    # Identity 0 has 1 image and identity 1 has 3, both fewer than K = 4.
    pids = [0] * 1 + [1] * 3 + [2] * 5 + [3] * 8 + [4] * 4 + [5] * 4
    dataset = _stub_dataset(pids)
    batch_size, num_instances = 8, 4
    sampler = RandomIdentitySampler(dataset, batch_size=batch_size, num_instances=num_instances)

    seen_short = set()
    for seed in range(20):
        random.seed(seed)
        indices = list(sampler)
        assert len(indices) % batch_size == 0
        assert len(indices) <= len(sampler)
        assert all(0 <= i < len(pids) for i in indices)
        labels = [pids[i] for i in indices]
        for start in range(0, len(labels), batch_size):
            batch = labels[start : start + batch_size]
            assert len(set(batch)) == batch_size // num_instances
            for grp in range(0, batch_size, num_instances):
                block = batch[grp : grp + num_instances]
                assert len(set(block)) == 1
        # A short identity only ever appears as one complete block of K.
        for short in (0, 1):
            count = labels.count(short)
            assert count in (0, num_instances)
            if count:
                seen_short.add(short)
    # Over twenty epochs both short identities are drawn at least once.
    assert seen_short == {0, 1}
