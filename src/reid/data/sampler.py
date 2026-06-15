"""Identity-balanced PK sampler for batch-hard triplet training.

Batch-hard triplet mining requires that every mini-batch contains several
instances of each sampled identity so that valid positive and negative pairs
exist. :class:`RandomIdentitySampler` implements the standard ``PK`` sampling
scheme: it draws ``P`` identities per batch and ``K`` instances per identity,
yielding ``P * K`` indices per batch.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING

from torch.utils.data import Sampler

if TYPE_CHECKING:
    from reid.data.dataset import Market1501

__all__ = ["RandomIdentitySampler"]


class RandomIdentitySampler(Sampler[int]):
    """Sample ``P`` identities and ``K`` instances per identity per batch.

    For each batch ``P = batch_size // num_instances`` identities are selected
    and ``K = num_instances`` images are drawn for each. Identities with fewer
    than ``K`` images are oversampled (with replacement) so every identity can
    contribute a full group; identities with more than ``K`` images contribute
    multiple disjoint groups across an epoch.

    Args:
        dataset: A :class:`reid.data.dataset.Market1501` (train subset). Its
            ``pids`` and ``pid2label`` attributes are used to group indices by
            contiguous label.
        batch_size: Total number of samples per batch. Must be a multiple of
            ``num_instances``.
        num_instances: Number of instances (``K``) sampled per identity.

    Raises:
        ValueError: If ``batch_size`` is not divisible by ``num_instances`` or
            if ``batch_size < num_instances``.
    """

    def __init__(
        self,
        dataset: Market1501,
        batch_size: int,
        num_instances: int,
    ) -> None:
        if batch_size < num_instances:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= num_instances ({num_instances})."
            )
        if batch_size % num_instances != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by num_instances ({num_instances})."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances

        # Group dataset indices by contiguous identity label. Falling back to
        # the raw pid keeps the sampler usable even on non-train subsets.
        pid2label = getattr(dataset, "pid2label", None) or {}
        self.index_dic: dict[int, list[int]] = defaultdict(list)
        for index, pid in enumerate(dataset.pids):
            label = pid2label.get(pid, pid)
            self.index_dic[label].append(index)
        self.labels: list[int] = list(self.index_dic.keys())

        # Epoch length = number of complete PK batches the sampler can form,
        # times the batch size. Each identity contributes ``floor(n / K)`` full
        # K-sized groups (short identities are oversampled up to one group), and
        # every batch consumes ``num_pids_per_batch`` groups. Computed once here
        # so ``len()`` is stable across epochs and aligned to ``batch_size``.
        total_groups = sum(
            max(len(idxs), self.num_instances) // self.num_instances
            for idxs in self.index_dic.values()
        )
        self.length = (total_groups // self.num_pids_per_batch) * self.batch_size

    def __iter__(self) -> Iterator[int]:
        """Yield a flat sequence of indices forming ``P * K`` batches.

        Returns:
            An iterator over dataset indices whose length is a multiple of
            ``batch_size``.
        """
        # Build per-identity groups of exactly ``num_instances`` indices.
        batch_idxs_dict: dict[int, list[list[int]]] = defaultdict(list)
        for label in self.labels:
            idxs = copy.deepcopy(self.index_dic[label])
            if len(idxs) < self.num_instances:
                idxs = list(random.choices(idxs, k=self.num_instances))
            random.shuffle(idxs)
            batch_idxs: list[int] = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[label].append(batch_idxs)
                    batch_idxs = []

        avail_labels = copy.deepcopy(self.labels)
        final_idxs: list[int] = []

        while len(avail_labels) >= self.num_pids_per_batch:
            selected_labels = random.sample(avail_labels, self.num_pids_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avail_labels.remove(label)

        return iter(final_idxs)

    def __len__(self) -> int:
        """Return the number of indices produced per epoch.

        Equal to the number of complete ``P * K`` batches that can be formed
        multiplied by ``batch_size`` — a stable, batch-aligned value computed
        once at construction time (never mutated during iteration).
        """
        return self.length
