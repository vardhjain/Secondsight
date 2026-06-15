"""Batch-hard triplet loss with optional soft margin.

This module ports and hardens the triplet loss from the original research
notebook. The key idea (Hermans et al., 2017, "In Defense of the Triplet
Loss for Person Re-Identification") is *batch-hard* mining: for every anchor
in the batch we select the hardest positive (the same-identity sample that is
*farthest* away) and the hardest negative (the different-identity sample that
is *closest*), then push the negative farther than the positive by a margin.

The improved recipe pairs this loss with the
:class:`reid.data.sampler.RandomIdentitySampler` so that every mini-batch
contains ``P`` identities with ``K`` instances each, guaranteeing that valid
positives exist for the hard mining.

Only :mod:`torch` (plus :mod:`reid.utils.distance`) is required.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from reid.utils.distance import euclidean_distance


class TripletLoss(nn.Module):
    r"""Batch-hard triplet loss.

    For each anchor the loss mines the hardest positive and hardest negative
    within the mini-batch. With a fixed margin the loss is

    .. math::

        \mathcal{L} = \frac{1}{N} \sum_{i} \big[ m + d^{+}_i - d^{-}_i \big]_+

    implemented via :class:`torch.nn.MarginRankingLoss`. With ``soft_margin``
    the hinge is replaced by a smooth softplus surrogate
    ``log(1 + exp(d^{+} - d^{-}))`` which has no margin hyper-parameter and
    behaves better when distances are small.

    Args:
        margin: Margin ``m`` for the hard hinge. Ignored when
            ``soft_margin`` is ``True``. Defaults to ``0.3``.
        soft_margin: If ``True``, use the softplus soft-margin formulation
            instead of a fixed margin. Defaults to ``False``.
    """

    def __init__(self, margin: float = 0.3, soft_margin: bool = False) -> None:
        super().__init__()
        self.margin = margin
        self.soft_margin = soft_margin
        # Only the hard-margin path uses MarginRankingLoss; in soft-margin mode
        # this stays None so repr()/state_dict() reflect the active path.
        self.ranking_loss: nn.MarginRankingLoss | None = (
            None if soft_margin else nn.MarginRankingLoss(margin=margin)
        )

    def forward(self, features: Tensor, targets: Tensor) -> Tensor:
        """Compute the batch-hard triplet loss.

        Args:
            features: Embedding tensor of shape ``(N, D)``. These are the
                pre-BNNeck global features in the strong baseline.
            targets: Identity labels of shape ``(N,)``. The batch must contain
                at least two distinct identities (the ``RandomIdentitySampler``
                guarantees this with ``P >= 2``).

        Returns:
            Scalar loss tensor.
        """
        n = features.size(0)

        # Pairwise Euclidean distance matrix (N, N).
        dist = euclidean_distance(features, features)

        # Boolean masks for same-identity (positive) and different-identity
        # (negative) pairs.
        targets = targets.view(n, 1)
        is_pos = targets.eq(targets.t())
        is_neg = ~is_pos

        # Batch-hard mining requires every anchor to have at least one negative
        # (a different-identity sample). This holds by construction when the
        # batch is built with RandomIdentitySampler (P >= 2 identities). Guard
        # the out-of-contract single-identity batch so it fails loudly instead
        # of silently producing +inf / NaN.
        if not bool(is_neg.any(dim=1).all()):
            raise ValueError(
                "TripletLoss requires at least two distinct identities per "
                "batch so every anchor has a negative; got a single-identity "
                "batch. Use reid.data.sampler.RandomIdentitySampler (P >= 2)."
            )

        # Hardest positive: largest distance among same-identity pairs.
        # Hardest negative: smallest distance among different-identity pairs.
        # Using masked_fill keeps the computation numerically clean and avoids
        # mixing in invalid entries.
        dist_ap = dist.masked_fill(~is_pos, float("-inf")).max(dim=1)[0]
        dist_an = dist.masked_fill(~is_neg, float("inf")).min(dim=1)[0]

        if self.soft_margin:
            # Soft-margin (softplus) formulation: smooth, margin-free.
            return torch.nn.functional.softplus(dist_ap - dist_an).mean()

        # Standard margin ranking: we want dist_an > dist_ap by `margin`.
        assert self.ranking_loss is not None  # always built in hard-margin mode
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
