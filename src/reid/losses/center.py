"""Center loss for compact intra-class feature clustering.

This module implements :class:`CenterLoss` (Wen et al., 2016, "A Discriminative
Feature Learning Approach for Deep Face Recognition"). The loss maintains one
learnable center per identity and penalizes the squared distance between each
sample's feature and its class center, encouraging tighter intra-class
clusters. It complements the triplet loss in the strong baseline and is gated
behind a config flag (default on).

The class centers are stored as an :class:`torch.nn.Parameter`; they are
**not** moved to a device inside ``__init__`` so that the standard
``model.to(device)`` / ``loss.to(device)`` pattern (driven by the
:class:`reid.engine.trainer.Trainer`) remains the single source of truth for
device placement. The trainer also creates a dedicated optimizer for these
centers.

Only :mod:`torch` is required.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CenterLoss(nn.Module):
    """Center loss with learnable per-class centers.

    Args:
        num_classes: Number of identity classes ``C``.
        feat_dim: Dimensionality ``D`` of the feature embedding. Defaults to
            ``2048`` (ResNet-50 global feature size).
    """

    def __init__(self, num_classes: int, feat_dim: int = 2048) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # Learnable centers; left on CPU until .to(device) is called by the
        # caller. Intentionally not relocated here (see module docstring).
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """Compute the center loss.

        Args:
            features: Feature embeddings of shape ``(N, D)``.
            labels: Identity labels of shape ``(N,)``.

        Returns:
            Scalar loss tensor.
        """
        batch_size = features.size(0)

        # Squared Euclidean distance between every feature and every center,
        # giving a (N, C) matrix.
        distmat = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )
        distmat = torch.addmm(distmat, features, self.centers.t(), beta=1, alpha=-2)

        # Select, for each sample, the distance to its own class center.
        classes = torch.arange(self.num_classes, device=features.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # Pick out, for each sample, only the distance to its own class center
        # (one True per row), so masked-out cells contribute nothing.
        dist = distmat[mask]
        # Clamp the true per-sample distances for numerical stability before
        # averaging over the batch.
        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size
        return loss
