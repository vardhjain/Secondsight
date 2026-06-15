"""Combined Re-ID loss and its factory.

This module wires together the three loss components of the strong baseline:

* identity classification via label-smoothing cross-entropy on the BNNeck
  classifier logits,
* metric learning via batch-hard triplet loss on the pre-BNNeck global
  features, and
* optional center loss for compact intra-class clustering.

The combined :class:`ReIDLoss` returns both the scalar total and a dictionary
of detached per-component values for logging. The center loss has its own set
of learnable parameters that must be optimized separately (with the gradients
"un-scaled" by ``1 / center_weight``); to keep that responsibility with the
training engine, :class:`ReIDLoss` exposes the ``center_loss`` submodule and a
``use_center`` flag rather than owning the optimizer itself.

Only :mod:`torch` (plus the sibling loss modules) is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from reid.losses.center import CenterLoss
from reid.losses.cross_entropy import CrossEntropyLabelSmooth
from reid.losses.triplet import TripletLoss

if TYPE_CHECKING:
    from reid.config import Config, LossConfig


class ReIDLoss(nn.Module):
    """Weighted sum of identity, triplet, and optional center losses.

    Args:
        num_classes: Number of training identities ``C``.
        cfg: Loss configuration controlling weights, the triplet margin /
            soft-margin mode, the label-smoothing factor, and whether center
            loss is enabled.
        feat_dim: Feature width used to size the center-loss centers. Should
            match the model's pre-BNNeck feature dimensionality.

    Attributes:
        cross_entropy: The label-smoothing cross-entropy module.
        triplet: The batch-hard triplet loss module.
        center_loss: The center loss module when ``cfg.center_loss`` is
            ``True``, otherwise ``None``.
        use_center: Whether center loss is active.
        center_weight: Weight applied to the center loss term. The trainer uses
            this to rescale center-loss gradients.
    """

    def __init__(self, num_classes: int, cfg: LossConfig, feat_dim: int = 2048) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.id_weight = cfg.id_weight
        self.triplet_weight = cfg.triplet_weight
        self.use_center = cfg.center_loss
        self.center_weight = cfg.center_weight

        self.cross_entropy = CrossEntropyLabelSmooth(
            num_classes=num_classes, epsilon=cfg.label_smoothing
        )
        self.triplet = TripletLoss(margin=cfg.triplet_margin, soft_margin=cfg.soft_margin)
        # Center loss owns a learnable parameter per identity, sized to the
        # model's feature width. It is built once here (not lazily) so the
        # trainer's dedicated center optimizer binds to a stable parameter.
        self.center_loss: CenterLoss | None = None
        if self.use_center:
            self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)

    def forward(
        self, cls_score: Tensor, global_feat: Tensor, target: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the combined loss and per-component breakdown.

        Args:
            cls_score: Classifier logits of shape ``(N, C)``.
            global_feat: Pre-BNNeck global features of shape ``(N, D)`` used by
                the triplet (and center) losses.
            target: Identity labels of shape ``(N,)``.

        Returns:
            A tuple ``(total_loss, components)`` where ``total_loss`` is the
            scalar weighted sum and ``components`` maps the keys ``"id"``,
            ``"triplet"``, ``"center"``, and ``"total"`` to detached Python
            floats for logging.
        """
        id_loss = self.cross_entropy(cls_score, target)
        triplet_loss = self.triplet(global_feat, target)

        total = self.id_weight * id_loss + self.triplet_weight * triplet_loss

        center_value = 0.0
        if self.use_center:
            assert self.center_loss is not None  # for type-checkers
            center_loss = self.center_loss(global_feat, target)
            total = total + self.center_weight * center_loss
            center_value = float(center_loss.detach().item())

        components: dict[str, float] = {
            "id": float(id_loss.detach().item()),
            "triplet": float(triplet_loss.detach().item()),
            "center": center_value,
            "total": float(total.detach().item()),
        }
        return total, components


def build_loss(cfg: Config, num_classes: int, feat_dim: int | None = None) -> ReIDLoss:
    """Build the combined Re-ID loss from a full configuration.

    Args:
        cfg: The complete experiment configuration. Only ``cfg.loss`` (and, as a
            fallback for ``feat_dim``, ``cfg.model``) is used.
        num_classes: Number of training identities.
        feat_dim: Feature width for the center-loss centers. Defaults to
            ``cfg.model.feat_dim``; pass the model's resolved ``feat_dim`` to
            guarantee the centers match the backbone output exactly.

    Returns:
        A configured :class:`ReIDLoss` instance.
    """
    return ReIDLoss(
        num_classes=num_classes,
        cfg=cfg.loss,
        feat_dim=feat_dim if feat_dim is not None else cfg.model.feat_dim,
    )
