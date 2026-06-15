"""Tests for the loss functions (:mod:`reid.losses`).

Light tests: only ``torch`` (and ``reid.utils.distance``) are required. They
cover the batch-hard triplet loss, label-smoothing cross-entropy, center loss,
and the combined :class:`reid.losses.ReIDLoss` -- all on tiny random tensors.
"""

from __future__ import annotations

import pytest
import torch

from reid.config import Config
from reid.losses.build import ReIDLoss, build_loss
from reid.losses.center import CenterLoss
from reid.losses.cross_entropy import CrossEntropyLabelSmooth
from reid.losses.triplet import TripletLoss


def _pk_batch(num_ids: int = 2, k: int = 4, dim: int = 16, seed: int = 0) -> tuple:
    """Build a tiny PK-structured feature batch and its labels.

    Args:
        num_ids: Number of identities ``P`` in the batch.
        k: Instances per identity ``K``.
        dim: Feature dimensionality.
        seed: RNG seed.

    Returns:
        A ``(features, targets)`` tuple with ``features`` of shape ``(P*K, dim)``
        and integer ``targets`` of shape ``(P*K,)``.
    """
    torch.manual_seed(seed)
    features = torch.randn(num_ids * k, dim)
    targets = torch.arange(num_ids).repeat_interleave(k)
    return features, targets


# --------------------------------------------------------------------------- #
# TripletLoss
# --------------------------------------------------------------------------- #
def test_triplet_loss_scalar_and_finite() -> None:
    """Triplet loss returns a finite, non-negative scalar."""
    features, targets = _pk_batch()
    loss = TripletLoss(margin=0.3)(features, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_triplet_loss_is_zero_when_well_separated() -> None:
    """A batch satisfying the margin everywhere yields zero loss."""
    # Two identities pushed far apart; intra-class spread tiny.
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.01],
            [10.0, 10.0],
            [10.0, 10.01],
        ]
    )
    targets = torch.tensor([0, 0, 1, 1])
    loss = TripletLoss(margin=0.3)(features, targets)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_triplet_soft_margin_nonnegative() -> None:
    """The soft-margin (softplus) variant is finite and non-negative."""
    features, targets = _pk_batch(seed=1)
    loss = TripletLoss(soft_margin=True)(features, targets)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_triplet_loss_is_differentiable() -> None:
    """Gradients flow back to the input features."""
    features, targets = _pk_batch(seed=2)
    features.requires_grad_(True)
    loss = TripletLoss(margin=0.3)(features, targets)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


# --------------------------------------------------------------------------- #
# CrossEntropyLabelSmooth
# --------------------------------------------------------------------------- #
def test_label_smooth_matches_plain_ce_when_epsilon_zero() -> None:
    """With ``epsilon == 0`` the loss equals plain cross-entropy."""
    torch.manual_seed(3)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    smoothed = CrossEntropyLabelSmooth(num_classes=5, epsilon=0.0)(logits, targets)
    reference = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(smoothed, reference, atol=1e-6)


def test_label_smooth_is_scalar_and_positive() -> None:
    """The smoothed loss is a positive scalar."""
    torch.manual_seed(4)
    logits = torch.randn(6, 4)
    targets = torch.randint(0, 4, (6,))
    loss = CrossEntropyLabelSmooth(num_classes=4, epsilon=0.1)(logits, targets)
    assert loss.dim() == 0
    assert float(loss) > 0.0


def test_label_smooth_raises_floor_on_confident_logits() -> None:
    """Smoothing keeps a non-zero loss even on perfectly confident logits.

    For confident correct predictions plain CE -> 0, but label smoothing keeps a
    positive floor, so the smoothed loss should strictly exceed the plain one.
    """
    logits = torch.full((4, 3), -10.0)
    targets = torch.tensor([0, 1, 2, 0])
    logits[torch.arange(4), targets] = 10.0  # very confident, correct.
    smoothed = CrossEntropyLabelSmooth(num_classes=3, epsilon=0.1)(logits, targets)
    plain = torch.nn.functional.cross_entropy(logits, targets)
    assert float(smoothed) > float(plain)


# --------------------------------------------------------------------------- #
# CenterLoss
# --------------------------------------------------------------------------- #
def test_center_loss_scalar_and_nonnegative() -> None:
    """Center loss returns a finite, non-negative scalar."""
    torch.manual_seed(5)
    features = torch.randn(8, 32)
    labels = torch.arange(4).repeat_interleave(2)
    loss = CenterLoss(num_classes=4, feat_dim=32)(features, labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_center_loss_centers_are_parameter_on_cpu() -> None:
    """Centers are a learnable parameter, left on CPU at construction."""
    loss = CenterLoss(num_classes=3, feat_dim=8)
    assert isinstance(loss.centers, torch.nn.Parameter)
    assert loss.centers.shape == (3, 8)
    assert loss.centers.device.type == "cpu"


def test_center_loss_zero_when_features_equal_centers() -> None:
    """If every feature equals its class center the loss is zero."""
    loss = CenterLoss(num_classes=3, feat_dim=4)
    labels = torch.tensor([0, 1, 2])
    features = loss.centers.detach()[labels]
    out = loss(features, labels)
    assert float(out) == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# ReIDLoss (combined)
# --------------------------------------------------------------------------- #
def test_reid_loss_components_dict_keys() -> None:
    """The combined loss returns the four expected component keys."""
    cfg = Config()
    cfg.loss.center_loss = False
    loss_fn = build_loss(cfg, num_classes=4)

    torch.manual_seed(6)
    cls_score = torch.randn(8, 4)
    global_feat = torch.randn(8, 2048)
    target = torch.arange(4).repeat_interleave(2)

    total, components = loss_fn(cls_score, global_feat, target)
    assert total.dim() == 0
    assert torch.isfinite(total)
    assert set(components) == {"id", "triplet", "center", "total"}
    assert all(isinstance(v, float) for v in components.values())


def test_reid_loss_center_disabled_has_no_center_module() -> None:
    """With center loss off, the submodule is ``None`` and its value is 0."""
    cfg = Config()
    cfg.loss.center_loss = False
    loss_fn = build_loss(cfg, num_classes=4)
    assert loss_fn.use_center is False
    assert loss_fn.center_loss is None

    cls_score = torch.randn(8, 4)
    global_feat = torch.randn(8, 2048)
    target = torch.arange(4).repeat_interleave(2)
    _, components = loss_fn(cls_score, global_feat, target)
    assert components["center"] == 0.0


def test_reid_loss_center_enabled_builds_module() -> None:
    """With center loss on, the submodule exists and contributes a value."""
    cfg = Config()
    cfg.loss.center_loss = True
    loss_fn = build_loss(cfg, num_classes=4)
    assert isinstance(loss_fn, ReIDLoss)
    assert loss_fn.use_center is True
    assert loss_fn.center_loss is not None

    torch.manual_seed(7)
    cls_score = torch.randn(8, 4)
    global_feat = torch.randn(8, 2048)
    target = torch.arange(4).repeat_interleave(2)
    total, components = loss_fn(cls_score, global_feat, target)
    assert torch.isfinite(total)
    assert components["center"] > 0.0
