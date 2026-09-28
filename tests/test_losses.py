"""Tests for the loss functions (:mod:`reid.losses`).

Light tests: only ``torch`` (and ``reid.utils.distance``) are required. They
cover the batch-hard triplet loss, label-smoothing cross-entropy, center loss,
and the combined :class:`reid.losses.ReIDLoss` on tiny tensors, including exact
values checked against brute-force references and hand computations.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 - conventional PyTorch alias

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
    assert float(loss.detach()) >= 0.0


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
    assert float(loss.detach()) >= 0.0


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
    assert float(loss.detach()) >= 0.0


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
    torch.manual_seed(7)
    loss_fn = build_loss(cfg, num_classes=4)
    assert isinstance(loss_fn, ReIDLoss)
    assert loss_fn.use_center is True
    assert loss_fn.center_loss is not None

    cls_score = torch.randn(8, 4)
    global_feat = torch.randn(8, 2048)
    target = torch.arange(4).repeat_interleave(2)
    total, components = loss_fn(cls_score, global_feat, target)
    assert torch.isfinite(total)
    assert components["center"] > 0.0


# --------------------------------------------------------------------------- #
# Exact values and references
# --------------------------------------------------------------------------- #
def _bruteforce_hard_pairs(
    features: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mine the hardest positive and negative distance per anchor with a loop.

    Args:
        features: ``(N, D)`` embeddings.
        targets: ``(N,)`` identity labels.

    Returns:
        A ``(dist_ap, dist_an)`` tuple of ``(N,)`` tensors.
    """
    dist = torch.cdist(features.double(), features.double())
    ap, an = [], []
    for i in range(features.size(0)):
        same = targets == targets[i]
        ap.append(dist[i][same].max())
        an.append(dist[i][~same].min())
    return torch.stack(ap).float(), torch.stack(an).float()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_triplet_matches_bruteforce_reference(seed: int) -> None:
    """Batch-hard mining matches an explicit loop over every anchor."""
    features, targets = _pk_batch(num_ids=3, k=4, seed=seed)
    ap, an = _bruteforce_hard_pairs(features, targets)
    hard = TripletLoss(margin=0.3)(features, targets)
    soft = TripletLoss(soft_margin=True)(features, targets)
    assert torch.allclose(hard, F.relu(ap - an + 0.3).mean(), atol=1e-5)
    assert torch.allclose(soft, F.softplus(ap - an).mean(), atol=1e-5)


def test_triplet_hard_margin_exact_value() -> None:
    """Hand-computed batch-hard hinge on 1-D points.

    For points ``[0, 2, 3, 6]`` with labels ``[0, 0, 1, 1]`` the hardest
    positive and negative distances per anchor are ``(2, 3)``, ``(2, 1)``,
    ``(3, 1)`` and ``(3, 4)``, so with margin 0.3 the hinges are
    ``[0, 1.3, 2.3, 0]`` and their mean is ``0.9``.
    """
    features = torch.tensor([[0.0], [2.0], [3.0], [6.0]])
    targets = torch.tensor([0, 0, 1, 1])
    loss = TripletLoss(margin=0.3)(features, targets)
    assert float(loss) == pytest.approx(0.9, abs=1e-5)


def test_triplet_soft_margin_exact_value() -> None:
    """The soft margin averages ``softplus(d_ap - d_an)`` over the anchors."""
    features = torch.tensor([[0.0], [2.0], [3.0], [6.0]])
    targets = torch.tensor([0, 0, 1, 1])
    loss = TripletLoss(soft_margin=True)(features, targets)
    assert float(loss) == pytest.approx(1.016678, abs=1e-5)


def test_triplet_single_identity_raises() -> None:
    """A batch without negatives fails loudly instead of returning inf or NaN."""
    features = torch.randn(4, 8)
    with pytest.raises(ValueError, match="two distinct identities"):
        TripletLoss()(features, torch.zeros(4, dtype=torch.long))


@pytest.mark.parametrize("epsilon", [0.1, 0.3])
def test_label_smooth_matches_torch_label_smoothing(epsilon: float) -> None:
    """Smoothing matches the target definition of PyTorch ``label_smoothing``."""
    torch.manual_seed(8)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    ours = CrossEntropyLabelSmooth(num_classes=5, epsilon=epsilon)(logits, targets)
    reference = F.cross_entropy(logits, targets, label_smoothing=epsilon)
    assert torch.allclose(ours, reference, atol=1e-6)


def test_label_smooth_hand_computation() -> None:
    """Hand-computed smoothed cross-entropy for two small cases."""
    # Uniform logits give log(C) whatever the smoothing factor.
    logits = torch.zeros(2, 4)
    loss = CrossEntropyLabelSmooth(num_classes=4, epsilon=0.2)(logits, torch.tensor([0, 3]))
    assert float(loss) == pytest.approx(float(torch.log(torch.tensor(4.0))), abs=1e-6)

    # Logits [2, 0] with epsilon 0.2 and two classes give the target [0.9, 0.1].
    logits = torch.tensor([[2.0, 0.0]])
    log_p = torch.log_softmax(logits, dim=1)[0]
    expected = -(0.9 * log_p[0] + 0.1 * log_p[1])
    loss = CrossEntropyLabelSmooth(num_classes=2, epsilon=0.2)(logits, torch.tensor([0]))
    assert float(loss) == pytest.approx(float(expected), abs=1e-6)


def test_center_loss_exact_value_and_gradients() -> None:
    """Hand-computed value and gradients of the center loss.

    Centers ``[[0, 0], [1, 1]]``, features ``[[1, 0], [1, 3]]`` and labels
    ``[0, 1]`` give squared distances 1 and 4, so the loss is ``5 / 2``. A
    third center that no sample uses must receive a zero gradient.
    """
    loss_fn = CenterLoss(num_classes=3, feat_dim=2)
    with torch.no_grad():
        loss_fn.centers.copy_(torch.tensor([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]]))
    features = torch.tensor([[1.0, 0.0], [1.0, 3.0]], requires_grad=True)
    labels = torch.tensor([0, 1])

    loss = loss_fn(features, labels)
    assert float(loss) == pytest.approx(2.5, abs=1e-5)

    loss.backward()
    centers = loss_fn.centers.detach()
    center_grad = loss_fn.centers.grad
    assert center_grad is not None
    assert features.grad is not None
    batch = features.size(0)
    for cls in range(3):
        members = features.detach()[labels == cls]
        expected = -2.0 * (members - centers[cls]).sum(dim=0) / batch
        assert torch.allclose(center_grad[cls], expected, atol=1e-5)
    assert torch.all(center_grad[2] == 0)
    expected_feat_grad = 2.0 * (features.detach() - centers[labels]) / batch
    assert torch.allclose(features.grad, expected_feat_grad, atol=1e-5)


def test_build_loss_sizes_centers_from_feat_dim() -> None:
    """``feat_dim`` sizes the center-loss centers."""
    cfg = Config()
    cfg.loss.center_loss = True
    loss_fn = build_loss(cfg, num_classes=4, feat_dim=16)
    assert loss_fn.center_loss is not None
    assert loss_fn.center_loss.centers.shape == (4, 16)


def _weighted_loss_inputs() -> tuple[ReIDLoss, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a center-enabled loss with non-default weights and its inputs.

    Returns:
        A ``(loss_fn, cls_score, global_feat, target)`` tuple.
    """
    cfg = Config()
    cfg.loss.id_weight = 2.0
    cfg.loss.triplet_weight = 0.5
    cfg.loss.center_loss = True
    cfg.loss.center_weight = 0.1
    torch.manual_seed(0)
    loss_fn = build_loss(cfg, num_classes=4, feat_dim=16)
    cls_score = torch.randn(8, 4)
    global_feat = torch.randn(8, 16)
    target = torch.arange(4).repeat_interleave(2)
    return loss_fn, cls_score, global_feat, target


def test_reid_loss_respects_weights() -> None:
    """The total is the weighted sum of the independently computed components."""
    loss_fn, cls_score, global_feat, target = _weighted_loss_inputs()
    total, parts = loss_fn(cls_score, global_feat, target)

    expected = 2.0 * parts["id"] + 0.5 * parts["triplet"] + 0.1 * parts["center"]
    assert float(total) == pytest.approx(expected, rel=1e-6)
    assert parts["total"] == pytest.approx(float(total), rel=1e-6)
    defaults = Config().loss
    ce = CrossEntropyLabelSmooth(4, epsilon=defaults.label_smoothing)(cls_score, target)
    assert parts["id"] == pytest.approx(float(ce), rel=1e-6)
    triplet = TripletLoss(margin=defaults.triplet_margin)(global_feat, target)
    assert parts["triplet"] == pytest.approx(float(triplet), rel=1e-6)
    assert parts["center"] > 0.0


def test_reid_loss_computes_in_float32_under_autocast() -> None:
    """Low-precision inputs inside autocast still yield a float32 loss."""
    loss_fn, cls_score, global_feat, target = _weighted_loss_inputs()
    cls_low = cls_score.to(torch.bfloat16)
    feat_low = global_feat.to(torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        total, parts = loss_fn(cls_low, feat_low, target)
    reference, _ = loss_fn(cls_low.float(), feat_low.float(), target)

    assert total.dtype == torch.float32
    assert torch.allclose(total, reference, rtol=1e-6, atol=1e-6)
    assert parts["total"] == pytest.approx(float(reference), rel=1e-6)


def test_reid_loss_gradients_reach_features_and_centers() -> None:
    """Backward populates gradients for the logits, features and centers."""
    loss_fn, cls_score, global_feat, target = _weighted_loss_inputs()
    cls_score.requires_grad_(True)
    global_feat.requires_grad_(True)
    total, _ = loss_fn(cls_score, global_feat, target)
    total.backward()
    assert cls_score.grad is not None
    assert global_feat.grad is not None
    assert loss_fn.center_loss is not None
    assert loss_fn.center_loss.centers.grad is not None
