"""Tests for the Re-ID model (:mod:`reid.models.reid_model`).

These are *heavy* tests: :class:`reid.models.reid_model.ReIDModel` builds a
ResNet-50 backbone via ``torchvision``, so the whole module is guarded with
:func:`pytest.importorskip`. The backbone is constructed with
``pretrained=False`` to avoid any network access, and all forward passes use a
tiny CPU input so the tests stay fast.
"""

from __future__ import annotations

import pytest
import torch

# Skip the entire module if torchvision (needed by the backbone) is unavailable.
pytest.importorskip("torchvision")

from reid.config import Config  # noqa: E402  (after importorskip by design)
from reid.models.reid_model import ReIDModel, build_model  # noqa: E402

NUM_CLASSES = 10
FEAT_DIM = 2048


@pytest.fixture(scope="module")
def model() -> ReIDModel:
    """Build a small, randomly-initialized Re-ID model once per module.

    Returns:
        A :class:`ReIDModel` with a non-pretrained ResNet-50 backbone.
    """
    return ReIDModel(num_classes=NUM_CLASSES, pretrained=False, pooling="gem")


def _dummy_batch(batch: int = 2, height: int = 64, width: int = 32) -> torch.Tensor:
    """Create a tiny random image batch.

    Args:
        batch: Number of images.
        height: Image height.
        width: Image width.

    Returns:
        A ``(batch, 3, height, width)`` float tensor.
    """
    torch.manual_seed(0)
    return torch.randn(batch, 3, height, width)


def test_model_components_present(model: ReIDModel) -> None:
    """The model exposes backbone, pool, BNNeck and classifier components."""
    assert hasattr(model, "backbone")
    assert hasattr(model, "pool")
    assert isinstance(model.bottleneck, torch.nn.BatchNorm1d)
    assert isinstance(model.classifier, torch.nn.Linear)
    assert model.num_classes == NUM_CLASSES
    assert model.feat_dim == FEAT_DIM


def test_bnneck_bias_is_frozen(model: ReIDModel) -> None:
    """The BNNeck bias is frozen and the classifier has no bias."""
    assert model.bottleneck.bias.requires_grad is False
    assert model.classifier.bias is None


def test_train_forward_returns_score_and_feature(model: ReIDModel) -> None:
    """Training-mode forward returns ``(cls_score, global_feat)``."""
    model.train()
    inputs = _dummy_batch()
    out = model(inputs)
    assert isinstance(out, tuple)
    cls_score, global_feat = out
    assert cls_score.shape == (inputs.size(0), NUM_CLASSES)
    assert global_feat.shape == (inputs.size(0), FEAT_DIM)


def test_eval_forward_returns_feature(model: ReIDModel) -> None:
    """Eval-mode forward returns only the post-BNNeck feature tensor."""
    model.eval()
    inputs = _dummy_batch()
    with torch.no_grad():
        out = model(inputs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (inputs.size(0), FEAT_DIM)


def test_extract_features_shape_and_mode_restoration(model: ReIDModel) -> None:
    """``extract_features`` returns post-BN feats and restores training mode."""
    model.train()
    inputs = _dummy_batch()
    with torch.no_grad():
        feats = model.extract_features(inputs)
    assert feats.shape == (inputs.size(0), FEAT_DIM)
    # The training flag must be restored after extraction.
    assert model.training is True


def test_build_model_from_config() -> None:
    """``build_model`` constructs a model honouring the config values."""
    cfg = Config()
    cfg.model.pretrained = False
    cfg.model.pooling = "avg"
    built = build_model(cfg, num_classes=NUM_CLASSES)
    assert isinstance(built, ReIDModel)
    assert built.num_classes == NUM_CLASSES

    built.eval()
    with torch.no_grad():
        feats = built(_dummy_batch())
    assert feats.shape == (2, FEAT_DIM)
