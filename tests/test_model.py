"""Tests for the Re-ID model (:mod:`reid.models.reid_model` and its backbone).

These tests build a ResNet-50 via ``torchvision``, so the whole module is
guarded with :func:`pytest.importorskip`. Every backbone is constructed with
``pretrained=False`` and every ``torch.hub`` call is monkeypatched, so nothing
touches the network, and all forward passes use a tiny CPU input so the tests
stay fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

torchvision = pytest.importorskip("torchvision")

from reid.config import Config  # noqa: E402  (after importorskip by design)
from reid.models import backbone as backbone_module  # noqa: E402
from reid.models.backbone import IBN_HUB_REPO, build_backbone  # noqa: E402
from reid.models.reid_model import (  # noqa: E402
    ReIDModel,
    build_model,
    load_trained_model,
)
from reid.utils.checkpoint import load_model, save_checkpoint, save_model  # noqa: E402

NUM_CLASSES = 10
FEAT_DIM = 2048


@pytest.fixture
def model() -> ReIDModel:
    """Build a fresh, randomly initialized Re-ID model for each test.

    Returns:
        A :class:`ReIDModel` with a non-pretrained ResNet-50 backbone.
    """
    torch.manual_seed(0)
    return ReIDModel(num_classes=NUM_CLASSES, pretrained=False, pooling="gem")


def _dummy_batch(batch: int = 2, height: int = 64, width: int = 32) -> torch.Tensor:
    """Create a tiny deterministic image batch.

    Args:
        batch: Number of images.
        height: Image height.
        width: Image width.

    Returns:
        A ``(batch, 3, height, width)`` float tensor.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch, 3, height, width, generator=generator)


# --------------------------------------------------------------------------- #
# Model structure and forward contract
# --------------------------------------------------------------------------- #
def test_model_components_present(model: ReIDModel) -> None:
    """The model exposes backbone, pool, BNNeck and classifier components."""
    assert isinstance(model.backbone, nn.Module)
    assert isinstance(model.pool, nn.Module)
    assert isinstance(model.bottleneck, nn.BatchNorm1d)
    assert isinstance(model.classifier, nn.Linear)
    assert model.num_classes == NUM_CLASSES
    assert model.feat_dim == FEAT_DIM


def test_bnneck_bias_is_frozen(model: ReIDModel) -> None:
    """The BNNeck bias is frozen and the classifier has no bias."""
    assert model.bottleneck.bias.requires_grad is False
    assert model.classifier.bias is None


def test_head_initialization(model: ReIDModel) -> None:
    """BNNeck starts at unit scale and zero shift, the classifier at std 1e-3."""
    assert torch.all(model.bottleneck.weight == 1)
    assert torch.all(model.bottleneck.bias == 0)
    assert abs(float(model.classifier.weight.detach().std()) - 1e-3) < 2e-4
    assert abs(float(model.classifier.weight.detach().mean())) < 1e-4


def test_train_forward_returns_logits_and_pre_bn_feature(model: ReIDModel) -> None:
    """Training forward returns classifier logits and the pre-BNNeck feature."""
    model.train()
    inputs = _dummy_batch(batch=4)
    with torch.no_grad():
        cls_score, global_feat = model(inputs)
        pooled = model.pool(model.backbone(inputs))
    assert cls_score.shape == (4, NUM_CLASSES)
    assert torch.allclose(global_feat, pooled, atol=1e-5)
    # The pre-BN feature must differ from its batch-normalized version.
    assert not torch.allclose(global_feat, model.bottleneck(pooled), atol=1e-3)


def test_eval_forward_returns_post_bn_feature(model: ReIDModel) -> None:
    """Eval forward returns the BNNeck output, matching ``extract_features``."""
    model.eval()
    inputs = _dummy_batch()
    with torch.no_grad():
        out = model(inputs)
        expected = model.bottleneck(model.pool(model.backbone(inputs)))
        extracted = model.extract_features(inputs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (inputs.size(0), FEAT_DIM)
    assert torch.allclose(out, expected, atol=1e-5)
    assert torch.allclose(out, extracted, atol=1e-5)


def test_extract_features_restores_training_mode(model: ReIDModel) -> None:
    """``extract_features`` returns post-BN features and restores train mode."""
    model.train()
    with torch.no_grad():
        feats = model.extract_features(_dummy_batch())
    assert feats.shape == (2, FEAT_DIM)
    assert model.training is True


def test_feat_dim_mismatch_warns_and_uses_backbone_width(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A feat_dim that differs from the backbone width is ignored with a warning."""
    with caplog.at_level("WARNING", logger="reid"):
        built = ReIDModel(num_classes=3, pretrained=False, feat_dim=512)
    assert built.feat_dim == FEAT_DIM
    assert "does not match backbone output width" in caplog.text


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("last_stride", "spatial"), [(1, (4, 2)), (2, (2, 1))])
def test_last_stride_sets_output_resolution(last_stride: int, spatial: tuple[int, int]) -> None:
    """``last_stride=1`` doubles the layer4 resolution relative to stride 2."""
    extractor, dim = build_backbone(pretrained=False, last_stride=last_stride)
    extractor.eval()
    with torch.no_grad():
        feat_map = extractor(_dummy_batch(batch=1))
    assert dim == FEAT_DIM
    assert feat_map.shape == (1, FEAT_DIM, *spatial)


@pytest.mark.parametrize("bad_stride", [0, 3, True])
def test_unsupported_last_stride_raises(bad_stride: int) -> None:
    """Only strides 1 and 2 are accepted."""
    with pytest.raises(ValueError, match="last_stride"):
        build_backbone(pretrained=False, last_stride=bad_stride)


def test_unsupported_backbone_name_raises() -> None:
    """Only ResNet-50 is supported."""
    with pytest.raises(ValueError, match="resnet101"):
        build_backbone(name="resnet101", pretrained=False)


def test_ibn_hub_call_is_pinned_and_non_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """The IBN hub load uses the pinned repo and trusts it without prompting."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    hub_model = torchvision.models.resnet50(weights=None)

    def fake_load(*args: Any, **kwargs: Any) -> nn.Module:
        calls.append((args, kwargs))
        return hub_model

    monkeypatch.setattr(torch.hub, "load", fake_load)
    extractor, _ = build_backbone(pretrained=True, ibn=True)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == IBN_HUB_REPO
    repo, _, ref = IBN_HUB_REPO.partition(":")
    assert repo == "XingangPan/IBN-Net"
    assert len(ref) == 40
    assert args[1] == "resnet50_ibn_a"
    assert kwargs == {"pretrained": True, "trust_repo": True}
    assert extractor.layer4 is hub_model.layer4


def test_ibn_failure_raises_instead_of_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hub failure raises rather than silently building a plain ResNet-50."""

    def failing_load(*args: Any, **kwargs: Any) -> nn.Module:
        raise OSError("no network")

    def forbidden_resnet(pretrained: bool) -> nn.Module:
        raise AssertionError("must not fall back to the plain ResNet-50")

    monkeypatch.setattr(torch.hub, "load", failing_load)
    monkeypatch.setattr(backbone_module, "_build_torchvision_resnet50", forbidden_resnet)
    with pytest.raises(RuntimeError, match="model.ibn") as excinfo:
        build_backbone(pretrained=False, ibn=True)
    assert isinstance(excinfo.value.__cause__, OSError)


# --------------------------------------------------------------------------- #
# build_model
# --------------------------------------------------------------------------- #
def test_build_model_from_config() -> None:
    """``build_model`` constructs a model honouring the config values."""
    cfg = Config()
    cfg.model.pretrained = False
    cfg.model.pooling = "avg"
    built = build_model(cfg, num_classes=NUM_CLASSES)
    assert isinstance(built, ReIDModel)
    assert built.num_classes == NUM_CLASSES
    assert not any(True for _ in built.pool.parameters())

    built.eval()
    with torch.no_grad():
        feats = built(_dummy_batch())
    assert feats.shape == (2, FEAT_DIM)


@pytest.mark.parametrize(("override", "expected"), [(None, True), (False, False), (True, True)])
def test_build_model_pretrained_override(
    monkeypatch: pytest.MonkeyPatch, override: bool | None, expected: bool
) -> None:
    """The ``pretrained`` keyword overrides ``cfg.model.pretrained`` when given."""
    seen: list[bool] = []

    def fake_resnet(pretrained: bool) -> nn.Module:
        seen.append(pretrained)
        return torchvision.models.resnet50(weights=None)

    monkeypatch.setattr(backbone_module, "_build_torchvision_resnet50", fake_resnet)
    cfg = Config()
    cfg.model.pretrained = True
    build_model(cfg, num_classes=3, pretrained=override)
    assert seen == [expected]


def test_build_model_pretrained_false_never_requests_imagenet_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``pretrained=False`` torchvision is asked for no weights at all."""
    real_resnet50 = torchvision.models.resnet50
    requested: list[Any] = []

    def recording_resnet50(*, weights: Any = None, **kwargs: Any) -> nn.Module:
        requested.append(weights)
        return real_resnet50(weights=None, **kwargs)

    monkeypatch.setattr(torchvision.models, "resnet50", recording_resnet50)
    cfg = Config()
    cfg.model.pretrained = True
    build_model(cfg, num_classes=3, pretrained=False)
    assert requested == [None]


# --------------------------------------------------------------------------- #
# load_trained_model and strict loading
# --------------------------------------------------------------------------- #
def _train_config() -> Config:
    """Return a config whose architecture differs from the defaults.

    Returns:
        A :class:`Config` with average pooling, ``last_stride=2`` and a small
        input size, as a trained checkpoint might embed.
    """
    cfg = Config()
    cfg.model.pooling = "avg"
    cfg.model.last_stride = 2
    cfg.data.height = 64
    cfg.data.width = 32
    return cfg


def _trained_source(cfg: Config) -> ReIDModel:
    """Build a deterministic source model for a checkpoint round trip.

    Args:
        cfg: Config describing the architecture.

    Returns:
        The model in eval mode.
    """
    torch.manual_seed(1)
    src = build_model(cfg, num_classes=5, pretrained=False)
    return src.eval()


@pytest.mark.parametrize("writer", ["trainer", "save_model"])
def test_load_trained_model_uses_embedded_architecture(tmp_path: Path, writer: str) -> None:
    """The checkpoint config, not the caller's config, defines the architecture."""
    train_cfg = _train_config()
    src = _trained_source(train_cfg)
    path = tmp_path / "ckpt.pth"
    if writer == "trainer":
        state = {"epoch": 3, "model": src.state_dict(), "config": train_cfg.to_dict()}
        save_checkpoint(state, path)
    else:
        save_model(src, path, mAP=0.5, config=train_cfg.to_dict())

    caller_cfg = Config()
    caller_cfg.eval.batch_size = 7
    loaded, effective = load_trained_model(path, caller_cfg)

    assert effective.model.pooling == "avg"
    assert effective.model.last_stride == 2
    assert (effective.data.height, effective.data.width) == (64, 32)
    assert effective.eval.batch_size == 7
    assert caller_cfg.model.pooling == "gem"  # the caller's config is untouched
    assert loaded.training is False
    assert loaded.num_classes == 5
    assert next(loaded.parameters()).device.type == "cpu"

    inputs = _dummy_batch()
    with torch.no_grad():
        assert torch.allclose(loaded(inputs), src(inputs), atol=1e-5)


def test_load_trained_model_without_embedded_config_uses_given_config(tmp_path: Path) -> None:
    """A bare state_dict is rebuilt from the passed config."""
    cfg = _train_config()
    src = _trained_source(cfg)
    path = tmp_path / "bare.pth"
    torch.save(src.state_dict(), path)

    loaded, effective = load_trained_model(path, cfg)
    assert effective.model.pooling == "avg"
    inputs = _dummy_batch()
    with torch.no_grad():
        assert torch.allclose(loaded(inputs), src(inputs), atol=1e-5)


def test_load_trained_model_mismatch_raises(tmp_path: Path) -> None:
    """Weights that do not fit the configured architecture raise clearly."""
    src = _trained_source(_train_config())  # average pooling, no ``pool.p``
    path = tmp_path / "bare.pth"
    torch.save(src.state_dict(), path)

    gem_cfg = Config()  # GeM pooling expects ``pool.p``
    with pytest.raises(RuntimeError, match="pool.p"):
        load_trained_model(path, gem_cfg)


def test_load_trained_model_missing_file_raises(tmp_path: Path) -> None:
    """A missing checkpoint raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        load_trained_model(tmp_path / "missing.pth")


def test_strict_load_model_rejects_other_pooling(tmp_path: Path) -> None:
    """``load_model`` refuses a GeM checkpoint for an average-pooling model."""
    torch.manual_seed(2)
    gem = ReIDModel(num_classes=3, pretrained=False, pooling="gem")
    path = tmp_path / "gem.pth"
    save_model(gem, path)

    avg = ReIDModel(num_classes=3, pretrained=False, pooling="avg")
    with pytest.raises(RuntimeError, match="Unexpected keys"):
        load_model(avg, path)
