"""Tests for checkpoint (de)serialization (:mod:`reid.utils.checkpoint`).

These tests need only ``torch`` and run on the CPU. Tiny ``nn.Linear`` modules
stand in for a real model, so the save and load round trips, ``state_dict`` key
handling, strict loading, embedded-config lookup and classifier-size inference
can be checked without a GPU, dataset, or torchvision.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import nn

from reid.config import Config
from reid.utils.checkpoint import (
    infer_num_classes_from_checkpoint,
    load_checkpoint,
    load_model,
    read_checkpoint_config,
    save_checkpoint,
    save_model,
)


def test_save_load_model_round_trip(tmp_path: Path) -> None:
    """Weights written by ``save_model`` load back identically via ``load_model``."""
    src = nn.Linear(4, 3)
    path = tmp_path / "weights.pth"
    save_model(src, path, mAP=0.5, epoch=7)

    dst = nn.Linear(4, 3)
    nn.init.zeros_(dst.weight)
    nn.init.zeros_(dst.bias)
    load_model(dst, path)

    assert torch.equal(dst.weight, src.weight)
    assert torch.equal(dst.bias, src.bias)


def test_save_model_creates_parent_dirs(tmp_path: Path) -> None:
    """``save_model`` creates missing parent directories."""
    path = tmp_path / "nested" / "dir" / "weights.pth"
    save_model(nn.Linear(2, 2), path)
    assert path.is_file()


def test_load_model_strips_module_prefix(tmp_path: Path) -> None:
    """``DataParallel``-style ``module.`` prefixes are stripped on load."""
    src = nn.Linear(4, 3)
    wrapped = {f"module.{k}": v for k, v in src.state_dict().items()}
    path = tmp_path / "wrapped.pth"
    torch.save({"state_dict": wrapped}, path)

    dst = nn.Linear(4, 3)
    load_model(dst, path)
    assert torch.equal(dst.weight, src.weight)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """``save_checkpoint`` / ``load_checkpoint`` preserve a mixed state dict."""
    path = tmp_path / "ckpt.pth"
    state = {"epoch": 3, "weights": torch.tensor([1.0, 2.0, 3.0])}
    save_checkpoint(state, path)

    loaded = load_checkpoint(path)
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["weights"], state["weights"])


def test_load_checkpoint_missing_raises(tmp_path: Path) -> None:
    """Loading a non-existent checkpoint raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pth")


def test_load_model_missing_raises(tmp_path: Path) -> None:
    """Loading weights from a missing path raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        load_model(nn.Linear(2, 2), tmp_path / "nope.pth")


def test_infer_num_classes_from_state_dict(tmp_path: Path) -> None:
    """The classifier row count is read from ``*.classifier.weight``."""
    state = {
        "backbone.fc.weight": torch.zeros(10, 4),
        "classifier.weight": torch.zeros(751, 2048),
    }
    path = tmp_path / "model.pth"
    torch.save({"state_dict": state}, path)
    assert infer_num_classes_from_checkpoint(path, fallback=0) == 751


def test_infer_num_classes_falls_back(tmp_path: Path) -> None:
    """A ``None`` path, a missing file, or no classifier weight returns the fallback."""
    assert infer_num_classes_from_checkpoint(None, fallback=42) == 42
    assert infer_num_classes_from_checkpoint(tmp_path / "missing.pth", fallback=42) == 42

    path = tmp_path / "no_clf.pth"
    torch.save({"state_dict": {"backbone.weight": torch.zeros(3, 3)}}, path)
    assert infer_num_classes_from_checkpoint(path, fallback=42) == 42


def _classifier_net() -> nn.Module:
    """Return a tiny module with a ``classifier`` head of five classes.

    Returns:
        A seeded ``nn.Sequential`` with ``backbone`` and ``classifier`` layers.
    """
    torch.manual_seed(0)
    return nn.Sequential(
        OrderedDict(backbone=nn.Linear(4, 8), classifier=nn.Linear(8, 5, bias=False))
    )


def _assert_same_weights(a: nn.Module, b: nn.Module) -> None:
    """Assert two modules hold identical tensors.

    Args:
        a: First module.
        b: Second module.
    """
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert ka == kb
        assert torch.equal(va, vb)


def test_trainer_checkpoint_round_trip_weights_only(tmp_path: Path) -> None:
    """The trainer format loads with the safe unpickler, config included."""
    src = _classifier_net()
    config = Config().to_dict()
    path = tmp_path / "best.pth"
    save_checkpoint(
        {"epoch": 1, "model": src.state_dict(), "best_mAP": 0.5, "config": config}, path
    )

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["config"] == config
    assert load_checkpoint(path)["best_mAP"] == 0.5
    assert infer_num_classes_from_checkpoint(path, fallback=0) == 5
    assert read_checkpoint_config(path) == config

    dst = nn.Sequential(
        OrderedDict(backbone=nn.Linear(4, 8), classifier=nn.Linear(8, 5, bias=False))
    )
    load_model(dst, path)
    _assert_same_weights(src, dst)


def test_save_model_checkpoint_round_trip_weights_only(tmp_path: Path) -> None:
    """The ``save_model`` format round-trips its metadata and embedded config."""
    src = _classifier_net()
    config = Config().to_dict()
    path = tmp_path / "model_final.pth"
    save_model(src, path, mAP=0.5, epoch=7, config=config)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["meta"] == {"mAP": 0.5, "epoch": 7, "config": config}
    assert infer_num_classes_from_checkpoint(path, fallback=0) == 5
    assert read_checkpoint_config(path) == config


def test_load_model_bare_state_dict(tmp_path: Path) -> None:
    """A bare ``state_dict`` file loads directly."""
    src = _classifier_net()
    path = tmp_path / "bare.pth"
    torch.save(src.state_dict(), path)

    dst = nn.Sequential(
        OrderedDict(backbone=nn.Linear(4, 8), classifier=nn.Linear(8, 5, bias=False))
    )
    load_model(dst, path)
    _assert_same_weights(src, dst)
    assert read_checkpoint_config(path) is None


def test_load_model_strict_mismatch_raises(tmp_path: Path) -> None:
    """Missing or unexpected keys raise by default and name the keys."""
    path = tmp_path / "linear.pth"
    save_model(nn.Linear(4, 3), path)

    with pytest.raises(RuntimeError, match=r"Missing keys: \['0.weight', '0.bias'\]"):
        load_model(nn.Sequential(nn.Linear(4, 3)), path)


def test_load_model_shape_mismatch_raises(tmp_path: Path) -> None:
    """A tensor of the wrong shape raises even when not strict."""
    path = tmp_path / "linear.pth"
    save_model(nn.Linear(4, 3), path)
    with pytest.raises(RuntimeError, match="do not fit the model"):
        load_model(nn.Linear(4, 2), path, strict=False)


def test_load_model_non_strict_logs_mismatch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With ``strict=False`` mismatched keys are only logged."""
    path = tmp_path / "linear.pth"
    save_model(nn.Linear(4, 3), path)

    with caplog.at_level(logging.WARNING, logger="reid"):
        load_model(nn.Sequential(nn.Linear(4, 3)), path, strict=False)
    assert "Missing keys" in caplog.text
    assert "Unexpected keys" in caplog.text


def test_read_checkpoint_config_handles_absent_and_unreadable(tmp_path: Path) -> None:
    """Missing files, corrupt files and config-free checkpoints give ``None``."""
    assert read_checkpoint_config(tmp_path / "missing.pth") is None

    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"not a checkpoint")
    assert read_checkpoint_config(corrupt) is None

    no_config = tmp_path / "no_config.pth"
    save_model(nn.Linear(2, 2), no_config, mAP=0.1)
    assert read_checkpoint_config(no_config) is None


def test_save_leaves_no_temporary_files(tmp_path: Path) -> None:
    """The atomic write leaves no ``*.tmp`` file behind."""
    save_checkpoint({"epoch": 1}, tmp_path / "ckpt.pth")
    save_model(nn.Linear(2, 2), tmp_path / "model.pth")
    assert not list(tmp_path.glob("*.tmp"))
