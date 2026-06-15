"""Tests for checkpoint (de)serialization (:mod:`reid.utils.checkpoint`).

CPU-only and ``torch``-only: a tiny ``nn.Linear`` stands in for a real model so
the save/load round-trips, ``state_dict`` key handling, and classifier-size
inference can be checked without a GPU, dataset, or torchvision.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from reid.utils.checkpoint import (
    infer_num_classes_from_checkpoint,
    load_checkpoint,
    load_model,
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
