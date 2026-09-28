"""End-to-end test of the training CLI (:mod:`reid.cli.train`).

``main()`` runs for one epoch on a tiny synthetic Market-1501 folder tree written
to ``tmp_path``, with the ResNet-50 builder monkeypatched to a tiny linear model,
so the whole pipeline (config loading and overrides, data loading, training,
periodic and final evaluation, and every output file) is exercised on CPU in a
few seconds and without any download.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from reid.cli import train as train_cli
from reid.config import Config

HEIGHT, WIDTH = 16, 8


class TinyReIDModel(nn.Module):
    """Minimal stand-in for ``ReIDModel`` with the same train/eval contract."""

    def __init__(self, num_classes: int, feat_dim: int = 8) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.embed = nn.Linear(3 * HEIGHT * WIDTH, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Return ``(logits, feat)`` in train mode and ``feat`` in eval mode."""
        feat = self.embed(x.flatten(1))
        if self.training:
            return self.classifier(feat), feat
        return feat


def _write_market(root: Path) -> None:
    """Write a tiny Market-1501 tree with canonical file names."""
    layout = {
        # Six identities, each seen by cameras 1 and 2 with two images apiece.
        "bounding_box_train": [
            (pid, cam, k) for pid in range(1, 7) for cam in (1, 2) for k in (0, 1)
        ],
        "query": [(1, 1, 0), (2, 1, 0), (3, 1, 0)],
        "bounding_box_test": [(1, 2, 0), (2, 2, 0), (3, 2, 0), (4, 2, 0)],
    }
    for subdir, entries in layout.items():
        folder = root / subdir
        folder.mkdir(parents=True)
        for pid, cam, k in entries:
            color = ((pid * 40) % 256, (cam * 90) % 256, (k * 120) % 256)
            name = f"{pid:04d}_c{cam}s1_{151 + k:06d}_00.jpg"
            Image.new("RGB", (WIDTH, HEIGHT), color).save(folder / name, format="JPEG")


def _write_config(path: Path) -> None:
    """Write a tiny but valid training config."""
    cfg = Config()
    cfg.data.height, cfg.data.width = HEIGHT, WIDTH
    cfg.data.batch_size, cfg.data.num_instances = 4, 2
    cfg.data.num_workers = 0
    cfg.data.pad = 2
    cfg.model.pretrained = False
    cfg.loss.center_loss = True
    cfg.train.log_period = 1
    cfg.to_yaml(path)


@pytest.fixture(autouse=True)
def _restore_global_state() -> Iterator[None]:
    """Undo the CLI's process-wide side effects after each test.

    The CLI toggles PyTorch's deterministic mode and attaches a ``log.txt``
    handler to the ``reid`` logger. Both are restored so later tests are not
    affected and ``tmp_path`` can be removed on Windows.
    """
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    yield
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    reid_logger = logging.getLogger("reid")
    for handler in list(reid_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            reid_logger.removeHandler(handler)
            handler.close()


@pytest.fixture
def tiny_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the ResNet-50 builder with the tiny stand-in model."""

    def fake_build_model(cfg: Config, num_classes: int, **_: Any) -> nn.Module:
        return TinyReIDModel(num_classes)

    monkeypatch.setattr(train_cli, "build_model", fake_build_model)


def _run(tmp_path: Path, *extra: str) -> tuple[int, Path]:
    """Run the CLI on a fresh synthetic dataset and return its exit code and output dir."""
    data_root = tmp_path / "market"
    _write_market(data_root)
    config = tmp_path / "tiny.yaml"
    _write_config(config)
    out = tmp_path / "run"
    argv = [
        "--config",
        str(config),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(out),
        "--device",
        "cpu",
        "--max-epochs",
        "1",
        "--no-rerank",
        *extra,
    ]
    return train_cli.main(argv), out


@pytest.mark.usefixtures("tiny_model")
def test_main_trains_end_to_end(tmp_path: Path) -> None:
    """One epoch writes the final weights, metrics, checkpoints and history."""
    code, out = _run(tmp_path, "--no-deterministic")
    assert code == 0

    for name in ("model_final.pth", "results.json", "best.pth", "last.pth", "history.json"):
        assert (out / name).is_file(), name

    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert {"mAP", "rank1", "rank5", "rank10"} <= set(results)
    assert not any(key.endswith("cmc") for key in results)
    assert 0.0 <= results["mAP"] <= 1.0

    final = torch.load(out / "model_final.pth", map_location="cpu", weights_only=True)
    assert final["meta"]["num_classes"] == 6
    assert final["meta"]["mAP"] == pytest.approx(results["mAP"])
    assert final["meta"]["config"]["data"]["root"] is None
    assert final["meta"]["config"]["train"]["output_dir"] == "run"
    assert final["meta"]["config"]["train"]["deterministic"] is False

    saved = Config.from_yaml(out / "config.yaml")
    assert saved.train.max_epochs == 1


@pytest.mark.usefixtures("tiny_model")
def test_main_with_validation_split(tmp_path: Path) -> None:
    """Holding out identities for validation still yields test-split results."""
    code, out = _run(tmp_path, "--val-ids", "2")
    assert code == 0
    assert (out / "best.pth").is_file()
    assert (out / "results.json").is_file()


def test_invalid_override_fails_cleanly(tmp_path: Path) -> None:
    """An override that breaks validation returns exit code 1 before any work."""
    config = tmp_path / "tiny.yaml"
    _write_config(config)
    code = train_cli.main(["--config", str(config), "--max-epochs", "0"])
    assert code == 1


def test_missing_config_fails_cleanly(tmp_path: Path) -> None:
    """A config path that does not exist returns exit code 1."""
    assert train_cli.main(["--config", str(tmp_path / "missing.yaml")]) == 1


def test_parser_defaults_resolve_the_shipped_config() -> None:
    """``--config`` defaults to ``None`` and the other overrides are unset."""
    args = train_cli.build_parser().parse_args([])
    assert args.config is None
    assert args.val_ids is None
    assert args.no_deterministic is False
