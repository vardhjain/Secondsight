"""Tests for the ``reid-evaluate`` and ``reid-visualize`` command-line interfaces.

The CLIs are driven through ``main([...])`` on the tiny fake Market-1501 from
``conftest.py`` with a checkpoint written into ``tmp_path``. Every model is
built with ``pretrained=False`` so nothing is downloaded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchvision")

from reid.cli import evaluate as eval_cli  # noqa: E402
from reid.cli import visualize as viz_cli  # noqa: E402
from reid.config import Config  # noqa: E402
from reid.models.reid_model import build_model  # noqa: E402
from reid.utils.checkpoint import save_model  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_global_state(monkeypatch: pytest.MonkeyPatch):
    """Undo the logger and determinism changes that ``main`` makes globally."""
    monkeypatch.delenv("REID_DATA_ROOT", raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    reid_logger = logging.getLogger("reid")
    handlers = reid_logger.handlers[:]
    propagate, level = reid_logger.propagate, reid_logger.level
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    for handler in reid_logger.handlers[:]:
        if handler not in handlers:
            reid_logger.removeHandler(handler)
            handler.close()
    reid_logger.propagate, reid_logger.level = propagate, level
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


def _tiny_config() -> Config:
    cfg = Config()
    cfg.data.height, cfg.data.width = 32, 16
    cfg.data.batch_size, cfg.data.num_instances, cfg.data.num_workers = 4, 2, 0
    cfg.train.device = "cpu"
    cfg.eval.rerank_k1, cfg.eval.rerank_k2, cfg.eval.max_rank = 2, 1, 3
    return cfg


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.yaml"
    _tiny_config().to_yaml(path)
    return path


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    cfg = _tiny_config()
    torch.manual_seed(0)
    model = build_model(cfg, num_classes=4, pretrained=False)
    path = tmp_path / "model_final.pth"
    save_model(model, path, config=cfg.to_dict())
    return path


def test_evaluate_parser_defaults() -> None:
    args = eval_cli.build_parser().parse_args(["--weights", "w.pth"])
    assert args.rerank is None
    assert args.config is None
    assert eval_cli.build_parser().parse_args(["--weights", "w", "--no-rerank"]).rerank is False
    with pytest.raises(SystemExit):
        eval_cli.build_parser().parse_args([])
    with pytest.raises(SystemExit):
        eval_cli.build_parser().parse_args(["--weights", "w", "--rerank", "--no-rerank"])


def test_evaluate_error_exits(tmp_path: Path, config_file: Path, fake_market_root: Path) -> None:
    missing = str(tmp_path / "missing.pth")
    assert eval_cli.main(["--config", str(tmp_path / "nope.yaml"), "--weights", missing]) == 1
    assert eval_cli.main(["--config", str(config_file), "--weights", missing]) == 1
    args = ["--config", str(config_file), "--data-root", str(fake_market_root)]
    assert eval_cli.main([*args, "--weights", missing]) == 1
    garbage = tmp_path / "garbage.pth"
    garbage.write_bytes(b"not a checkpoint")
    assert eval_cli.main([*args, "--weights", str(garbage)]) == 1


def test_evaluate_end_to_end(
    tmp_path: Path,
    config_file: Path,
    weights: Path,
    fake_market_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REID_DATA_ROOT", str(fake_market_root))
    out = tmp_path / "out" / "results.json"
    argv = ["--config", str(config_file), "--weights", str(weights), "--results-json", str(out)]
    assert eval_cli.main([*argv, "--rerank"]) == 0
    metrics = json.loads(out.read_text(encoding="utf-8"))
    assert {"mAP", "rank1", "rerank_mAP"} <= metrics.keys()
    assert "cmc" not in metrics
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_visualize_rejects_missing_weights(
    tmp_path: Path, config_file: Path, fake_market_root: Path
) -> None:
    code = viz_cli.main(
        [
            "--config", str(config_file),
            "--data-root", str(fake_market_root),
            "--weights", str(tmp_path / "missing.pth"),
            "--output", str(tmp_path / "figs"),
        ]
    )  # fmt: skip
    assert code == 1
    assert not (tmp_path / "figs").exists()


def test_visualize_end_to_end(
    tmp_path: Path, config_file: Path, weights: Path, fake_market_root: Path
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    out = tmp_path / "figs"
    code = viz_cli.main(
        [
            "--config", str(config_file),
            "--data-root", str(fake_market_root),
            "--weights", str(weights),
            "--output", str(out),
            "--num-gradcam", "1",
            "--rerank",
        ]
    )  # fmt: skip
    assert code == 0
    for name in ("cmc_curve", "ap_distribution", "camera_heatmap", "distance_distributions"):
        assert (out / f"{name}.png").is_file(), name
    assert (out / "ranked_results.png").is_file()
    assert (out / "success_failure.png").is_file()


def test_safe_plot_swallows_failures(caplog: pytest.LogCaptureFixture) -> None:
    def missing() -> None:
        raise ImportError("no cv2")

    def broken() -> None:
        raise RuntimeError("boom")

    assert viz_cli._safe_plot("a", missing) is False
    assert viz_cli._safe_plot("b", broken) is False
    assert viz_cli._safe_plot("c", lambda: None) is False
    assert viz_cli._safe_plot("d", lambda: object()) is True
