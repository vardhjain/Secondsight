"""Tests for the local Gradio demo in ``app/gradio_app.py``.

The engine runs on the tiny fake Market-1501 with ``pretrained=False`` models,
and only the UI smoke test imports gradio (with analytics disabled so it makes
no network calls).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from PIL import Image

pytest.importorskip("torchvision")

from app import gradio_app  # noqa: E402
from app.gradio_app import ReIDDemoEngine, build_parser, parse_auth, resolve_server  # noqa: E402

from reid.config import Config  # noqa: E402
from reid.models.reid_model import build_model  # noqa: E402
from reid.utils.checkpoint import save_model  # noqa: E402


def _tiny_config() -> Config:
    cfg = Config()
    cfg.data.height, cfg.data.width, cfg.data.batch_size = 32, 16, 2
    cfg.train.device = "cpu"
    return cfg


@pytest.fixture
def engine(fake_market_root: Path) -> ReIDDemoEngine:
    torch.manual_seed(0)
    return ReIDDemoEngine(
        _tiny_config(), None, fake_market_root, torch.device("cpu"), gallery_limit=2
    )


def test_server_defaults_follow_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRADIO_SERVER_NAME", raising=False)
    monkeypatch.delenv("GRADIO_SERVER_PORT", raising=False)
    args = build_parser().parse_args([])
    assert (args.server_name, args.server_port) == (None, None)
    assert resolve_server(None, None) == ("127.0.0.1", 7860)

    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "8080")
    assert resolve_server(None, None) == ("0.0.0.0", 8080)
    assert resolve_server("localhost", 9000) == ("localhost", 9000)

    monkeypatch.setenv("GRADIO_SERVER_PORT", "not-a-port")
    with pytest.raises(ValueError):
        resolve_server(None, None)


def test_parse_auth() -> None:
    assert parse_auth("u:p:q") == ("u", "p:q")
    for bad in ("admin", ":p", "u:", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_auth(bad)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--auth", "admin"])
    assert build_parser().parse_args(["--auth", "a:b"]).auth == ("a", "b")


def test_resolve_path_env_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("REID_WEIGHTS", raising=False)
    assert gradio_app._resolve_path(None, "REID_WEIGHTS") is None
    monkeypatch.setenv("REID_WEIGHTS", str(tmp_path / "w.pth"))
    assert gradio_app._resolve_path(None, "REID_WEIGHTS") == tmp_path / "w.pth"
    assert gradio_app._resolve_path(Path("x.pth"), "REID_WEIGHTS") == Path("x.pth")


def test_engine_subsamples_gallery_consistently(engine: ReIDDemoEngine) -> None:
    assert engine.gallery is not None and len(engine.gallery) == 2
    for path, pid in zip(engine.gallery.img_paths, engine.gallery.pids, strict=True):
        assert int(Path(path).name[:4]) == pid
    assert engine.gallery_features is not None
    assert engine.gallery_features.shape == (2, 2048)
    assert not engine.has_weights
    assert "random initialization" in engine.status_message


def test_search_orders_by_similarity(engine: ReIDDemoEngine) -> None:
    assert engine.search(None, 5) == ([], "Please upload a probe image.")
    items, message = engine.search(Image.new("RGB", (16, 32), (200, 10, 10)), 10)
    assert len(items) == 2
    assert [c.split(" |")[0] for _, c in items] == ["#1", "#2"]
    sims = [float(c.rsplit("sim ", 1)[1]) for _, c in items]
    assert sims == sorted(sims, reverse=True)
    assert "top-2" in message


def test_engine_without_gallery() -> None:
    engine = ReIDDemoEngine(_tiny_config(), None, None, torch.device("cpu"))
    items, message = engine.search(Image.new("RGB", (16, 32)), 3)
    assert items == []
    assert "not available" in message


def test_engine_loads_checkpoint_architecture(tmp_path: Path) -> None:
    trained = _tiny_config()
    trained.data.height, trained.data.width = 64, 32
    torch.manual_seed(0)
    path = tmp_path / "model_final.pth"
    save_model(build_model(trained, 751, pretrained=False), path, config=trained.to_dict())

    engine = ReIDDemoEngine(_tiny_config(), path, None, torch.device("cpu"))
    assert engine.has_weights
    assert engine.model.classifier.out_features == 751
    # The input size comes from the checkpoint, not from the passed config.
    assert (engine.cfg.data.height, engine.cfg.data.width) == (64, 32)
    assert engine.transform(Image.new("RGB", (10, 10))).shape == (3, 64, 32)


def test_build_demo_smoke(engine: ReIDDemoEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    gr = pytest.importorskip("gradio")
    assert isinstance(gradio_app.build_demo(engine), gr.Blocks)
