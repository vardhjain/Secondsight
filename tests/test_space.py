"""Tests for the Hugging Face Space entry point in ``space/app.py``.

The module is loaded from its file path, which must have no side effects. The
helpers are exercised with a stub model, and checkpoint loading uses a tiny
``pretrained=False`` model written to ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from PIL import Image
from torch import nn

SPACE_APP = Path(__file__).resolve().parents[1] / "space" / "app.py"


@pytest.fixture
def space(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    monkeypatch.chdir(tmp_path)  # prove nothing depends on the working directory
    monkeypatch.delenv("REID_WEIGHTS", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    spec = importlib.util.spec_from_file_location("secondsight_space_app", SPACE_APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _Stub(nn.Module):
    """Returns queued feature vectors, one per call."""

    def __init__(self, *features: list[float]) -> None:
        super().__init__()
        self.features = [torch.tensor([f]) for f in features]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features.pop(0)


def _stub_loaded(space: ModuleType, *features: list[float]):
    return space.LoadedModel(_Stub(*features), lambda img: torch.zeros(3, 4, 2), flip_tta=False)


def test_import_has_no_side_effects(space: ModuleType) -> None:
    assert "demo" not in vars(space)
    assert space.get_model.cache_info().currsize == 0
    assert space.weights_path() == SPACE_APP.parent / "best.pth"
    examples_dir = space.EXAMPLES_DIR
    assert examples_dir == SPACE_APP.parent / "examples"


def test_weights_env_override(space: ModuleType, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REID_WEIGHTS", str(tmp_path / "w.pth"))
    assert space.weights_path() == tmp_path / "w.pth"


def test_compare_with_messages(space: ModuleType) -> None:
    img = Image.new("RGB", (8, 16))
    assert space.compare_with(None, img, img) == space.MISSING_WEIGHTS_MESSAGE
    assert space.compare_with(None, img, img, space.LOAD_FAILED_MESSAGE) == (
        space.LOAD_FAILED_MESSAGE
    )
    assert "both boxes" in space.compare_with(_stub_loaded(space), img, None)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], "1.000"),
        ([1.0, 0.0], [0.0, 1.0], "0.000"),
        ([1.0, 0.0], [-1.0, 0.0], "-1.000"),
    ],
)
def test_compare_with_reports_similarity_without_identity_verdict(
    space: ModuleType, first: list[float], second: list[float], expected: str
) -> None:
    """Similarity is descriptive, never a calibrated same-person verdict."""
    img = Image.new("RGB", (8, 16))
    output = space.compare_with(_stub_loaded(space, first, second), img, img)
    assert f"Cosine similarity: {expected} (range: -1 to +1)" in output
    assert "not a probability" in output
    assert "No decision threshold has been calibrated" in output
    assert "does not classify people" in output
    assert "Likely the SAME person" not in output
    assert "Likely DIFFERENT people" not in output
    assert "demo threshold" not in output


def test_space_has_no_identity_threshold(space: ModuleType) -> None:
    """The demo does not expose an arbitrary identity decision boundary."""
    assert not hasattr(space, "SIM_THRESHOLD")
    assert not hasattr(space, "similarity_verdict")


def test_pair_examples(space: ModuleType, tmp_path: Path) -> None:
    assert space.pair_examples(["a.jpg", "b.PNG", "c.jpg"]) == [["a.jpg", "b.PNG"]]
    assert space.pair_examples(["a.jpg"]) is None
    assert space.find_examples(tmp_path / "missing") is None
    for name in ("B.jpg", "a.png", "notes.txt", "c.webp"):
        (tmp_path / name).write_bytes(b"")
    pairs = space.find_examples(tmp_path)
    assert pairs == [[str(tmp_path / "a.png"), str(tmp_path / "B.jpg")]]


def test_load_missing_and_corrupt(space: ModuleType, tmp_path: Path) -> None:
    assert space.load(tmp_path / "none.pth") == (None, space.MISSING_WEIGHTS_MESSAGE)
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"garbage")
    assert space.load(bad) == (None, space.LOAD_FAILED_MESSAGE)


def test_load_checkpoint_uses_its_config(space: ModuleType, tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    from reid.config import Config
    from reid.models.reid_model import build_model
    from reid.utils.checkpoint import save_checkpoint

    cfg = Config()
    cfg.data.height, cfg.data.width = 64, 32
    cfg.model.pretrained = True  # must not trigger an ImageNet download on load
    torch.manual_seed(0)
    model = build_model(cfg, 5, pretrained=False)
    path = tmp_path / "best.pth"
    save_checkpoint({"epoch": 1, "model": model.state_dict(), "config": cfg.to_dict()}, path)

    loaded, error = space.load(path)
    assert error is None and loaded is not None
    assert loaded.transform(Image.new("RGB", (10, 10))).shape == (3, 64, 32)
    feat = space.embed(loaded, Image.new("RGB", (16, 32), (10, 20, 30)))
    assert feat.shape == (1, 2048)
    assert float(feat.norm()) == pytest.approx(1.0, abs=1e-5)


def test_build_demo_smoke(space: ModuleType) -> None:
    gr = pytest.importorskip("gradio")
    demo = space.build_demo(examples=None)
    assert isinstance(demo, gr.Interface)
    assert space.demo is space.demo  # lazily built once for reload tooling


def test_download_release_weights_validates_digest_and_cleans_up(
    space: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Download once; a bad checksum must not leave a partial weight file."""
    payload = b"test-pinned-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        space.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(payload)
    )
    target = tmp_path / "nested" / "best.pth"
    space.download_release_weights(target, expected_sha256=digest, expected_bytes=len(payload))
    assert target.read_bytes() == payload
    assert not list(target.parent.glob("*.part"))
    # An existing, manually provisioned checkpoint is not overwritten.
    space.download_release_weights(target, expected_sha256="bad", expected_bytes=1)
    assert target.read_bytes() == payload

    bad = tmp_path / "bad.pth"
    with pytest.raises(ValueError, match="checksum"):
        space.download_release_weights(bad, expected_sha256="0" * 64, expected_bytes=len(payload))
    assert not bad.exists()
    assert not list(tmp_path.glob("*.part"))


def test_hf_space_provisions_weights_only_when_running_on_hub(
    space: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No startup network access locally; a Space may fetch pinned weights."""
    target = tmp_path / "released.pth"
    monkeypatch.setenv("REID_WEIGHTS", str(target))
    assert space.get_model() == (None, space.MISSING_WEIGHTS_MESSAGE)
    space.get_model.cache_clear()
    monkeypatch.setenv("SPACE_ID", "example/secondsight")
    calls: list[Path] = []

    def fake_download(path: Path) -> None:
        calls.append(path)
        raise ValueError("test checksum mismatch")

    monkeypatch.setattr(space, "download_release_weights", fake_download)
    assert space.get_model() == (None, space.LOAD_FAILED_MESSAGE)
    assert calls == [target]
