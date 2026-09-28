"""Tests for the dataset download CLI (:mod:`reid.cli.download`).

``kagglehub`` is replaced by a stub module, so these tests never touch the
network and do not need the ``data`` extra installed.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reid.cli import download
from reid.cli.download import _is_valid_root, _resolve_data_root, build_parser, main


def _stub_kagglehub(monkeypatch: pytest.MonkeyPatch, path: Path) -> list[str]:
    """Install a fake ``kagglehub`` whose download returns ``path``.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        path: Directory the fake download reports.

    Returns:
        A list that records every requested dataset slug.
    """
    calls: list[str] = []

    def dataset_download(slug: str) -> str:
        calls.append(slug)
        return str(path)

    monkeypatch.setitem(
        sys.modules, "kagglehub", SimpleNamespace(dataset_download=dataset_download)
    )
    return calls


def _nest(market: Path, parent: Path) -> Path:
    """Move a fake Market-1501 root under ``parent`` using the canonical folder name.

    Args:
        market: A complete fake Market-1501 root.
        parent: Directory that will contain ``Market-1501-v15.09.15``.

    Returns:
        The new data root.
    """
    target = parent / "Market-1501-v15.09.15"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(market), str(target))
    return target


def test_resolve_data_root_layouts(fake_market_root: Path, tmp_path: Path) -> None:
    """The canonical folder is found directly, nested, or falls back to the path."""
    direct = tmp_path / "direct"
    root = _nest(fake_market_root, direct)
    assert _resolve_data_root(direct) == root

    nested = tmp_path / "nested"
    moved = _nest(root, nested / "versions" / "1")
    assert _resolve_data_root(nested) == moved

    plain = tmp_path / "plain"
    plain.mkdir()
    assert _resolve_data_root(plain) == plain


def test_is_valid_root_requires_populated_splits(fake_market_root: Path, tmp_path: Path) -> None:
    """Empty or partial split folders are rejected."""
    assert _is_valid_root(fake_market_root)

    shell = tmp_path / "shell"
    for name in ("bounding_box_train", "query", "bounding_box_test"):
        (shell / name).mkdir(parents=True)
    assert not _is_valid_root(shell)

    for path in (fake_market_root / "query").iterdir():
        path.unlink()
    assert not _is_valid_root(fake_market_root)


def test_main_prints_only_the_bare_root(
    fake_market_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stdout holds exactly the data root, and logs go to stderr."""
    calls = _stub_kagglehub(monkeypatch, fake_market_root)
    assert main([]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"{fake_market_root}\n"
    assert "Market-1501 data root" in captured.err
    assert calls == ["pengcw1/market-1501"]


def test_main_fails_on_invalid_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A download without the expected folders exits non-zero and prints nothing."""
    _stub_kagglehub(monkeypatch, tmp_path)
    assert main([]) == 2
    assert capsys.readouterr().out == ""


def test_main_fails_when_download_raises(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A kagglehub error exits with status 1."""

    def dataset_download(slug: str) -> str:
        raise ConnectionError(slug)

    monkeypatch.setitem(
        sys.modules, "kagglehub", SimpleNamespace(dataset_download=dataset_download)
    )
    assert main([]) == 1
    assert capsys.readouterr().out == ""


def test_main_reports_missing_kagglehub(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without kagglehub the CLI names the extra to install and exits with status 1."""
    monkeypatch.setitem(sys.modules, "kagglehub", None)
    assert main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "secondsight[data]" in captured.err


def test_main_copies_to_output(
    fake_market_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--output`` copies the used folders, skips the unused ones, and reuses a copy."""
    (fake_market_root / "gt_bbox").mkdir()
    (fake_market_root / "gt_bbox" / "0001_c1s1_000001_00.jpg").write_bytes(b"")
    _stub_kagglehub(monkeypatch, fake_market_root)
    destination = tmp_path / "out" / "market"

    assert main(["--output", str(destination)]) == 0
    assert capsys.readouterr().out == f"{destination.resolve()}\n"
    assert _is_valid_root(destination)
    assert not (destination / "gt_bbox").exists()
    assert not destination.with_name("market.partial").exists()

    # A second run reuses the complete copy.
    assert main(["--output", str(destination)]) == 0
    assert capsys.readouterr().out == f"{destination.resolve()}\n"


def test_main_copies_into_empty_output(
    fake_market_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An existing empty directory (for example a Docker bind mount) is filled."""
    _stub_kagglehub(monkeypatch, fake_market_root)
    destination = tmp_path / "data"
    destination.mkdir()
    assert main(["--output", str(destination)]) == 0
    assert capsys.readouterr().out == f"{destination.resolve()}\n"
    assert _is_valid_root(destination)


def test_main_rejects_partial_output(
    fake_market_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-empty but incomplete destination is left alone and the CLI fails."""
    _stub_kagglehub(monkeypatch, fake_market_root)
    destination = tmp_path / "partial"
    for name in ("bounding_box_train", "query", "bounding_box_test"):
        (destination / name).mkdir(parents=True)
    assert main(["--output", str(destination)]) == 2
    assert capsys.readouterr().out == ""
    assert not any((destination / "query").iterdir())


def test_interrupted_copy_leaves_no_valid_looking_output(
    fake_market_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy that fails midway leaves neither the destination nor a partial folder."""

    def failing_copytree(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(download.shutil, "copytree", failing_copytree)
    _stub_kagglehub(monkeypatch, fake_market_root)
    destination = tmp_path / "out" / "market"
    assert main(["--output", str(destination)]) == 1
    assert not destination.exists()
    assert not destination.with_name("market.partial").exists()


def test_parser_defaults() -> None:
    """``--output`` is optional."""
    assert build_parser().parse_args([]).output is None


def test_script_shim_exposes_main() -> None:
    """The repository script re-exports the library entry point."""
    from scripts.download_data import main as shim_main

    assert shim_main is main


def test_network_guard_blocks_weight_downloads() -> None:
    """The autouse conftest guard makes pretrained weight downloads fail loudly."""
    import urllib.request

    import torch.hub

    with pytest.raises(RuntimeError, match="network access is disabled"):
        torch.hub.load_state_dict_from_url("https://example.invalid/weights.pth")
    with pytest.raises(RuntimeError, match="network access is disabled"):
        urllib.request.urlopen("https://example.invalid")
    torchvision_models = pytest.importorskip("torchvision.models")
    with pytest.raises(RuntimeError, match="network access is disabled"):
        torchvision_models.resnet18(weights="IMAGENET1K_V1")
