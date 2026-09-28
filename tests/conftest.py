"""Shared pytest fixtures for the Re-ID test suite.

The central fixture here, :func:`fake_market_root`, materializes a tiny but
structurally faithful Market-1501 dataset on disk under ``tmp_path``. It writes a
handful of small valid JPEG images named with the canonical
``<pid>_c<camid>s<seq>_<frame>_<n>.jpg`` convention into the three expected
sub-directories (``bounding_box_train``, ``query`` and ``bounding_box_test``).
Like the real dataset it includes junk images named ``-1_...``, which must be
dropped, and pid ``0000`` distractors in the gallery, which must be kept.

Two autouse fixtures keep every test hermetic. One seeds the Python, NumPy and
torch random generators so tests are deterministic. The other blocks network
downloads (``torch.hub``, torchvision weight downloads and ``urllib``), so an
accidental download of pretrained weights fails loudly instead of silently
fetching hundreds of megabytes. Tests that genuinely need the network can opt
out with ``@pytest.mark.network``.

Only ``Pillow`` is required to create the images, so the fixture works in the
lightweight test environment (no ``torchvision``).
"""

from __future__ import annotations

import importlib
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest
import torch
from PIL import Image

# Logical subset name mapped to its on-disk Market-1501 sub-directory name.
_SUBSET_DIRS: dict[str, str] = {
    "train": "bounding_box_train",
    "query": "query",
    "gallery": "bounding_box_test",
}

NETWORK_DISABLED_MESSAGE = "network access is disabled in tests"

# Module attributes that perform downloads. Several libraries bind these names at
# import time, so each binding is patched where it lives rather than only at the
# source in ``torch.hub``.
_DOWNLOAD_TARGETS: tuple[tuple[str, str], ...] = (
    ("torch.hub", "load_state_dict_from_url"),
    ("torch.hub", "download_url_to_file"),
    ("torch.hub", "urlopen"),
    ("torch.utils.model_zoo", "load_url"),
    ("torchvision.models._api", "load_state_dict_from_url"),
    ("urllib.request", "urlopen"),
)


def pytest_configure(config: pytest.Config) -> None:
    """Register the custom markers used by this suite.

    Args:
        config: The pytest configuration object.
    """
    config.addinivalue_line(
        "markers", "network: the test may access the network (disables the download guard)"
    )


def _blocked(*_args: Any, **_kwargs: Any) -> NoReturn:
    """Stand-in for download functions that always refuses.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError(NETWORK_DISABLED_MESSAGE)


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every download entry point raise unless the test is marked ``network``.

    Args:
        request: The requesting test, inspected for the ``network`` marker.
        monkeypatch: Pytest monkeypatch fixture used to install the guards.
    """
    if request.node.get_closest_marker("network") is not None:
        return
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    for module_name, attribute in _DOWNLOAD_TARGETS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, attribute):
            monkeypatch.setattr(module, attribute, _blocked)


@pytest.fixture(autouse=True)
def _seed_rngs() -> Iterator[None]:
    """Seed the Python, NumPy and torch random generators before each test."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    yield


@dataclass(frozen=True)
class FakeMarketSpec:
    """Description of the synthetic Market-1501 dataset written to disk.

    Attributes:
        root: Filesystem root containing the three subset directories.
        train_pids: Sorted unique non-junk training identities.
        train_images_per_id: Number of training images written per identity.
        train_cams: Camera ids cycled through for the training images.
        query_pids: Identities of the query images, in file order.
        gallery_pids: Identities of the kept gallery images (distractors
            included, junk excluded), in file order.
        image_size: ``(width, height)`` of every generated JPEG.
    """

    root: Path
    train_pids: tuple[int, ...]
    train_images_per_id: int
    train_cams: tuple[int, ...]
    query_pids: tuple[int, ...]
    gallery_pids: tuple[int, ...]
    image_size: tuple[int, int]


def _write_jpeg(path: Path, color: tuple[int, int, int], size: tuple[int, int]) -> None:
    """Write a small solid-color RGB JPEG to ``path``.

    Args:
        path: Destination file path (parent directory must already exist).
        color: ``(r, g, b)`` fill color in ``[0, 255]``.
        size: ``(width, height)`` of the image in pixels.
    """
    Image.new("RGB", size, color).save(path, format="JPEG")


def _filename(pid: int, camid: int, seq: int, frame: int, n: int) -> str:
    """Build a canonical Market-1501 image filename.

    Junk images use the real dataset's ``-1_`` prefix, and every other identity
    is zero padded to four digits (so distractors start with ``0000_``).

    Args:
        pid: Person identity (``-1`` for junk, ``0`` for distractors).
        camid: Camera identity (1-indexed in real Market-1501).
        seq: Sequence index.
        frame: Frame number.
        n: Per-frame detection index.

    Returns:
        A filename of the form ``<pid>_c<camid>s<seq>_<frame>_<n>.jpg``.
    """
    pid_str = "-1" if pid == -1 else f"{pid:04d}"
    return f"{pid_str}_c{camid}s{seq}_{frame:06d}_{n:02d}.jpg"


@pytest.fixture
def fake_market_spec(tmp_path: Path) -> FakeMarketSpec:
    """Create a tiny on-disk fake Market-1501 dataset and describe it.

    The dataset is written under ``tmp_path/market`` with the following layout.

    * ``bounding_box_train`` holds identities ``{1, 2, 3, 4}`` with 4 images
      each, cameras alternating between ``1`` and ``2`` so every identity is seen
      by two cameras (suitable for the identity sampler and the validation
      split), plus two junk ``-1`` images that must be filtered out.
    * ``query`` holds one probe image each for identities ``1`` and ``2``.
    * ``bounding_box_test`` (the gallery) holds identities ``{1, 2, 3}`` under
      camera 2, two pid ``0000`` distractors that must be kept, and one junk
      ``-1`` image that must be dropped.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        A :class:`FakeMarketSpec` describing what was written.
    """
    root = tmp_path / "market"
    size = (16, 32)  # (width, height), tiny but non-degenerate.

    for subdir in _SUBSET_DIRS.values():
        (root / subdir).mkdir(parents=True, exist_ok=True)

    train_pids = (1, 2, 3, 4)
    train_images_per_id = 4
    train_cams = (1, 2)

    train_dir = root / _SUBSET_DIRS["train"]
    for i, pid in enumerate(train_pids):
        for k in range(train_images_per_id):
            camid = train_cams[k % len(train_cams)]
            color = ((pid * 37) % 256, (i * 53) % 256, (k * 29) % 256)
            _write_jpeg(train_dir / _filename(pid, camid, 1, 1000 + k, k), color, size)
    _write_jpeg(train_dir / _filename(-1, 1, 1, 9001, 0), (0, 0, 0), size)
    _write_jpeg(train_dir / _filename(-1, 2, 1, 9002, 0), (0, 0, 0), size)

    query_dir = root / _SUBSET_DIRS["query"]
    _write_jpeg(query_dir / _filename(1, 1, 1, 2001, 0), (200, 10, 10), size)
    _write_jpeg(query_dir / _filename(2, 1, 1, 2002, 0), (10, 200, 10), size)

    gallery_dir = root / _SUBSET_DIRS["gallery"]
    _write_jpeg(gallery_dir / _filename(0, 1, 1, 4001, 0), (5, 5, 5), size)
    _write_jpeg(gallery_dir / _filename(0, 3, 1, 4002, 1), (6, 6, 6), size)
    _write_jpeg(gallery_dir / _filename(1, 2, 1, 3001, 0), (210, 20, 20), size)
    _write_jpeg(gallery_dir / _filename(2, 2, 1, 3002, 0), (20, 210, 20), size)
    _write_jpeg(gallery_dir / _filename(3, 2, 1, 3003, 0), (20, 20, 210), size)
    _write_jpeg(gallery_dir / _filename(-1, 2, 1, 9003, 0), (0, 0, 0), size)

    return FakeMarketSpec(
        root=root,
        train_pids=train_pids,
        train_images_per_id=train_images_per_id,
        train_cams=train_cams,
        query_pids=(1, 2),
        gallery_pids=(0, 0, 1, 2, 3),
        image_size=size,
    )


@pytest.fixture
def fake_market_root(fake_market_spec: FakeMarketSpec) -> Path:
    """Return just the filesystem root of the fake Market-1501 dataset.

    A convenience wrapper around :func:`fake_market_spec` for tests that only
    need the path.

    Args:
        fake_market_spec: The full dataset specification fixture.

    Returns:
        The dataset root directory path.
    """
    return fake_market_spec.root
