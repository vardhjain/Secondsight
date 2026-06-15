"""Shared pytest fixtures for the Re-ID test suite.

The central fixture here, :func:`fake_market_root`, materializes a *tiny* but
structurally faithful Market-1501 dataset on disk under ``tmp_path``. It writes a
handful of small valid JPEG images named with the canonical
``<pid>_c<camid>s<seq>_<frame>_<n>.jpg`` convention into the three expected
sub-directories (``bounding_box_train``, ``query`` and ``bounding_box_test``),
including a couple of junk (``pid == -1``) distractors so the junk-filtering
logic is exercised.

Only ``Pillow`` is required to create the images, so the fixture works in the
lightweight test environment (no ``torchvision``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

# Logical subset name -> on-disk Market-1501 sub-directory name.
_SUBSET_DIRS: dict[str, str] = {
    "train": "bounding_box_train",
    "query": "query",
    "gallery": "bounding_box_test",
}


@dataclass(frozen=True)
class FakeMarketSpec:
    """Description of the synthetic Market-1501 dataset written to disk.

    Attributes:
        root: Filesystem root containing the three subset directories.
        train_pids: Sorted unique non-junk training identities.
        train_images_per_id: Number of training images written per identity.
        train_cams: Camera ids cycled through for the training images.
        image_size: ``(width, height)`` of every generated JPEG.
    """

    root: Path
    train_pids: tuple[int, ...]
    train_images_per_id: int
    train_cams: tuple[int, ...]
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

    Args:
        pid: Person identity (may be ``-1`` for junk).
        camid: Camera identity (1-indexed in real Market-1501).
        seq: Sequence index.
        frame: Frame number.
        n: Per-frame detection index.

    Returns:
        A filename of the form ``<pid>_c<camid>s<seq>_<frame>_<n>.jpg``.
    """
    return f"{pid:04d}_c{camid}s{seq}_{frame:06d}_{n:02d}.jpg"


@pytest.fixture
def fake_market_spec(tmp_path: Path) -> FakeMarketSpec:
    """Create a tiny on-disk fake Market-1501 dataset and describe it.

    Layout written under ``tmp_path/market``:

    * ``bounding_box_train``: identities ``{1, 2, 3, 4}`` with 4 images each,
      cameras cycled over ``{1, 2}`` so every identity appears under more than
      one camera (suitable for the identity sampler). Two junk ``-1`` images are
      added and must be filtered out.
    * ``query``: a couple of probe images for identities ``{1, 2}``.
    * ``bounding_box_test`` (gallery): images for identities ``{1, 2, 3}`` plus a
      junk ``-1`` distractor.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        A :class:`FakeMarketSpec` describing what was written.
    """
    root = tmp_path / "market"
    size = (16, 32)  # (width, height) -- tiny but non-degenerate.

    for subdir in _SUBSET_DIRS.values():
        (root / subdir).mkdir(parents=True, exist_ok=True)

    train_pids = (1, 2, 3, 4)
    train_images_per_id = 4
    train_cams = (1, 2)

    # --- Training images: 4 ids x 4 images, cameras alternating. ---
    train_dir = root / _SUBSET_DIRS["train"]
    for i, pid in enumerate(train_pids):
        for k in range(train_images_per_id):
            camid = train_cams[k % len(train_cams)]
            color = ((pid * 37) % 256, (i * 53) % 256, (k * 29) % 256)
            _write_jpeg(train_dir / _filename(pid, camid, 1, 1000 + k, k), color, size)
    # Junk distractors that must be dropped on load.
    _write_jpeg(train_dir / _filename(-1, 1, 1, 9001, 0), (0, 0, 0), size)
    _write_jpeg(train_dir / _filename(-1, 2, 1, 9002, 0), (0, 0, 0), size)

    # --- Query images: probes for ids 1 and 2 (camera 1). ---
    query_dir = root / _SUBSET_DIRS["query"]
    _write_jpeg(query_dir / _filename(1, 1, 1, 2001, 0), (200, 10, 10), size)
    _write_jpeg(query_dir / _filename(2, 1, 1, 2002, 0), (10, 200, 10), size)

    # --- Gallery images: ids 1, 2, 3 under camera 2 plus a junk distractor. ---
    gallery_dir = root / _SUBSET_DIRS["gallery"]
    _write_jpeg(gallery_dir / _filename(1, 2, 1, 3001, 0), (210, 20, 20), size)
    _write_jpeg(gallery_dir / _filename(2, 2, 1, 3002, 0), (20, 210, 20), size)
    _write_jpeg(gallery_dir / _filename(3, 2, 1, 3003, 0), (20, 20, 210), size)
    _write_jpeg(gallery_dir / _filename(-1, 2, 1, 9003, 0), (0, 0, 0), size)

    return FakeMarketSpec(
        root=root,
        train_pids=train_pids,
        train_images_per_id=train_images_per_id,
        train_cams=train_cams,
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
