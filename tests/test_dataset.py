"""Tests for the Market-1501 dataset (:mod:`reid.data.dataset`).

The :class:`reid.data.dataset.Market1501` class deliberately avoids importing
``torchvision``, so the core metadata and parsing tests are light and run with
only ``Pillow`` installed. The single test that applies a real transform pipeline
imports ``torchvision`` via :func:`pytest.importorskip` and is skipped when it is
unavailable.

A tiny on-disk fake dataset (the ``fake_market_root`` and ``fake_market_spec``
fixtures) backs these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from reid.data.dataset import ImageListDataset, Market1501, parse_market_filename

# Local import of the fixture dataclass for type-checked attribute access.
from tests.conftest import FakeMarketSpec


def test_train_subset_metadata(fake_market_spec: FakeMarketSpec) -> None:
    """The train subset parses pids, drops junk, and builds a label map."""
    dataset = Market1501(fake_market_spec.root, subset="train")

    expected_count = len(fake_market_spec.train_pids) * fake_market_spec.train_images_per_id
    assert len(dataset) == expected_count
    assert -1 not in dataset.pids

    # Contiguous label map over the sorted unique identities.
    assert dataset.pid2label == {pid: i for i, pid in enumerate(fake_market_spec.train_pids)}
    assert dataset.num_classes == len(fake_market_spec.train_pids)


def test_train_subset_drops_real_style_junk(fake_market_root: Path) -> None:
    """Junk files named ``-1_...`` (as in the real dataset) are dropped."""
    train_dir = fake_market_root / "bounding_box_train"
    assert any(p.name.startswith("-1_") for p in train_dir.iterdir())
    dataset = Market1501(fake_market_root, subset="train")
    assert all(not p.name.startswith("-1_") for p in dataset.img_paths)


def test_query_subset_uses_raw_pids(fake_market_spec: FakeMarketSpec) -> None:
    """Non-train subsets expose an empty label map and raw-pid targets."""
    dataset = Market1501(fake_market_spec.root, subset="query")
    assert dataset.pid2label == {}
    assert tuple(dataset.pids) == fake_market_spec.query_pids
    assert dataset.num_classes == 2

    _, label, camid = dataset[0]
    # For non-train subsets the returned label is the raw pid.
    assert (label, camid) == (1, 1)


def test_gallery_drops_junk_and_keeps_distractors(fake_market_spec: FakeMarketSpec) -> None:
    """The gallery drops ``-1`` junk but keeps pid ``0000`` distractors."""
    dataset = Market1501(fake_market_spec.root, subset="gallery")
    assert tuple(dataset.pids) == fake_market_spec.gallery_pids
    assert 0 in dataset.pids
    assert -1 not in dataset.pids
    assert len(dataset) == 5


def test_getitem_returns_pil_image_without_transform(fake_market_root: Path) -> None:
    """Without a transform, ``__getitem__`` returns an RGB ``PIL.Image``."""
    dataset = Market1501(fake_market_root, subset="train")
    img, label, camid = dataset[0]
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert isinstance(label, int)
    assert isinstance(camid, int)


def test_getitem_label_is_contiguous_for_train(fake_market_spec: FakeMarketSpec) -> None:
    """Train labels are the contiguous ``pid2label`` values, not raw pids."""
    dataset = Market1501(fake_market_spec.root, subset="train")
    labels = {dataset[idx][1] for idx in range(len(dataset))}
    assert labels == set(range(len(fake_market_spec.train_pids)))


def test_camera_ids_parsed(fake_market_spec: FakeMarketSpec) -> None:
    """Camera ids are parsed from the filename and lie in the expected set."""
    dataset = Market1501(fake_market_spec.root, subset="train")
    assert set(dataset.camids) == set(fake_market_spec.train_cams)


def test_scan_skips_non_matching_files(fake_market_root: Path) -> None:
    """Files that do not follow the naming convention or are not JPEGs are ignored."""
    train_dir = fake_market_root / "bounding_box_train"
    Image.new("RGB", (4, 4)).save(train_dir / "readme.jpg", format="JPEG")
    Image.new("RGB", (4, 4)).save(train_dir / "0001_c1s1_000001_00.png", format="PNG")
    (train_dir / "Thumbs.db").write_bytes(b"")

    dataset = Market1501(fake_market_root, subset="train")
    names = {p.name for p in dataset.img_paths}
    assert "readme.jpg" not in names
    assert "0001_c1s1_000001_00.png" not in names
    assert len(dataset) == 16


def test_invalid_subset_raises_value_error(fake_market_root: Path) -> None:
    """An unknown subset name raises :class:`ValueError`."""
    with pytest.raises(ValueError, match="Unknown subset"):
        Market1501(fake_market_root, subset="not_a_subset")


def test_missing_directory_raises_file_not_found(tmp_path: Path) -> None:
    """A root without the expected sub-directory raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        Market1501(tmp_path / "does_not_exist", subset="train")


def test_empty_subset_raises_value_error(tmp_path: Path) -> None:
    """A subset folder without Market-1501 images fails fast with ``ValueError``."""
    (tmp_path / "bounding_box_train").mkdir()
    with pytest.raises(ValueError, match="No Market-1501 images"):
        Market1501(tmp_path, subset="train")


def test_img_paths_are_retrievable(fake_market_root: Path) -> None:
    """The raw image path for each sample is exposed via ``img_paths``."""
    dataset = Market1501(fake_market_root, subset="train")
    assert len(dataset.img_paths) == len(dataset)
    assert all(p.suffix == ".jpg" for p in dataset.img_paths)
    assert all(p.exists() for p in dataset.img_paths)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("0002_c1s1_000451_03.jpg", (2, 1)),
        ("-1_c3s2_001234_01.jpg", (-1, 3)),
        ("0000_c6s4_002202_07.jpg", (0, 6)),
        ("0123_c12s1_000001_00.jpg", (123, 12)),
        ("readme.jpg", None),
        ("x0002_c1s1_000451_03.jpg", None),
    ],
)
def test_parse_market_filename(name: str, expected: tuple[int, int] | None) -> None:
    """Filenames are parsed into ``(pid, camid)`` or rejected."""
    assert parse_market_filename(name) == expected


def test_image_list_dataset_relabels_only_when_asked(fake_market_spec: FakeMarketSpec) -> None:
    """``ImageListDataset`` maps identities to contiguous labels only when asked to."""
    source = Market1501(fake_market_spec.root, subset="train")
    keep = [i for i, pid in enumerate(source.pids) if pid in (2, 4)]
    paths = [source.img_paths[i] for i in keep]
    pids = [source.pids[i] for i in keep]
    camids = [source.camids[i] for i in keep]

    relabelled = ImageListDataset(paths, pids, camids, relabel=True)
    assert relabelled.pid2label == {2: 0, 4: 1}
    assert {relabelled[i][1] for i in range(len(relabelled))} == {0, 1}

    raw = ImageListDataset(paths, pids, camids)
    assert raw.pid2label == {}
    assert {raw[i][1] for i in range(len(raw))} == {2, 4}


def test_image_list_dataset_rejects_mismatched_lengths() -> None:
    """Misaligned path, pid and camid lists raise ``ValueError``."""
    with pytest.raises(ValueError, match="equal lengths"):
        ImageListDataset(["a.jpg"], [1, 2], [1])


def test_transform_pipeline_yields_tensor(fake_market_root: Path) -> None:
    """With a real transform pipeline ``__getitem__`` returns a tensor.

    This exercises the integration between the dataset and the torchvision-backed
    transforms, so it is skipped when ``torchvision`` is not installed.
    """
    pytest.importorskip("torchvision")
    from reid.config import DataConfig
    from reid.data.transforms import build_transforms

    cfg = DataConfig(height=32, width=16, pad=2)
    transform = build_transforms(cfg, is_train=False)
    dataset = Market1501(fake_market_root, subset="train", transform=transform)

    img, _, _ = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, cfg.height, cfg.width)
