"""Tests for the Market-1501 dataset (:mod:`reid.data.dataset`).

The :class:`reid.data.dataset.Market1501` class deliberately avoids importing
``torchvision``, so the core metadata / parsing tests are *light* and run with
only ``Pillow`` installed. The single test that applies a real transform pipeline
imports ``torchvision`` via :func:`pytest.importorskip` and is skipped when it is
unavailable.

A tiny on-disk fake dataset (the ``fake_market_root`` / ``fake_market_spec``
fixtures) backs these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from reid.data.dataset import Market1501

# Local import of the fixture dataclass for type-checked attribute access.
from tests.conftest import FakeMarketSpec


def test_train_subset_metadata(fake_market_spec: FakeMarketSpec) -> None:
    """The train subset parses pids, drops junk, and builds a label map."""
    dataset = Market1501(fake_market_spec.root, subset="train")

    expected_count = len(fake_market_spec.train_pids) * fake_market_spec.train_images_per_id
    assert len(dataset) == expected_count

    # Junk (pid == -1) images were filtered out.
    assert all(pid != -1 for pid in dataset.pids)

    # Contiguous label map over the sorted unique identities.
    assert dataset.pid2label == {pid: i for i, pid in enumerate(fake_market_spec.train_pids)}
    assert dataset.num_classes == len(fake_market_spec.train_pids)


def test_query_subset_uses_raw_pids(fake_market_root: Path) -> None:
    """Non-train subsets expose an empty label map and raw-pid targets."""
    dataset = Market1501(fake_market_root, subset="query")
    assert dataset.pid2label == {}
    assert dataset.num_classes == len(set(dataset.pids))

    _, label, _ = dataset[0]
    # For non-train subsets the returned label is the raw pid.
    assert label in dataset.pids


def test_gallery_subset_drops_junk(fake_market_root: Path) -> None:
    """The gallery subset (``bounding_box_test``) drops junk distractors."""
    dataset = Market1501(fake_market_root, subset="gallery")
    assert len(dataset) > 0
    assert all(pid != -1 for pid in dataset.pids)


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
    num_classes = len(fake_market_spec.train_pids)
    for idx in range(len(dataset)):
        _, label, _ = dataset[idx]
        assert 0 <= label < num_classes


def test_camera_ids_parsed(fake_market_spec: FakeMarketSpec) -> None:
    """Camera ids are parsed from the filename and lie in the expected set."""
    dataset = Market1501(fake_market_spec.root, subset="train")
    assert set(dataset.camids) == set(fake_market_spec.train_cams)


def test_invalid_subset_raises_value_error(fake_market_root: Path) -> None:
    """An unknown subset name raises :class:`ValueError`."""
    with pytest.raises(ValueError):
        Market1501(fake_market_root, subset="not_a_subset")


def test_missing_directory_raises_file_not_found(tmp_path: Path) -> None:
    """A root without the expected sub-directory raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        Market1501(tmp_path / "does_not_exist", subset="train")


def test_img_paths_are_retrievable(fake_market_root: Path) -> None:
    """The raw image path for each sample is exposed via ``img_paths``."""
    dataset = Market1501(fake_market_root, subset="train")
    assert len(dataset.img_paths) == len(dataset)
    assert all(p.suffix == ".jpg" for p in dataset.img_paths)
    assert all(p.exists() for p in dataset.img_paths)


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
