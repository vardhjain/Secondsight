"""Tests for the transform pipelines (:mod:`reid.data.transforms`).

These tests need ``torchvision`` and are skipped when it is not installed.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

pytest.importorskip("torchvision")
from torchvision import transforms  # noqa: E402

from reid.config import DataConfig  # noqa: E402
from reid.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transforms  # noqa: E402


def _cfg(**overrides: object) -> DataConfig:
    """Return a tiny data configuration for fast transform tests.

    Args:
        **overrides: Field values that replace the defaults.

    Returns:
        A :class:`DataConfig` producing ``32 x 16`` images.
    """
    values: dict[str, object] = {"height": 32, "width": 16, "pad": 2}
    values.update(overrides)
    return DataConfig(**values)  # type: ignore[arg-type]


def test_eval_pipeline_normalises_with_imagenet_statistics() -> None:
    """A solid image at the ImageNet mean colour normalises to roughly zero."""
    mean_rgb = tuple(round(m * 255) for m in IMAGENET_MEAN)
    out = build_transforms(_cfg(), is_train=False)(Image.new("RGB", (20, 40), mean_rgb))
    assert out.shape == (3, 32, 16)
    assert out.abs().max().item() < 0.02

    white = build_transforms(_cfg(), is_train=False)(Image.new("RGB", (20, 40), (255, 255, 255)))
    expected = torch.tensor([(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)])
    torch.testing.assert_close(white[:, 0, 0], expected, atol=1e-5, rtol=0)


def test_eval_pipeline_is_deterministic() -> None:
    """The evaluation pipeline has no randomness."""
    image = Image.new("RGB", (50, 120), (30, 90, 150))
    pipeline = build_transforms(_cfg(), is_train=False)
    assert torch.equal(pipeline(image), pipeline(image))


@pytest.mark.parametrize("size", [(50, 120), (10, 10)])
def test_train_pipeline_output_shape(size: tuple[int, int]) -> None:
    """The training pipeline resizes any input to the configured geometry."""
    out = build_transforms(_cfg(), is_train=True)(Image.new("RGB", size, (10, 20, 30)))
    assert out.shape == (3, 32, 16)
    assert out.dtype == torch.float32


def test_train_pipeline_ends_with_bag_of_tricks_random_erasing() -> None:
    """Random erasing comes last, after normalisation, with the reference area range."""
    pipeline = build_transforms(_cfg(random_erasing=True, re_prob=0.7), is_train=True)
    steps = pipeline.transforms
    erasing = steps[-1]
    assert isinstance(erasing, transforms.RandomErasing)
    assert isinstance(steps[-2], transforms.Normalize)
    assert erasing.p == 0.7
    assert erasing.scale == (0.02, 0.4)
    assert erasing.ratio == pytest.approx((0.3, 1 / 0.3))
    assert erasing.value == 0


def test_train_pipeline_without_random_erasing() -> None:
    """Disabling random erasing removes it from the pipeline."""
    pipeline = build_transforms(_cfg(random_erasing=False), is_train=True)
    assert not any(isinstance(t, transforms.RandomErasing) for t in pipeline.transforms)


def test_train_pipeline_is_reproducible_under_seed() -> None:
    """Seeding torch reproduces the random augmentations exactly."""
    image = Image.new("RGB", (50, 120), (30, 90, 150))
    pipeline = build_transforms(_cfg(re_prob=1.0), is_train=True)
    torch.manual_seed(0)
    first = pipeline(image)
    torch.manual_seed(0)
    second = pipeline(image)
    assert torch.equal(first, second)
