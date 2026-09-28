"""Image transform pipelines for Re-ID training and evaluation.

This module builds ``torchvision`` transform pipelines from a
:class:`reid.config.DataConfig`. ``torchvision`` is imported here (and not in
:mod:`reid.data.dataset`) to keep the dataset importable in lightweight
environments.

Train pipeline (strong baseline): resize, random horizontal flip, pad +
random crop (spatial jitter), tensor conversion, ImageNet normalisation and
optional random erasing. Test pipeline: resize, tensor conversion and
normalisation only.

Random erasing follows the Bag of Tricks recipe, erasing a rectangle that covers
between 2% and 40% of the image with an aspect ratio between 0.3 and 1/0.3. It
runs after normalisation and fills the erased pixels with 0 in normalised space,
which corresponds to the ImageNet mean pixel. The reference implementation
writes fixed per-channel constants instead, a difference with negligible effect.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reid.config import DataConfig

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "build_transforms"]

# ImageNet channel statistics used to normalise inputs for the pretrained
# ResNet backbone.
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# Random erasing area range (fraction of the image) and aspect ratio range, as in
# the reid-strong-baseline reference (sl=0.02, sh=0.4, r1=0.3).
_RE_SCALE: tuple[float, float] = (0.02, 0.4)
_RE_RATIO: tuple[float, float] = (0.3, 1 / 0.3)


def build_transforms(cfg: DataConfig, is_train: bool) -> Callable:
    """Build a transform pipeline for the requested phase.

    Args:
        cfg: Data configuration carrying image geometry and augmentation flags
            (``height``, ``width``, ``pad``, ``random_erasing``, ``re_prob``).
        is_train: When ``True`` the training pipeline (with augmentations) is
            returned; otherwise the deterministic evaluation pipeline.

    Returns:
        A callable mapping a ``PIL.Image`` to a normalised ``torch.Tensor`` of
        shape ``(3, cfg.height, cfg.width)``.
    """
    from torchvision import transforms

    size = (cfg.height, cfg.width)

    if is_train:
        pipeline: list[Callable] = [
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(cfg.pad),
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        if cfg.random_erasing:
            pipeline.append(
                transforms.RandomErasing(
                    p=cfg.re_prob,
                    scale=_RE_SCALE,
                    ratio=_RE_RATIO,
                    value=0,
                )
            )
        return transforms.Compose(pipeline)

    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
