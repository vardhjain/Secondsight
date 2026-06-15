"""ResNet-50 backbone construction for Re-ID.

This module builds the convolutional feature extractor used by the Re-ID model.
It is a ResNet-50 truncated before the global pooling and fully-connected
classification head, so its forward pass returns the raw ``layer4`` feature map
of shape ``[N, 2048, h, w]``.

Two project-specific modifications are supported:

* **last_stride=1** -- removing the stride in the first block of ``layer4``
  doubles the spatial resolution of the final feature map (e.g. ``16x8``
  instead of ``8x4`` for a ``256x128`` input). This is a standard Re-ID trick
  that improves retrieval accuracy at a modest compute cost.
* **IBN backbone** -- optionally load a ResNet-50-IBN-a model (Instance-Batch
  Normalization) from ``torch.hub``. If the hub model cannot be fetched (for
  example because there is no network access) the build falls back to a plain
  ResNet-50 and logs a warning.

``torchvision`` is imported lazily inside :func:`build_backbone` so that simply
importing this module (or the top-level :mod:`reid` package) does not pull in
``torchvision``.
"""

from __future__ import annotations

import logging

from torch import Tensor, nn

logger = logging.getLogger(__name__)

FEAT_DIM: int = 2048
"""Output channel dimension of a ResNet-50 ``layer4`` feature map."""


def _set_last_stride_one(layer4: nn.Module) -> None:
    """Sets the stride of the first ``layer4`` block to 1.

    This modifies the ``3x3`` convolution and the downsampling shortcut of the
    first bottleneck block in-place so that ``layer4`` no longer downsamples,
    increasing the spatial resolution of the final feature map.

    Args:
        layer4: The ``layer4`` :class:`~torch.nn.Sequential` of a ResNet-50.
    """
    block = layer4[0]
    # Bottleneck downsampling happens in conv2 (the 3x3 conv) and in the
    # downsample shortcut. Setting both strides to 1 keeps the resolution.
    if hasattr(block, "conv2") and block.conv2 is not None:
        block.conv2.stride = (1, 1)
    if getattr(block, "downsample", None) is not None:
        block.downsample[0].stride = (1, 1)


class ResNetFeatureExtractor(nn.Module):
    """Wraps a ResNet-50 to expose only its convolutional feature map.

    The wrapped network runs ``conv1 -> bn1 -> relu -> maxpool`` followed by
    ``layer1 .. layer4`` and returns the ``layer4`` output. The original global
    average pool and fully-connected classifier are intentionally omitted.

    Attributes:
        conv1: Initial ``7x7`` convolution.
        bn1: Batch-norm following ``conv1``.
        relu: ReLU activation following ``bn1``.
        maxpool: Initial ``3x3`` max-pool.
        layer1: First residual stage.
        layer2: Second residual stage.
        layer3: Third residual stage.
        layer4: Fourth residual stage (optionally with ``last_stride=1``).
    """

    def __init__(self, resnet: nn.Module) -> None:
        """Builds the feature extractor from a constructed ResNet-50.

        Args:
            resnet: A ResNet-50 module exposing the standard attribute names
                (``conv1``, ``bn1``, ``relu``, ``maxpool``, ``layer1`` ..
                ``layer4``).
        """
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: Tensor) -> Tensor:
        """Runs the convolutional stem and residual stages.

        Args:
            x: Input image batch of shape ``[N, 3, H, W]``.

        Returns:
            The ``layer4`` feature map of shape ``[N, 2048, h, w]``.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def _build_torchvision_resnet50(pretrained: bool) -> nn.Module:
    """Constructs a torchvision ResNet-50, with or without ImageNet weights.

    Args:
        pretrained: Whether to load ImageNet-pretrained weights.

    Returns:
        A torchvision ResNet-50 :class:`~torch.nn.Module`.
    """
    from torchvision import models

    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    return models.resnet50(weights=weights)


def _build_ibn_resnet50(pretrained: bool) -> nn.Module:
    """Attempts to construct a ResNet-50-IBN-a from ``torch.hub``.

    Args:
        pretrained: Whether to request pretrained weights from the hub.

    Returns:
        A ResNet-50-IBN-a :class:`~torch.nn.Module`.

    Raises:
        Exception: Propagated from :func:`torch.hub.load` if the model cannot
            be fetched or constructed. The caller is expected to catch this and
            fall back to a standard ResNet-50.
    """
    import torch

    return torch.hub.load(
        "XingangPan/IBN-Net",
        "resnet50_ibn_a",
        pretrained=pretrained,
    )


def build_backbone(
    name: str = "resnet50",
    pretrained: bool = True,
    last_stride: int = 1,
    ibn: bool = False,
) -> tuple[nn.Module, int]:
    """Builds a ResNet-50 feature-extractor backbone.

    Args:
        name: Backbone identifier. Only ``"resnet50"`` is currently supported.
        pretrained: Whether to initialize from ImageNet-pretrained weights.
        last_stride: Stride of the first ``layer4`` block. When set to ``1`` the
            stride is removed to increase the final feature-map resolution; any
            other value leaves the default stride of ``2`` untouched.
        ibn: If ``True``, attempt to load a ResNet-50-IBN-a backbone from
            ``torch.hub``. On failure, fall back to a standard ResNet-50 and log
            a warning.

    Returns:
        A tuple ``(feature_extractor, feat_dim)`` where ``feature_extractor`` is
        an :class:`~torch.nn.Module` mapping ``[N, 3, H, W]`` to
        ``[N, 2048, h, w]`` and ``feat_dim`` is ``2048``.

    Raises:
        ValueError: If ``name`` is not a supported backbone.
    """
    if name != "resnet50":
        raise ValueError(f"Unsupported backbone {name!r}. Only 'resnet50' is supported.")

    resnet: nn.Module | None = None
    if ibn:
        try:
            resnet = _build_ibn_resnet50(pretrained)
            logger.info("Loaded ResNet-50-IBN-a backbone from torch.hub.")
        except Exception as exc:  # noqa: BLE001 - any hub failure -> fallback
            logger.warning(
                "Failed to load ResNet-50-IBN-a backbone (%s); falling back to standard ResNet-50.",
                exc,
            )
            resnet = None

    if resnet is None:
        resnet = _build_torchvision_resnet50(pretrained)

    if last_stride == 1:
        _set_last_stride_one(resnet.layer4)

    feature_extractor = ResNetFeatureExtractor(resnet)
    return feature_extractor, FEAT_DIM
