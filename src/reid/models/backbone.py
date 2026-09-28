"""ResNet-50 backbone construction for Re-ID.

This module builds the convolutional feature extractor used by the Re-ID model.
It is a ResNet-50 truncated before the global pooling and fully-connected
classification head, so its forward pass returns the raw ``layer4`` feature map
of shape ``[N, 2048, h, w]``.

Two project-specific modifications are supported. Setting ``last_stride=1``
removes the stride in the first block of ``layer4``, which doubles the spatial
resolution of the final feature map (``16x8`` instead of ``8x4`` for a
``256x128`` input). This is a standard Re-ID trick that improves retrieval
accuracy at a modest compute cost. Setting ``ibn=True`` swaps in a
ResNet-50-IBN-a (Instance-Batch Normalization) loaded from ``torch.hub`` at a
pinned IBN-Net commit. Enabling it downloads and executes that pinned hub code,
and any failure to load it raises instead of silently building a plain
ResNet-50, so a saved config never claims an IBN backbone that was not used.

``torchvision`` is imported lazily inside :func:`build_backbone` so that simply
importing this module (or the top-level :mod:`reid` package) does not pull in
``torchvision``.
"""

from __future__ import annotations

import logging
from typing import Any

from torch import Tensor, nn

logger = logging.getLogger(__name__)

FEAT_DIM: int = 2048
"""Output channel dimension of a ResNet-50 ``layer4`` feature map."""

IBN_HUB_REPO: str = "XingangPan/IBN-Net:d1673389b36c1180cf9bc35ea8260d84046da915"
"""The ``torch.hub`` spec for IBN-Net, pinned to an audited commit."""

IBN_HUB_ENTRYPOINT: str = "resnet50_ibn_a"
"""Name of the ResNet-50-IBN-a entrypoint in the IBN-Net ``hubconf.py``."""

_SUPPORTED_LAST_STRIDES: tuple[int, ...] = (1, 2)


def _set_last_stride_one(layer4: nn.Sequential) -> None:
    """Sets the stride of the first ``layer4`` block to 1.

    This modifies the ``3x3`` convolution and the downsampling shortcut of the
    first bottleneck block in place so that ``layer4`` no longer downsamples,
    increasing the spatial resolution of the final feature map.

    Args:
        layer4: The ``layer4`` :class:`~torch.nn.Sequential` of a ResNet-50.
    """
    block = layer4[0]
    # Bottleneck downsampling happens in conv2 (the 3x3 conv) and in the
    # downsample shortcut. Setting both strides to 1 keeps the resolution.
    conv2 = getattr(block, "conv2", None)
    if isinstance(conv2, nn.Conv2d):
        conv2.stride = (1, 1)
    downsample = getattr(block, "downsample", None)
    if isinstance(downsample, nn.Sequential) and isinstance(downsample[0], nn.Conv2d):
        downsample[0].stride = (1, 1)


def _child(resnet: nn.Module, name: str) -> nn.Module:
    """Returns a named submodule of ``resnet``, failing clearly when absent.

    Args:
        resnet: A ResNet-style network.
        name: Attribute name of the submodule, for example ``"layer4"``.

    Returns:
        The submodule.

    Raises:
        TypeError: If ``resnet`` has no submodule called ``name``.
    """
    module = getattr(resnet, name, None)
    if not isinstance(module, nn.Module):
        raise TypeError(f"Backbone network has no submodule {name!r}.")
    return module


class ResNetFeatureExtractor(nn.Module):
    """Wraps a ResNet-50 to expose only its convolutional feature map.

    The wrapped network runs ``conv1``, ``bn1``, ``relu`` and ``maxpool``
    followed by ``layer1`` through ``layer4`` and returns the ``layer4``
    output. The original global average pool and fully-connected classifier
    are intentionally omitted.

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
                (``conv1``, ``bn1``, ``relu``, ``maxpool`` and ``layer1``
                through ``layer4``).
        """
        super().__init__()
        self.conv1: nn.Module = _child(resnet, "conv1")
        self.bn1: nn.Module = _child(resnet, "bn1")
        self.relu: nn.Module = _child(resnet, "relu")
        self.maxpool: nn.Module = _child(resnet, "maxpool")
        self.layer1: nn.Module = _child(resnet, "layer1")
        self.layer2: nn.Module = _child(resnet, "layer2")
        self.layer3: nn.Module = _child(resnet, "layer3")
        self.layer4: nn.Module = _child(resnet, "layer4")

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
    """Constructs a ResNet-50-IBN-a from ``torch.hub`` at a pinned commit.

    The repository is trusted explicitly with ``trust_repo=True`` because
    setting ``model.ibn`` is the opt-in. Without it, recent PyTorch releases ask
    for confirmation on stdin, which blocks interactive runs and fails in
    headless ones.

    Args:
        pretrained: Whether to request ImageNet-pretrained weights from the hub.

    Returns:
        A ResNet-50-IBN-a :class:`~torch.nn.Module`.

    Raises:
        TypeError: If the hub entrypoint does not return a module.
    """
    import torch

    model: Any = torch.hub.load(
        IBN_HUB_REPO,
        IBN_HUB_ENTRYPOINT,
        pretrained=pretrained,
        trust_repo=True,
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"torch.hub entrypoint {IBN_HUB_ENTRYPOINT!r} did not return a module.")
    return model


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
        last_stride: Stride of the first ``layer4`` block, either ``1`` (the
            stride is removed to increase the final feature-map resolution) or
            ``2`` (the standard ResNet-50 behavior).
        ibn: If ``True``, load a ResNet-50-IBN-a backbone from ``torch.hub`` at
            the pinned commit in :data:`IBN_HUB_REPO`.

    Returns:
        A tuple ``(feature_extractor, feat_dim)`` where ``feature_extractor`` is
        an :class:`~torch.nn.Module` mapping ``[N, 3, H, W]`` to
        ``[N, 2048, h, w]`` and ``feat_dim`` is ``2048``.

    Raises:
        ValueError: If ``name`` is not a supported backbone or ``last_stride``
            is neither ``1`` nor ``2``.
        RuntimeError: If ``ibn`` is ``True`` and the IBN backbone cannot be
            loaded.
    """
    if name != "resnet50":
        raise ValueError(f"Unsupported backbone {name!r}. Only 'resnet50' is supported.")
    if isinstance(last_stride, bool) or last_stride not in _SUPPORTED_LAST_STRIDES:
        raise ValueError(f"Unsupported last_stride {last_stride!r}; expected 1 or 2.")

    if ibn:
        try:
            resnet = _build_ibn_resnet50(pretrained)
        except Exception as exc:
            raise RuntimeError(
                "model.ibn is true but ResNet-50-IBN-a could not be loaded from torch.hub "
                f"({IBN_HUB_REPO}). Check network access, or set model.ibn to false to use "
                "the plain ResNet-50."
            ) from exc
        logger.info("Loaded ResNet-50-IBN-a backbone from torch.hub (%s).", IBN_HUB_REPO)
    else:
        resnet = _build_torchvision_resnet50(pretrained)

    if last_stride == 1:
        layer4 = _child(resnet, "layer4")
        if not isinstance(layer4, nn.Sequential):
            raise TypeError("Expected the backbone's layer4 to be an nn.Sequential.")
        _set_last_stride_one(layer4)

    feature_extractor = ResNetFeatureExtractor(resnet)
    return feature_extractor, FEAT_DIM
