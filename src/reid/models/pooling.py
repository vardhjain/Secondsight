"""Global pooling layers for Re-ID feature extraction.

This module provides the pooling operations used to collapse a convolutional
feature map ``[N, C, H, W]`` produced by the backbone into a global feature
vector ``[N, C]``. Three variants are supported:

* **Average pooling** -- the classic global average pooling used by most
  ResNet-style classifiers.
* **Max pooling** -- global max pooling, occasionally useful as a baseline.
* **Generalized-Mean (GeM) pooling** -- a learnable interpolation between
  average and max pooling that consistently improves retrieval metrics and is
  the recommended default for this project.

All pooling modules expose the same contract: a callable mapping a
``[N, C, H, W]`` tensor to a ``[N, C]`` tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - conventional PyTorch alias
from torch import Tensor, nn


class GeMPooling(nn.Module):
    """Generalized-Mean (GeM) pooling.

    GeM pooling generalizes average and max pooling through a learnable
    exponent ``p``. For an input feature map, the pooled value of each channel
    is computed as::

        gem(x) = (mean(x ** p)) ** (1 / p)

    As ``p -> 1`` this reduces to average pooling and as ``p -> inf`` it
    approaches max pooling. The exponent is exposed as a learnable parameter so
    the network can adapt it during training.

    Reference:
        Radenovic et al., "Fine-tuning CNN Image Retrieval with No Human
        Annotation", TPAMI 2018.

    Attributes:
        p: The learnable Generalized-Mean exponent, stored as a single-element
            :class:`torch.nn.Parameter`.
        eps: Lower bound applied to the input before exponentiation to avoid
            numerical issues with zero or negative activations.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        """Initializes the GeM pooling layer.

        Args:
            p: Initial value of the Generalized-Mean exponent. Defaults to
                ``3.0``, a commonly used starting point.
            eps: Numerical-stability epsilon used to clamp the input prior to
                raising it to the power ``p``. Defaults to ``1e-6``.
        """
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Applies GeM pooling to a feature map.

        Args:
            x: Input feature map of shape ``[N, C, H, W]``.

        Returns:
            A tensor of shape ``[N, C]`` containing the pooled features.
        """
        clamped = x.clamp(min=self.eps).pow(self.p)
        pooled = F.adaptive_avg_pool2d(clamped, output_size=1)
        pooled = pooled.pow(1.0 / self.p)
        return pooled.flatten(1)

    def extra_repr(self) -> str:
        """Returns a human-readable representation of the layer's parameters."""
        return f"p={self.p.data.item():.4f}, eps={self.eps}"


class GlobalAvgPool(nn.Module):
    """Global average pooling mapping ``[N, C, H, W]`` to ``[N, C]``."""

    def forward(self, x: Tensor) -> Tensor:
        """Applies global average pooling.

        Args:
            x: Input feature map of shape ``[N, C, H, W]``.

        Returns:
            A tensor of shape ``[N, C]`` containing the pooled features.
        """
        return F.adaptive_avg_pool2d(x, output_size=1).flatten(1)


class GlobalMaxPool(nn.Module):
    """Global max pooling mapping ``[N, C, H, W]`` to ``[N, C]``."""

    def forward(self, x: Tensor) -> Tensor:
        """Applies global max pooling.

        Args:
            x: Input feature map of shape ``[N, C, H, W]``.

        Returns:
            A tensor of shape ``[N, C]`` containing the pooled features.
        """
        return F.adaptive_max_pool2d(x, output_size=1).flatten(1)


def build_pooling(name: str) -> nn.Module:
    """Builds a global pooling module by name.

    Args:
        name: Identifier of the pooling type. One of ``"avg"``, ``"gem"`` or
            ``"max"`` (case-insensitive).

    Returns:
        A :class:`torch.nn.Module` mapping ``[N, C, H, W]`` to ``[N, C]``.

    Raises:
        ValueError: If ``name`` does not correspond to a known pooling type.
    """
    key = name.strip().lower()
    if key == "avg":
        return GlobalAvgPool()
    if key == "gem":
        return GeMPooling()
    if key == "max":
        return GlobalMaxPool()
    raise ValueError(f"Unknown pooling type {name!r}. Expected one of 'avg', 'gem', 'max'.")
