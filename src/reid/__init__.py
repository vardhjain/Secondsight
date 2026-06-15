"""Person Re-Identification (Re-ID) toolkit for the Market-1501 dataset.

This package provides a clean, config-driven implementation of a strong Re-ID
baseline: a ResNet-50 + BNNeck model trained with a combination of
label-smoothed cross-entropy, batch-hard triplet loss, and optional center
loss, evaluated with the standard CMC / mAP protocol and optional
k-reciprocal re-ranking.

The top-level package is intentionally lightweight: importing :mod:`reid` must
not pull in heavy third-party dependencies such as ``torchvision``, ``cv2``,
``gradio`` or ``kagglehub``. Those are imported lazily inside the submodules
that actually need them, so that lightweight code paths (for example
:mod:`reid.evaluation.metrics`) work with only ``numpy`` and ``torch``
installed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
