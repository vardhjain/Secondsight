"""Test suite for the :mod:`reid` person re-identification package.

The tests are split into two tiers:

* **Light tests** depend only on ``numpy``, ``torch`` and ``Pillow`` and must run
  in a minimal environment (no ``torchvision``, ``cv2``, ``gradio`` or
  ``kagglehub`` installed). These cover the configuration system, the dataset /
  sampler metadata logic, the loss functions, the distance utilities, the
  Market-1501 metrics, and the k-reciprocal re-ranking.
* **Heavy tests** exercise code paths that need ``torchvision`` (the ResNet-50
  backbone). They are guarded with :func:`pytest.importorskip` so the suite still
  passes when ``torchvision`` is absent.

Everything is intentionally tiny and CPU-only so the suite runs in seconds.
"""
