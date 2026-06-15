"""Grad-CAM attention visualization for Re-ID models.

This module ports the Grad-CAM implementation from the original research
notebook into a reusable, hook-safe :class:`GradCAM` class. Grad-CAM
(Gradient-weighted Class Activation Mapping) highlights the spatial regions
of an input image that most strongly influence the model's feature response,
giving an interpretable view of *where* the network "looks" when computing a
person's embedding.

For Re-ID there is no single classification logit to back-propagate from at
inference time (the model returns an embedding), so — following the notebook —
the scalar used to seed the backward pass is the sum of the output feature
activations. This produces a class-agnostic saliency map over the embedding.

Grad-CAM rendering (resizing the activation map and building the color
overlay) is inherently OpenCV-specific, so ``cv2`` is imported lazily inside
the functions/methods that need it rather than at module scope. Importing
:mod:`reid.visualization.gradcam` therefore does not require OpenCV to be
installed; it is only needed when :class:`GradCAM` is called or
:func:`overlay_heatmap` is used.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor, nn


class GradCAM:
    """Grad-CAM saliency generator backed by forward/backward hooks.

    The instance registers a forward hook (to capture the activations of the
    target layer) and a full backward hook (to capture the gradients flowing
    into that layer). Calling the instance on an input tensor runs a forward
    and backward pass, combines gradients and activations into a class
    activation map, applies ReLU, and normalizes the result to ``[0, 1]``.

    Hooks are removed by :meth:`remove_hooks`. The class also supports the
    context-manager protocol so that hooks are guaranteed to be cleaned up::

        with GradCAM(model, model.backbone.layer4) as cam:
            heatmap = cam(input_tensor)

    Attributes:
        model: The Re-ID model to inspect.
        target_layer: The layer whose activations/gradients are captured
            (typically the last convolutional block, e.g. ``layer4``).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Initializes the Grad-CAM hooks on ``target_layer``.

        Args:
            model: The trained Re-ID model. It is switched to eval mode when
                the saliency map is computed.
            target_layer: The convolutional module to hook. For the default
                ResNet-50 backbone this is ``model.backbone.layer4`` (or the
                equivalent last-stage feature map).
        """
        self.model = model
        self.target_layer = target_layer
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None
        self._handles: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Registers the forward and backward hooks on the target layer."""

        def forward_hook(_module: nn.Module, _inp: tuple, output: Tensor) -> None:
            self._activations = output.detach()

        def backward_hook(_module: nn.Module, _grad_in: tuple, grad_out: tuple) -> None:
            self._gradients = grad_out[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Removes all registered hooks.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> GradCAM:
        """Enters the context manager, returning ``self``."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exits the context manager, removing all hooks."""
        self.remove_hooks()

    def __call__(self, input_tensor: Tensor) -> np.ndarray:
        """Computes the Grad-CAM heatmap for a single input image.

        Args:
            input_tensor: A pre-processed image tensor of shape ``[1, 3, H, W]``
                on the same device as the model. A batch size of exactly 1 is
                expected; the first element is used if a larger batch is given.

        Returns:
            A ``float32`` numpy array of shape ``[H, W]`` with values in
            ``[0, 1]``, where ``H`` and ``W`` are the spatial dimensions of the
            input tensor. Higher values indicate regions of greater importance.
        """
        import cv2

        self.model.eval()

        # Ensure gradients can flow back to the captured activations.
        inp = input_tensor.clone().detach().requires_grad_(True)

        # Reset any state from a previous call.
        self._activations = None
        self._gradients = None

        output = self.model(inp)
        if isinstance(output, (tuple, list)):
            # In training mode the model may return (cls_score, global_feat);
            # use the feature/embedding component for a class-agnostic map.
            output = output[-1]

        # Class-agnostic seed: sum of the embedding activations.
        score = output.sum()
        self.model.zero_grad(set_to_none=True)
        if inp.grad is not None:
            inp.grad = None
        score.backward()

        if self._activations is None or self._gradients is None:
            msg = "Grad-CAM hooks did not capture activations/gradients; check the target layer."
            raise RuntimeError(msg)

        # First sample only: [C, H, W].
        grads = self._gradients[0].cpu().numpy()
        fmap = self._activations[0].cpu().numpy()

        # Channel weights = global-average-pooled gradients.
        weights = np.mean(grads, axis=(1, 2))

        # Weighted combination of activation maps.
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for channel_idx, weight in enumerate(weights):
            cam += weight * fmap[channel_idx]

        # ReLU: keep only features with a positive influence.
        cam = np.maximum(cam, 0.0)

        # Resize back to the input spatial size (cv2 expects (width, height)).
        target_h, target_w = int(inp.shape[2]), int(inp.shape[3])
        cam = cv2.resize(cam, (target_w, target_h))

        # Normalize to [0, 1].
        cam -= cam.min()
        cam_max = float(cam.max())
        if cam_max != 0.0:
            cam /= cam_max

        return cam.astype(np.float32)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlays a Grad-CAM heatmap onto an RGB image.

    The heatmap is colorized with the JET colormap and alpha-blended with the
    original image. Both inputs are matched in spatial size by resizing the
    heatmap to the image's dimensions if necessary.

    Args:
        image: RGB image as a numpy array of shape ``[H, W, 3]``. Values may be
            in ``[0, 1]`` (float) or ``[0, 255]`` (uint8); the output matches
            the float ``[0, 1]`` convention.
        heatmap: Single-channel saliency map of shape ``[H, W]`` with values in
            ``[0, 1]`` (as produced by :class:`GradCAM`).
        alpha: Blending weight for the heatmap in ``[0, 1]``. ``0`` returns the
            original image; ``1`` returns the pure heatmap.

    Returns:
        A ``float32`` RGB image of shape ``[H, W, 3]`` with values in
        ``[0, 1]`` representing the blended overlay.
    """
    import cv2

    img = image.astype(np.float32)
    # Normalize uint8-style images to [0, 1].
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    target_h, target_w = img.shape[0], img.shape[1]
    hmap = heatmap.astype(np.float32)
    if hmap.shape[:2] != (target_h, target_w):
        hmap = cv2.resize(hmap, (target_w, target_h))
    hmap = np.clip(hmap, 0.0, 1.0)

    # Colorize: cv2 produces BGR, convert to RGB and scale to [0, 1].
    heatmap_uint8 = np.uint8(255 * hmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = (1.0 - alpha) * img + alpha * colored
    return np.clip(overlay, 0.0, 1.0).astype(np.float32)
