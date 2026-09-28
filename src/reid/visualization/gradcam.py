"""Grad-CAM attention visualization for Re-ID models.

This module ports the Grad-CAM implementation from the original research
notebook into a reusable, hook-safe :class:`GradCAM` class. Grad-CAM
(Gradient-weighted Class Activation Mapping) highlights the spatial regions
of an input image that most strongly influence the model's feature response,
giving an interpretable view of *where* the network "looks" when computing a
person's embedding.

For Re-ID there is no single classification logit to back-propagate from at
inference time (the model returns an embedding), so, following the notebook,
the scalar used to seed the backward pass is the sum of the output feature
activations. This produces a class-agnostic saliency map over the embedding.

Grad-CAM rendering (resizing the activation map and building the colour
overlay) uses OpenCV, which ships in the optional ``viz`` extra. ``cv2`` is
imported only when :class:`GradCAM` is called or :func:`overlay_heatmap` is
used, so importing this module does not require it, and a missing install
raises an :class:`ImportError` that names the extra.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from reid.visualization._common import finalize_figure, pyplot, require

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from matplotlib.figure import Figure
    from torch import Tensor, nn
    from torch.utils.data import Dataset
    from torch.utils.hooks import RemovableHandle


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
            model: The trained Re-ID model. It is switched to eval mode while
                the saliency map is computed and restored afterwards.
            target_layer: The convolutional module to hook. For the default
                ResNet-50 backbone this is ``model.backbone.layer4`` (or the
                equivalent last-stage feature map).
        """
        self.model = model
        self.target_layer = target_layer
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None
        self._handles: list[RemovableHandle] = []
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

        The call also works inside ``torch.no_grad()`` or
        ``torch.inference_mode()`` blocks, because gradients are re-enabled
        locally. The model's train or eval mode is restored afterwards and the
        parameter gradients produced by the backward pass are released.

        Args:
            input_tensor: A pre-processed image tensor of shape ``[1, 3, H, W]``
                on the same device as the model. A batch size of exactly 1 is
                expected; the first element is used if a larger batch is given.

        Returns:
            A ``float32`` numpy array of shape ``[H, W]`` with values in
            ``[0, 1]``, where ``H`` and ``W`` are the spatial dimensions of the
            input tensor. Higher values indicate regions of greater importance.

        Raises:
            ImportError: If OpenCV is not installed.
            RuntimeError: If the hooks captured no activations or gradients.
        """
        cv2 = require("cv2", "Grad-CAM")

        was_training = self.model.training
        self.model.eval()
        self._activations = None
        self._gradients = None
        try:
            with torch.inference_mode(False), torch.enable_grad():
                # A fresh leaf tensor lets gradients flow back to the activations.
                inp = input_tensor.clone().detach().requires_grad_(True)
                output = self.model(inp)
                if isinstance(output, (tuple, list)):
                    # In training mode the model may return (cls_score, feat);
                    # use the feature component for a class-agnostic map.
                    output = output[-1]
                self.model.zero_grad(set_to_none=True)
                # Class-agnostic seed: sum of the embedding activations.
                output.sum().backward()
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

        if self._activations is None or self._gradients is None:
            msg = "Grad-CAM hooks did not capture activations/gradients; check the target layer."
            raise RuntimeError(msg)

        # First sample only: [C, H, W].
        grads = self._gradients[0].float().cpu().numpy()
        fmap = self._activations[0].float().cpu().numpy()

        # Channel weights are the global-average-pooled gradients. The map is
        # the weighted sum of activation channels passed through a ReLU.
        weights = grads.mean(axis=(1, 2))
        cam = np.maximum(np.tensordot(weights, fmap, axes=1), 0.0).astype(np.float32)

        # Resize back to the input spatial size (cv2 expects (width, height)).
        target_h, target_w = int(input_tensor.shape[2]), int(input_tensor.shape[3])
        cam = cv2.resize(cam, (target_w, target_h))

        # Normalize to [0, 1].
        cam -= cam.min()
        cam_max = float(cam.max())
        if cam_max != 0.0:
            cam /= cam_max
        return np.asarray(cam, dtype=np.float32)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlays a Grad-CAM heatmap onto an RGB image.

    The heatmap is colorized with the JET colormap and alpha-blended with the
    original image. The heatmap is resized to the image's dimensions if needed.

    Args:
        image: RGB image as a numpy array of shape ``[H, W, 3]``. Values may be
            in ``[0, 1]`` (float) or ``[0, 255]`` (uint8); the output follows
            the float ``[0, 1]`` convention.
        heatmap: Single-channel saliency map of shape ``[H, W]`` with values in
            ``[0, 1]`` (as produced by :class:`GradCAM`).
        alpha: Blending weight for the heatmap in ``[0, 1]``. ``0`` returns the
            original image and ``1`` returns the pure heatmap.

    Returns:
        A ``float32`` RGB image of shape ``[H, W, 3]`` with values in
        ``[0, 1]`` representing the blended overlay.

    Raises:
        ImportError: If OpenCV is not installed.
    """
    cv2 = require("cv2", "overlay_heatmap")

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

    # Colorize: cv2 produces BGR, so convert to RGB and scale to [0, 1].
    heatmap_uint8 = np.uint8(255 * hmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = (1.0 - alpha) * img + alpha * colored
    return np.clip(overlay, 0.0, 1.0).astype(np.float32)


def plot_gradcam_samples(
    model: nn.Module,
    dataset: Dataset,
    num_samples: int = 5,
    save_path: str | Path | None = None,
    *,
    target_layer: nn.Module | None = None,
    device: torch.device | str = "cpu",
    seed: int | None = 0,
) -> Figure | None:
    """Render Grad-CAM overlays for a random sample of dataset images.

    The figure has two rows, the original crops on top and their Grad-CAM
    overlays underneath, with one column per sampled image.

    Args:
        model: The Re-ID model, already on ``device``.
        dataset: A sized dataset yielding normalized ``(tensor, pid, camid)``
            samples, such as :class:`~reid.data.dataset.Market1501`.
        num_samples: Number of images to render, clamped to the dataset size.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.
        target_layer: Layer to hook. Defaults to ``model.backbone.layer4``.
        device: Device on which to run the forward and backward passes.
        seed: Seed for the image sampling.

    Returns:
        The matplotlib figure, or ``None`` when there is nothing to render.

    Raises:
        ImportError: If matplotlib or OpenCV is not installed.
    """
    from reid.visualization.ranking import _denormalize_to_image

    plt = pyplot()
    require("cv2", "Grad-CAM")

    size = len(dataset)  # type: ignore[arg-type]
    num_samples = min(num_samples, size)
    if num_samples <= 0:
        return None
    if target_layer is None:
        backbone = cast("nn.Module", model.backbone)
        target_layer = cast("nn.Module", backbone.layer4)
    rng = np.random.default_rng(seed)
    indices = rng.choice(size, size=num_samples, replace=False)

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.4, 5.2), squeeze=False)
    try:
        with GradCAM(model, target_layer) as cam:
            for col, idx in enumerate(indices):
                img_t, pid, camid = dataset[int(idx)]
                heatmap = cam(img_t.unsqueeze(0).to(device))
                original = _denormalize_to_image(img_t)

                axes[0, col].imshow(original)
                axes[0, col].set_title(f"ID {int(pid)}\nCam {int(camid)}", fontsize=9)
                axes[0, col].axis("off")

                axes[1, col].imshow(overlay_heatmap(original, heatmap, alpha=0.5))
                axes[1, col].set_title("Grad-CAM", fontsize=9)
                axes[1, col].axis("off")
    except BaseException:
        plt.close(fig)
        raise

    fig.suptitle("Grad-CAM Attention", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return finalize_figure(fig, save_path)
