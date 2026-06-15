"""Visualization subpackage for Person Re-Identification.

This package provides interpretability and analysis tooling for the Re-ID
system, including Grad-CAM attention maps, ranked-result galleries, and a
suite of analysis plots (CMC curves, AP distributions, camera heatmaps,
t-SNE embeddings, and distance distributions).

The heavy/optional third-party dependencies used by these modules are kept
local to the submodule that needs them so that importing this package never
forces ``cv2``, ``sklearn``, ``seaborn`` or other optional libraries to be
present unless the corresponding function is actually called:

* :mod:`reid.visualization.gradcam` imports ``cv2`` lazily inside the
  functions/methods that require it.
* :mod:`reid.visualization.analysis` imports ``seaborn`` / ``sklearn``
  lazily inside the functions that require them.

Public API:
    * :class:`GradCAM`, :func:`overlay_heatmap`
    * :func:`visualize_ranked_results`, :func:`plot_success_failure`
    * :func:`plot_cmc_curve`, :func:`plot_ap_distribution`,
      :func:`plot_camera_heatmap`, :func:`plot_tsne`,
      :func:`plot_distance_distributions`
"""

from __future__ import annotations

from reid.visualization.analysis import (
    plot_ap_distribution,
    plot_camera_heatmap,
    plot_cmc_curve,
    plot_distance_distributions,
    plot_tsne,
)
from reid.visualization.gradcam import GradCAM, overlay_heatmap
from reid.visualization.ranking import plot_success_failure, visualize_ranked_results

__all__ = [
    "GradCAM",
    "overlay_heatmap",
    "visualize_ranked_results",
    "plot_success_failure",
    "plot_cmc_curve",
    "plot_ap_distribution",
    "plot_camera_heatmap",
    "plot_tsne",
    "plot_distance_distributions",
]
