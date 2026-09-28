"""Visualization subpackage for Person Re-Identification.

This package provides interpretability and analysis tooling for the Re-ID
system. It covers Grad-CAM attention maps, ranked-result galleries, Rank-1
success and failure panels, a before-versus-after re-ranking comparison, and a
suite of analysis plots (CMC curve, AP distribution, camera heatmap, t-SNE
embedding and distance distributions).

The plotting libraries ship in the optional ``viz`` extra
(``pip install "secondsight[viz]"``). They are imported only inside the
functions that use them, so importing this package needs nothing beyond the
core install, and calling a function whose dependency is missing raises an
:class:`ImportError` that names the extra.
"""

from __future__ import annotations

from reid.visualization.analysis import (
    plot_ap_distribution,
    plot_camera_heatmap,
    plot_cmc_curve,
    plot_distance_distributions,
    plot_tsne,
)
from reid.visualization.gradcam import GradCAM, overlay_heatmap, plot_gradcam_samples
from reid.visualization.ranking import (
    plot_rerank_impact,
    plot_success_failure,
    visualize_ranked_results,
)

__all__ = [
    "GradCAM",
    "overlay_heatmap",
    "plot_gradcam_samples",
    "visualize_ranked_results",
    "plot_success_failure",
    "plot_rerank_impact",
    "plot_cmc_curve",
    "plot_ap_distribution",
    "plot_camera_heatmap",
    "plot_tsne",
    "plot_distance_distributions",
]
