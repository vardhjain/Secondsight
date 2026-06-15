"""Evaluation subpackage for person re-identification.

Exposes the Market-1501 evaluation building blocks:

* :func:`compute_cmc_map` / :func:`compute_ap_per_query` -- NumPy-only CMC and
  mAP metrics under the Market-1501 single-query protocol.
* :func:`extract_features` / :class:`Evaluator` -- feature extraction (with
  optional flip TTA and L2 normalisation) and the end-to-end evaluation driver.
* :func:`re_ranking` -- robust k-reciprocal encoding re-ranking.

The metrics module is importable with only ``numpy`` available; the evaluator
and re-ranking modules additionally require ``torch``.
"""

from __future__ import annotations

from reid.evaluation.evaluator import Evaluator, extract_features
from reid.evaluation.metrics import compute_ap_per_query, compute_cmc_map
from reid.evaluation.reporting import format_results_table
from reid.evaluation.rerank import re_ranking

__all__ = [
    "Evaluator",
    "compute_ap_per_query",
    "compute_cmc_map",
    "extract_features",
    "format_results_table",
    "re_ranking",
]
