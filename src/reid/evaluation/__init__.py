"""Evaluation subpackage for person re-identification.

The package exposes the Market-1501 evaluation building blocks. The functions
:func:`compute_cmc_map` and :func:`compute_ap_per_query` compute CMC and mAP
under the Market-1501 single-query protocol using NumPy alone. The function
:func:`extract_features` and the class :class:`Evaluator` handle feature
extraction (with optional flip TTA and L2 normalisation) and drive the
end-to-end evaluation. The function :func:`re_ranking` implements k-reciprocal
encoding re-ranking, and :func:`format_results_table` renders results as text.

Submodules are imported lazily on first attribute access (PEP 562), so
``import reid.evaluation.metrics`` needs only NumPy, while the evaluator and
re-ranking modules additionally require ``torch``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reid.evaluation.evaluator import Evaluator, extract_features
    from reid.evaluation.metrics import compute_ap_per_query, compute_cmc_map
    from reid.evaluation.reporting import format_results_table
    from reid.evaluation.rerank import re_ranking

_EXPORTS = {
    "Evaluator": "reid.evaluation.evaluator",
    "extract_features": "reid.evaluation.evaluator",
    "compute_ap_per_query": "reid.evaluation.metrics",
    "compute_cmc_map": "reid.evaluation.metrics",
    "format_results_table": "reid.evaluation.reporting",
    "re_ranking": "reid.evaluation.rerank",
}

__all__ = [
    "Evaluator",
    "compute_ap_per_query",
    "compute_cmc_map",
    "extract_features",
    "format_results_table",
    "re_ranking",
]


def __getattr__(name: str) -> Any:
    """Import the public attribute ``name`` from its submodule on first use."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the lazily exported names alongside the module globals."""
    return sorted(set(globals()) | set(__all__))
