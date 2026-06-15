"""Human-readable rendering of evaluation results."""

from __future__ import annotations

from typing import Any


def format_results_table(results: dict[str, Any]) -> str:
    """Render an evaluation results dict as an aligned, monospace text table.

    Args:
        results: A metrics dict as returned by
            :meth:`reid.evaluation.evaluator.Evaluator.evaluate`. The re-ranked
            row is included only when ``rerank_*`` keys are present.

    Returns:
        A multi-line, monospace-friendly table string.
    """
    header = f"{'Setting':<14}{'mAP':>10}{'Rank-1':>10}{'Rank-5':>10}{'Rank-10':>10}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    lines.append(
        f"{'Baseline':<14}"
        f"{results['mAP']:>9.2%} "
        f"{results['rank1']:>9.2%} "
        f"{results['rank5']:>9.2%} "
        f"{results['rank10']:>9.2%}"
    )
    if "rerank_mAP" in results:
        lines.append(
            f"{'Re-ranked':<14}"
            f"{results['rerank_mAP']:>9.2%} "
            f"{results['rerank_rank1']:>9.2%} "
            f"{results['rerank_rank5']:>9.2%} "
            f"{results['rerank_rank10']:>9.2%}"
        )
    lines.append(sep)
    return "\n".join(lines)


__all__ = ["format_results_table"]
