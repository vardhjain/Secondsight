"""Tests for the results-table renderer (:mod:`reid.evaluation.reporting`)."""

from __future__ import annotations

from reid.evaluation.reporting import format_results_table

_BASELINE = {"mAP": 0.7, "rank1": 0.9, "rank5": 0.95, "rank10": 0.97}


def test_baseline_only_table() -> None:
    """A baseline-only result renders the baseline row and no re-ranked row."""
    table = format_results_table(_BASELINE)
    assert "Baseline" in table
    assert "Re-ranked" not in table
    # 0.70 formatted as a percentage with two decimals.
    assert "70.00%" in table
    assert "mAP" in table and "Rank-10" in table
    assert len({len(line) for line in table.splitlines()}) == 1


def test_table_includes_rerank_row_when_present() -> None:
    """Re-rank metrics add a second row to the table."""
    results = {
        **_BASELINE,
        "rerank_mAP": 0.88,
        "rerank_rank1": 0.92,
        "rerank_rank5": 0.96,
        "rerank_rank10": 0.98,
    }
    table = format_results_table(results)
    assert "Re-ranked" in table
    assert "88.00%" in table
    lines = table.splitlines()
    assert len(lines) == 6
    assert len({len(line) for line in lines}) == 1


def test_values_right_align_under_headers() -> None:
    """Each value ends in the same column as its header label."""
    lines = format_results_table(_BASELINE).splitlines()
    header, row = lines[1], lines[3]
    for label, value in (("mAP", "70.00%"), ("Rank-10", "97.00%")):
        assert header.index(label) + len(label) == row.index(value) + len(value)
