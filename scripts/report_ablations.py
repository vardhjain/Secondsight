"""Render an honest ablation results report from a plan and recorded metrics.

Missing results are marked pending; this command never invents scores.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _metric(value: Any, name: str, path: Path) -> float:
    """Check that a recorded metric is a valid fraction."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric {name} in {path}")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"Invalid {name}={result!r} in {path}")
    return result


def render_report(manifest_path: Path, output_path: Path) -> dict[str, int]:
    """Write a Markdown report; raise on malformed existing metric artifacts."""
    plan = json.loads(manifest_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or not isinstance(plan.get("runs"), list):
        raise ValueError("Unsupported ablation manifest schema.")
    observed: dict[tuple[int, str, str], tuple[float, float]] = {}
    rows: list[tuple[str, int, str, bool, str, str, str]] = []
    complete = 0
    for run in plan["runs"]:
        for trial in run["evaluations"]:
            file = Path(trial["results"])
            if not file.is_file():
                rows.append(
                    (run["variant"], run["seed"], trial["name"], trial["rerank"],
                     "pending", "—", "—")
                )
                continue
            result = json.loads(file.read_text(encoding="utf-8"))
            prefix = "rerank_" if trial["rerank"] else ""
            map_value = _metric(result.get(prefix + "mAP"), prefix + "mAP", file)
            rank1 = _metric(result.get(prefix + "rank1"), prefix + "rank1", file)
            observed[(run["seed"], run["variant"], trial["name"])] = (
                map_value, rank1
            )
            complete += 1
            rows.append(
                (run["variant"], run["seed"], trial["name"], trial["rerank"],
                 "recorded", f"{map_value * 100:.2f}%", f"{rank1 * 100:.2f}%")
            )

    lines = [
        "# Secondsight ablation report",
        "",
        f"**Recorded evaluations:** {complete}/{len(rows)}. "
        "Missing results are explicitly marked pending.",
        "",
        "These are metrics loaded from generated result files, not new "
        "measurements performed by the report generator. Comparisons are "
        "exploratory; the benchmark test partition must not be used for "
        "hyperparameter selection. See docs/EXPERIMENTS.md.",
        "",
        "| Training variant | Seed | Evaluation | Status | mAP | Rank-1 | "
        "Δ mAP vs matching baseline |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for variant, seed, name, _rerank, status, map_str, rank_str in rows:
        if status == "recorded":
            score = observed[(seed, variant, name)][0]
            baseline = observed.get((seed, "baseline", name))
            delta = f"{(score - baseline[0]) * 100:+.2f} pp" if baseline else "—"
        else:
            delta = "—"
        lines.append(
            f"| {variant} | {seed} | {name} | {status} | {map_str} | "
            f"{rank_str} | {delta} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Each delta compares the *same seed and evaluation setting* against "
        "the baseline. Training variants change one training factor; "
        "flip-TTA and re-ranking vary evaluation only. The comparison is "
        "not a confidence interval or a statistical significance test.",
        "",
        "## Provenance",
        "",
        f"Plan: \`{manifest_path}\`. Config hashes and exact CLI argument "
        "arrays are recorded in the manifest. No GPU run is required to "
        "produce this report; only existing result files are read.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return {"recorded": complete, "total": len(rows)}


def main(argv: list[str] | None = None) -> int:
    """Render Markdown from existing ablation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = render_report(args.manifest, args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Report: {args.output} ({summary['recorded']}/{summary['total']} recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
