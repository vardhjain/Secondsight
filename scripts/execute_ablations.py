"""Execute a prepared ablation manifest on user-supplied compute and data.

Dry-run is the default. Training is separate from benchmark-test evaluation
and execution requires an explicit --apply flag.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def planned_actions(
    manifest: dict[str, Any],
    *,
    stage: str,
    data_root: Path,
    device: str = "auto",
    max_runs: int | None = None,
) -> list[tuple[Path, list[str]]]:
    """Return missing output paths and their reproducible CLI commands."""
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("runs"), list):
        raise ValueError("Unsupported ablation manifest schema.")
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be >= 1.")
    jobs: list[tuple[Path, list[str]]] = []
    runs = manifest["runs"] if max_runs is None else manifest["runs"][:max_runs]
    for run in runs:
        weights = Path(run["weights"])
        if stage in ("train", "all") and not weights.is_file():
            jobs.append(
                (
                    weights,
                    [
                        *run["train_argv"],
                        "--data-root",
                        str(data_root),
                        "--device",
                        device,
                    ],
                )
            )
        if stage in ("evaluate", "all"):
            for item in run["evaluations"]:
                result = Path(item["results"])
                if not result.is_file():
                    jobs.append(
                        (
                            result,
                            [
                                *item["argv"],
                                "--data-root",
                                str(data_root),
                                "--device",
                                device,
                            ],
                        )
                    )
    return jobs


def main(argv: list[str] | None = None) -> int:
    """List or execute pending ablation commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="train")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--allow-test-evaluation",
        action="store_true",
        help="Explicit acknowledgment that repeated test evaluation is exploratory.",
    )
    args = parser.parse_args(argv)
    if args.stage != "train" and not args.allow_test_evaluation:
        parser.error(
            "Evaluation requires --allow-test-evaluation; do not select using test scores."
        )
    root_value = args.data_root or os.environ.get("REID_DATA_ROOT")
    if not root_value:
        parser.error("Provide --data-root or REID_DATA_ROOT.")
    root = Path(root_value)
    if args.apply and not root.is_dir():
        parser.error(f"Market-1501 data directory does not exist: {root}")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        actions = planned_actions(
            manifest,
            stage=args.stage,
            data_root=root,
            device=args.device,
            max_runs=args.max_runs,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(f"{len(actions)} pending {args.stage} commands.")
    for index, (output, command) in enumerate(actions, start=1):
        print(f"[{index}/{len(actions)}] {' '.join(command)}")
        if args.apply:
            if args.stage in ("evaluate", "all") and command[0] == "reid-evaluate":
                weights = Path(command[command.index("--weights") + 1])
                if not weights.is_file():
                    parser.error(f"Training output is missing: {weights}")
            subprocess.run(command, check=True)
            if not output.is_file():
                raise RuntimeError(f"Command completed without expected output: {output}")
    if not args.apply:
        print("Preview only; --apply runs the commands on your local compute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
