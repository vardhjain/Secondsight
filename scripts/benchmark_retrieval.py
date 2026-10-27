"""Synthetic, CPU-only retrieval latency benchmark for Secondsight.

This measures distance and re-ranking algorithms on randomly generated
embeddings; it does NOT benchmark the backbone or Market-1501 accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.functional import normalize

from reid.evaluation.rerank import re_ranking
from reid.utils.distance import compute_distance_matrix


def benchmark(
    queries: int = 64,
    gallery: int = 128,
    dim: int = 128,
    repeats: int = 5,
    warmups: int = 1,
    seed: int = 42,
    threads: int = 1,
    k1: int = 20,
    k2: int = 6,
    allow_large: bool = False,
) -> dict[str, Any]:
    """Measure retrieval stages separately with a fixed synthetic input seed.

    Timing is hardware-specific and not an accuracy or production SLA measure.
    The reported dense array footprint is a partial analytical lower bound,
    not measured peak memory or total allocator usage.
    """
    if min(queries, gallery, dim, repeats, threads, k1, k2) < 1:
        raise ValueError("All sizes, repeats, threads, k1 and k2 must be positive.")
    if warmups < 0 or seed < 0:
        raise ValueError("warmups and seed must be non-negative.")
    total = queries + gallery
    # float32 NxN distances, float16 NxN V, and float32 QxG output,
    # excluding activations, full/partial sorting and transient copies.
    rerank_min_bytes = 6 * total * total + 4 * queries * gallery
    if rerank_min_bytes > 512 * 1024 * 1024 and not allow_large:
        raise ValueError(
            "Dense re-ranking arrays exceed 512 MiB (lower bound); "
            "pass --allow-large only after checking available RAM."
        )

    torch.set_num_threads(threads)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    qf = normalize(torch.randn(queries, dim, generator=generator), dim=1)
    gf = normalize(torch.randn(gallery, dim, generator=generator), dim=1)
    results: dict[str, Any] = {}

    def run_stage(name: str) -> None:
        def operation() -> np.ndarray:
            if name == "base_distance":
                return compute_distance_matrix(qf, gf).numpy()
            return re_ranking(qf, gf, k1=k1, k2=k2)

        for _ in range(warmups):
            operation()
        durations: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            values = operation()
            durations.append((time.perf_counter() - start) * 1000)
            if values.shape != (queries, gallery) or not np.isfinite(values).all():
                raise RuntimeError(f"{name} returned invalid distances.")
        durations.sort()
        p95 = durations[math.ceil(0.95 * len(durations)) - 1]
        results[name] = {
            "median_ms": round(statistics.median(durations), 4),
            "p95_ms": round(p95, 4),
            "min_ms": round(durations[0], 4),
            "samples_ms": [round(t, 4) for t in durations],
        }

    run_stage("base_distance")
    run_stage("reranking")
    return {
        "schema_version": 1,
        "dataset": "synthetic_embeddings_only",
        "queries": queries,
        "gallery": gallery,
        "feature_dim": dim,
        "seed": seed,
        "repeats": repeats,
        "warmups": warmups,
        "threads": threads,
        "k1": k1,
        "k2": k2,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "estimated_dense_bytes_lower_bound": {
            "base_distance": 4 * queries * gallery,
            "reranking": rerank_min_bytes,
        },
        "memory_note": (
            "Analytical dense-array lower bounds, not measured peak RSS. "
            "They omit intermediate tensors, sorting buffers and allocator overhead."
        ),
        "timing_note": (
            "CPU-only, synthetic L2-normalized embeddings; excludes image decoding, "
            "model inference and data loading. Compare timings only on equivalent hardware."
        ),
        "stages": results,
    }


def main(argv: list[str] | None = None) -> int:
    """Run a reproducible CPU benchmark and persist JSON measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--gallery", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--k1", type=int, default=20)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/retrieval.json"))
    args = parser.parse_args(argv)
    try:
        result = benchmark(
            queries=args.queries,
            gallery=args.gallery,
            dim=args.dim,
            repeats=args.repeats,
            warmups=args.warmups,
            seed=args.seed,
            threads=args.threads,
            k1=args.k1,
            k2=args.k2,
            allow_large=args.allow_large,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Synthetic CPU benchmark saved: {args.output}; "
        f"base median {result['stages']['base_distance']['median_ms']:.3f} ms; "
        f"rerank median {result['stages']['reranking']['median_ms']:.3f} ms."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
