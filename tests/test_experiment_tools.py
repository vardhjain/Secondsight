"""No-dataset/no-GPU coverage for the experiment planning and benchmark tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from reid.config import Config
from scripts.benchmark_retrieval import benchmark, main as benchmark_main
from scripts.plan_ablations import build_plan, main as plan_main
from scripts.report_ablations import render_report

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "market1501_strong_baseline.yaml"


def test_plan_is_deterministic_and_only_writes_configs(tmp_path: Path) -> None:
    """Three training variants and four evaluation variants per seed."""
    output = tmp_path / "ablations"
    plan = build_plan(BASE_CONFIG, output, seeds=[43, 42])
    assert plan["schema_version"] == 1
    assert plan["val_ids"] == 0
    assert len(plan["runs"]) == 6
    assert [run["id"] for run in plan["runs"]] == [
        "baseline_seed42", "avg_pool_seed42", "no_center_loss_seed42",
        "baseline_seed43", "avg_pool_seed43", "no_center_loss_seed43",
    ]
    assert (output / "manifest.json").is_file()
    for run in plan["runs"]:
        assert len(run["evaluations"]) == 4
        assert not Path(run["weights"]).exists()
        training = Config.from_yaml(run["train_config"])
        assert training.train.seed == run["seed"]
        assert training.data.root is None
        assert training.eval.rerank is False
        assert training.model.pooling == (
            "avg" if run["variant"] == "avg_pool" else "gem"
        )
        assert training.loss.center_loss == (run["variant"] != "no_center_loss")
        assert run["train_argv"][0] == "reid-train"
        for evaluation in run["evaluations"]:
            config = Config.from_yaml(evaluation["config"])
            assert config.model.pooling == training.model.pooling
            assert config.eval.flip_tta == evaluation["flip_tta"]
            assert config.eval.rerank == evaluation["rerank"]
            assert evaluation["argv"][0] == "reid-evaluate"
            assert not Path(evaluation["results"]).exists()


def test_plan_validates_inputs_and_does_not_overwrite(tmp_path: Path) -> None:
    for seeds in ([], [42, 42], [-1]):
        with pytest.raises(ValueError, match="Seeds"):
            build_plan(BASE_CONFIG, tmp_path / "invalid", seeds)
    with pytest.raises(ValueError, match="val_ids"):
        build_plan(BASE_CONFIG, tmp_path / "invalid", [42], -1)
    root = tmp_path / "plan"
    build_plan(BASE_CONFIG, root, [42], val_ids=1)
    assert Config.from_yaml(root / "baseline_seed42" / "train.yaml").data.val_ids == 1
    with pytest.raises(ValueError, match="empty"):
        build_plan(BASE_CONFIG, root, [42])


def test_report_marks_missing_results_and_reads_only_recorded_files(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan = build_plan(BASE_CONFIG, plan_dir, [42])
    report = tmp_path / "report.md"
    status = render_report(plan_dir / "manifest.json", report)
    assert status == {"recorded": 0, "total": 12}
    assert "0/12" in report.read_text(encoding="utf-8")
    trial = plan["runs"][0]["evaluations"][0]
    path = Path(trial["results"])
    path.write_text(
        json.dumps({"rerank_mAP": 0.25, "rerank_rank1": 0.5}) + "\n",
        encoding="utf-8",
    )
    status = render_report(plan_dir / "manifest.json", report)
    assert status == {"recorded": 1, "total": 12}
    output = report.read_text(encoding="utf-8")
    assert "25.00%" in output and "50.00%" in output
    assert "| pending |" in output
    path.write_text(json.dumps({"mAP": 0.25, "rank1": 0.5}), encoding="utf-8")
    with pytest.raises(ValueError, match="rerank_mAP"):
        render_report(plan_dir / "manifest.json", report)


def test_synthetic_cpu_benchmark_records_actual_timings_and_memory_bound() -> None:
    out = benchmark(queries=4, gallery=8, dim=16, repeats=2, warmups=0, k1=3, k2=2)
    assert out["dataset"] == "synthetic_embeddings_only"
    assert out["estimated_dense_bytes_lower_bound"]["base_distance"] == 4 * 4 * 8
    assert out["estimated_dense_bytes_lower_bound"]["reranking"] == (
        6 * 12 * 12 + 4 * 4 * 8
    )
    for stage in ("base_distance", "reranking"):
        assert len(out["stages"][stage]["samples_ms"]) == 2
        assert out["stages"][stage]["median_ms"] >= 0.0
    with pytest.raises(ValueError, match="positive"):
        benchmark(queries=0)
    with pytest.raises(ValueError, match="512 MiB"):
        benchmark(queries=10_000, gallery=10_000)


def test_experiment_commands_run_without_dataset(tmp_path: Path) -> None:
    out = tmp_path / "plans"
    assert plan_main(["--config", str(BASE_CONFIG), "--output-dir", str(out)]) == 0
    bench_path = tmp_path / "bench.json"
    assert benchmark_main([
        "--queries", "4", "--gallery", "8", "--dim", "16",
        "--repeats", "1", "--warmups", "0", "--k1", "3", "--k2", "2",
        "--output", str(bench_path),
    ]) == 0
    assert json.loads(bench_path.read_text(encoding="utf-8"))["repeats"] == 1
