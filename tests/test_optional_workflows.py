"""No-network tests for free Space deployment and resumable ablation commands."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.deploy_space import FILES, publish
from scripts.deploy_space import main as deploy_main
from scripts.execute_ablations import main as execute_main
from scripts.execute_ablations import planned_actions
from scripts.plan_ablations import build_plan

ROOT = Path(__file__).resolve().parents[1]


class FakeHub:
    """Capture the exact allowed Space API operations."""

    def __init__(self, hardware: str = "zero-a10g") -> None:
        self.hardware = hardware
        self.created: list[dict[str, object]] = []
        self.files: list[str] = []

    def create_repo(self, **kwargs) -> None:
        self.created.append(kwargs)

    def get_space_runtime(self, repo_id: str) -> SimpleNamespace:
        return SimpleNamespace(requested_hardware=self.hardware, hardware=self.hardware)

    def upload_file(self, **kwargs) -> None:
        self.files.append(kwargs["path_in_repo"])


def test_space_deployment_is_zerogpu_only_and_dry_run_by_default() -> None:
    assert deploy_main(["--repo-id", "owner/secondsight"]) == 0
    api = FakeHub()
    result = publish(api, "owner/secondsight")
    assert result == "https://huggingface.co/spaces/owner/secondsight"
    assert api.created[0]["space_hardware"] == "zero-a10g"
    assert set(api.files) == set(FILES)
    bad = FakeHub("t4-medium")
    with pytest.raises(RuntimeError, match="Refusing"):
        publish(bad, "owner/secondsight")
    assert bad.files == []


def test_ablation_actions_are_resume_safe_and_require_test_opt_in(tmp_path: Path) -> None:
    plan_dir = tmp_path / "ablations"
    manifest = build_plan(
        ROOT / "configs" / "market1501_strong_baseline.yaml", plan_dir, [42, 123, 456]
    )
    train = planned_actions(manifest, stage="train", data_root=tmp_path, max_runs=2)
    assert len(train) == 2
    assert all(argv[0] == "reid-train" for _, argv in train)
    train[0][0].write_bytes(b"simulated output marker")
    assert len(planned_actions(manifest, stage="train", data_root=tmp_path, max_runs=2)) == 1
    eval_jobs = planned_actions(manifest, stage="evaluate", data_root=tmp_path, max_runs=1)
    assert len(eval_jobs) == 4
    assert all(argv[0] == "reid-evaluate" for _, argv in eval_jobs)
    assert (
        execute_main(
            [
                "--manifest",
                str(plan_dir / "manifest.json"),
                "--data-root",
                str(tmp_path),
                "--max-runs",
                "1",
            ]
        )
        == 0
    )
    with pytest.raises(SystemExit):
        execute_main(
            [
                "--manifest",
                str(plan_dir / "manifest.json"),
                "--stage",
                "evaluate",
                "--data-root",
                str(tmp_path),
            ]
        )
    assert len(json.loads((plan_dir / "manifest.json").read_text())["runs"]) == 9
