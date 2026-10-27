"""Regression checks for the PyTorch-specific Dependabot version-update policy."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORED_VERSION_UPDATES = {
    "version-update:semver-major",
    "version-update:semver-minor",
    "version-update:semver-patch",
}


def test_pytorch_paired_upgrades_are_manual_without_disabling_security_updates() -> None:
    """Only routine PyTorch version bumps are excluded from uv Dependabot."""
    config = yaml.safe_load((REPO_ROOT / ".github" / "dependabot.yml").read_text("utf-8"))
    uv_updates = [
        update
        for update in config["updates"]
        if update["package-ecosystem"] == "uv" and update["directory"] == "/"
    ]
    assert len(uv_updates) == 1

    uv_config = uv_updates[0]
    assert uv_config["schedule"]["interval"] == "weekly"
    assert uv_config["groups"]["python"]["update-types"] == ["minor", "patch"]

    ignored = {item["dependency-name"]: item for item in uv_config["ignore"]}
    assert set(ignored) == {"torch", "torchvision"}
    for package in ("torch", "torchvision"):
        # Bare ignore rules also suppress security updates: enforce explicit
        # version-update-only filters so security advisories still surface.
        assert set(ignored[package]["update-types"]) == IGNORED_VERSION_UPDATES
        assert "versions" not in ignored[package]
