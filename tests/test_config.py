"""Tests for the typed configuration system (:mod:`reid.config`).

These are light tests: they require only ``PyYAML`` (a hard dependency of
``reid.config``) and the standard library. They verify dataclass defaults, the
dict / YAML round-trips, the forgiving construction (unknown keys ignored,
missing keys defaulted), and that the shipped ``configs/default.yaml`` mirrors
the dataclass defaults exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reid.config import (
    Config,
    DataConfig,
    EvalConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)

# Repo root = two levels up from this file (tests/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"


def test_default_construction_has_expected_sections() -> None:
    """A default ``Config`` composes all six typed sub-configs."""
    cfg = Config()
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.loss, LossConfig)
    assert isinstance(cfg.optim, OptimConfig)
    assert isinstance(cfg.eval, EvalConfig)
    assert isinstance(cfg.train, TrainConfig)


def test_default_values_match_contract() -> None:
    """Spot-check the headline default values from the contract."""
    cfg = Config()
    assert cfg.data.batch_size == 64
    assert cfg.data.num_instances == 4
    assert cfg.model.pooling == "gem"
    assert cfg.model.last_stride == 1
    assert cfg.loss.label_smoothing == pytest.approx(0.1)
    assert cfg.loss.center_loss is False
    assert cfg.optim.scheduler == "warmup_multistep"
    assert cfg.optim.milestones == [30, 50]
    assert cfg.eval.feat_norm is True
    assert cfg.eval.flip_tta is True
    assert cfg.eval.rerank is True
    assert cfg.train.amp is True
    assert cfg.train.seed == 42


def test_milestones_default_factory_is_not_shared() -> None:
    """Each config gets its own ``milestones`` list (no shared mutable state)."""
    a = Config()
    b = Config()
    a.optim.milestones.append(99)
    assert b.optim.milestones == [30, 50]


def test_to_dict_from_dict_roundtrip() -> None:
    """``from_dict(to_dict(cfg))`` reproduces an equivalent config."""
    cfg = Config()
    restored = Config.from_dict(cfg.to_dict())
    assert restored == cfg


def test_from_dict_ignores_unknown_keys_and_sections() -> None:
    """Unknown sections and unknown keys are dropped, not raised on."""
    payload = {
        "data": {"batch_size": 32, "totally_unknown": 123},
        "mystery_section": {"foo": "bar"},
    }
    cfg = Config.from_dict(payload)
    assert cfg.data.batch_size == 32
    # Unknown key did not leak onto the dataclass.
    assert not hasattr(cfg.data, "totally_unknown")
    # Missing sections fall back to defaults.
    assert cfg.model == ModelConfig()


def test_from_dict_with_empty_mapping_yields_defaults() -> None:
    """An empty (or ``None``) mapping produces an all-default config."""
    assert Config.from_dict({}) == Config()


def test_yaml_roundtrip(tmp_path: Path) -> None:
    """Writing then reading a config via YAML preserves all values."""
    cfg = Config()
    cfg.data.batch_size = 48
    cfg.model.pooling = "avg"
    cfg.optim.milestones = [10, 20, 40]
    out = tmp_path / "nested" / "cfg.yaml"

    cfg.to_yaml(out)
    assert out.exists()  # parent dirs created automatically.

    loaded = Config.from_yaml(out)
    assert loaded == cfg


def test_default_yaml_file_exists() -> None:
    """The shipped default config file is present in the repo."""
    assert DEFAULT_YAML.is_file(), f"Missing config file: {DEFAULT_YAML}"


def test_default_yaml_mirrors_dataclass_defaults() -> None:
    """``configs/default.yaml`` round-trips to the default ``Config``."""
    loaded = Config.from_yaml(DEFAULT_YAML)
    assert loaded == Config()
