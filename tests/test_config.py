"""Tests for the typed configuration system (:mod:`reid.config`).

These are light tests: they require only ``PyYAML`` (a hard dependency of
``reid.config``) and the standard library. They verify dataclass defaults, the
dict / YAML round-trips, the forgiving construction (unknown keys ignored with a
warning, missing keys defaulted, numeric strings coerced), every validation
rule, and that the shipped YAML files match the dataclass defaults and the
documented strong-baseline recipe exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from reid.config import (
    Config,
    DataConfig,
    EvalConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    default_config_path,
)

# Repo root = two levels up from this file (tests/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"
STRONG_YAML = REPO_ROOT / "configs" / "market1501_strong_baseline.yaml"


def _raw_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file without going through the forgiving ``Config`` loader."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
    assert cfg.data.val_ids == 0
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
    assert cfg.train.deterministic is True


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


# ---------------------------------------------------------------------------
# Shipped YAML files
# ---------------------------------------------------------------------------


def test_default_yaml_file_exists() -> None:
    """The shipped default config file is present in the repo."""
    assert DEFAULT_YAML.is_file(), f"Missing config file: {DEFAULT_YAML}"


def test_default_yaml_mirrors_dataclass_defaults() -> None:
    """``configs/default.yaml`` round-trips to the default ``Config``."""
    assert Config.from_yaml(DEFAULT_YAML) == Config()


def test_default_yaml_is_an_exact_mirror_of_the_dataclass_defaults() -> None:
    """Missing, extra or misspelled keys in ``default.yaml`` fail this comparison."""
    assert _raw_yaml(DEFAULT_YAML) == Config().to_dict()


def test_strong_baseline_yaml_differs_from_defaults_only_in_center_loss() -> None:
    """The headline recipe is exactly the defaults with center loss switched on."""
    expected = Config().to_dict()
    expected["loss"]["center_loss"] = True
    assert _raw_yaml(STRONG_YAML) == expected


def test_strong_baseline_recipe_values() -> None:
    """The headline recipe matches the documented strong-baseline settings."""
    cfg = Config.from_yaml(STRONG_YAML)
    assert (cfg.data.batch_size, cfg.data.num_instances) == (64, 4)
    assert cfg.model.pooling == "gem"
    assert cfg.model.last_stride == 1
    assert cfg.model.ibn is False
    assert cfg.loss.center_loss is True
    assert cfg.loss.label_smoothing == pytest.approx(0.1)
    assert cfg.loss.triplet_margin == pytest.approx(0.3)
    assert cfg.optim.milestones == [30, 50]
    assert cfg.optim.warmup_epochs == 10
    assert cfg.train.max_epochs == 60
    assert cfg.eval.feat_norm and cfg.eval.flip_tta and cfg.eval.rerank


def test_default_config_path_finds_the_shipped_configs() -> None:
    """``default_config_path`` resolves the shipped recipes in a checkout."""
    path = default_config_path()
    assert path.is_file()
    assert path.name == "market1501_strong_baseline.yaml"
    assert default_config_path("default.yaml").is_file()


# ---------------------------------------------------------------------------
# Coercion and unknown-key warnings
# ---------------------------------------------------------------------------


def test_scientific_notation_strings_are_coerced(tmp_path: Path) -> None:
    """YAML 1.1 reads ``1e-4`` as a string, which must become a float."""
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "optim:\n  lr: 1e-4\n  weight_decay: 5e-4\n"
        "loss:\n  center_weight: 5e-4\n"
        "train:\n  max_epochs: '80'\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(path)
    assert isinstance(cfg.optim.lr, float)
    assert cfg.optim.lr == pytest.approx(1e-4)
    assert isinstance(cfg.optim.weight_decay, float)
    assert isinstance(cfg.loss.center_weight, float)
    assert cfg.train.max_epochs == 80


def test_integers_for_float_fields_become_floats() -> None:
    """An integer given for a float field is stored as a float."""
    cfg = Config.from_dict({"loss": {"id_weight": 2}})
    assert isinstance(cfg.loss.id_weight, float)


def test_unparseable_number_raises_type_error() -> None:
    """A non-numeric string for a numeric field fails with a clear message."""
    with pytest.raises(TypeError, match=r"OptimConfig\.lr"):
        Config.from_dict({"optim": {"lr": "fast"}})


def test_unknown_keys_and_sections_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Typos are ignored but reported, so they do not go unnoticed."""
    with caplog.at_level(logging.WARNING, logger="reid.config"):
        Config.from_dict({"train": {"max_epoch": 120}, "optimizer": {}})
    assert "max_epoch" in caplog.text
    assert "optimizer" in caplog.text


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"num_instances": 0}},
        {"data": {"batch_size": 8, "num_instances": 1}},
        {"data": {"batch_size": 4, "num_instances": 4}},
        {"data": {"batch_size": 2, "num_instances": 4}},
        {"data": {"batch_size": 30, "num_instances": 4}},
        {"data": {"val_ids": -1}},
        {"model": {"pooling": "median"}},
        {"model": {"last_stride": 3}},
        {"optim": {"scheduler": "step"}},
        {"optim": {"name": "rmsprop"}},
        {"optim": {"warmup_epochs": -1}},
        {"train": {"max_epochs": 0}},
        {"train": {"eval_period": 0}},
        {"train": {"log_period": 0}},
        {"eval": {"max_rank": 0}},
        {"eval": {"rerank_k1": 0}},
        {"eval": {"rerank_k2": 0}},
        {"eval": {"rerank_lambda": 1.5}},
        {"eval": {"rerank_lambda": -0.1}},
    ],
)
def test_validate_rejects(payload: dict[str, Any]) -> None:
    """Every hard validation rule raises ``ValueError``."""
    with pytest.raises(ValueError):
        Config.from_dict(payload)


def test_validate_is_case_insensitive() -> None:
    """Enum-like fields accept any casing, like the downstream builders."""
    cfg = Config.from_dict({"model": {"pooling": "GeM"}, "optim": {"name": "ADAM"}})
    assert cfg.model.pooling == "GeM"


def test_late_milestones_only_warn(caplog: pytest.LogCaptureFixture) -> None:
    """A smoke run shorter than the milestones stays loadable, with a warning."""
    with caplog.at_level(logging.WARNING, logger="reid.config"):
        cfg = Config.from_dict({"train": {"max_epochs": 5}})
    assert cfg.train.max_epochs == 5
    assert "never fire" in caplog.text


def test_milestone_equal_to_max_epochs_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A milestone at ``max_epochs`` never fires, so it is reported too."""
    with caplog.at_level(logging.WARNING, logger="reid.config"):
        Config.from_dict({"train": {"max_epochs": 50}, "optim": {"milestones": [30, 50]}})
    assert "[50]" in caplog.text


def test_cosine_schedule_ignores_milestones(caplog: pytest.LogCaptureFixture) -> None:
    """Unused multistep milestones cause no warning under a cosine schedule."""
    with caplog.at_level(logging.WARNING, logger="reid.config"):
        Config.from_dict({"optim": {"scheduler": "cosine"}, "train": {"max_epochs": 40}})
    assert "milestones" not in caplog.text


def test_saved_smoke_run_config_reloads(tmp_path: Path) -> None:
    """A config with a shortened ``max_epochs`` round-trips through YAML."""
    cfg = Config()
    cfg.train.max_epochs = 5
    cfg.validate()
    cfg.to_yaml(tmp_path / "config.yaml")
    assert Config.from_yaml(tmp_path / "config.yaml").train.max_epochs == 5


def test_non_mapping_yaml_raises_type_error(tmp_path: Path) -> None:
    """A top-level YAML list is rejected."""
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(TypeError):
        Config.from_yaml(path)


def test_non_mapping_section_raises_type_error() -> None:
    """A section that is not a mapping is rejected."""
    with pytest.raises(TypeError):
        Config.from_dict({"data": [1]})


def test_to_portable_dict_scrubs_local_paths(tmp_path: Path) -> None:
    """Checkpoint configs carry no absolute data root or output directory."""
    cfg = Config()
    cfg.data.root = str(tmp_path / "Market-1501")
    cfg.train.output_dir = str(tmp_path / "runs" / "strong")
    portable = cfg.to_portable_dict()
    assert portable["data"]["root"] is None
    assert portable["train"]["output_dir"] == "strong"
    # The live config is untouched.
    assert cfg.data.root == str(tmp_path / "Market-1501")
