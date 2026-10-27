"""No-GPU checks for strict config auditing and pinned Colab reproducibility."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from scripts.validate_config import main, validate_config_file

ROOT = Path(__file__).resolve().parents[1]
STRONG = ROOT / "configs" / "market1501_strong_baseline.yaml"
NOTEBOOK = ROOT / "notebooks" / "train_colab.ipynb"


def test_shipped_recipe_passes_strict_check() -> None:
    """The shipped strong baseline has no unknown configuration keys."""
    assert validate_config_file(STRONG).loss.center_loss is True
    assert main(["--config", str(STRONG)]) == 0


def test_rejects_unknown_field_with_suggestion(tmp_path: Path) -> None:
    """A typo that the standard loader would ignore becomes an error."""
    path = tmp_path / "typo.yaml"
    path.write_text("train:\n  max_epoch: 30\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"train\.max_epoch.*max_epochs"):
        validate_config_file(path)


def test_rejects_unknown_section_with_suggestion(tmp_path: Path) -> None:
    """Mistyped section headings must not silently fall back to defaults."""
    path = tmp_path / "typo.yaml"
    path.write_text("modle:\n  pooling: avg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration section"):
        validate_config_file(path)


def test_standard_value_validation_still_applies(tmp_path: Path) -> None:
    """The audit also catches invalid known values and malformed YAML."""
    path = tmp_path / "invalid.yaml"
    path.write_text("model:\n  pooling: made_up\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pooling"):
        validate_config_file(path)
    path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        validate_config_file(path)


def test_colab_uses_immutable_source_and_locked_dependencies() -> None:
    """The notebook must not silently track main or install floating extras."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    setup = "".join(nb["cells"][4]["source"])
    assert 'PINNED_COMMIT = "4faaecb949736a4c6306716082fb2ccc5031c2ed"' in setup
    assert "git" in setup and "checkout" in setup and "--detach" in setup
    assert "uv" in setup and "--locked" in setup
    assert "--extra" in setup and "cuda" in setup
    assert 'pip", "install", "-q", "-e"' not in setup
    assert "sys.path.insert" in setup
    assert all(not cell.get("outputs") for cell in nb["cells"] if cell["cell_type"] == "code")
    assert all(
        cell.get("execution_count") is None for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    sources = "\n".join("".join(cell["source"]) for cell in nb["cells"])
    assert "uv.lock" in sources and "provenance.json" in sources


def test_colab_python_cells_are_syntactically_valid() -> None:
    """Check notebook Python cells without executing paid-GPU workloads."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.lstrip().startswith("!"):
            continue  # IPython shell escape, not Python syntax.
        ast.parse(source)
