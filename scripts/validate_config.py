"""Strictly audit YAML configuration keys before an experiment is run.

The regular Config loader remains backward compatible (warns and ignores
unknown keys). This opt-in check rejects those same typos, including nested
field names, to prevent silently using defaults in published experiments.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import yaml

from reid.config import Config


def _suggest(key: str, candidates: list[str]) -> str:
    """Offer a likely spelling correction when one exists."""
    matches = difflib.get_close_matches(key, candidates, n=1, cutoff=0.6)
    return f" Did you mean {matches[0]!r}?" if matches else ""


def validate_config_file(path: Path) -> Config:
    """Reject unknown sections/keys then apply the normal typed validation.

    Args:
        path: YAML configuration file to inspect.

    Returns:
        The validated project configuration.

    Raises:
        ValueError: For malformed YAML, unknown sections or unknown fields.
        TypeError: For invalid field types in the standard Config loader.
    """
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration must contain a YAML mapping.")
    sections = Config().to_dict()
    for name, values in raw.items():
        if name not in sections:
            raise ValueError(
                f"Unknown configuration section {name!r}."
                + _suggest(str(name), list(sections))
            )
        if not isinstance(values, dict):
            raise ValueError(f"Section {name!r} must be a mapping.")
        allowed = sections[name]
        for key in values:
            if key not in allowed:
                raise ValueError(
                    f"Unknown configuration key {name}.{key}."
                    + _suggest(str(key), list(allowed))
                )
    return Config.from_yaml(path)


def main(argv: list[str] | None = None) -> int:
    """Run an opt-in strict configuration audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config to validate")
    args = parser.parse_args(argv)
    try:
        validate_config_file(args.config)
    except (ValueError, TypeError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(f"Valid configuration (all keys recognized): {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
