"""Command-line interfaces for Secondsight.

Each module in this package backs one console script and exposes
``build_parser()`` and ``main(argv=None) -> int``, so the commands can be
driven from tests as well as from the shell. The modules are ``download``
(``reid-download``), ``train`` (``reid-train``), ``evaluate``
(``reid-evaluate``) and ``visualize`` (``reid-visualize``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from reid.config import Config, default_config_path

DATA_ROOT_ENV = "REID_DATA_ROOT"


def load_cli_config(path: Path | None, logger: logging.Logger) -> Config | None:
    """Load the ``--config`` YAML, defaulting to the packaged strong baseline.

    Args:
        path: The ``--config`` value, or ``None`` for the default config.
        logger: Logger that receives the error message.

    Returns:
        The loaded configuration, or ``None`` when the file does not exist.
    """
    resolved = path if path is not None else default_config_path()
    if not resolved.is_file():
        logger.error("Config file not found: %s", resolved)
        return None
    return Config.from_yaml(resolved)


def resolve_data_root(cfg: Config, cli_value: Path | None, logger: logging.Logger) -> Path | None:
    """Resolve the Market-1501 root from the CLI, the config or the environment.

    The command-line value wins, then ``cfg.data.root``, then the
    ``REID_DATA_ROOT`` environment variable. The resolved path is written back
    to ``cfg.data.root``. Problems are logged as errors.

    Args:
        cfg: Configuration to read from and update.
        cli_value: The ``--data-root`` value, or ``None`` when not given.
        logger: Logger that receives the error messages.

    Returns:
        The existing data root directory, or ``None`` when no root is
        configured or the configured path is not a directory.
    """
    if cli_value is not None:
        cfg.data.root = str(cli_value)
    if cfg.data.root is None:
        cfg.data.root = os.environ.get(DATA_ROOT_ENV) or None
    if cfg.data.root is None:
        logger.error(
            "No dataset root configured. Pass --data-root, set data.root in the "
            "config, or set the %s environment variable.",
            DATA_ROOT_ENV,
        )
        return None
    root = Path(cfg.data.root)
    if not root.is_dir():
        logger.error("Dataset root does not exist: %s", root)
        return None
    return root
