"""Logging configuration for the Re-ID package.

Provides a single idempotent :func:`setup_logger` factory that wires up a
console handler and an optional rotating-free file handler. Repeated calls with
the same logger name reuse the existing handlers instead of duplicating output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "reid",
    output_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or retrieve a configured logger.

    The logger writes to ``stdout`` and, when ``output_dir`` is provided, also
    to ``<output_dir>/log.txt``. The function is idempotent: calling it again
    with the same ``name`` updates the level but does not add duplicate console
    handlers, and only adds a file handler if one for that path is not already
    attached.

    Args:
        name: Logger name.
        output_dir: Optional directory for a ``log.txt`` file. Created if
            missing. When ``None``, only console logging is configured.
        level: Logging level (e.g. :data:`logging.INFO`).

    Returns:
        The configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "log.txt"
        resolved = str(log_path.resolve())
        already = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == resolved
            for h in logger.handlers
        )
        if not already:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


__all__ = ["setup_logger"]
