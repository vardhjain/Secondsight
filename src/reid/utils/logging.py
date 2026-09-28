"""Logging configuration for the Re-ID package.

Provides a single idempotent :func:`setup_logger` factory that wires up a
console handler and an optional plain (non-rotating) file handler. Repeated
calls with the same logger name reuse the existing handlers instead of
duplicating output.
"""

from __future__ import annotations

import logging
import os
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
    to ``<output_dir>/log.txt``. The function is idempotent. Calling it again
    with the same ``name`` applies the new level to the logger and to every
    attached handler, never adds a second console handler, and keeps exactly
    one file handler: a handler for a different ``log.txt`` (for example from
    an earlier run in the same notebook kernel) is closed and replaced.

    Args:
        name: Logger name.
        output_dir: Optional directory for a ``log.txt`` file. Created if
            missing. When ``None``, only console logging is configured and any
            existing file handler is left in place.
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
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "log.txt"
        # FileHandler stores os.path.abspath(filename), which does not resolve
        # symlinks or mapped drives, so compare against exactly that form.
        target = os.path.abspath(log_path)
        already = False
        for handler in list(logger.handlers):
            if not isinstance(handler, logging.FileHandler):
                continue
            if handler.baseFilename == target:
                already = True
            else:
                logger.removeHandler(handler)
                handler.close()
        if not already:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    return logger


__all__ = ["setup_logger"]
