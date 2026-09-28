"""Download the Market-1501 dataset via kagglehub.

This CLI downloads the ``pengcw1/market-1501`` dataset from Kaggle using
``kagglehub``, resolves the canonical ``Market-1501-v15.09.15`` sub-directory
(the directory that contains ``bounding_box_train``, ``query`` and
``bounding_box_test``), checks that every split holds Market-1501 images, and
prints the resulting data root. Only the bare path is written to stdout, and all
log lines go to stderr, so the output can be captured directly, for example
``export REID_DATA_ROOT=$(reid-download)``. The command exits non-zero whenever
it cannot produce a complete data root.

``kagglehub`` is an optional dependency (the ``data`` extra) and is imported
lazily inside :func:`main`, so importing this module never requires it.

Example:
    Download the dataset and print its root path::

        reid-download
        reid-download --output data/market1501
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from reid.data.dataset import parse_market_filename

__all__ = ["build_parser", "main"]

# The Kaggle dataset slug and the canonical extracted sub-directory name.
_DATASET_SLUG = "pengcw1/market-1501"
_CANONICAL_SUBDIR = "Market-1501-v15.09.15"
# Sub-directories that identify a valid Market-1501 data root.
_REQUIRED_DIRS = ("bounding_box_train", "query", "bounding_box_test")
# Folders shipped with Market-1501 that the pipeline never reads.
_UNUSED_DIRS = ("gt_bbox", "gt_query")

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Exit codes: download or copy failures, and an unusable dataset layout.
_EXIT_FAILURE = 1
_EXIT_INVALID = 2

logger = logging.getLogger("reid.cli.download")


def _configure_logging() -> None:
    """Send this CLI's log records to stderr so stdout carries only the path.

    Repeated calls reuse the handler but point it at the current ``sys.stderr``,
    which may have been replaced since the last call (for example by a test
    harness capturing output).
    """
    handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    if handlers:
        handlers[0].stream = sys.stderr
    else:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _resolve_data_root(download_path: Path) -> Path:
    """Resolve the Market-1501 data root from a kagglehub download path.

    kagglehub returns the directory the archive was extracted into. Depending on
    the archive that directory may itself be the data root, or it may contain a
    ``Market-1501-v15.09.15`` sub-directory (possibly nested one level deeper)
    which is the real root. This helper handles all of these layouts.

    Args:
        download_path: The path returned by ``kagglehub.dataset_download``.

    Returns:
        The directory that should directly contain the Market-1501 split folders.
    """
    candidate = download_path / _CANONICAL_SUBDIR
    if candidate.is_dir():
        return candidate
    for path in sorted(download_path.rglob(_CANONICAL_SUBDIR)):
        if path.is_dir():
            return path
    return download_path


def _has_market_images(directory: Path) -> bool:
    """Return whether ``directory`` holds at least one Market-1501 image.

    Args:
        directory: Candidate split folder.

    Returns:
        ``True`` if a ``.jpg`` named with the Market-1501 convention is present.
    """
    return any(parse_market_filename(p.name) is not None for p in directory.glob("*.jpg"))


def _is_valid_root(root: Path) -> bool:
    """Return whether ``root`` is a complete Market-1501 data root.

    Every required split folder must exist and contain Market-1501 images, so
    empty or hand-made shells are rejected.

    Args:
        root: Candidate data-root directory.

    Returns:
        ``True`` if all required split folders are present and populated.
    """
    return all(
        (root / name).is_dir() and _has_market_images(root / name) for name in _REQUIRED_DIRS
    )


def _copy_dataset(source: Path, destination: Path) -> None:
    """Copy the split folders of a data root to ``destination`` atomically.

    The copy is written to a sibling ``<name>.partial`` directory and renamed
    into place only once it is complete, so an interrupted copy never leaves a
    directory that looks valid. The ``gt_bbox`` and ``gt_query`` folders are
    skipped because the pipeline never reads them.

    Args:
        source: A valid Market-1501 data root.
        destination: Target path, which must not exist.
    """
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    try:
        shutil.copytree(source, partial, ignore=shutil.ignore_patterns(*_UNUSED_DIRS))
        os.replace(partial, destination)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the download CLI.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="reid-download",
        description=(
            "Download the Market-1501 dataset via kagglehub and print its data root "
            "on stdout (logs go to stderr)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Directory that will become the Market-1501 data root. It must not "
            "exist, be empty, or already hold a complete copy. When omitted the "
            "dataset stays in the kagglehub cache and only its path is printed."
        ),
    )
    return parser


def _materialize(data_root: Path, destination: Path) -> bool:
    """Make ``destination`` a complete copy of ``data_root`` when needed.

    Args:
        data_root: A valid Market-1501 data root.
        destination: Requested output directory.

    Returns:
        ``True`` if ``destination`` now holds a complete data root, ``False`` if
        it already exists with other contents and was left untouched.
    """
    if destination.exists():
        if _is_valid_root(destination):
            logger.info("Destination already holds Market-1501; reusing %s", destination)
            return True
        if not destination.is_dir() or any(destination.iterdir()):
            logger.error(
                "Destination %s already exists but is not a complete Market-1501 data "
                "root. Remove it or choose another --output and re-run.",
                destination,
            )
            return False
        destination.rmdir()

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Copying dataset from %s to %s ...", data_root, destination)
    _copy_dataset(data_root, destination)
    return True


def main(argv: list[str] | None = None) -> int:
    """Download Market-1501 and print (or copy to) its data root.

    Args:
        argv: Optional list of command-line arguments (defaults to
            ``sys.argv[1:]``).

    Returns:
        Process exit code. ``0`` on success, ``1`` when the download, the copy or
        the ``kagglehub`` import fails, and ``2`` when the dataset layout is
        incomplete or ``--output`` points at an unusable directory.
    """
    args = build_parser().parse_args(argv)
    _configure_logging()

    try:
        import kagglehub
    except ImportError:
        logger.error('kagglehub is not installed. Install it with: pip install "secondsight[data]"')
        return _EXIT_FAILURE

    logger.info("Downloading Market-1501 (%s) via kagglehub...", _DATASET_SLUG)
    try:
        download_path = Path(kagglehub.dataset_download(_DATASET_SLUG))
    except Exception:  # noqa: BLE001  # any kagglehub or network failure is reported cleanly
        logger.exception("Failed to download the dataset via kagglehub.")
        return _EXIT_FAILURE

    data_root = _resolve_data_root(download_path)
    if not _is_valid_root(data_root):
        logger.error(
            "Downloaded data at %s does not contain populated Market-1501 folders %s. "
            "The dataset layout may have changed.",
            data_root,
            ", ".join(_REQUIRED_DIRS),
        )
        return _EXIT_INVALID

    if args.output is not None:
        destination = args.output.expanduser().resolve()
        try:
            if not _materialize(data_root, destination):
                return _EXIT_INVALID
        except OSError:
            logger.exception("Failed to copy the dataset to %s.", destination)
            return _EXIT_FAILURE
        data_root = destination

    logger.info("Market-1501 data root: %s", data_root)
    sys.stdout.write(f"{data_root}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
