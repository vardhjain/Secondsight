"""Download the Market-1501 dataset via kagglehub.

This CLI downloads the ``pengcw1/market-1501`` dataset from Kaggle using
``kagglehub``, resolves the canonical ``Market-1501-v15.09.15`` sub-directory
(the directory that contains ``bounding_box_train``, ``query`` and
``bounding_box_test``), and prints the resulting data root so it can be fed to
the training / evaluation CLIs (e.g. via ``--data-root``).

``kagglehub`` is imported lazily inside :func:`main` so that merely importing
this module (for example to expose its ``main`` entry point) does not require
the dependency to be installed.

Example:
    Download the dataset and print its root path::

        python -m scripts.download_data
        # or, after `pip install -e .`:
        reid-download
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from reid.utils.logging import setup_logger

# The Kaggle dataset slug and the canonical extracted sub-directory name.
_DATASET_SLUG = "pengcw1/market-1501"
_CANONICAL_SUBDIR = "Market-1501-v15.09.15"
# Sub-directories that identify a valid Market-1501 data root.
_REQUIRED_DIRS = ("bounding_box_train", "query", "bounding_box_test")

logger = logging.getLogger("reid.scripts.download")


def _resolve_data_root(download_path: Path) -> Path:
    """Resolve the Market-1501 data root from a kagglehub download path.

    kagglehub returns the directory the archive was extracted into. Depending
    on the archive that directory may *be* the data root, or it may contain a
    single ``Market-1501-v15.09.15`` sub-directory which is the real root. This
    helper handles both layouts.

    Args:
        download_path: The path returned by ``kagglehub.dataset_download``.

    Returns:
        The directory that directly contains the Market-1501 split folders.
    """
    candidate = download_path / _CANONICAL_SUBDIR
    if candidate.is_dir():
        return candidate
    # Some mirrors nest the data one extra level deep; search for the canonical
    # folder anywhere beneath the download path as a fallback.
    for path in download_path.rglob(_CANONICAL_SUBDIR):
        if path.is_dir():
            return path
    return download_path


def _is_valid_root(root: Path) -> bool:
    """Return whether ``root`` looks like a Market-1501 data root.

    Args:
        root: Candidate data-root directory.

    Returns:
        ``True`` if all required split sub-directories are present.
    """
    return all((root / name).is_dir() for name in _REQUIRED_DIRS)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the download CLI.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="reid-download",
        description="Download the Market-1501 dataset via kagglehub and print its data root.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional directory to copy the resolved dataset into. When omitted "
            "the dataset stays in the kagglehub cache and only its path is printed."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Download Market-1501 and print (or copy to) its data root.

    Args:
        argv: Optional list of command-line arguments (defaults to
            ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` on success, non-zero on failure.
    """
    args = build_parser().parse_args(argv)
    setup_logger("reid")

    try:
        import kagglehub
    except ImportError:
        logger.error(
            "kagglehub is not installed. Install it with `pip install kagglehub` "
            "(or `pip install -e .`) and retry."
        )
        return 1

    logger.info("Downloading Market-1501 (%s) via kagglehub...", _DATASET_SLUG)
    try:
        download_path = Path(kagglehub.dataset_download(_DATASET_SLUG))
    except Exception:  # noqa: BLE001 - surface any kagglehub/network failure cleanly
        logger.exception("Failed to download the dataset via kagglehub.")
        return 1

    data_root = _resolve_data_root(download_path)
    if not _is_valid_root(data_root):
        logger.warning(
            "Downloaded data at %s does not contain the expected Market-1501 "
            "sub-directories %s. The dataset layout may have changed.",
            data_root,
            _REQUIRED_DIRS,
        )

    if args.output is not None:
        destination = args.output.expanduser().resolve()
        logger.info("Copying dataset from %s to %s ...", data_root, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if _is_valid_root(destination):
                logger.info("Destination already exists; reusing %s", destination)
            else:
                logger.warning(
                    "Destination %s already exists but does not contain the "
                    "expected Market-1501 sub-directories %s; it may be a leftover "
                    "or partially-copied directory. Remove it and re-run, or copy "
                    "the dataset there manually.",
                    destination,
                    _REQUIRED_DIRS,
                )
        else:
            shutil.copytree(data_root, destination)
        data_root = destination

    logger.info("Market-1501 data root: %s", data_root)
    # Print the bare path to stdout so it can be captured by shell pipelines.
    sys.stdout.write(f"{data_root}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
