"""Market-1501 dataset.

This module implements a lightweight :class:`torch.utils.data.Dataset` for the
Market-1501 person re-identification benchmark. It deliberately avoids importing
``torchvision`` so that the dataset can be constructed (and image-free metadata
inspected) in environments where only ``numpy``, ``Pillow`` and ``torch`` are
installed.

The Market-1501 directory layout is::

    <root>/
        bounding_box_train/   # training images
        query/                # query (probe) images
        bounding_box_test/    # gallery images

Image filenames follow the convention ``<pid>_c<camid>s<seq>_<frame>_<n>.jpg``,
for example ``0002_c1s1_000451_03.jpg``. Images with pid ``-1`` are junk and are
dropped on load. Images with pid ``0000`` are distractors and are kept, because
the standard evaluation protocol leaves them in the gallery as hard negatives.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

__all__ = ["ImageListDataset", "Market1501", "parse_market_filename"]

# Maps a logical subset name to its on-disk sub-directory.
_SUBSET_DIRS: dict[str, str] = {
    "train": "bounding_box_train",
    "query": "query",
    "gallery": "bounding_box_test",
}

# Parses ``<pid>_c<camid>`` from the start of a Market-1501 filename. The pid may
# be ``-1`` (junk), hence the optional minus sign. ``\d+`` on the camera id keeps
# the pattern valid for datasets with more than nine cameras (Market-1501 has six).
_PATTERN = re.compile(r"^(-?\d+)_c(\d+)")

# The pid Market-1501 uses for junk crops, which the protocol discards.
_JUNK_PID = -1


def parse_market_filename(name: str) -> tuple[int, int] | None:
    """Parse the person and camera identity from a Market-1501 filename.

    Args:
        name: A bare filename such as ``0002_c1s1_000451_03.jpg``.

    Returns:
        A ``(pid, camid)`` tuple, or ``None`` when the name does not follow the
        Market-1501 convention.
    """
    match = _PATTERN.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


class ImageListDataset(Dataset):
    """Re-ID dataset backed by explicit lists of image paths and identities.

    This is the shared implementation behind :class:`Market1501` and the
    validation splits carved out of the training set. With ``relabel=True`` the
    identities are mapped to contiguous labels (``pid2label``) so they can be
    used directly as cross-entropy targets. Otherwise ``__getitem__`` returns the
    raw ``pid``, as the evaluation protocol requires.

    Args:
        img_paths: Image file paths.
        pids: Person identities aligned with ``img_paths``.
        camids: Camera identities aligned with ``img_paths``.
        relabel: Whether to build a contiguous ``pid2label`` mapping and return
            contiguous labels.
        transform: Optional callable applied to each loaded ``PIL.Image`` (for
            example a ``torchvision`` pipeline). When ``None`` the raw RGB
            ``PIL.Image`` is returned.

    Attributes:
        img_paths: List of image file paths (``pathlib.Path``).
        pids: Raw person identities aligned with ``img_paths``.
        camids: Camera identities aligned with ``img_paths``.
        pid2label: Mapping from raw ``pid`` to contiguous label (``{}`` unless
            ``relabel`` is ``True``).
        num_classes: Number of unique identities.

    Raises:
        ValueError: If the three sequences differ in length.
    """

    def __init__(
        self,
        img_paths: Sequence[str | Path],
        pids: Sequence[int],
        camids: Sequence[int],
        *,
        relabel: bool = False,
        transform: Callable | None = None,
    ) -> None:
        if not len(img_paths) == len(pids) == len(camids):
            raise ValueError(
                f"img_paths, pids and camids must have equal lengths; got "
                f"{len(img_paths)}, {len(pids)} and {len(camids)}."
            )
        self.img_paths: list[Path] = [Path(p) for p in img_paths]
        self.pids: list[int] = list(pids)
        self.camids: list[int] = list(camids)
        self.transform = transform
        self.relabel = relabel
        self.pid2label: dict[int, int] = {}
        if relabel:
            self.pid2label = {pid: label for label, pid in enumerate(sorted(set(self.pids)))}
        self.num_classes = len(set(self.pids))

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[object, int, int]:
        """Load and return one sample.

        Args:
            idx: Index into the dataset.

        Returns:
            A ``(image, label, camid)`` tuple where ``image`` is the (optionally
            transformed) RGB image, ``label`` is the contiguous label
            (``pid2label[pid]``) when relabelling is enabled or the raw ``pid``
            otherwise, and ``camid`` is the camera identity.
        """
        with Image.open(self.img_paths[idx]) as raw:
            img = raw.convert("RGB")

        sample: object = img if self.transform is None else self.transform(img)
        pid = self.pids[idx]
        label = self.pid2label[pid] if self.relabel else pid
        return sample, label, self.camids[idx]


class Market1501(ImageListDataset):
    """Market-1501 image dataset.

    The dataset parses person identity (``pid``) and camera identity (``camid``)
    from each filename. For the ``train`` subset a contiguous label mapping
    (``pid2label``) is built so that identities can be used directly as
    cross-entropy targets. For the ``query`` and ``gallery`` subsets the raw
    ``pid`` is returned, as the evaluation protocol requires.

    Args:
        root: Path to the Market-1501 dataset root (the directory that contains
            ``bounding_box_train``, ``query`` and ``bounding_box_test``).
        subset: One of ``{"train", "query", "gallery"}``.
        transform: Optional callable applied to each loaded ``PIL.Image``. When
            ``None`` the raw RGB ``PIL.Image`` is returned.

    Attributes:
        root: Dataset root path.
        subset: The subset name passed at construction.
        data_dir: Path to the subset's image directory.

    Raises:
        ValueError: If ``subset`` is not a recognised name, or if the subset
            directory holds no parsable Market-1501 images.
        FileNotFoundError: If the subset directory does not exist.
    """

    def __init__(
        self,
        root: str | Path,
        subset: str = "train",
        transform: Callable | None = None,
    ) -> None:
        if subset not in _SUBSET_DIRS:
            raise ValueError(f"Unknown subset {subset!r}; expected one of {sorted(_SUBSET_DIRS)}.")

        self.root = Path(root)
        self.subset = subset
        self.data_dir = self.root / _SUBSET_DIRS[subset]

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Subset directory not found: {self.data_dir}. Expected a "
                f"Market-1501 layout under {self.root}."
            )

        img_paths, pids, camids = self._scan(self.data_dir)
        if not img_paths:
            raise ValueError(f"No Market-1501 images found in {self.data_dir}.")

        super().__init__(img_paths, pids, camids, relabel=subset == "train", transform=transform)

    @staticmethod
    def _scan(data_dir: Path) -> tuple[list[Path], list[int], list[int]]:
        """Scan a subset directory for Market-1501 images.

        Junk images (pid ``-1``) are skipped, while distractors (pid ``0000``)
        are kept so they stay in the gallery as hard negatives. Files whose names
        do not follow the Market-1501 convention are ignored. Files are processed
        in sorted order so indexing is deterministic.

        Args:
            data_dir: Directory holding the subset's ``.jpg`` images.

        Returns:
            Aligned lists of image paths, pids and camera ids.
        """
        img_paths: list[Path] = []
        pids: list[int] = []
        camids: list[int] = []
        for path in sorted(data_dir.glob("*.jpg")):
            parsed = parse_market_filename(path.name)
            if parsed is None:
                continue
            pid, camid = parsed
            if pid == _JUNK_PID:
                continue
            img_paths.append(path)
            pids.append(pid)
            camids.append(camid)
        return img_paths, pids, camids
