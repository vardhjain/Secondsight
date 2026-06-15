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
for example ``0002_c1s1_000451_03.jpg``. Identities labelled ``-1`` are *junk*
distractors and are dropped on load.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

__all__ = ["Market1501"]

# Maps a logical subset name to its on-disk sub-directory.
_SUBSET_DIRS: dict[str, str] = {
    "train": "bounding_box_train",
    "query": "query",
    "gallery": "bounding_box_test",
}

# Parses ``<pid>_c<camid>`` from a Market-1501 filename. ``pid`` may be ``-1``
# (junk) hence the ``[-\d]+`` character class; ``\d+`` on the camera id is
# future-proof for datasets with more than nine cameras (Market-1501 has six).
_PATTERN = re.compile(r"([-\d]+)_c(\d+)")


class Market1501(Dataset):
    """Market-1501 image dataset.

    The dataset parses person identity (``pid``) and camera identity (``camid``)
    from each filename. For the ``train`` subset a contiguous label mapping
    (``pid2label``) is built so that identities can be used directly as
    cross-entropy targets. For ``query`` and ``gallery`` subsets the raw ``pid``
    is returned (required by the evaluation protocol).

    Args:
        root: Path to the Market-1501 dataset root (the directory that contains
            ``bounding_box_train``, ``query`` and ``bounding_box_test``).
        subset: One of ``{"train", "query", "gallery"}``.
        transform: Optional callable applied to each loaded ``PIL.Image`` (e.g.
            a ``torchvision`` transform pipeline). When ``None`` the raw RGB
            ``PIL.Image`` is returned.

    Attributes:
        root: Resolved dataset root path.
        subset: The subset name passed at construction.
        data_dir: Resolved path to the subset's image directory.
        img_paths: List of image file paths (``pathlib.Path``).
        pids: Raw person identities aligned with ``img_paths``.
        camids: Camera identities aligned with ``img_paths``.
        pid2label: Mapping from raw ``pid`` to contiguous label (``{}`` unless
            ``subset == "train"``).
        num_classes: Number of identities. For ``train`` this is
            ``len(pid2label)``; otherwise the number of unique raw identities.

    Raises:
        ValueError: If ``subset`` is not a recognised name.
        FileNotFoundError: If the resolved subset directory does not exist.
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
        self.transform = transform
        self.data_dir = self.root / _SUBSET_DIRS[subset]

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Subset directory not found: {self.data_dir}. Expected a "
                f"Market-1501 layout under {self.root}."
            )

        self.img_paths: list[Path] = []
        self.pids: list[int] = []
        self.camids: list[int] = []
        self._scan()

        self.pid2label: dict[int, int] = {}
        if subset == "train":
            unique_pids = sorted(set(self.pids))
            self.pid2label = {pid: label for label, pid in enumerate(unique_pids)}
            self.num_classes = len(self.pid2label)
        else:
            self.num_classes = len(set(self.pids))

    def _scan(self) -> None:
        """Scan the subset directory and populate path / pid / camid lists.

        Junk identities (``pid == -1``) are skipped. Files are processed in
        sorted order for deterministic indexing.
        """
        for path in sorted(self.data_dir.glob("*.jpg")):
            match = _PATTERN.search(path.name)
            if match is None:
                continue
            pid = int(match.group(1))
            if pid == -1:
                continue
            camid = int(match.group(2))
            self.img_paths.append(path)
            self.pids.append(pid)
            self.camids.append(camid)

    def __len__(self) -> int:
        """Return the number of (non-junk) images in the subset."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[object, int, int]:
        """Load and return one sample.

        Args:
            idx: Index into the dataset.

        Returns:
            A ``(image, label, camid)`` tuple where ``image`` is the (optionally
            transformed) RGB image, ``label`` is the contiguous training label
            (``pid2label[pid]``) for the ``train`` subset or the raw ``pid``
            otherwise, and ``camid`` is the camera identity.
        """
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        pid = self.pids[idx]
        camid = self.camids[idx]

        label = self.pid2label[pid] if self.subset == "train" else pid

        return img, label, camid
