"""Reader for the per-camera mask archive written by ma_masks.

Verbatim copy of the reader half of ``segmentation/core/mask_archive.py``
(ma_2d cannot import ``segmentation``). The contract is the file spec, not
a shared import; keep the two in sync.

Format contract (v1):

* ``<cam>/masks/masks.zip`` is a ``ZIP_STORED`` archive; each entry is a
  grayscale PNG named exactly like the legacy loose file
  (``mask_{frame:04d}_{pid:02d}.png``, pids 1-based).
* Legacy layout: the same names as loose files directly in ``<cam>/masks/``.
* Readers auto-detect: a ``masks.zip`` wins over loose PNGs in the same dir.
"""
from __future__ import annotations

import glob
import os
import threading
import zipfile
from typing import List, Optional

import cv2
import numpy as np

ARCHIVE_NAME = "masks.zip"


class MaskArchiveReader:
    """Auto-detecting reader: packed ``masks.zip`` if present, else loose PNGs."""

    def __init__(self, mask_dir: str):
        self.mask_dir = mask_dir
        self._lock = threading.Lock()
        archive = os.path.join(mask_dir, ARCHIVE_NAME)
        self._zip = zipfile.ZipFile(archive, "r") if os.path.isfile(archive) else None

    @property
    def packed(self) -> bool:
        return self._zip is not None

    def names(self) -> List[str]:
        """Sorted mask entry basenames (``mask_FFFF_PP.png``)."""
        if self._zip is not None:
            return sorted(n for n in self._zip.namelist()
                          if n.startswith("mask_") and n.endswith(".png"))
        return sorted(os.path.basename(p)
                      for p in glob.glob(os.path.join(self.mask_dir, "mask_*.png")))

    def read(self, name: str) -> Optional[np.ndarray]:
        """Grayscale uint8 mask, or None when the entry does not exist."""
        if self._zip is not None:
            with self._lock:
                try:
                    data = self._zip.read(name)
                except KeyError:
                    return None
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        path = os.path.join(self.mask_dir, name)
        if not os.path.exists(path):
            return None
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
