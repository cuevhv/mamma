"""Per-camera mask archive: one packed file instead of thousands of PNGs.

Format contract (v1):

* ``<cam>/masks/masks.zip`` is a ``ZIP_STORED`` archive; each entry is a
  grayscale PNG named exactly like the legacy loose file
  (``mask_{frame:04d}_{pid:02d}.png``, pids 1-based).
* Legacy layout: the same names as loose files directly in ``<cam>/masks/``.
* Readers auto-detect: a ``masks.zip`` wins over loose PNGs in the same dir.

Entries hold already-deflated PNG bytes, so they are STORED without
re-compression and the archive matches the loose layout byte-for-byte in
size, while cutting ~8k files per camera down to one.

Not to be confused with :mod:`core.mask_store` (the in-memory/spill store
for masks during tracking); this module is the on-disk output format.

NOTE: ``landmarks/utils/mask_archive.py`` carries a verbatim copy of the
reader half (ma_2d cannot import ``segmentation``). The contract is the
file spec above; keep the two in sync.
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


class MaskArchiveWriter:
    """Thread-safe writer for per-frame mask PNGs.

    ``packed=True`` (the default) collects entries into a private temp file
    and atomically publishes ``masks.zip`` on :meth:`close`; loose mode
    writes the legacy one-file-per-mask layout. Both modes raise ``IOError``
    on a failed encode/write: the masks are the step's primary output, so a
    silent drop must fail the run.
    """

    def __init__(self, mask_dir: str, packed: bool = True):
        self.mask_dir = mask_dir
        self.packed = packed
        self._lock = threading.Lock()
        self._zip = None
        self._tmp = None
        os.makedirs(mask_dir, exist_ok=True)
        if packed:
            self._tmp = os.path.join(mask_dir, f"{ARCHIVE_NAME}.tmp.{os.getpid()}")
            self._zip = zipfile.ZipFile(self._tmp, "w", zipfile.ZIP_STORED)
        else:
            # A --png re-run must not leave a stale archive behind: readers
            # prefer masks.zip over loose files in the same dir.
            try:
                os.remove(os.path.join(mask_dir, ARCHIVE_NAME))
            except OSError:
                pass

    def write(self, name: str, mask_uint8: np.ndarray) -> None:
        if not self.packed:
            path = os.path.join(self.mask_dir, name)
            if not cv2.imwrite(path, mask_uint8):
                raise IOError(f"failed to write mask PNG: {path}")
            return
        ok, buf = cv2.imencode(".png", mask_uint8)
        if not ok:
            raise IOError(f"failed to encode mask PNG: {name}")
        with self._lock:
            self._zip.writestr(name, buf.tobytes())

    def close(self) -> None:
        """Publish the archive (packed mode). Idempotent."""
        if self._zip is None:
            return
        self._zip.close()
        os.replace(self._tmp, os.path.join(self.mask_dir, ARCHIVE_NAME))
        self._zip = None

    def abort(self) -> None:
        """Drop the partial archive without publishing. Idempotent."""
        if self._zip is None:
            return
        try:
            self._zip.close()
        finally:
            self._zip = None
            try:
                os.remove(self._tmp)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()


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
