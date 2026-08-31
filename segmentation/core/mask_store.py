"""Disk-backed store for per-frame, per-object segmentation masks.

Background
----------
``run_propagation`` used to collect SAM's output into an in-RAM dict shaped
``{frame_idx: {obj_id: bool_ndarray}}`` holding **every** frame's
full-resolution masks for an entire camera at once. For a 40 s, multi-person
1080p clip that is ~5 GB *per camera* of host RAM (the user-reported
"15 GB per camera" once the surrounding full-video image copies are included),
which OOMs modest machines on long / many-person sequences.

``MaskStore`` keeps the same logical content but spills it to disk: each frame's
masks are bit-packed (:func:`numpy.packbits`, 8x smaller than a bool array) and
written to a small ``.npz`` in a temporary directory. At most a handful of
frames are ever unpacked in RAM, so peak host memory is O(1 frame) instead of
O(whole sequence). Masks are restored with their **exact original shape**
(2-D ``(H, W)`` or ``(1, H, W)``) so downstream consumers are byte-identical.

The methodology is unchanged: this is purely a memory/plumbing layer. The
forward+reverse propagation, tracklet merge/discard thresholds, and the emitted
mask PNGs all behave exactly as before.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np


def _as_stored(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (packed_bits, shape) for a boolean-ish mask, preserving shape.

    Accepts bool or numeric masks of any shape (typically ``(H, W)`` or
    ``(1, H, W)``); values are thresholded ``> 0`` to bool before packing.
    """
    arr = np.asarray(mask)
    shape = np.asarray(arr.shape, dtype=np.int64)
    bits = np.packbits((arr > 0).reshape(-1))
    return bits, shape


def _from_stored(bits: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_as_stored`: unpack bits back to a bool array."""
    shape = tuple(int(s) for s in shape)
    n = int(np.prod(shape)) if shape else 0
    flat = np.unpackbits(np.asarray(bits, dtype=np.uint8))[:n].astype(bool)
    return flat.reshape(shape)


class MaskStore:
    """Disk-backed ``{frame_idx: {obj_id: mask}}`` with bounded RAM use.

    Only the access patterns the segmentation pipeline actually needs are
    exposed. Persistent mutations (drop an object, merge two objects) go through
    explicit methods rather than in-place dict edits, because each frame lives on
    disk and must be rewritten to persist a change.
    """

    def __init__(self, root: str | None = None, prefix: str = "ma_masks_store_"):
        self._dir = tempfile.mkdtemp(prefix=prefix, dir=root)
        # frame_idx -> filename (relative to self._dir). Sorted views derive from
        # this small index, so iteration never touches the on-disk payloads.
        self._index: dict[int, str] = {}
        self._closed = False

    # ---- internals ------------------------------------------------------
    def _path(self, fidx: int) -> str:
        return os.path.join(self._dir, f"frame_{int(fidx):08d}.npz")

    # ---- writing --------------------------------------------------------
    def set_frame(self, fidx: int, masks: dict) -> None:
        """Write (or overwrite) all masks for one frame.

        ``masks`` maps obj_id -> mask ndarray. ``None`` masks are skipped. An
        empty mapping still records the frame (so it appears in iteration with
        no objects), matching the old dict's ``video_segments[fidx] = {}``.
        """
        if self._closed:
            raise RuntimeError("MaskStore is closed")
        fidx = int(fidx)
        payload: dict[str, np.ndarray] = {}
        ids = []
        for oid, mask in masks.items():
            if mask is None:
                continue
            bits, shape = _as_stored(mask)
            payload[f"p{int(oid)}"] = bits
            payload[f"s{int(oid)}"] = shape
            ids.append(int(oid))
        payload["ids"] = np.asarray(sorted(ids), dtype=np.int64)
        # Compressed: packed binary person masks deflate extremely well
        # (measured 756 KB -> 4 KB/frame on a salsa-resolution blob at
        # +1.7 ms/frame, <1% of the propagation budget), which keeps the
        # per-camera store in the tens-of-MB range instead of several GB
        # (a single ~4900-frame camera can fill the node's /tmp with ~3 GB uncompressed).
        # Reads are transparently decompressed by np.load.
        np.savez_compressed(self._path(fidx), **payload)
        self._index[fidx] = self._path(fidx)

    # ---- reading --------------------------------------------------------
    def frame(self, fidx: int) -> dict:
        """Return ``{obj_id: bool_ndarray}`` for one frame (unpacked in RAM).

        Returns an empty dict for an unknown frame. This is the only place a
        frame's full-resolution masks are materialized; callers should let the
        result go out of scope promptly to keep peak RAM low.
        """
        fidx = int(fidx)
        path = self._index.get(fidx)
        if path is None:
            return {}
        with np.load(path) as data:
            ids = [int(o) for o in data["ids"]]
            return {oid: _from_stored(data[f"p{oid}"], data[f"s{oid}"]) for oid in ids}

    def frame_obj_ids(self, fidx: int) -> list[int]:
        """Object ids present on a frame, without unpacking the masks."""
        path = self._index.get(int(fidx))
        if path is None:
            return []
        with np.load(path) as data:
            return [int(o) for o in data["ids"]]

    def obj_area(self, fidx: int, oid: int) -> int | None:
        """Pixel area (set bits) of one object's mask on one frame, or None.

        Unpacks a single mask; used by the tiny-tracklet / merge stats so they
        never hold more than one mask at a time.
        """
        path = self._index.get(int(fidx))
        if path is None:
            return None
        oid = int(oid)
        with np.load(path) as data:
            key = f"p{oid}"
            if key not in data.files:
                return None
            return int(np.unpackbits(data[key].astype(np.uint8)).sum())

    # ---- frame-set views ------------------------------------------------
    def frames(self) -> list[int]:
        """Sorted list of frame indices present in the store."""
        return sorted(self._index.keys())

    def all_obj_ids(self) -> list[int]:
        """Sorted union of object ids across all frames."""
        ids: set[int] = set()
        for fidx in self._index:
            ids.update(self.frame_obj_ids(fidx))
        return sorted(ids)

    def image_size(self) -> tuple[int, int] | None:
        """(W, H) inferred from the first stored mask, or None if empty."""
        for fidx in self.frames():
            path = self._index[fidx]
            with np.load(path) as data:
                ids = [int(o) for o in data["ids"]]
                if not ids:
                    continue
                shape = tuple(int(s) for s in data[f"s{ids[0]}"])
            h, w = (shape[-2], shape[-1])
            return (w, h)
        return None

    def __contains__(self, fidx) -> bool:
        return int(fidx) in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self):
        return iter(self.frames())

    # ---- dict-compatible read API --------------------------------------
    # These let the (many) read-only consumers that previously indexed the
    # in-RAM ``video_segments`` dict keep working unchanged: each access loads
    # just one frame from disk. Only writers and in-place mutators were ported
    # to set_frame() / discard_obj() / merge_obj().
    def __getitem__(self, fidx) -> dict:
        return self.frame(fidx)

    def __setitem__(self, fidx, masks: dict) -> None:
        self.set_frame(fidx, masks)

    def keys(self) -> list[int]:
        return self.frames()

    def get(self, fidx, default=None):
        if int(fidx) in self._index:
            return self.frame(fidx)
        return default

    # ---- persistent mutations ------------------------------------------
    def discard_obj(self, oid: int) -> None:
        """Remove an object id from every frame (tiny-tracklet discard)."""
        oid = int(oid)
        for fidx in self.frames():
            frame = self.frame(fidx)
            if oid in frame:
                frame.pop(oid, None)
                self.set_frame(fidx, frame)

    def merge_obj(self, drop: int, keep: int) -> None:
        """Absorb ``drop`` into ``keep`` across all frames (duplicate merge).

        Mirrors the original in-place logic exactly: on frames where only
        ``drop`` exists its mask is transferred to ``keep``; on frames where
        both exist ``keep``'s mask is retained; ``drop`` is then removed.
        """
        drop, keep = int(drop), int(keep)
        for fidx in self.frames():
            frame = self.frame(fidx)
            has_keep = keep in frame and frame[keep] is not None
            has_drop = drop in frame and frame[drop] is not None
            if not has_drop:
                continue
            if not has_keep:
                frame[keep] = frame[drop]
            frame.pop(drop, None)
            self.set_frame(fidx, frame)

    # ---- lifecycle ------------------------------------------------------
    def close(self) -> None:
        """Delete the temp directory and all spilled frames."""
        if self._closed:
            return
        shutil.rmtree(self._dir, ignore_errors=True)
        self._index.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
