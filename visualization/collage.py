"""Per-sequence preview collage video.

Tiles up to N per-camera overlay videos into a single grid mp4 for quick
inspection. Vendored from ``run_ma_vis.py::_create_preview_collage``.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .overlay import _reencode_to_h264

log = logging.getLogger(__name__)


def _fit_tile(frame, tile_w: int, tile_h: int):
    """Aspect-preserving resize + centered black padding onto a tile canvas.

    Mixed-orientation rigs (portrait + landscape overlays) share one tile
    size, so a plain resize would stretch every source whose aspect differs
    from the tile's — letterbox/pillarbox instead.
    """
    import cv2

    h, w = frame.shape[:2]
    if (h, w) == (tile_h, tile_w):
        return frame
    scale = min(tile_w / w, tile_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((tile_h, tile_w, 3), dtype=frame.dtype)
    x0 = (tile_w - new_w) // 2
    y0 = (tile_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _grid_shape(n: int) -> Tuple[int, int]:
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _ordered_videos(
    video_paths: Sequence, selected_cam_names: Optional[Sequence[str]]
) -> List[Tuple[str, Path]]:
    """Map ``[<dir>/<cam>.mp4, ...]`` to ``[(cam_name, path), ...]``.

    If ``selected_cam_names`` is given, only those cameras are returned, in
    that order. Otherwise sort alphabetically.
    """
    by_cam = {}
    for raw in video_paths:
        path = Path(raw)
        by_cam[path.stem] = path

    if selected_cam_names:
        return [(c, by_cam[c]) for c in selected_cam_names if c in by_cam]
    return [(c, by_cam[c]) for c in sorted(by_cam)]


def make_preview_collage(
    video_paths: Sequence,
    out_path,
    *,
    cam_names: Optional[Sequence[str]] = None,
    max_videos: int = 4,
    label_color: Tuple[int, int, int] = (255, 255, 255),
) -> bool:
    """Tile up to ``max_videos`` overlay videos into a single grid mp4.

    The output canvas is ``cols * tile_w`` by ``rows * tile_h``, where
    ``tile_w/h`` is the smallest source size across the chosen videos.
    Each tile is annotated with its camera name in the top-left corner.

    Args:
        video_paths: Per-camera mp4 paths.
        out_path: Where to write the collage mp4 (parent dir is created).
        cam_names: Optional whitelist; otherwise alphabetic order is used.
        max_videos: Cap on number of tiles. Default 4.
        label_color: BGR colour for the per-tile camera-name label.

    Returns:
        True if the collage was written, False otherwise (no readable
        videos, or the writer couldn't be opened).
    """
    import cv2

    selected = _ordered_videos(video_paths, cam_names)
    if max_videos > 0:
        selected = selected[:max_videos]
    if not selected:
        log.info("no overlay videos to collage")
        return False

    log.info("preview collage: cameras = %s", [c for c, _ in selected])

    caps = []
    try:
        for cam_name, path in selected:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                log.warning("cannot open %s for preview collage", path)
                cap.release()
                continue
            caps.append((cam_name, cap))

        if not caps:
            return False

        widths, heights, fps_values = [], [], []
        for _, cap in caps:
            widths.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            heights.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps_values.append(float(cap.get(cv2.CAP_PROP_FPS)))

        valid_w = [w for w in widths if w > 0]
        valid_h = [h for h in heights if h > 0]
        valid_fps = [f for f in fps_values if f > 0]
        tile_w = min(valid_w) if valid_w else 640
        tile_h = min(valid_h) if valid_h else 360
        out_fps = min(valid_fps) if valid_fps else 30.0

        rows, cols = _grid_shape(len(caps))
        out_w, out_h = cols * tile_w, rows * tile_h

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(out_fps), (out_w, out_h)
        )
        if not writer.isOpened():
            log.warning("preview collage: failed to open writer at %s", out_path)
            return False

        try:
            n_written = 0
            while True:
                tiles = []
                for cam_name, cap in caps:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        tiles = None
                        break
                    frame = _fit_tile(frame, tile_w, tile_h)
                    cv2.putText(
                        frame, cam_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, label_color, 2, cv2.LINE_AA,
                    )
                    tiles.append(frame)
                if tiles is None:
                    break

                canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                for i, frame in enumerate(tiles):
                    r, c = divmod(i, cols)
                    canvas[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = frame
                writer.write(canvas)
                n_written += 1
        finally:
            writer.release()

        log.info(
            "preview collage: %s (%d frames, grid %dx%d, %d sources)",
            out_path, n_written, rows, cols, len(caps),
        )
        if n_written == 0:
            return False
        _reencode_to_h264(out_path)
        return True
    finally:
        for _, cap in caps:
            cap.release()
