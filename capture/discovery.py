"""Multi-camera discovery helpers: videos-dir and images-root layouts.

Lifted from ``segmentation/process_sequence.py`` so every step
(``ma_cap``, ``ma_masks``, ``ma_2d``, ``ma_3d``, ``ma_vis``) can share
the same on-disk conventions:

- ``<videos_dir>/<cam_name>.mp4`` (one MP4 per camera; filename stem
  is the camera name)
- ``<images_root_dir>/<cam_name>/<frame>.{jpg,png,...}`` (one
  subdirectory per camera)

Returns plain ``cam_data`` dicts compatible with
:func:`capture.frame_source.frame_source_from_cam_data`.
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def sort_frame_files(names: Iterable[str]) -> List[str]:
    """Frame order shared with the SAM frame loader: numeric when every file
    stem is an integer (1, 2, 10), lexicographic otherwise."""
    out = sorted(names)
    try:
        out.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        pass
    return out


def normalize_cam_name(name: str) -> str:
    """Strip ``.npz`` / ``.mp4`` extensions and surrounding whitespace."""
    name = name.strip()
    for ext in (".npz", ".mp4"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
    return name


def normalize_cam_names(cam_names: Optional[Iterable[str]]) -> Optional[List[str]]:
    """Parse a list of camera names (supports comma-separated or list)."""
    if not cam_names:
        return None
    cam_names = list(cam_names)
    if len(cam_names) == 1 and "," in cam_names[0]:
        cam_names = cam_names[0].split(",")
    cleaned = [c.strip() for c in cam_names if c and c.strip()]
    return cleaned or None


def find_video_files(
    videos_dir: str,
    cam_names: Optional[Iterable[str]] = None,
) -> List[str]:
    """Discover MP4 video files under ``videos_dir``, optionally filtered.

    The filename stem (e.g. ``cam04.mp4`` -> ``cam04``) is the camera
    name. Looks one level deep first, then recurses.
    """
    cam_names_set = None
    if cam_names:
        cam_names_set = {normalize_cam_name(c) for c in cam_names}

    video_files = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))
    if not video_files:
        video_files = sorted(
            glob.glob(os.path.join(videos_dir, "**", "*.mp4"), recursive=True)
        )

    if cam_names_set:
        video_files = [
            v for v in video_files
            if os.path.splitext(os.path.basename(v))[0] in cam_names_set
        ]
    logger.info("Discovered %d video files under '%s'.", len(video_files), videos_dir)
    return video_files


def find_image_cam_dirs(
    images_root_dir: str,
    cam_names: Optional[Iterable[str]] = None,
) -> List[str]:
    """Discover camera subdirectories containing image frames.

    Expected structure: ``<images_root_dir>/<cam_name>/<frame>.jpg``.

    Args:
        images_root_dir: Root directory containing one subdirectory per camera.
        cam_names: Optional list of camera names to filter by.

    Returns:
        Sorted list of camera directory paths (only dirs that contain images).
    """
    cam_names_set = None
    if cam_names:
        cam_names_set = {normalize_cam_name(c) for c in cam_names}

    cam_dirs = sorted(
        d for d in glob.glob(os.path.join(images_root_dir, "*"))
        if os.path.isdir(d)
    )

    if cam_names_set:
        cam_dirs = [d for d in cam_dirs if os.path.basename(d) in cam_names_set]

    cam_dirs = [
        d for d in cam_dirs
        if any(os.path.splitext(f)[1] in IMAGE_EXTENSIONS for f in os.listdir(d))
    ]

    logger.info(
        "Discovered %d camera image directories under '%s'.",
        len(cam_dirs), images_root_dir,
    )
    return cam_dirs


def cam_data_from_image_dir(
    cam_dir: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a ``cam_data`` dict from a directory of image frames.

    Args:
        cam_dir: Directory containing image files (jpg/png).
        start: First frame index (0-based, inclusive). None = 0.
        end: Last frame index (0-based, exclusive). None = all frames.

    Returns:
        Dict compatible with
        :func:`capture.frame_source.frame_source_from_cam_data`.
    """
    image_files = sort_frame_files(
        f for f in os.listdir(cam_dir)
        if os.path.splitext(f)[1] in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No image files found in '{cam_dir}'")

    if start is not None or end is not None:
        s = max(0, start or 0)
        e = min(len(image_files), end or len(image_files))
        image_files = image_files[s:e]

    cam_name = os.path.basename(cam_dir)
    abs_paths = [os.path.join(os.path.abspath(cam_dir), f) for f in image_files]

    logger.info(
        "ImageDir: '%s' - %d frames%s",
        cam_name, len(abs_paths),
        (f" (range [{start}:{end}])" if (start is not None or end is not None) else ""),
    )

    return {
        'cam_name': np.array(cam_name),
        'img_abs_path': np.array(abs_paths),
    }
