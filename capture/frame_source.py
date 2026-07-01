"""Unified frame access abstraction for image-file and video (MP4) sources.

This module provides a common interface so that pipeline code can read frames
without knowing whether the underlying source is a directory of images or an
MP4 video file.

Usage::

    from capture.frame_source import ImageFileSource, VideoSource, frame_source_from_cam_data

    # From image file paths (NPZ mode)
    frames = ImageFileSource(sorted_path_list)
    img = frames.read_pil(0)

    # From MP4 video (video mode)
    from capture.video_reader import VideoFrameReader
    frames = VideoSource(VideoFrameReader("/path/to/video.mp4", start=10, end=50))
    img = frames.read_pil(0)   # frame 0 within the range (global frame 10)

    # Automatically from cam_data dict
    frames = frame_source_from_cam_data(cam_data)
"""

import os
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from PIL import Image

from .calibration import Camera
from .undistort import undistort_rgb


@runtime_checkable
class FrameSource(Protocol):
    """Protocol for reading video/image frames by index."""

    def read_pil(self, idx: int) -> Image.Image:
        """Read frame as a PIL Image (RGB)."""
        ...

    def read_rgb(self, idx: int) -> np.ndarray:
        """Read frame as an RGB numpy array (H, W, 3), uint8."""
        ...

    def __len__(self) -> int:
        """Number of frames available."""
        ...

    @property
    def frame_names(self) -> List[str]:
        """Ordered list of frame filenames (e.g. ``['000010.jpg', ...]``)."""
        ...

    @property
    def image_size(self) -> Tuple[int, int]:
        """(width, height) of frames."""
        ...

    @property
    def cam_name(self) -> str:
        """Camera / source name (e.g. ``'IOI_09'``)."""
        ...


class ImageFileSource:
    """Frame source backed by image files on disk.

    Args:
        paths: Sorted list of absolute image file paths.
        camera_name: Camera name override. Derived from directory if *None*.
        camera: Optional :class:`Camera` for undistortion. Required when
            ``undistort=True``.
        undistort: When ``True``, apply Vicon-radial-2 undistortion to
            every frame returned by :meth:`read_pil` / :meth:`read_rgb`.
            Default ``False`` (frames returned as-is).
    """

    def __init__(self, paths: List[str], camera_name: str = None,
                 *, camera: Optional[Camera] = None, undistort: bool = False):
        if not paths:
            raise ValueError("ImageFileSource requires at least one file path.")
        if undistort and camera is None:
            raise ValueError("ImageFileSource(undistort=True) requires a camera.")
        self._paths = list(paths)
        self._cam_name = camera_name or os.path.basename(os.path.dirname(paths[0]))
        self._image_size = None
        self._camera = camera
        self._undistort = undistort

    def read_pil(self, idx: int) -> Image.Image:
        if idx < 0 or idx >= len(self._paths):
            raise IndexError(f"Frame index {idx} out of range [0, {len(self._paths)})")
        img = Image.open(self._paths[idx]).convert("RGB")
        if self._undistort:
            arr = undistort_rgb(np.asarray(img), self._camera)
            img = Image.fromarray(arr)
        return img

    def read_rgb(self, idx: int) -> np.ndarray:
        arr = np.asarray(Image.open(self._paths[idx]).convert("RGB"))
        if self._undistort:
            arr = undistort_rgb(arr, self._camera)
        return arr

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def frame_names(self) -> List[str]:
        return [os.path.basename(p) for p in self._paths]

    @property
    def image_size(self) -> Tuple[int, int]:
        if self._image_size is None:
            with Image.open(self._paths[0]) as img:
                self._image_size = img.size  # (width, height)
        return self._image_size

    @property
    def cam_name(self) -> str:
        return self._cam_name

    @property
    def paths(self) -> List[str]:
        """Access underlying file paths (for backward compatibility)."""
        return self._paths

    def __repr__(self):
        return f"ImageFileSource(n={len(self)}, cam={self._cam_name})"


class VideoSource:
    """Frame source backed by an MP4 video via :class:`VideoFrameReader`.

    Args:
        reader: A ``VideoFrameReader`` instance (from ``capture.video_reader``).
        camera_name: Camera name override. Derived from video filename if *None*.
        camera: Optional :class:`Camera` for undistortion. Required when
            ``undistort=True``.
        undistort: When ``True``, apply Vicon-radial-2 undistortion to
            every frame returned by :meth:`read_pil` / :meth:`read_rgb`.
            Default ``False``.
    """

    def __init__(self, reader, camera_name: str = None,
                 *, camera: Optional[Camera] = None, undistort: bool = False):
        if undistort and camera is None:
            raise ValueError("VideoSource(undistort=True) requires a camera.")
        self._reader = reader
        self._cam_name = camera_name or os.path.splitext(os.path.basename(reader.video_path))[0]
        self._camera = camera
        self._undistort = undistort

    def read_pil(self, idx: int) -> Image.Image:
        if not self._undistort:
            return self._reader.read_pil(idx)
        arr = undistort_rgb(self._reader.read_rgb(idx), self._camera)
        return Image.fromarray(arr)

    def read_rgb(self, idx: int) -> np.ndarray:
        arr = self._reader.read_rgb(idx)
        if self._undistort:
            arr = undistort_rgb(arr, self._camera)
        return arr

    def __len__(self) -> int:
        return len(self._reader)

    @property
    def frame_names(self) -> List[str]:
        """Synthetic frame names matching the global frame index pattern."""
        return [f"{self._reader.start + i:06d}.jpg" for i in range(len(self._reader))]

    @property
    def image_size(self) -> Tuple[int, int]:
        return (self._reader.width, self._reader.height)

    @property
    def cam_name(self) -> str:
        return self._cam_name

    @property
    def video_path(self) -> str:
        """Absolute path to the MP4 file."""
        return self._reader.video_path

    @property
    def reader(self):
        """Access the underlying VideoFrameReader."""
        return self._reader

    def __repr__(self):
        return (
            f"VideoSource(n={len(self)}, cam={self._cam_name}, "
            f"range=[{self._reader.start}:{self._reader.end}])"
        )


def _cam_data_field(cam_data: dict, key: str):
    """Read ``key`` from ``cam_data``, unwrapping 0-d numpy scalars."""
    v = cam_data.get(key)
    if v is None:
        return None
    try:
        # numpy 0-d arrays / scalars
        if hasattr(v, "shape") and v.shape == ():
            return v.item()
    except Exception:  # noqa: BLE001
        pass
    return v


def _camera_from_cam_data(cam_data: dict) -> Optional[Camera]:
    """Build a minimal :class:`Camera` for undistortion from a cam_data dict.

    Reads the distortion fields ``ma_cap`` writes into each per-camera NPZ --
    the generic ``distortion_model`` / ``distortion_coeffs`` pair first, falling
    back to the legacy ``vicon_radial_2`` key. Intrinsics come from ``cam_int``
    (needed for the OpenCV radtan / opencv_brown models; the Vicon pixel-space
    model only needs width/height). ``T_cam_world`` / ``T_world_cam`` are not
    used by undistortion, so identity placeholders are fine.

    Returns ``None`` when no usable distortion information is present, so callers
    can no-op gracefully. This is what lets ``ma_2d`` / ``ma_masks`` undistort
    straight from an ``--ma_cap_dir`` NPZ without a separate ``--calibration``.
    """
    model: Optional[str] = None
    coeffs: Optional[tuple] = None
    dm = _cam_data_field(cam_data, 'distortion_model')
    dc = cam_data.get('distortion_coeffs')
    if dm is not None and dc is not None:
        dc_arr = np.asarray(dc)
        if dc_arr.ndim == 1 and dc_arr.size >= 4:
            model = str(dm)
            coeffs = tuple(float(v) for v in dc_arr.tolist())
    if model is None:                                  # legacy NPZs
        v2 = cam_data.get('vicon_radial_2')
        if v2 is not None:
            v2_arr = np.asarray(v2)
            if v2_arr.shape == (5,):
                model = 'vicon_radial_2'
                coeffs = tuple(float(v) for v in v2_arr.tolist())
    if model is None:
        return None

    K = cam_data.get('cam_int')
    intrinsics = (
        np.asarray(K, dtype=np.float64).reshape(3, 3) if K is not None else None
    )
    w = _cam_data_field(cam_data, 'cam_img_w')
    h = _cam_data_field(cam_data, 'cam_img_h')
    return Camera(
        name=str(cam_data.get('cam_name', '')),
        width=int(w) if w is not None else 0,
        height=int(h) if h is not None else 0,
        intrinsics=intrinsics,
        distortion_model=model,
        distortion_coeffs=coeffs,
        T_cam_world=np.eye(4, dtype=np.float64),
        T_world_cam=np.eye(4, dtype=np.float64),
    )


def frame_source_from_cam_data(
    cam_data: dict,
    *,
    camera: Optional[Camera] = None,
    undistort: bool = False,
) -> "FrameSource":
    """Build the appropriate FrameSource from a cam_data dictionary.

    Precedence (first match wins):

    1. ``cam_data['frame_reader']`` — caller pre-built a VideoFrameReader
       (e.g. ma_masks's videos_dir path) → :class:`VideoSource`.
    2. ``cam_data['video_path']`` (non-empty) — lazily construct a
       :class:`VideoFrameReader` using the optional ``frame_start`` /
       ``frame_end`` fields from the same dict. Used by downstream
       steps that load a ma_cap NPZ: the NPZ already encodes the
       canonical range, so the step inherits it automatically.
    3. ``cam_data['img_abs_path']`` — :class:`ImageFileSource`.

    Pass ``camera`` + ``undistort=True`` to undistort every frame read
    (any supported lens model). As a convenience for callers that don't
    see the FrameSource construction site (e.g. ma_masks goes through
    several layers of indirection), the same values may be pre-stuffed
    into the ``cam_data`` dict under ``_undistort_camera`` / ``_undistort``;
    explicit kwargs win when set.

    When ``undistort`` is requested but no ``camera`` is supplied, the
    per-camera distortion that ma_cap wrote into the NPZ (``distortion_model``
    / ``distortion_coeffs``, or the legacy ``vicon_radial_2`` key) is used via
    :func:`_camera_from_cam_data`. If the dict carries no usable distortion,
    undistortion is silently disabled (no-op) rather than raising.
    """
    if camera is None:
        camera = cam_data.get('_undistort_camera')
    if not undistort:
        undistort = bool(cam_data.get('_undistort', False))
    # Fall back to the per-camera distortion the ma_cap NPZ carries when no
    # explicit camera was supplied (e.g. --undistort without --calibration in
    # chained --ma_cap_dir mode). If the dict has no usable distortion, disable
    # undistortion rather than erroring downstream -- a graceful no-op.
    if undistort and camera is None:
        camera = _camera_from_cam_data(cam_data)
        if camera is None:
            undistort = False
    cam_name = str(cam_data.get('cam_name', ''))

    reader = cam_data.get('frame_reader')
    if reader is None:
        # Lazy: cam_data has video_path (e.g. loaded from ma_cap NPZ).
        # Construct a VideoFrameReader that honours the canonical range.
        vp = _cam_data_field(cam_data, 'video_path')
        vp_str = str(vp) if vp is not None else ""
        if vp_str:
            from .video_reader import VideoFrameReader
            start = _cam_data_field(cam_data, 'frame_start')
            end = _cam_data_field(cam_data, 'frame_end')
            reader = VideoFrameReader(
                vp_str,
                start=int(start) if start is not None else None,
                end=int(end) if end is not None else None,
            )
    if reader is not None:
        return VideoSource(reader, camera_name=cam_name or None,
                           camera=camera, undistort=undistort)
    paths = cam_data['img_abs_path'].tolist()
    return ImageFileSource(paths, camera_name=cam_name or None,
                           camera=camera, undistort=undistort)
