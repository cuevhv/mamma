"""Per-camera metadata loading from the ``ma_cap`` step's npz output.

The ``ma_cap`` step writes one ``<cam_name>.npz`` per camera into
``<seq>/gt/`` with this schema:

* ``cam_name`` (string-like) — camera identifier
* ``cam_int`` ``(3, 3)`` — intrinsic ``K``
* ``cam_ext`` ``(4, 4)`` — world->cam transform (``T_cam_world``)
* ``cam_img_w``, ``cam_img_h`` (int-like) — image dimensions
* ``img_abs_path`` ``(F,)`` strings — absolute frame paths
* ``img_rel_path`` ``(F,)`` strings — frame paths relative to a dataset root
* (optional) ``fps`` (int-like)

Vendored and cleaned from the upstream ``engine/systems_mv.py::
MultiViewSystem._load_cameras``. Differences:

* ``Camera`` is a frozen dataclass with native numpy types.
* No silent ``cam_int * resize_factor`` mutation; resize for display is
  done explicitly via :meth:`Camera.scaled` when needed.
* The legacy "global" pseudo-camera (cluster artifact) is dropped silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _to_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return str(value.tolist())
    return str(value)


@dataclass(frozen=True)
class Camera:
    """A single camera at its original (un-rescaled) resolution.

    ``distortion_model`` + ``distortion_coeffs`` are duck-typed to match
    ``capture.calibration.Camera`` so :func:`capture.undistort_rgb` can
    consume this dataclass directly (no shim needed).
    """

    name: str
    intrinsics: np.ndarray              # (3, 3) float64
    extrinsics: np.ndarray              # (4, 4) float64, T_cam_world
    width: int
    height: int
    image_paths: Optional[List[str]] = None    # absolute paths; None if not present
    video_path: Optional[str] = None           # absolute path to MP4 if frames live in a video
    # Canonical frame range owned by ma_cap. When set, downstream readers
    # (overlay.VideoFrameReader) slice the source video to [frame_start, frame_end)
    # so that local-index 0 maps to source-video frame `frame_start` —
    # matching the indexing of meshes/joints produced by ma_3d.
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    fps: Optional[int] = None
    distortion_model: str = "radtan"           # "radtan"/"opencv_brown" (OpenCV) or "vicon_radial_2"
    distortion_coeffs: tuple = (0.0, 0.0, 0.0, 0.0)   # zeros => undistortion is a no-op

    def scaled(self, factor: float) -> "Camera":
        """Return a copy with intrinsics, width, and height scaled by ``factor``.

        Useful when the Rerun viewer or overlay renderer wants a smaller
        display image. The bottom-right of ``intrinsics`` is held at 1.0.
        """
        if factor == 1.0:
            return self
        K = self.intrinsics.copy()
        K[0, 0] *= factor
        K[1, 1] *= factor
        K[0, 2] *= factor
        K[1, 2] *= factor
        K[2, 2] = 1.0
        return Camera(
            name=self.name,
            intrinsics=K,
            extrinsics=self.extrinsics,
            width=max(1, int(round(self.width * factor))),
            height=max(1, int(round(self.height * factor))),
            image_paths=self.image_paths,
            video_path=self.video_path,
            frame_start=self.frame_start,
            frame_end=self.frame_end,
            fps=self.fps,
            distortion_model=self.distortion_model,
            distortion_coeffs=self.distortion_coeffs,
        )


def load_cameras(
    gt_dir,
    *,
    cam_names: Optional[Sequence[str]] = None,
    drop_global: bool = True,
) -> List[Camera]:
    """Load every ``<cam>.npz`` file from a ``ma_cap`` ``gt/`` directory.

    Args:
        gt_dir: Path to ``<seq>/gt/`` containing per-camera npz files.
        cam_names: Optional whitelist of camera names. Missing names raise
            ``KeyError`` (early failure, easier to debug than silent drop).
        drop_global: If True (default), the cluster-only ``"global"``
            pseudo-camera entry is filtered out.

    Returns:
        Cameras sorted by name. Raises ``FileNotFoundError`` if no per-camera
        npz files exist in ``gt_dir``.
    """
    gt = Path(gt_dir)
    if not gt.is_dir():
        raise FileNotFoundError(f"camera metadata dir not found: {gt}")

    npz_paths = sorted(glob(str(gt / "*.npz")))
    cameras: List[Camera] = []
    for npz_path in npz_paths:
        cam = _load_one(npz_path)
        if cam is None:
            continue
        if drop_global and cam.name.lower() == "global":
            continue
        cameras.append(cam)

    if not cameras:
        raise FileNotFoundError(f"no readable camera npz files in {gt}")

    if cam_names is not None:
        wanted = list(cam_names)
        by_name = {c.name: c for c in cameras}
        missing = [n for n in wanted if n not in by_name]
        if missing:
            raise KeyError(
                f"camera(s) not found in {gt}: {missing} "
                f"(available: {sorted(by_name)})"
            )
        cameras = [by_name[n] for n in wanted]

    cameras.sort(key=lambda c: c.name)
    return cameras


def _load_one(npz_path: str) -> Optional[Camera]:
    data = np.load(npz_path, allow_pickle=True)
    try:
        files = set(data.files)
        if not {"cam_name", "cam_int", "cam_ext"}.issubset(files):
            return None

        name = _to_str(data["cam_name"])
        K = np.asarray(data["cam_int"], dtype=np.float64).reshape(3, 3)
        ext = np.asarray(data["cam_ext"], dtype=np.float64).reshape(4, 4)
        width = int(np.asarray(data["cam_img_w"]).item()) if "cam_img_w" in files else 0
        height = int(np.asarray(data["cam_img_h"]).item()) if "cam_img_h" in files else 0

        image_paths: Optional[List[str]] = None
        if "img_abs_path" in files:
            image_paths = [_to_str(p) for p in data["img_abs_path"].tolist()]
        elif "img_rel_path" in files:
            image_paths = [_to_str(p) for p in data["img_rel_path"].tolist()]

        # Videos-mode: ma_cap (capture/run_ma_cap.py) writes video_path on
        # per-camera NPZs when frames came from MP4s. Used by the overlay
        # renderer to read backgrounds via VideoFrameReader.
        video_path: Optional[str] = None
        if "video_path" in files:
            raw = data["video_path"]
            s = _to_str(raw)
            if s:  # ignore empty-string fallback written in image-mode
                video_path = s

        # ma_cap-owned canonical frame range. Used by the overlay renderer
        # so video frames stay aligned with ma_3d's per-frame meshes when
        # only a sub-range was processed.
        frame_start: Optional[int] = None
        frame_end: Optional[int] = None
        if "frame_start" in files:
            try:
                frame_start = int(np.asarray(data["frame_start"]).item())
            except (ValueError, TypeError):
                frame_start = None
        if "frame_end" in files:
            try:
                frame_end = int(np.asarray(data["frame_end"]).item())
            except (ValueError, TypeError):
                frame_end = None

        fps: Optional[int] = None
        if "fps" in files:
            try:
                fps = int(np.asarray(data["fps"]).item())
            except (ValueError, TypeError):
                fps = None

        # Distortion: ma_cap writes vicon_radial_2 as a 5-float array,
        # or as None (stored as a 0-d object array) when not applicable.
        distortion_model = "radtan"
        distortion_coeffs: tuple = (0.0, 0.0, 0.0, 0.0)
        if "vicon_radial_2" in files:
            v2 = np.asarray(data["vicon_radial_2"])
            if v2.shape == (5,):
                distortion_model = "vicon_radial_2"
                distortion_coeffs = tuple(float(v) for v in v2.tolist())

        return Camera(
            name=name,
            intrinsics=K,
            extrinsics=ext,
            width=width,
            height=height,
            image_paths=image_paths,
            video_path=video_path,
            frame_start=frame_start,
            frame_end=frame_end,
            fps=fps,
            distortion_model=distortion_model,
            distortion_coeffs=distortion_coeffs,
        )
    finally:
        data.close()


@dataclass(frozen=True)
class MultiViewCameras:
    """Convenience container over an ordered list of :class:`Camera`.

    Iteration, ``len``, ``__getitem__`` (by name), and ``names`` are
    supported. Most callers just want ``MultiViewCameras.load(...).cameras``.
    """

    cameras: Tuple[Camera, ...] = field(default_factory=tuple)

    @classmethod
    def load(
        cls,
        gt_dir,
        *,
        cam_names: Optional[Sequence[str]] = None,
        drop_global: bool = True,
    ) -> "MultiViewCameras":
        return cls(tuple(load_cameras(gt_dir, cam_names=cam_names, drop_global=drop_global)))

    @classmethod
    def from_calibration(
        cls,
        calibration_path: str,
        *,
        cam_names: Sequence[str],
        videos_dir: Optional[str] = None,
        images_root_dir: Optional[str] = None,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
    ) -> "MultiViewCameras":
        """Build :class:`Camera` objects in-memory from a calibration file.

        Skips the per-camera NPZ scaffolding when ma_vis is invoked
        standalone (videos/images-root workflows). For each requested
        ``cam_name``:

        - intrinsics + extrinsics + width/height come from the calibration,
        - ``video_path`` is set to ``<videos_dir>/<cam_name>.mp4`` if that
          file exists (overlay reads frames via :class:`VideoFrameReader`),
        - ``image_paths`` is set to a sorted list under
          ``<images_root_dir>/<cam_name>/`` if that dir contains images.

        Cameras present in the calibration but not listed in ``cam_names``
        are silently dropped; the reverse (a name in ``cam_names`` missing
        from the calibration) emits a warning and is skipped.
        """
        # Lazy import — capture/ lives in the superproject. Bump sys.path
        # so this works when invoked with cwd=visualization/ or from a
        # multiprocessing worker.
        import os as _os
        import sys as _sys
        _repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from capture import load_calibration  # noqa: E402

        calib = load_calibration(calibration_path)

        _IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        cameras: List[Camera] = []
        for cam_name in cam_names:
            capt_cam = calib.cameras.get(cam_name)
            if capt_cam is None:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "from_calibration: camera %r not in calibration; skipping", cam_name,
                )
                continue

            video_path: Optional[str] = None
            image_paths: Optional[List[str]] = None
            if videos_dir:
                vp = _os.path.join(videos_dir, f"{cam_name}.mp4")
                if _os.path.isfile(vp):
                    video_path = _os.path.abspath(vp)
            elif images_root_dir:
                cam_dir = _os.path.join(images_root_dir, cam_name)
                if _os.path.isdir(cam_dir):
                    paths = sorted(
                        _os.path.join(cam_dir, f)
                        for f in _os.listdir(cam_dir)
                        if f.endswith(_IMAGE_EXTS)
                    )
                    if paths:
                        image_paths = [_os.path.abspath(p) for p in paths]

            cameras.append(Camera(
                name=cam_name,
                intrinsics=np.asarray(capt_cam.intrinsics, dtype=np.float64).reshape(3, 3),
                extrinsics=np.asarray(capt_cam.T_cam_world, dtype=np.float64).reshape(4, 4),
                width=int(capt_cam.width),
                height=int(capt_cam.height),
                image_paths=image_paths,
                video_path=video_path,
                frame_start=frame_start,
                frame_end=frame_end,
                fps=None,
                distortion_model=str(capt_cam.distortion_model),
                distortion_coeffs=tuple(float(v) for v in capt_cam.distortion_coeffs),
            ))
        return cls(tuple(cameras))

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.cameras]

    def __iter__(self) -> Iterable[Camera]:
        return iter(self.cameras)

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, name: str) -> Camera:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(f"camera not found: {name!r}; available: {self.names}")
