"""OpenCV FileStorage calibration loader.

Reads the widely-used OpenCV ``%YAML:1.0`` / ``!!opencv-matrix`` calibration
format: per-camera ``K`` (3x3 intrinsics), ``D`` (Brown-Conrady distortion),
``R`` (3x3 rotation), ``T`` (3x1 translation), using OpenCV's standard
**world->camera** extrinsic convention (``X_cam = R @ X_world + T``).

Accepts either:
  * a single file holding one camera (named by the file stem), or
  * a directory of per-camera files (camera name = each file's stem) — the common
    "one calibration file per camera" rig export.

Parsing is done with a tolerant PyYAML reader rather than ``cv2.FileStorage``,
which chokes on real-world files that contain tabs or the non-standard
``%YAML:1.0`` directive. Distortion is normalized to 5-parameter ``opencv_brown``
(k1, k2, p1, p2, k3). Resolution is read from ``image_width``/``image_height``
when present, otherwise approximated from the principal point (with a warning),
since OpenCV calibration files frequently omit it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..calibration import Camera, CalibrationError

log = logging.getLogger(__name__)

_CV_EXTS = (".yml", ".yaml")
_OPENCV_MATRIX_TAG = "tag:yaml.org,2002:opencv-matrix"


def looks_like_opencv_fs(path) -> bool:
    """True if ``path`` is an OpenCV FileStorage file (header / matrix markers)."""
    try:
        with open(path, "r", errors="ignore") as f:
            head = f.read(4096)
    except OSError:
        return False
    return ("%YAML:1.0" in head) or ("opencv-matrix" in head)


def _opencv_matrix(loader, node):
    d = loader.construct_mapping(node, deep=True)
    rows, cols = int(d.get("rows", 0)), int(d.get("cols", 0))
    data = np.asarray(d.get("data", []), dtype=np.float64)
    return data.reshape(rows, cols) if (rows and cols) else data


def parse_fs(path) -> Dict[str, object]:
    """Tolerant OpenCV FileStorage reader -> ``{key: value}``.

    Matrices come back as ``np.ndarray``; scalars as int/float/str/list. Handles
    the non-standard ``%YAML:1.0`` directive, tab indentation, and the
    ``!!opencv-matrix`` tag — all of which trip ``cv2.FileStorage`` and a plain
    ``yaml.safe_load``."""
    import yaml
    text = Path(path).read_text(errors="ignore")
    text = text.replace("%YAML:1.0", "")   # non-standard directive -> drop
    text = text.replace("\t", " ")          # tabs are illegal YAML indentation

    class _Loader(yaml.SafeLoader):
        pass
    _Loader.add_constructor(_OPENCV_MATRIX_TAG, _opencv_matrix)
    _Loader.add_constructor("!opencv-matrix", _opencv_matrix)

    try:
        data = yaml.load(text, Loader=_Loader)
    except yaml.YAMLError as e:
        raise CalibrationError(f"{path}: could not parse OpenCV FileStorage: {e}") from e
    if not isinstance(data, dict):
        raise CalibrationError(f"{path}: unexpected OpenCV FileStorage structure")
    return data


def _as_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _camera_from_KDRT(name, K, D, R, T, *, width=None, height=None) -> Camera:
    """Build a Camera from OpenCV-style world->cam K/D/R/T."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(3)

    d = np.asarray(D, dtype=np.float64).reshape(-1).tolist() if D is not None else []
    d = (d + [0.0] * 5)[:5]   # pad/truncate to opencv_brown (k1,k2,p1,p2,k3)

    # world->cam: T_cam_world = [R | T]; invert for the camera pose in world.
    T_cam_world = np.eye(4, dtype=np.float64)
    T_cam_world[:3, :3] = R
    T_cam_world[:3, 3] = T
    T_world_cam = np.eye(4, dtype=np.float64)
    T_world_cam[:3, :3] = R.T
    T_world_cam[:3, 3] = -R.T @ T

    if not width or not height:
        width = int(round(2 * K[0, 2]))
        height = int(round(2 * K[1, 2]))
        log.warning(
            "camera %s: resolution not in calibration; inferred %dx%d from the "
            "principal point. Add 'image_width'/'image_height' for an exact size.",
            name, width, height,
        )

    return Camera(
        name=name,
        width=int(width),
        height=int(height),
        intrinsics=K,
        distortion_model="opencv_brown",
        distortion_coeffs=tuple(float(v) for v in d),
        T_cam_world=T_cam_world,
        T_world_cam=T_world_cam,
    )


def _load_one_file(path: Path, name: str) -> Camera:
    fs = parse_fs(path)
    K = fs.get("K")
    if K is None:
        K = fs.get("camera_matrix")
    D = fs.get("D")
    if D is None:
        D = fs.get("distortion_coefficients")
    R, T = fs.get("R"), fs.get("T")
    width = _as_int(fs.get("image_width"))
    height = _as_int(fs.get("image_height"))
    if K is None or R is None or T is None:
        raise CalibrationError(
            f"{path}: OpenCV calibration needs K, R and T "
            f"(found K={K is not None}, R={R is not None}, T={T is not None})"
        )
    return _camera_from_KDRT(name, K, D, R, T, width=width, height=height)


def load(path) -> Dict[str, Camera]:
    """Parse an OpenCV FileStorage calibration (file or directory of files)."""
    p = Path(path)
    cameras: Dict[str, Camera] = {}
    if p.is_dir():
        files: List[Path] = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in _CV_EXTS and looks_like_opencv_fs(f)
        )
        if not files:
            raise CalibrationError(
                f"{p}: no OpenCV FileStorage (.yml/.yaml) calibration files found"
            )
        for f in files:
            cameras[f.stem] = _load_one_file(f, f.stem)
    else:
        cameras[p.stem] = _load_one_file(p, p.stem)
    if not cameras:
        raise CalibrationError(f"{p}: no cameras parsed")
    return cameras
