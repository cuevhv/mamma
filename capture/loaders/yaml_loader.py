"""YAML calibration loader.

Schema (single rig per file, this iteration)::

    # Optional, recommended — declare your extrinsics convention explicitly:
    extrinsics_convention: cam2world   # default; translation/quaternion = camera
                                       # pose in world. Use 'world2cam' if they
                                       # describe the world->cam transform instead.
    quaternion_order: wxyz             # default Hamilton; use 'xyzw' if needed.
    cameras:
      cam0:
        camera_model: pinhole
        # One of:
        distortion_model: radtan
        distortion_coeffs: [k1, k2, p1, p2]     # Brown-Conrady radial-tangential
        # ...or:
        distortion_model: vicon_radial_2
        distortion_coeffs: [pp_x, pp_y, rad_1, rad_2, rad_3]  # Vicon XCP format
        intrinsics: [fx, fy, cx, cy]            # pixels
        resolution: [W, H]                       # ints
        translation: [tx, ty, tz]                # meters; camera position in world
        rotation_quaternion: [w, x, y, z]        # Hamilton, unit norm; camera orientation in world

Conventions: OpenCV-style camera frame (+X right, +Y down, +Z forward).
The translation/quaternion encode ``T_world_cam`` (camera's pose in the
world); the loader inverts to also produce ``T_cam_world`` as expected
by the projection pipeline.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .._quat import hamilton_quat_to_rotmat
from ..calibration import Camera, CalibrationError


_REQUIRED_CAM_FIELDS = (
    "camera_model",
    "distortion_model",
    "intrinsics",
    "distortion_coeffs",
    "resolution",
    "translation",
    "rotation_quaternion",
)


def load(path: Path) -> Dict[str, Camera]:
    """Parse a YAML calibration file and return ``{cam_name: Camera}``."""
    try:
        import yaml  # type: ignore
    except ImportError as e:  # pragma: no cover  (env error, not a code bug)
        raise CalibrationError(
            "YAML calibration support requires pyyaml; "
            "install it with `pip install pyyaml` or add it to your env."
        ) from e

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise CalibrationError(f"{path}: invalid YAML: {e}") from e

    if not isinstance(data, dict):
        raise CalibrationError(
            f"{path}: top-level must be a mapping with a 'cameras' key"
        )

    cameras_raw = data.get("cameras")
    if not isinstance(cameras_raw, dict) or not cameras_raw:
        raise CalibrationError(
            f"{path}: 'cameras' must be a non-empty mapping"
        )

    # Optional, explicit extrinsics convention (default keeps the historical
    # behavior). Declaring this removes the #1 source of calibration confusion.
    convention = str(data.get("extrinsics_convention", "cam2world")).lower()
    if convention not in ("cam2world", "world2cam"):
        raise CalibrationError(
            f"{path}: extrinsics_convention must be 'cam2world' or 'world2cam', "
            f"got {convention!r}"
        )
    quat_order = str(data.get("quaternion_order", "wxyz")).lower()
    if quat_order not in ("wxyz", "xyzw"):
        raise CalibrationError(
            f"{path}: quaternion_order must be 'wxyz' or 'xyzw', got {quat_order!r}"
        )

    errors: List[str] = []
    cameras: Dict[str, Camera] = {}
    for cam_name, cam in cameras_raw.items():
        if not isinstance(cam, dict):
            errors.append(f"cameras.{cam_name}: must be a mapping")
            continue
        cam_errors: List[str] = []
        camera = _parse_camera(str(cam_name), cam, cam_errors,
                               convention=convention, quat_order=quat_order)
        if cam_errors:
            errors.extend(cam_errors)
        elif camera is not None:
            cameras[str(cam_name)] = camera

    if errors:
        raise CalibrationError(
            f"{path}: YAML calibration validation failed:\n  - "
            + "\n  - ".join(errors)
        )

    return cameras


def _parse_camera(
    cam_name: str, cam: Dict[str, Any], errors: List[str],
    *, convention: str = "cam2world", quat_order: str = "wxyz",
) -> Camera | None:
    missing = [k for k in _REQUIRED_CAM_FIELDS if k not in cam]
    if missing:
        for k in missing:
            errors.append(f"cameras.{cam_name}.{k}: required")
        return None

    if cam["camera_model"] != "pinhole":
        errors.append(
            f"cameras.{cam_name}.camera_model: only \"pinhole\" supported, "
            f"got {cam['camera_model']!r}"
        )

    distortion_model = cam["distortion_model"]
    if distortion_model == "radtan":
        n_dist, dist_label = 4, "[k1,k2,p1,p2] for radtan"
    elif distortion_model == "vicon_radial_2":
        n_dist, dist_label = 5, "[pp_x,pp_y,rad_1,rad_2,rad_3] for vicon_radial_2"
    else:
        errors.append(
            f"cameras.{cam_name}.distortion_model: must be one of "
            f"['radtan', 'vicon_radial_2'], got {distortion_model!r}"
        )
        n_dist, dist_label = 4, ""

    intr = _check_finite_seq(cam["intrinsics"], 4,
                             f"cameras.{cam_name}.intrinsics", errors,
                             label="[fx,fy,cx,cy]")
    dist = _check_finite_seq(cam["distortion_coeffs"], n_dist,
                             f"cameras.{cam_name}.distortion_coeffs", errors,
                             label=dist_label)
    res = cam["resolution"]
    width = height = None
    if not isinstance(res, (list, tuple)) or len(res) != 2:
        errors.append(
            f"cameras.{cam_name}.resolution: expected length 2 [W,H], "
            f"got {res!r}"
        )
    else:
        try:
            width, height = int(res[0]), int(res[1])
            if width <= 0 or height <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                f"cameras.{cam_name}.resolution: width and height must be "
                f"positive integers, got {res!r}"
            )
            width = height = None

    trans = _check_finite_seq(cam["translation"], 3,
                              f"cameras.{cam_name}.translation", errors,
                              label="[tx,ty,tz]")
    quat_raw = cam["rotation_quaternion"]
    quat = _check_finite_seq(quat_raw, 4,
                             f"cameras.{cam_name}.rotation_quaternion", errors,
                             label="[w,x,y,z], Hamilton")
    if quat is not None:
        norm = math.sqrt(sum(v * v for v in quat))
        if not math.isfinite(norm) or norm < 1e-12 or abs(norm - 1.0) > 1e-3:
            errors.append(
                f"cameras.{cam_name}.rotation_quaternion: not unit norm "
                f"(|q|={norm:.6g}); fix the calibration or pass a unit quaternion"
            )

    if errors:
        return None

    fx, fy, cx, cy = intr  # type: ignore[misc]
    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    q = list(quat)                                  # type: ignore[arg-type]
    if quat_order == "xyzw":
        q = [q[3], q[0], q[1], q[2]]                # -> Hamilton [w,x,y,z]
    R = hamilton_quat_to_rotmat(q)
    t = np.asarray(trans, dtype=np.float64)

    T_world_cam = np.eye(4, dtype=np.float64)
    T_cam_world = np.eye(4, dtype=np.float64)
    if convention == "world2cam":
        # translation/quaternion describe world->cam (R·X_world + t = X_cam).
        T_cam_world[:3, :3] = R
        T_cam_world[:3, 3] = t
        T_world_cam[:3, :3] = R.T
        T_world_cam[:3, 3] = -R.T @ t
    else:
        # cam2world (default): they describe the camera's pose in the world.
        T_world_cam[:3, :3] = R
        T_world_cam[:3, 3] = t
        T_cam_world[:3, :3] = R.T
        T_cam_world[:3, 3] = -R.T @ t

    extra_keys = set(cam.keys()) - set(_REQUIRED_CAM_FIELDS) - {"cam_name"}
    if extra_keys:
        import logging
        logging.getLogger(__name__).warning(
            "cameras.%s: ignoring unknown keys %s", cam_name, sorted(extra_keys)
        )

    return Camera(
        name=cam_name,
        width=width,           # type: ignore[arg-type]
        height=height,         # type: ignore[arg-type]
        intrinsics=K,
        distortion_model=distortion_model,
        distortion_coeffs=tuple(float(v) for v in dist),  # type: ignore[arg-type]
        T_cam_world=T_cam_world,
        T_world_cam=T_world_cam,
    )


def _check_finite_seq(
    seq: Any, expected_len: int, path: str, errors: List[str], *, label: str = ""
) -> Sequence[float] | None:
    """Validate a length-N sequence of finite floats. Returns the coerced list, or None."""
    if not isinstance(seq, (list, tuple)):
        errors.append(f"{path}: expected list of length {expected_len} {label}, "
                      f"got {type(seq).__name__}")
        return None
    if len(seq) != expected_len:
        errors.append(f"{path}: expected length {expected_len} {label}, "
                      f"got {len(seq)}")
        return None
    out: List[float] = []
    for i, v in enumerate(seq):
        try:
            f = float(v)
        except (TypeError, ValueError):
            errors.append(f"{path}[{i}]: not a number ({v!r})")
            return None
        if not math.isfinite(f):
            errors.append(f"{path}[{i}]: not finite ({v!r})")
            return None
        out.append(f)
    return out
