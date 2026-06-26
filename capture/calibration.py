"""Calibration data model and format-dispatching loader.

Three calibration file formats are supported:

* ``.yaml`` / ``.yml`` — pinhole + radtan, Hamilton ``[w,x,y,z]`` quaternion,
  meters, single-rig per file. The format we recommend for new users.
* ``.xcp`` — Vicon XML calibration export. Vicon 5-parameter radial
  distortion, JPL ``[x,y,z,w]`` quaternion, millimetre units (converted
  to metres at parse time).
* ``.json`` — OpenCV-style 3x3 intrinsics + 3x4 extrinsics + 5-param
  Brown-Conrady distortion. Both nested ``{seq: {cam: ...}}`` and flat
  ``{cam: ...}`` layouts are accepted, plus a legacy flat layout that
  uses ``focal`` / ``princpt`` / ``rotation`` / ``position`` keys.

All loaders return the same in-memory representation (:class:`Calibration`
holding a mapping of :class:`Camera`). Both ``T_cam_world`` (world->cam,
the projection-ready transform) and ``T_world_cam`` (camera pose in
the world) are stored verbatim — every downstream consumer wants one or
the other and round-tripping needs both.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np


VALID_DISTORTION_MODELS: Tuple[str, ...] = (
    "radtan",            # YAML: 4-param Brown-Conrady (k1, k2, p1, p2)
    "opencv_brown",      # OpenCV JSON: 5-param (k1, k2, p1, p2, k3)
    "vicon_radial_2",    # Vicon XCP / legacy JSON: (pp_x, pp_y, rad_1, rad_2, rad_3)
)


class CalibrationError(ValueError):
    """Raised by any loader when a calibration file is malformed or unsupported."""


@dataclass(frozen=True)
class Camera:
    """A single calibrated camera.

    Attributes:
        name: Camera identifier (must be unique within a :class:`Calibration`).
        width: Image width in pixels.
        height: Image height in pixels.
        intrinsics: 3x3 ``K`` matrix in float64. ``K[0,0] = fx``, ``K[1,1] = fy``,
            ``K[0,2] = cx``, ``K[1,2] = cy``.
        distortion_model: One of :data:`VALID_DISTORTION_MODELS`.
        distortion_coeffs: Tuple of distortion coefficients. Length depends on
            the model (4 for ``radtan``, 5 for ``opencv_brown`` and
            ``vicon_radial_2``).
        T_cam_world: 4x4 float64 transform. Multiplying a homogeneous world
            point by this gives the point in the camera frame
            (``p_cam = T_cam_world @ p_world``). This is what most projection
            code expects.
        T_world_cam: 4x4 float64 transform. The camera's pose in the world
            (its inverse is ``T_cam_world``). Stored verbatim so we never
            lose the "natural" representation of a given source format.
    """

    name: str
    width: int
    height: int
    intrinsics: np.ndarray
    distortion_model: str
    distortion_coeffs: Tuple[float, ...]
    T_cam_world: np.ndarray
    T_world_cam: np.ndarray


@dataclass(frozen=True)
class Calibration:
    """A multi-camera rig loaded from a calibration file."""

    cameras: Mapping[str, Camera]
    source_format: str            # "yaml" | "xcp" | "json"
    source_path: Path


def load_calibration(path) -> Calibration:
    """Load a calibration file by path. Format is selected from the extension.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` / ``.xcp`` / ``.json`` file.
            Both ``str`` and :class:`pathlib.Path` are accepted; extension
            matching is case-insensitive.

    Returns:
        A :class:`Calibration` whose ``cameras`` map is sorted by camera name.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        CalibrationError: For any other failure (unsupported extension,
            malformed file, missing required field, bad quaternion, ...).
    """
    p = Path(os.fspath(path))
    if not p.exists():
        raise FileNotFoundError(f"calibration file not found: {p}")

    # A directory holds either an EasyMocap rig (intri.yml + extri.yml) or a set
    # of per-camera OpenCV FileStorage files (camera name = file stem).
    if p.is_dir():
        from .loaders import easymocap_loader, opencv_loader
        if easymocap_loader.is_easymocap_dir(p):
            cameras, source_format = easymocap_loader.load(p), "easymocap"
        else:
            cameras, source_format = opencv_loader.load(p), "opencv"
        return Calibration(
            cameras=dict(sorted(cameras.items())),
            source_format=source_format,
            source_path=p,
        )

    ext = p.suffix.lower()
    # Lazy imports so optional deps (pyyaml) only matter when actually used.
    if ext == ".xcp":
        from .loaders import xcp_loader as loader
        source_format = "xcp"
    elif ext == ".json":
        from .loaders import json_loader as loader
        source_format = "json"
    elif ext in (".yaml", ".yml"):
        # `.yml` is ambiguous: an OpenCV FileStorage calibration or the MAMMA
        # YAML schema. Sniff the content to pick the right loader.
        from .loaders import opencv_loader
        if opencv_loader.looks_like_opencv_fs(p):
            loader, source_format = opencv_loader, "opencv"
        else:
            from .loaders import yaml_loader as loader
            source_format = "yaml"
    else:
        raise CalibrationError(
            f"unsupported calibration extension: {ext!r}; supported: "
            f".yaml/.yml/.xcp/.json (or a directory of OpenCV/EasyMocap files)"
        )

    cameras = loader.load(p)
    return Calibration(
        cameras=dict(sorted(cameras.items())),
        source_format=source_format,
        source_path=p,
    )


# ---------------------------------------------------------------------------
# World up-axis
# ---------------------------------------------------------------------------
# The up-axis is a property of the calibration's world frame, but it can't be
# declared inside every external format (OpenCV/EasyMocap/Vicon have fixed
# schemas). So we *auto-detect* it from camera geometry (poses only -> works for
# every format) and let it be overridden explicitly in capture.json.
#
# Representation: a signed axis string, one of "x","y","z","-x","-y","-z" (a
# leading "+" is accepted and normalized away). "auto" means "detect it".

VALID_UP_AXES: Tuple[str, ...] = ("x", "y", "z", "-x", "-y", "-z")

_VIEWCOORDS_BY_UP = {
    "x": "RIGHT_HAND_X_UP",  "-x": "RIGHT_HAND_X_DOWN",
    "y": "RIGHT_HAND_Y_UP",  "-y": "RIGHT_HAND_Y_DOWN",
    "z": "RIGHT_HAND_Z_UP",  "-z": "RIGHT_HAND_Z_DOWN",
}


def normalize_up_axis(up_axis: str) -> str:
    """Normalize a user/file up-axis string to one of :data:`VALID_UP_AXES`.

    Accepts "+y"/"Y"/" y " etc. Raises :class:`CalibrationError` on anything
    that isn't a signed cardinal axis (callers handle "auto" before this)."""
    s = str(up_axis).strip().lower().replace("+", "")
    if s not in VALID_UP_AXES:
        raise CalibrationError(
            f"up_axis must be one of {VALID_UP_AXES} (or 'auto'), got {up_axis!r}"
        )
    return s


def parse_up_axis(up_axis: str) -> Tuple[int, int]:
    """Return ``(index, sign)`` for a signed axis string, e.g. "-y" -> (1, -1)."""
    s = normalize_up_axis(up_axis)
    sign = -1 if s.startswith("-") else 1
    return {"x": 0, "y": 1, "z": 2}[s[-1]], sign


def up_axis_to_vector(up_axis: str) -> np.ndarray:
    """Signed unit vector for a signed axis string, e.g. "-y" -> [0,-1,0]."""
    idx, sign = parse_up_axis(up_axis)
    v = np.zeros(3, dtype=np.float64)
    v[idx] = float(sign)
    return v


def up_axis_to_viewcoords(up_axis: str) -> str:
    """Rerun ``ViewCoordinates`` name for a signed axis, e.g. "-y" ->
    "RIGHT_HAND_Y_DOWN"."""
    return _VIEWCOORDS_BY_UP[normalize_up_axis(up_axis)]


def detect_up_axis(calibration: "Calibration") -> Tuple[str, float]:
    """Estimate the world up-axis from camera geometry.

    Returns ``(signed_axis, confidence)`` where ``signed_axis`` is one of
    :data:`VALID_UP_AXES` and ``confidence`` is in ``[0, 1]`` (how close the
    averaged up direction is to a cardinal axis).

    Method (robust to roll/few cameras; validated empirically): each camera's
    image-up in world is ``R_wc @ [0,-1,0]`` (OpenCV image-up = camera -Y).
    Averaging across the rig cancels the horizontal components (azimuthal
    symmetry) and reinforces the vertical -> the mean points along world up. The
    sign is fixed by the fact that cameras look *down* at the subject
    (``up . mean_forward <= 0``). Then snap to the nearest signed cardinal axis.
    """
    cams = list(calibration.cameras.values())
    if not cams:
        return "z", 0.0
    Rwc = [np.asarray(c.T_world_cam[:3, :3], dtype=np.float64) for c in cams]
    up = np.mean([R @ np.array([0.0, -1.0, 0.0]) for R in Rwc], axis=0)
    n = float(np.linalg.norm(up))
    if n < 1e-9:                      # degenerate (e.g. all cams 90deg-rolled)
        return "z", 0.0
    up = up / n
    fwd = np.mean([R @ np.array([0.0, 0.0, 1.0]) for R in Rwc], axis=0)
    if np.dot(up, fwd) > 0:          # cameras look down -> up opposes forward
        up = -up
    idx = int(np.argmax(np.abs(up)))
    sign = "-" if up[idx] < 0 else ""
    return f"{sign}{'xyz'[idx]}", float(abs(up[idx]))


def resolve_up_axis(capture_cfg: Mapping, calibration: "Calibration") -> str:
    """Resolve the world up-axis for a capture.

    Order: an explicit, non-"auto" ``up_axis`` in ``capture_cfg`` wins; otherwise
    auto-detect from ``calibration``; falling back to ``"z"``.
    """
    raw = str((capture_cfg or {}).get("up_axis", "") or "").strip().lower()
    if raw and raw != "auto":
        return normalize_up_axis(raw)
    axis, _conf = detect_up_axis(calibration)
    return axis
