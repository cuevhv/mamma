"""EasyMocap calibration loader.

EasyMocap stores a rig as two OpenCV FileStorage files in one directory:
  * ``intri.yml`` — a ``names`` list and per-camera ``K_<name>`` (3x3) +
    ``dist_<name>`` (1x5 Brown), optionally ``W_<name>``/``H_<name>``.
  * ``extri.yml`` — per-camera ``Rot_<name>`` (3x3) or ``R_<name>`` (3x1
    Rodrigues rotation vector) and ``T_<name>`` (3x1).

Convention is **world->camera** (``R @ X_world + T = X_cam``), metres — the same
as plain OpenCV, so we reuse that builder and the tolerant FileStorage parser.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..calibration import Camera, CalibrationError
from .opencv_loader import _as_int, _camera_from_KDRT, parse_fs

INTRI_NAME = "intri.yml"
EXTRI_NAME = "extri.yml"


def is_easymocap_dir(path) -> bool:
    p = Path(path)
    return p.is_dir() and (p / INTRI_NAME).is_file() and (p / EXTRI_NAME).is_file()


def _names(fs: Dict[str, object]) -> List[str]:
    raw = fs.get("names")
    if isinstance(raw, (list, tuple)):
        return [str(n) for n in raw]
    return []


def load(path) -> Dict[str, Camera]:
    """Parse an EasyMocap ``intri.yml`` + ``extri.yml`` pair from a directory."""
    p = Path(path)
    intri, extri = p / INTRI_NAME, p / EXTRI_NAME
    if not intri.is_file() or not extri.is_file():
        raise CalibrationError(
            f"{p}: EasyMocap calibration needs both {INTRI_NAME} and {EXTRI_NAME}"
        )

    fi, fe = parse_fs(intri), parse_fs(extri)
    names = _names(fi) or _names(fe)
    if not names:
        raise CalibrationError(f"{intri}: missing a 'names' camera list")

    import cv2  # only for Rodrigues (rotvec -> matrix)
    cameras: Dict[str, Camera] = {}
    errors: List[str] = []
    for name in names:
        K = fi.get(f"K_{name}")
        D = fi.get(f"dist_{name}")
        if D is None:
            D = fi.get(f"D_{name}")
        Rot = fe.get(f"Rot_{name}")
        if Rot is None:
            rvec = fe.get(f"R_{name}")
            if rvec is not None:
                Rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        T = fe.get(f"T_{name}")
        if K is None or Rot is None or T is None:
            missing = " ".join(s for s, m in
                               (("K", K is None), ("R/Rot", Rot is None), ("T", T is None)) if m)
            errors.append(f"camera {name}: missing {missing}")
            continue
        width = _as_int(fi.get(f"W_{name}"))
        height = _as_int(fi.get(f"H_{name}"))
        cameras[name] = _camera_from_KDRT(name, K, D, Rot, T, width=width, height=height)

    if errors:
        raise CalibrationError(
            f"{p}: EasyMocap calibration validation failed:\n  - " + "\n  - ".join(errors)
        )
    return cameras
