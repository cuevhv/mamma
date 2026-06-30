"""Frame undistortion for the inference pipeline.

Supports two distortion model families:

* ``radtan`` / ``opencv_brown`` — the standard OpenCV Brown-Conrady model
  (``distortion_coeffs = [k1, k2, p1, p2(, k3)]``), undistorted via
  ``cv2.initUndistortRectifyMap`` using the camera's ``K`` (newCameraMatrix = K,
  so a mesh projected with ``K`` aligns with the undistorted background).
* ``vicon_radial_2`` — the Vicon pixel-space radial model below.

The Vicon distortion model is **not** OpenCV-compatible: it applies a
radial correction in raw pixel-space coordinates rather than in
normalized image coords. The math here is ported verbatim from
``capture-hall-toolkit/cameras/utils.py`` (``radial_distortion_correction``
+ ``undistort_image_custom``). See that file for the original Vicon
reference math.

Coefficient layout (matches :data:`capture.calibration.VALID_DISTORTION_MODELS`
entry ``"vicon_radial_2"``)::

    [pp_x, pp_y, rad_1, rad_2, rad_3]

The scale factor applied to ``(x - pp_x, y - pp_y)`` is::

    1 + rad_1 * d^2 + rad_2 * d^4 + rad_3 * d^6      where d^2 = (x-pp_x)^2 + (y-pp_y)^2

This module memoizes the ``(map_x, map_y)`` arrays per camera so the
3M-point meshgrid is built once per (camera, resolution) and reused for
every frame — important when processing 1000+ frames × 30+ cameras.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import numpy as np

from .calibration import Camera


_CACHE: Dict[Tuple[str, str, int, int], Tuple[np.ndarray, np.ndarray]] = {}
_CACHE_LOCK = threading.Lock()


def _radial_correction(
    x: np.ndarray, y: np.ndarray,
    pp_x: float, pp_y: float,
    rad_1: float, rad_2: float, rad_3: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vicon radial correction in pixel space (see module docstring)."""
    cx = x - pp_x
    cy = y - pp_y
    d_sq = cx * cx + cy * cy
    scale = 1.0 + rad_1 * d_sq + rad_2 * d_sq ** 2 + rad_3 * d_sq ** 3
    return cx * scale + pp_x, cy * scale + pp_y


def _build_maps(camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``cv2.remap``-compatible (map_x, map_y) for one camera.

    Supports both the Vicon ``vicon_radial_2`` pixel-space model and the
    standard OpenCV Brown-Conrady ``radtan`` / ``opencv_brown`` models. The
    OpenCV models undistort into the *same* pinhole ``K`` frame (newCameraMatrix
    = K), so a mesh projected with ``K`` aligns with the undistorted background.
    """
    w, h = int(camera.width), int(camera.height)
    if camera.distortion_model == "vicon_radial_2":
        pp_x, pp_y, rad_1, rad_2, rad_3 = camera.distortion_coeffs[:5]
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        map_x_flat, map_y_flat = _radial_correction(
            x_coords.flatten().astype(np.float64),
            y_coords.flatten().astype(np.float64),
            float(pp_x), float(pp_y),
            float(rad_1), float(rad_2), float(rad_3),
        )
        return (
            map_x_flat.reshape(h, w).astype(np.float32),
            map_y_flat.reshape(h, w).astype(np.float32),
        )
    # OpenCV Brown-Conrady: radtan = [k1, k2, p1, p2], opencv_brown = [..., k3].
    import cv2
    K = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    dist = np.asarray([float(c) for c in camera.distortion_coeffs], dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
    return map_x, map_y


def _is_noop(camera: Optional[Camera]) -> bool:
    """True if undistortion would be the identity (zero / missing / unsupported)."""
    if camera is None:
        return True
    model = getattr(camera, "distortion_model", None)
    coeffs = tuple(float(c) for c in (getattr(camera, "distortion_coeffs", ()) or ()))
    if model == "vicon_radial_2":
        if len(coeffs) < 5:
            return True
        _, _, rad_1, rad_2, rad_3 = coeffs[:5]
        return rad_1 == 0.0 and rad_2 == 0.0 and rad_3 == 0.0
    if model in ("radtan", "opencv_brown"):
        # radtan needs at least [k1,k2,p1,p2]; no-op when all coeffs are zero
        # or the camera lacks an intrinsic matrix to undistort into.
        if len(coeffs) < 4 or all(c == 0.0 for c in coeffs):
            return True
        return getattr(camera, "intrinsics", None) is None
    return True  # unknown / unsupported distortion model


def get_maps(camera: Camera) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return cached (map_x, map_y) for ``camera``, or ``None`` if no-op."""
    if _is_noop(camera):
        return None
    key = (
        camera.name,
        camera.distortion_model,
        int(camera.width),
        int(camera.height),
    )
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        maps = _build_maps(camera)
        _CACHE[key] = maps
    return maps


def undistort_rgb(image: np.ndarray, camera: Optional[Camera]) -> np.ndarray:
    """Return ``image`` undistorted via ``camera``'s coeffs (or unchanged).

    No-op when ``camera`` is ``None`` or the coefficients are all zero.
    Maps are built lazily on first call per camera and cached thereafter.
    """
    maps = get_maps(camera) if camera is not None else None
    if maps is None:
        return image
    import cv2
    map_x, map_y = maps
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
