"""Camera geometry helpers shared by the tracking pipeline and the
cross-camera consensus post-process.

Torch-free, numpy-only: importable without loading any model stack. The
fundamental-matrix chain and the point-to-epipolar-line distance follow the
standard formulation (E = [t]_x R, F = K2^-T E K1^-1) with ``cam_ext`` as the
world-to-camera transform; both were verified numerically against synthetic
camera rigs in the mathematics audit.
"""
from __future__ import annotations

import numpy as np


def as_intrinsics(cam_int):
    """3x3 float64 K from any of the stored intrinsics layouts, or None."""
    cam_int = np.asarray(cam_int)
    if cam_int.ndim == 2 and cam_int.shape[0] >= 3 and cam_int.shape[1] >= 3:
        return cam_int[:3, :3].astype(np.float64)
    if cam_int.ndim >= 3:
        mat = cam_int.reshape(-1, cam_int.shape[-2], cam_int.shape[-1])[0]
        return mat[:3, :3].astype(np.float64)
    return None


def as_extrinsics(cam_ext):
    """4x4 float64 world-to-camera transform, or None."""
    cam_ext = np.asarray(cam_ext)
    if cam_ext.ndim == 2:
        if cam_ext.shape == (4, 4):
            return cam_ext.astype(np.float64)
        if cam_ext.shape == (3, 4):
            out = np.eye(4, dtype=np.float64)
            out[:3, :4] = cam_ext
            return out
    if cam_ext.ndim >= 3:
        mat = cam_ext.reshape(-1, cam_ext.shape[-2], cam_ext.shape[-1])[0]
        return as_extrinsics(mat)
    return None


def compute_fundamental_matrix(source_cam_data, target_cam_data):
    """F mapping source-image points to target-image epipolar lines, or None
    when either camera lacks usable calibration."""
    if source_cam_data is None or target_cam_data is None:
        return None
    if "cam_int" not in source_cam_data or "cam_int" not in target_cam_data:
        return None
    if "cam_ext" not in source_cam_data or "cam_ext" not in target_cam_data:
        return None
    k1 = as_intrinsics(source_cam_data["cam_int"])
    k2 = as_intrinsics(target_cam_data["cam_int"])
    e1 = as_extrinsics(source_cam_data["cam_ext"])
    e2 = as_extrinsics(target_cam_data["cam_ext"])
    if k1 is None or k2 is None or e1 is None or e2 is None:
        return None
    r1, t1 = e1[:3, :3], e1[:3, 3]
    r2, t2 = e2[:3, :3], e2[:3, 3]
    r = r2 @ r1.T
    t = t2 - (r @ t1)
    tx = np.array([[0.0, -t[2], t[1]],
                   [t[2], 0.0, -t[0]],
                   [-t[1], t[0], 0.0]], dtype=np.float64)
    e = tx @ r
    try:
        f = np.linalg.inv(k2).T @ e @ np.linalg.inv(k1)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(f).all():
        return None
    return f


def epipolar_score(f_matrix, x1, x2, sigma_px, max_dist_px):
    """(gaussian score, point-to-line distance px) for homogeneous x1/x2."""
    if f_matrix is None or x1 is None:
        return None, None
    line = f_matrix @ x1
    denom = np.linalg.norm(line[:2])
    if denom < 1e-8:
        return None, None
    dist = float(np.abs(np.dot(line, x2)) / denom)
    if max_dist_px is not None and dist > max_dist_px:
        return 0.0, dist
    sigma = max(float(sigma_px), 1e-3)
    return float(np.exp(-0.5 * ((dist / sigma) ** 2))), dist


def epipolar_distance(f_matrix, x1, x2):
    """Point-to-epipolar-line distance in target pixels, or None."""
    if f_matrix is None or x1 is None or x2 is None:
        return None
    line = f_matrix @ x1
    denom = np.linalg.norm(line[:2])
    if denom < 1e-8:
        return None
    return float(np.abs(np.dot(line, x2)) / denom)


def line_to_bbox_distance(line, bbox):
    """Distance (px) from an epipolar line to an axis-aligned box.

    Zero when the line crosses the box (the correct-match case: a ray from a
    body-interior point in one view passes through the person's extent in the
    other, regardless of silhouette asymmetry); otherwise the smallest corner
    distance. ``line`` = (a, b, c) with ax+by+c=0; ``bbox`` = (x1, y1, x2, y2).
    """
    denom = np.linalg.norm(line[:2])
    if denom < 1e-8:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    signed = [(line[0] * x + line[1] * y + line[2]) / denom
              for x in (x1, x2) for y in (y1, y2)]
    if min(signed) <= 0.0 <= max(signed):
        return 0.0
    return float(min(abs(s) for s in signed))
