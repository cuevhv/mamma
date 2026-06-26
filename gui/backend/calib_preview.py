"""Write a small Rerun (.rrd) preview of a calibration's camera rig.

Lets a user *see* their multi-view setup and sanity-check the world2cam vs
cam2world convention: a correct rig shows the cameras arranged around the scene
and looking inward (frustums converging). A wrong convention scatters them or
points them outward.

Self-contained and best-effort: parses the calibration via
``capture.calibration.load_calibration`` (so every supported format works) and
logs camera frustums + a world-origin axis triad to a `.rrd` the GUI opens in
its existing Rerun viewers. Never raises into the request — returns the path or
raises a clear error the route turns into a 4xx.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """Snap a near-rotation to SO(3) via SVD (Rerun rejects non-orthonormal)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    M = U @ Vt
    if np.linalg.det(M) < 0:
        U[:, -1] *= -1
        M = U @ Vt
    return M


def _log_static(rr, path: str, entity) -> None:
    try:
        rr.log(path, entity, static=True)
    except TypeError:
        try:
            rr.log(path, entity, timeless=True)
        except TypeError:
            rr.log(path, entity)


def _flush(rr) -> None:
    for name in ("rerun_shutdown", "disconnect"):
        fn = getattr(rr, name, None)
        if fn is not None:
            try:
                fn()
                return
            except Exception:  # noqa: BLE001
                pass


def write_calib_rrd(calib_path, out_path, up_axis=None) -> int:
    """Parse ``calib_path`` and write a camera-rig ``.rrd`` to ``out_path``.

    ``up_axis`` is a signed axis string (``"x"``/``"y"``/``"z"``/``"-x"``/``"-y"``/
    ``"-z"``) setting which world axis points up in the viewer. ``None``/``"auto"``
    auto-detects it from the camera geometry. Returns the number of cameras
    logged. Raises on parse failure (the caller surfaces it); the Rerun logging
    itself is wrapped defensively.
    """
    from capture.calibration import (  # backend already imports capture
        load_calibration, detect_up_axis, normalize_up_axis, up_axis_to_viewcoords,
    )
    import rerun as rr

    cal = load_calibration(calib_path)
    cams = cal.cameras
    if not cams:
        raise ValueError("calibration has no cameras")

    up = (normalize_up_axis(up_axis)
          if up_axis and str(up_axis).lower() != "auto"
          else detect_up_axis(cal)[0])

    centers = np.array([c.T_world_cam[:3, 3] for c in cams.values()], dtype=np.float64)
    span = float(np.linalg.norm(centers.max(0) - centers.min(0))) if len(centers) > 1 else 1.0
    span = span or 1.0
    axis_len = 0.10 * span        # world-origin triad length, relative to rig size
    frustum_depth = 0.045 * span  # small frustums: visible but uncluttered
    pt_radius = 0.005 * span      # small marker at each camera centre (label anchor)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rr.init("mamma_calib_rig", spawn=False)
    rr.save(str(out_path))

    # Orient the viewer to the resolved up-axis (no fixed convention).
    vc = getattr(rr.ViewCoordinates, up_axis_to_viewcoords(up), None)
    if vc is not None:
        _log_static(rr, "world", vc)

    # World-origin axis triad (RGB = XYZ) so the user can locate the root frame.
    _log_static(rr, "world/origin", rr.Arrows3D(
        origins=[[0, 0, 0]] * 3,
        vectors=[[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]],
        colors=[[230, 60, 60], [60, 200, 60], [70, 120, 255]],
        labels=["x", "y", "z"],
    ))

    n = 0
    for name, cam in cams.items():
        Twc = cam.T_world_cam
        R = _orthonormalize(Twc[:3, :3])
        t = Twc[:3, 3]
        ent = f"world/cameras/{name}"
        try:
            _log_static(rr, ent, rr.Transform3D(mat3x3=R, translation=t))
            _log_static(rr, ent, rr.Pinhole(
                image_from_camera=np.asarray(cam.intrinsics, dtype=np.float64),
                resolution=[int(cam.width), int(cam.height)],
                camera_xyz=rr.ViewCoordinates.RDF,   # OpenCV: +X right, +Y down, +Z fwd
                image_plane_distance=frustum_depth,  # size the frustum to the rig
            ))
            # Labelled centre point — logged OUTSIDE the camera entity (anything
            # under a Pinhole is image-space, so a 3D point there can't render).
            _log_static(rr, f"world/camera_labels/{name}", rr.Points3D(
                [t], labels=[name], radii=[pt_radius],
                colors=[[245, 215, 70]],
            ))
            n += 1
        except Exception as e:  # noqa: BLE001 — never let one camera break the preview
            log.warning("calib preview: skipped camera %s: %s", name, e)

    _flush(rr)
    return n
