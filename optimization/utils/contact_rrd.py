"""Rerun (.rrd) visualization of the intermediate triangulated contact points.

A lighter, faster replacement for the Plotly ``_contact.html`` /
``_floor_contact.html`` debug outputs: logs the *same* per-frame triangulated 3D
landmarks, colored by contact probability, to a compact ``.rrd`` that opens in the
GUI's Rerun viewer (Browser + Native). Contact and floor-contact are separate,
toggleable entities on one ``frame`` timeline; a ground quad gives the
floor-contact layer a reference. Near-zero points stay small and dim so the few
real contact points stand out.

Self-contained inside the ``optimization`` package and fully best-effort: any
problem logs a warning and returns rather than raising into the ma_3d run.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Colour ramp endpoints (RGB). Low end is a light neutral so every point stays
# visible against Rerun's dark gradient background (a dim grey merged into it);
# contact "pops" via hue/saturation + radius, not by going dark.
_LO = (205, 210, 220)          # near-zero probability -> light cool grey
_HI_CONTACT = (255, 55, 45)    # contact -> red
_HI_FLOOR = (40, 145, 255)     # floor-contact -> blue

# Signed up-axis helpers, inlined so this stays self-contained: the ma_3d runtime
# imports from the ``optimization`` root and cannot see the top-level ``capture``
# package (mirrors capture.calibration.up_axis_to_* but with no cross-package dep).
_VIEWCOORDS_BY_UP = {
    "x": "RIGHT_HAND_X_UP",  "-x": "RIGHT_HAND_X_DOWN",
    "y": "RIGHT_HAND_Y_UP",  "-y": "RIGHT_HAND_Y_DOWN",
    "z": "RIGHT_HAND_Z_UP",  "-z": "RIGHT_HAND_Z_DOWN",
}


def _parse_up(up_axis):
    """Return ``(idx, up_vec, viewcoords_name)`` for a signed axis string
    (``x|y|z|-x|-y|-z``) or a legacy int index (0/1/2)."""
    if isinstance(up_axis, int):
        up_axis = {0: "x", 1: "y", 2: "z"}.get(up_axis, "z")
    s = str(up_axis).strip().lower().replace("+", "")
    if s not in _VIEWCOORDS_BY_UP:
        s = "z"
    idx = {"x": 0, "y": 1, "z": 2}[s[-1]]
    vec = np.zeros(3, dtype=np.float64)
    vec[idx] = -1.0 if s.startswith("-") else 1.0
    return idx, vec, _VIEWCOORDS_BY_UP[s]


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def _flush(rr) -> None:
    """Finalize the recording so the .rrd is complete on return (the ma_3d
    process keeps running, so we can't rely on interpreter-exit flushing)."""
    for name in ("rerun_shutdown", "disconnect"):
        fn = getattr(rr, name, None)
        if fn is not None:
            try:
                fn()
                return
            except Exception:  # noqa: BLE001
                pass


def _set_frame(rr, t: int) -> None:
    """Set the integer ``frame`` timeline, tolerant to rerun-sdk API drift."""
    try:
        rr.set_time("frame", sequence=int(t))          # rerun >= ~0.28
    except (AttributeError, TypeError):
        rr.set_time_sequence("frame", int(t))          # older


def _log_static(rr, path: str, entity) -> None:
    """Log an entity once for all time, tolerant to the static/timeless rename."""
    try:
        rr.log(path, entity, static=True)
    except TypeError:
        try:
            rr.log(path, entity, timeless=True)
        except TypeError:
            rr.log(path, entity)


def _rgba(prob: np.ndarray, hi_rgb) -> np.ndarray:
    """(N,) prob in [0,1] -> (N,4) uint8 RGBA ramping dim-grey -> hi_rgb."""
    p = np.clip(np.nan_to_num(prob.astype(np.float64)), 0.0, 1.0)[:, None]
    lo = np.asarray(_LO, np.float64)
    hi = np.asarray(hi_rgb, np.float64)
    rgb = lo * (1.0 - p) + hi * p
    rgba = np.concatenate([rgb, np.full((rgb.shape[0], 1), 255.0)], axis=1)
    return rgba.astype(np.uint8)


def _radii(prob: np.ndarray, r_min: float, r_max: float) -> np.ndarray:
    """Radius scales with probability so strong-contact points read as larger."""
    p = np.clip(np.nan_to_num(prob.astype(np.float64)), 0.0, 1.0)
    return (r_min + (r_max - r_min) * p).astype(np.float32)


def _log_ground(rr, floor_height: float, up_vec, half: float,
                center0: float = 0.0, center1: float = 0.0) -> None:
    """A square ground quad centred under the subjects (center0/center1 are the
    two non-up axes), sized to span them. ``floor_height`` is the up-axis
    projection value (``p·up_vec``); ``up_vec`` is the signed up unit vector."""
    up = np.asarray(up_vec, dtype=np.float64)
    idx = int(np.argmax(np.abs(up)))
    sign = 1.0 if up[idx] >= 0 else -1.0
    a0, a1 = [a for a in (0, 1, 2) if a != idx]
    plane_coord = floor_height * sign        # p·up = floor_height (axis-aligned)
    corners = [(-half, half), (half, half), (-half, -half), (half, -half)]
    coords = np.zeros((4, 3), dtype=np.float64)
    for i, (d0, d1) in enumerate(corners):
        coords[i, a0] = center0 + d0
        coords[i, a1] = center1 + d1
        coords[i, idx] = plane_coord
    normal = up / (np.linalg.norm(up) + 1e-12)
    _log_static(rr, "world/ground", rr.Mesh3D(
        vertex_positions=coords,
        triangle_indices=np.array([[0, 1, 2], [1, 3, 2]]),
        vertex_normals=np.tile(normal, (4, 1)),
        vertex_colors=np.tile(np.array([70, 72, 78], dtype=np.uint8), (4, 1)),
    ))


def write_contact_rrd(out_path, points_world, contact, floor_contact,
                      valid_mask=None, up_axis="z") -> None:
    """Write one ``.rrd`` with the triangulated points as two contact layers.

    ``points_world[b]``: (T, N, 3); ``contact[b]`` / ``floor_contact[b]``: (T, N)
    in [0, 1] or None; ``valid_mask[b]``: (T, N) bool or None. Lists are indexed
    per body. ``up_axis`` is a signed axis string (``x|y|z|-x|-y|-z``); a legacy
    int index (0/1/2) is accepted too. Best-effort — never raises.
    """
    try:
        import rerun as rr
    except Exception as e:  # noqa: BLE001
        log.warning("rerun unavailable; skipping contact .rrd: %s", e)
        return

    try:
        up_idx, up_vec, _vc_name = _parse_up(up_axis)
        pts = [_to_numpy(p) for p in (points_world or [])]
        pts = [p if (p is not None and p.ndim == 3 and p.shape[0] > 0) else None for p in pts]
        if not any(p is not None for p in pts):
            return
        con = [_to_numpy(c) for c in (contact or [])]
        flo = [_to_numpy(c) for c in (floor_contact or [])]
        msk = [_to_numpy(m) for m in (valid_mask or [])]
        T = min(p.shape[0] for p in pts if p is not None)

        # Scene scale + placement from the bounding box of all finite points
        # (robust to metres vs mm). The ground is centred under the subjects and
        # sized to span them so they don't look like they're floating away from
        # it, and so the viewer's auto-fit frames the people rather than an
        # off-origin floor.
        allpts = np.concatenate([p.reshape(-1, 3) for p in pts if p is not None], axis=0)
        allpts = allpts[np.isfinite(allpts).all(axis=1)]
        plane = [a for a in (0, 1, 2) if a != up_idx]
        if allpts.shape[0] >= 2:
            # Robust extent: a handful of degenerate triangulations can land
            # hundreds of metres away (finite, so NOT caught by the NaN mask).
            # Size the scene from a percentile box so those flyers can't blow up
            # the diagonal, the sphere radii and the ground quad — otherwise the
            # viewer auto-fits to a ~km scene and every subject merges into one
            # dot. Centre/height are taken from the in-box (inlier) points.
            lo = np.percentile(allpts, 1.0, axis=0)
            hi = np.percentile(allpts, 99.0, axis=0)
            inlier = np.all((allpts >= lo) & (allpts <= hi), axis=1)
            inpts = allpts[inlier] if inlier.any() else allpts
            bb = hi - lo
            diag = float(np.linalg.norm(bb)) or 1.0
            floor_h = float(np.percentile(inpts @ up_vec, 2))   # signed up projection
            xy_extent = float(max(bb[plane[0]], bb[plane[1]]))
            half = 0.5 * xy_extent + max(0.75, 0.3 * xy_extent)   # span subjects + margin
            c0 = float((lo[plane[0]] + hi[plane[0]]) * 0.5)
            c1 = float((lo[plane[1]] + hi[plane[1]]) * 0.5)
        else:
            diag, floor_h, half, c0, c1 = 1.0, 0.0, 1.0, 0.0, 0.0
        # Radii track the (robust) scene but are hard-capped so no residual
        # outlier can inflate the point spheres past a human scale.
        r_min, r_max = 0.004 * diag, min(0.016 * diag, 0.05)
        r_min = min(r_min, r_max)

        rr.init("mamma_contact", spawn=False)
        rr.save(str(out_path))
        # Make the scene upright so the camera orients sensibly on open.
        if hasattr(rr, "ViewCoordinates"):
            _vc = getattr(rr.ViewCoordinates, _vc_name, None)
            if _vc is not None:
                _log_static(rr, "/", _vc)
        _log_ground(rr, floor_h, up_vec, half, c0, c1)

        def _emit(path, pos, prob, hi):
            finite = np.isfinite(pos).all(axis=1)
            sel = finite & np.isfinite(prob)
            if not sel.any():
                return
            rr.log(path, rr.Points3D(
                pos[sel], colors=_rgba(prob[sel], hi), radii=_radii(prob[sel], r_min, r_max)))

        for t in range(T):
            _set_frame(rr, t)
            for b, p in enumerate(pts):
                if p is None or t >= p.shape[0]:
                    continue
                pos = np.asarray(p[t], dtype=np.float64)
                # validity mask for this (body, frame), if provided
                if b < len(msk) and msk[b] is not None and t < msk[b].shape[0]:
                    keep = np.asarray(msk[b][t], dtype=bool)
                    pos = np.where(keep[:, None], pos, np.nan)
                if b < len(con) and con[b] is not None and t < con[b].shape[0]:
                    _emit(f"contact/body_{b:02d}", pos, np.asarray(con[b][t], np.float64), _HI_CONTACT)
                if b < len(flo) and flo[b] is not None and t < flo[b].shape[0]:
                    _emit(f"floor_contact/body_{b:02d}", pos, np.asarray(flo[b][t], np.float64), _HI_FLOOR)
        _flush(rr)  # finalize so the .rrd is complete before ma_3d continues
    except Exception as e:  # noqa: BLE001 — viz must never break a run
        log.warning("failed to write contact .rrd %s: %s", out_path, e)
        try:
            _flush(rr)
        except Exception:  # noqa: BLE001
            pass
