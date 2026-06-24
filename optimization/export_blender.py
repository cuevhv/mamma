"""Export MAMMA SMPL-X fits to the SMPL-X Blender add-on ``.npz`` format.

Per person, reads ``ma_3d``'s ``smplx_params_body_id-NN.npz`` and writes an
AMASS-style npz (``poses`` / ``betas`` / ``trans`` / ``gender`` /
``mocap_frame_rate``) that the ``smplx_blender_addon`` imports. Pure-Python
(numpy + the ``smplx`` model, for the relaxed-hand mean) — no Blender needed.

Conversions (see docs/smplx-export-plan.md):
  * pose layout is already identical (165 axis-angle), so it copies straight over;
  * the relaxed-hand mean is BAKED into the hand angles so the result is absolute
    (import as the add-on's FLAT mode), which is convention-proof: the mean is
    zero when the fit used ``flat_hand_mean=True``;
  * ``global_orient`` + ``trans`` are rotated from the capture's world up-axis to
    the add-on's AMASS Y-up convention;
  * ``mocap_frame_rate`` comes from ma_cap's ``global.npz`` (or ``--fps``).

Self-describing where possible: reads ``smplx_export_*`` metadata stamped by
``run_ma_3d`` and falls back to safe defaults for older outputs.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re
from typing import Optional

import numpy as np

log = logging.getLogger("export_blender")

# Pose-vector slices (SMPL-X, 165 axis-angle) — identical in MAMMA and the add-on.
_GLOBAL = slice(0, 3)
_LHAND = slice(75, 120)
_RHAND = slice(120, 165)


# SMPL-X body joint indices: head, and feet (ankles + foot tips).
_J_HEAD, _J_FEET = 15, (7, 8, 10, 11)


def _rotation_aligning(a, b):
    """Shortest-arc 3x3 rotation mapping unit vector ``a`` onto unit vector ``b``."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:  # antiparallel: 180deg about any perpendicular axis
        p = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        v = np.cross(a, p); v /= np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if c < -0.999999:
        return np.eye(3) + 2.0 * vx @ vx
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _auto_orient(joints, fps=30.0):
    """Detect up + floor from SMPL-X joints ``(F, J, 3)`` in the capture's world.

    Fuses, robustly: foot-contact plane (E1) + body-vertical median (E3) +
    free-fall gravity (E2, only when airborne frames exist). Returns
    ``(R, floor_along_up, info)`` where R maps the detected up onto +Y. Pure numpy,
    posture-agnostic (jumps/cartwheels/lying are minority outliers, not the answer).
    """
    J = np.asarray(joints, dtype=np.float64)
    feet = J[:, _J_FEET, :].reshape(-1, 3)
    feet_c = J[:, _J_FEET, :].mean(1)
    centroid = J.reshape(-1, 3).mean(0)

    # E3 body-vertical: robust median of (head - feet) over frames.
    bv = J[:, _J_HEAD, :] - feet_c
    up = np.median(bv / (np.linalg.norm(bv, axis=1, keepdims=True) + 1e-12), axis=0)
    src = "body-vertical"

    # E1 foot-contact plane: smallest-variance direction of the foot cloud, used
    # only when the feet are spread out and lie near a plane (planar + moving).
    try:
        _, sv, vt = np.linalg.svd(feet - feet.mean(0), full_matrices=False)
        if sv[0] > 0.1 and sv[-1] / sv[0] < 0.25:
            n = vt[-1]
            up = n if np.dot(n, up) >= 0 else -n
            src = "foot-plane"
    except np.linalg.LinAlgError:
        pass

    # (Free-fall gravity from COM acceleration was evaluated and dropped: on real
    # captured motion — incl. breakdancing — clean ballistic flight is rare/short
    # while push-off/impact frames also hit ~9.8 m/s^2 in scattered directions, so
    # it either abstains or degrades the foot-plane estimate. E1+E3 are robust.)

    # Sign: up points from the floor toward the body centroid.
    if np.dot(centroid - feet_c.mean(0), up) < 0:
        up = -up
    up = up / (np.linalg.norm(up) + 1e-12)

    # Snap to the nearest world axis when nearly axis-aligned (typical rigs).
    ax = int(np.argmax(np.abs(up)))
    conf = float(abs(up[ax]))
    if conf > 0.93:
        snapped = np.zeros(3); snapped[ax] = np.sign(up[ax]); up = snapped

    floor = float(np.percentile(feet @ up, 5))  # robust floor coord along up
    return up, floor, {"axis": "xyz"[ax], "confidence": round(conf, 3), "source": src}


_AXIS_UNIT = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def _rotvec_to_mat(rotvecs: np.ndarray) -> np.ndarray:
    """(N,3) axis-angle -> (N,3,3) rotation matrices."""
    try:
        from scipy.spatial.transform import Rotation
        return Rotation.from_rotvec(rotvecs).as_matrix()
    except Exception:  # pragma: no cover - scipy fallback
        import cv2
        return np.stack([cv2.Rodrigues(r.astype(np.float64))[0] for r in rotvecs])


def _mat_to_rotvec(mats: np.ndarray) -> np.ndarray:
    """(N,3,3) rotation matrices -> (N,3) axis-angle."""
    try:
        from scipy.spatial.transform import Rotation
        return Rotation.from_matrix(mats).as_rotvec()
    except Exception:  # pragma: no cover
        import cv2
        return np.stack([cv2.Rodrigues(m.astype(np.float64))[0].ravel() for m in mats])


def _scalar(data, key, default):
    """Read a 0-d scalar (str/bool/int) from an npz, with a default if absent."""
    if key not in getattr(data, "files", []):
        return default
    v = data[key]
    try:
        return v.item()
    except Exception:
        return default


def _resolve_fps(ma_cap_dir: Optional[str], seq_name: str, fps_override: Optional[int]) -> int:
    if fps_override:
        return int(fps_override)
    if ma_cap_dir:
        g = os.path.join(ma_cap_dir, seq_name, "gt", "global.npz")
        if os.path.exists(g):
            try:
                with np.load(g, allow_pickle=True) as gd:
                    if "fps" in gd.files:
                        return int(np.asarray(gd["fps"]).item())
            except Exception as e:
                log.warning("could not read fps from %s: %s", g, e)
    log.warning("fps not found (no --fps and no ma_cap global.npz); defaulting to 30")
    return 30


def _build_model(model_dir, gender, num_betas, flat_hand_mean, batch_size):
    import smplx
    return smplx.create(
        model_dir, model_type="smplx", gender=gender, ext="npz",
        flat_hand_mean=bool(flat_hand_mean), num_betas=int(num_betas),
        use_pca=False, num_pca_comps=45, batch_size=int(batch_size),
    )


def _reconstruct_verts(model, pose, betas_row, trans):
    """Run SMPL-X forward on the stored pose split into its parts (the way MAMMA
    did), to compare against the saved ``pred_vertices``."""
    import torch
    F = pose.shape[0]
    betas_t = torch.tensor(np.tile(betas_row, (F, 1)), dtype=torch.float32)
    with torch.no_grad():
        return model(
            return_verts=True, betas=betas_t,
            global_orient=torch.tensor(pose[:, 0:3], dtype=torch.float32),
            body_pose=torch.tensor(pose[:, 3:66], dtype=torch.float32),
            jaw_pose=torch.tensor(pose[:, 66:69], dtype=torch.float32),
            leye_pose=torch.tensor(pose[:, 69:72], dtype=torch.float32),
            reye_pose=torch.tensor(pose[:, 72:75], dtype=torch.float32),
            left_hand_pose=torch.tensor(pose[:, 75:120], dtype=torch.float32),
            right_hand_pose=torch.tensor(pose[:, 120:165], dtype=torch.float32),
            transl=torch.tensor(trans, dtype=torch.float32),
        ).vertices.cpu().numpy()


def _write_addon_npz(out_path, poses, betas_row, trans, gender, fps):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path,
             poses=poses.astype(np.float32), betas=betas_row.astype(np.float32),
             trans=trans.astype(np.float32), gender=str(gender),
             mocap_frame_rate=int(fps), surface_model_type="smplx")


def _rotate_params(poses, trans, R, J0):
    """Apply world rotation R to a whole-body animation (global_orient + trans).
    global_orient rotates about the root, so trans needs the pivot offset (R-I)@J0."""
    p = poses.copy()
    p[:, _GLOBAL] = _mat_to_rotvec(R[None] @ _rotvec_to_mat(poses[:, _GLOBAL]))
    t = trans @ R.T + (R - np.eye(3)) @ J0
    return p, t


def export_person(params_path, model_dir, fps, out_path,
                  verts_path=None, validate=True, validate_tol_mm=2.0,
                  ground=True, up_axis="auto", blender_format="auto",
                  write_blender_npz=False, blender_npz_path=None):
    """Convert one ``smplx_params_body_id-NN.npz`` to an add-on npz at ``out_path``."""
    with np.load(params_path, allow_pickle=True) as d:
        pose = np.asarray(d["smplx_pose"], dtype=np.float64)          # (F,165)
        betas = np.asarray(d["smplx_betas"], dtype=np.float64)        # (1,nb)
        trans = np.asarray(d["smplx_translation"], dtype=np.float64)  # (F,3)
        model_type = _scalar(d, "smplx_export_model_type", "smplx")
        gender = _scalar(d, "smplx_export_gender", "neutral")
        flat_hint = bool(_scalar(d, "smplx_export_flat_hand_mean", False))
        num_betas = int(_scalar(d, "smplx_export_num_betas", betas.shape[-1]))

    if model_type != "smplx":
        raise ValueError(f"{params_path}: unsupported model_type {model_type!r}")
    F = pose.shape[0]
    betas_row = betas.reshape(-1)[:num_betas]

    ref = None
    ref_joints = None
    if verts_path and os.path.exists(verts_path):
        with np.load(verts_path, allow_pickle=True) as vd:
            if validate:
                ref = np.asarray(vd["pred_vertices"], dtype=np.float64)
            if "pred_joints" in vd.files:
                ref_joints = np.asarray(vd["pred_joints"], dtype=np.float64)

    # Determine flat_hand_mean. The fit's convention is whatever reconstructs the
    # saved vertices — so when they're available we AUTO-DETECT it (the stamped
    # metadata is only a first-try hint). This is bulletproof across datasets,
    # eval-vs-inference differences, and older files that predate the metadata.
    chosen_model, chosen_flat, chosen_mm = None, None, None
    tried = []
    for flat in ([flat_hint, not flat_hint] if ref is not None else [flat_hint]):
        if flat in tried:
            continue
        tried.append(flat)
        m = _build_model(model_dir, gender, num_betas, flat, F)
        if ref is None:
            chosen_model, chosen_flat = m, flat
            log.info("%s: F=%d nb=%d gender=%s; no verts to validate -> "
                     "flat_hand_mean=%s (hint)", os.path.basename(params_path),
                     F, num_betas, gender, flat)
            break
        mm = float(np.abs(_reconstruct_verts(m, pose, betas_row, trans) - ref).max() * 1000.0)
        log.info("%s: flat_hand_mean=%s -> round-trip max %.3f mm",
                 os.path.basename(params_path), flat, mm)
        if mm <= validate_tol_mm:
            chosen_model, chosen_flat, chosen_mm = m, flat, mm
            break
    if chosen_model is None:
        raise RuntimeError(
            f"{params_path}: neither flat_hand_mean reconstructs the saved vertices "
            f"within {validate_tol_mm} mm (tried {tried}) — model config likely wrong; "
            "refusing to export")
    if chosen_mm is not None:
        log.info("  detected flat_hand_mean=%s (round-trip %.3f mm)", chosen_flat, chosen_mm)

    # --- bake the relaxed-hand mean -> absolute (add-on FLAT mode) ---
    # model.left/right_hand_mean is the relaxed mean (0 when flat_hand_mean=True),
    # exactly what the SMPL-X forward adds internally, so this yields the absolute
    # hand pose for the detected convention.
    lhm = chosen_model.left_hand_mean.detach().cpu().numpy().reshape(-1)
    rhm = chosen_model.right_hand_mean.detach().cpu().numpy().reshape(-1)
    poses = pose.copy()
    poses[:, _LHAND] += lhm
    poses[:, _RHAND] += rhm

    # --- detect the source up-axis (works for any coordinate system, not just Z) ---
    if up_axis in ("x", "y", "z"):
        up_vec, up_idx, src, conf = _AXIS_UNIT[up_axis], {"x": 0, "y": 1, "z": 2}[up_axis], "forced", 1.0
    elif ref_joints is not None:
        up_vec, _f, info = _auto_orient(ref_joints, fps=fps)
        up_idx, src, conf = int(np.argmax(np.abs(up_vec))), info["source"], info["confidence"]
    else:
        up_vec, up_idx, src, conf = _AXIS_UNIT["z"], 2, "default-z", 0.0
    src_up = "xyz"[up_idx]

    import torch
    with torch.no_grad():
        J0 = chosen_model(betas=torch.tensor(np.tile(betas_row, (F, 1)), dtype=torch.float32)
                          ).joints[0, 0].cpu().numpy().astype(np.float64)

    # Floor grounding (optional) is done in the SOURCE frame first (a pure
    # translation along the up-axis), so both the user npz and the Z-up geometry
    # npz inherit feet-on-floor regardless of the orientation applied next.
    floor_val = 0.0
    if ground:
        if ref is not None:
            floor_val = float(np.percentile(ref[..., up_idx], 1))   # lowest verts (soles)
        elif ref_joints is not None:
            floor_val = float(np.percentile(ref_joints[:, _J_FEET, up_idx], 5))
        trans = trans.copy()
        trans[:, up_idx] -= floor_val

    # --- orient the npz so it imports UPRIGHT with the chosen add-on Format ---
    # AMASS reproduces the npz frame as-is (needs a Z-up npz); SMPL-X adds a fixed
    # +90deg X (needs a Y-up npz). 'auto' keeps the data's own axes and reports the
    # matching Format.
    if blender_format == "amass":
        R_user, import_fmt = _rotation_aligning(up_vec, _AXIS_UNIT["z"]), "AMASS"
    elif blender_format == "smplx":
        R_user, import_fmt = _rotation_aligning(up_vec, _AXIS_UNIT["y"]), "SMPL-X"
    else:  # auto: keep the data's own axes
        R_user, import_fmt = np.eye(3), {"z": "AMASS", "y": "SMPL-X"}.get(src_up, "AMASS")
    poses_u, trans_u = _rotate_params(poses, trans, R_user, J0)
    log.info("  up-axis=%s (%s, conf %.2f); npz prepared for Blender Format=%s%s",
             src_up, src, conf, import_fmt, ", grounded" if ground else "")

    # Validate the user npz end-to-end (bake + grounding + orientation).
    if ref is not None:
        flat_model = chosen_model if chosen_flat else _build_model(model_dir, gender, num_betas, True, F)
        v_out = _reconstruct_verts(flat_model, poses_u, betas_row, trans_u)
        expected = ref.copy()
        if ground:
            expected[..., up_idx] -= floor_val
        expected = expected @ R_user.T
        out_mm = float(np.abs(v_out - expected).max() * 1000.0)
        if out_mm > validate_tol_mm:
            raise RuntimeError(
                f"{params_path}: exported npz does not reproduce the fit "
                f"({out_mm:.3f} mm > {validate_tol_mm} mm) — bake/orient bug; refusing to export")
        log.info("  export round-trip OK: %.3f mm", out_mm)

    _write_addon_npz(out_path, poses_u, betas_row, trans_u, gender, fps)
    log.info("  wrote %s -> import in Blender with Format=%s", out_path, import_fmt)

    # --- Blender geometry npz: always normalize the source up onto +Z so the rigged
    # FBX/ABC/BVH/USD are correct for ANY input axes. Reuse the user npz when it is
    # already that Z-up file. ---
    blender_out = None
    if write_blender_npz:
        R_z = _rotation_aligning(up_vec, _AXIS_UNIT["z"])
        if np.allclose(R_user, R_z):
            blender_out = out_path
        else:
            poses_z, trans_z = _rotate_params(poses, trans, R_z, J0)
            _write_addon_npz(blender_npz_path, poses_z, betas_row, trans_z, gender, fps)
            blender_out = blender_npz_path
    return out_path, blender_out, src_up, import_fmt


_BLENDER_FORMATS = ("fbx", "abc", "bvh", "usd")  # need Blender; npz is pure-Python


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_blender(explicit=None):
    """Locate a Blender binary: --blender-bin / MAMMA_BLENDER_BIN / the portable
    download under data/blender/ / system `blender`. Returns None if none found."""
    import glob as _glob
    import shutil
    cands = [explicit, os.environ.get("MAMMA_BLENDER_BIN")]
    cands += sorted(_glob.glob(os.path.join(_repo_root(), "data", "blender", "blender-*", "blender")))
    cands.append(shutil.which("blender"))
    for c in cands:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return None


def _blender_export(blender_bin, addon_dir, npz_path, out_prefix, formats, fps, unit):
    """Drive the portable Blender headless (client of the add-on) to write the
    requested rigged formats from the (Z-up-normalized) add-on npz. ``unit`` is
    m or cm (scales the geometry)."""
    import subprocess
    script = os.path.join(_repo_root(), "scripts", "blender_smplx_export.py")
    cmd = [blender_bin, "--background", "--python", script, "--",
           "--addon-dir", addon_dir, "--npz", npz_path, "--out-prefix", out_prefix,
           "--formats", ",".join(formats), "--fps", str(int(fps)), "--unit", unit]
    log.info("  blender export [%s] -> %s.*", ",".join(formats), out_prefix)
    r = subprocess.run(cmd, capture_output=True, text=True)
    produced = [f for f in formats
                if os.path.exists(f"{out_prefix}.{f}") and os.path.getsize(f"{out_prefix}.{f}") > 0]
    missing = [f for f in formats if f not in produced]
    # Hard-fail only when nothing came out; otherwise keep the formats that did
    # (one format failing in Blender must not lose the rest — e.g. the add-on's
    # cm/UNREAL FBX path can throw on some Blender versions).
    if not produced:
        log.error("blender export failed (rc=%s):\n%s", r.returncode, (r.stderr or r.stdout)[-2000:])
        raise RuntimeError(f"blender export failed for {os.path.basename(npz_path)}")
    if missing:
        # Surface what didn't make it (and the Blender output that explains why)
        # via print so it reaches the streamed GUI job log.
        print(f"[export] WARNING: {os.path.basename(npz_path)}: these formats "
              f"were not produced: {','.join(missing)}")
        tail = (r.stdout or r.stderr or "")[-1500:]
        if tail.strip():
            print(tail)
    return [f"{out_prefix}.{f}" for f in produced]


def export_sequence(ma_3d_dir, seq_name, model_dir, out_dir, up_axis="auto",
                    ma_cap_dir=None, fps=None, validate=True,
                    formats=("npz",), blender_bin=None, addon_dir=None,
                    unit="m", ground=True, blender_format="auto"):
    seq_dir = os.path.join(ma_3d_dir, seq_name)
    params = sorted(glob.glob(os.path.join(seq_dir, "smplx_params_body_id-*.npz")))
    if not params:
        raise FileNotFoundError(f"no smplx_params_body_id-*.npz under {seq_dir}")
    fps = _resolve_fps(ma_cap_dir, seq_name, fps)

    # Resolve Blender once (graceful: fall back to npz-only if it / the add-on
    # is missing — the npz path never needs Blender).
    blender_fmts = [f for f in formats if f in _BLENDER_FORMATS]
    bin_ = None
    if blender_fmts:
        addon_dir = addon_dir or os.path.join(_repo_root(), "data", "blender_addon")
        bin_ = _find_blender(blender_bin)
        if bin_ is None:
            log.warning("Blender not found for %s — writing npz only. Run "
                        "data/download_blender.sh or set MAMMA_BLENDER_BIN.", blender_fmts)
            blender_fmts = []
        elif not os.path.isdir(os.path.join(addon_dir, "smplx_blender_addon")):
            log.warning("SMPL-X add-on not found under %s — writing npz only. Run "
                        "data/download_smplx_blender_addon.sh.", addon_dir)
            blender_fmts = []

    out = []
    for p in params:
        m = re.search(r"body_id-(\d+)", os.path.basename(p))
        bid = m.group(1) if m else "00"
        verts = os.path.join(seq_dir, f"verts_joints_body_id-{bid}.npz")
        npz_out = os.path.join(out_dir, f"{seq_name}_body-{bid}_smplx.npz")
        bz = os.path.join(out_dir, f".{seq_name}_body-{bid}_blenderZ.npz")  # temp Z-up
        faithful, bnpz, _src, _fmt = export_person(
            p, model_dir, fps, npz_out, verts_path=verts, validate=validate,
            ground=ground, up_axis=up_axis, blender_format=blender_format,
            write_blender_npz=bool(blender_fmts), blender_npz_path=bz)
        out.append(faithful)
        if blender_fmts and bnpz:
            out += _blender_export(bin_, addon_dir, bnpz,
                                   os.path.join(out_dir, f"{seq_name}_body-{bid}"),
                                   blender_fmts, fps, unit)
            if bnpz != faithful and os.path.exists(bnpz):
                os.remove(bnpz)  # clean the temp Z-up npz (geometry now baked)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ma-3d-dir", "--ma_3d_dir", required=True,
                    help="ma_3d output root (contains <seq>/smplx_params_body_id-*.npz).")
    ap.add_argument("--seq-name", "--seq_name", required=True)
    ap.add_argument("--out-dir", "--out_dir", required=True)
    ap.add_argument("--smplx-models", "--smplx_models",
                    default="data/body_models/smplx_locked_head",
                    help="Folder containing smplx/SMPLX_NEUTRAL.npz (for the hand mean).")
    ap.add_argument("--blender-format", "--blender_format", default="auto",
                    choices=["auto", "amass", "smplx"],
                    help="Which add-on import Format the npz is prepared for. 'auto' "
                         "(default) keeps the data's own axes and reports the matching "
                         "Format; 'amass' orients the npz Z-up (import as AMASS); "
                         "'smplx' orients it Y-up (import as SMPL-X). Both import upright.")
    ap.add_argument("--unit", default="m", choices=["m", "cm"],
                    help="Units for the rigged formats (FBX/ABC/USD/BVH): m (meters, "
                         "default — Blender/Unity/Maya) or cm (centimeters — Unreal). "
                         "The npz is always meters (SMPL-X convention).")
    ap.add_argument("--ground", dest="ground", action="store_true", default=True,
                    help="Drop the feet to the floor (0 along the up-axis). Default on.")
    ap.add_argument("--no-ground", dest="ground", action="store_false",
                    help="Keep the fit's original translation (exact, ungrounded).")
    ap.add_argument("--up-axis", "--up_axis", default="auto", choices=["auto", "x", "y", "z"],
                    help="Source up-axis. 'auto' (default) detects it from the motion "
                         "(foot-contact plane + body-vertical); x/y/z force it. Used for "
                         "grounding and for normalizing the geometry to the target.")
    ap.add_argument("--ma-cap-dir", "--ma_cap_dir", default=None,
                    help="ma_cap output root, to read fps from <seq>/gt/global.npz.")
    ap.add_argument("--fps", type=int, default=None,
                    help="Override mocap_frame_rate (else read from ma_cap, else 30).")
    ap.add_argument("--formats", default="npz",
                    help="Comma list: npz,fbx,abc,bvh,usd. npz is always written "
                         "(pure-Python); fbx/abc/bvh/usd need Blender (auto-detected; "
                         "falls back to npz-only if absent). Default: npz.")
    ap.add_argument("--blender-bin", "--blender_bin", default=None,
                    help="Blender executable (else MAMMA_BLENDER_BIN / data/blender/ / PATH).")
    ap.add_argument("--addon-dir", "--addon_dir", default=None,
                    help="Dir containing the smplx_blender_addon package "
                         "(default data/blender_addon).")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip the round-trip vertex check (not recommended).")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose == 0 else logging.DEBUG,
                        format="%(levelname)s %(name)s: %(message)s")

    formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())
    outs = export_sequence(
        args.ma_3d_dir, args.seq_name, args.smplx_models, args.out_dir,
        up_axis=args.up_axis, ma_cap_dir=args.ma_cap_dir, fps=args.fps,
        validate=not args.no_validate, formats=formats,
        blender_bin=args.blender_bin, addon_dir=args.addon_dir,
        unit=args.unit, ground=args.ground, blender_format=args.blender_format,
    )
    print(f"exported {len(outs)} file(s):")
    for o in outs:
        print(f"  {o}")


if __name__ == "__main__":
    main()
