"""Prepare a compact 3D-scene blob for the GUI's in-browser mesh preview.

The browser poses SMPL-X itself (three.js GPU skinning), so instead of
per-frame geometry (~126 KB/person/frame) we ship parameters: per person
a shaped rest template + rest joints + per-frame axis-angle poses and
translations, plus the shared topology (faces, top-4 skin weights,
kinematic parents). A 2-person 426-frame sequence is ~1.5 MB of JSON.

Server-side we bake everything the client shouldn't know about:
``v_template_pred`` overrides, betas x shapedirs, and the SMPL-X
``flat_hand_mean`` convention (hand means are added here when the run
used ``flat_hand_mean=False``). The client's job reduces to standard
skeletal animation: axis-angle -> quaternion, FK over ``parents``, LBS
with 4 influences. Validated against ``pred_vertices``: mean error
~2-3 mm (no pose correctives), max ~3 cm at bent joints.

Usage:
    python -m visualization.preview_scene --ma-3d-dir output/ma_3d/<tag>/<capture> \
        --seq-name <seq> --out scene3d.json
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import time
from glob import glob
from pathlib import Path

import numpy as np

from .overlay import _default_palette

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL = _REPO_ROOT / "data" / "body_models" / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz"

_NUM_JOINTS = 55


def _b64(arr: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def _top4_weights(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncate LBS weights to the 4 strongest influences per vertex
    (three.js' limit) and renormalize. Drops <5% weight mass worst-case."""
    idx = np.argsort(-weights, axis=1)[:, :4]
    rows = np.arange(weights.shape[0])[:, None]
    w = weights[rows, idx]
    w = w / w.sum(axis=1, keepdims=True)
    return idx.astype(np.uint16), w.astype(np.float32)


def _expand_pose(pose: np.ndarray, flat_hand: bool,
                 hands_meanl: np.ndarray, hands_meanr: np.ndarray) -> np.ndarray:
    """(T,165) packed pose -> (T,55,3) per-joint axis-angle in SMPL-X joint
    order, with hand means baked in when the model added them implicitly."""
    T = pose.shape[0]
    fp = np.zeros((T, _NUM_JOINTS, 3), dtype=np.float64)
    fp[:, 0] = pose[:, 0:3]                       # global orient (pelvis)
    fp[:, 1:22] = pose[:, 3:66].reshape(T, 21, 3)  # body
    fp[:, 22] = pose[:, 66:69]                    # jaw
    fp[:, 23] = pose[:, 69:72]                    # left eye
    fp[:, 24] = pose[:, 72:75]                    # right eye
    lh = pose[:, 75:120].copy()
    rh = pose[:, 120:165].copy()
    if not flat_hand:
        lh += hands_meanl[None]
        rh += hands_meanr[None]
    fp[:, 25:40] = lh.reshape(T, 15, 3)
    fp[:, 40:55] = rh.reshape(T, 15, 3)
    return fp


def _recorded_fps(ma_cap_dir, seq_name: str) -> int | None:
    """The capture's recorded frame rate from ``<ma_cap_dir>/<seq>/gt/global.npz``
    (same source ma_vis' CLI uses), or None when absent — older ma_cap runs
    predate the ``fps`` key."""
    if not ma_cap_dir:
        return None
    p = Path(ma_cap_dir) / seq_name / "gt" / "global.npz"
    try:
        with np.load(p, allow_pickle=True) as z:
            if "fps" in z.files and int(z["fps"]) > 0:
                return int(z["fps"])
    except (OSError, ValueError):
        pass
    return None


def build_scene(ma_3d_dir, seq_name: str, *, model_npz=None, fps: int = 30,
                ma_cap_dir=None, up_axis: str = "auto") -> dict:
    seq_dir = Path(ma_3d_dir) / seq_name
    params = sorted(glob(str(seq_dir / "smplx_params_body_id-*.npz")))
    if not params:
        raise FileNotFoundError(f"no smplx_params_body_id-*.npz under {seq_dir}")

    recorded = _recorded_fps(ma_cap_dir, seq_name)
    fps_source = "recorded" if recorded is not None else "fallback"
    if recorded is not None:
        fps = recorded

    model_npz = Path(model_npz or _DEFAULT_MODEL)
    m = np.load(model_npz, allow_pickle=True)
    v_template = np.asarray(m["v_template"], dtype=np.float64)
    shapedirs = np.asarray(m["shapedirs"], dtype=np.float64)
    j_reg = np.asarray(m["J_regressor"], dtype=np.float64)
    weights = np.asarray(m["weights"], dtype=np.float64)
    faces = np.asarray(m["f"], dtype=np.uint32)
    parents = np.asarray(m["kintree_table"][0], dtype=np.int64)[:_NUM_JOINTS]
    parents[0] = -1
    hands_meanl = np.asarray(m["hands_meanl"], dtype=np.float64)
    hands_meanr = np.asarray(m["hands_meanr"], dtype=np.float64)

    skin_idx, skin_w = _top4_weights(weights[:, :_NUM_JOINTS])

    # ONE scene-level up-axis + floor, same machinery the Blender exporter
    # uses. An explicit up_axis (the task config's global.up_axis, passed by
    # the exporter API) is authoritative; "auto" pose-based detection is the
    # fallback only, since it misreads floor-work sequences where the
    # dancers actually lie on the ground (task 139 post-mortem: detected -y
    # in a z-up scene and the viewer showed everyone flat).
    from optimization.export_blender import _scene_orient_and_floor
    (up_vec, up_idx, src, conf), floor = _scene_orient_and_floor(
        str(seq_dir), params, up_axis or "auto", fps)
    axis = "xyz"[up_idx]
    up_axis = axis if up_vec[up_idx] >= 0 else f"-{axis}"
    log.info("up-axis %s (source=%s confidence=%.2f) floor=%s", up_axis, src, conf,
             "n/a" if floor is None else f"{floor:.3f}")

    colors = _default_palette(max(10, len(params)))
    people = []
    for p in params:
        mm = re.search(r"body_id-(\d+)", Path(p).name)
        body_id = int(mm.group(1)) if mm else len(people)
        with np.load(p, allow_pickle=True) as d:
            pose = np.asarray(d["smplx_pose"], dtype=np.float64)
            betas = np.asarray(d["smplx_betas"], dtype=np.float64).reshape(-1)
            transl = np.asarray(d["smplx_translation"], dtype=np.float64)
            vt_pred = d["v_template_pred"] if "v_template_pred" in d.files else None
            flat_hand = (bool(d["smplx_export_flat_hand_mean"])
                         if "smplx_export_flat_hand_mean" in d.files else False)

        base = v_template
        if vt_pred is not None and vt_pred.dtype != object and vt_pred.size > 3:
            base = np.asarray(vt_pred, dtype=np.float64).reshape(-1, 3)
        v_shaped = base + np.einsum("vij,j->vi", shapedirs[:, :, :betas.shape[0]], betas)
        rest_joints = j_reg[:_NUM_JOINTS] @ v_shaped
        full_pose = _expand_pose(pose, flat_hand, hands_meanl, hands_meanr)

        c = np.clip(np.array(colors[body_id % len(colors)]) * 255, 0, 255).astype(int)
        people.append({
            "bodyId": body_id,
            "color": f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}",
            "frameCount": int(pose.shape[0]),
            "vShaped": _b64(v_shaped, "<f4"),        # (V,3) f32
            "restJoints": _b64(rest_joints, "<f4"),  # (55,3) f32
            "poses": _b64(full_pose, "<f4"),         # (T,55,3) f32 axis-angle
            "transl": _b64(transl, "<f4"),           # (T,3) f32
        })

    return {
        "version": 3,
        "seq": seq_name,
        "fps": int(fps),
        # "recorded" = the capture's own rate (gt/global.npz) is baked in;
        # "fallback" = no recorded rate — the viewer may apply a client-side
        # hint (e.g. the capture.json FPS) instead of this default.
        "fpsSource": fps_source,
        "upAxis": up_axis,
        "floor": None if floor is None else float(floor),
        "vertexCount": int(v_template.shape[0]),
        "jointCount": _NUM_JOINTS,
        "parents": [int(x) for x in parents],
        "faces": _b64(faces, "<u4"),                 # (F,3) u32
        "skinIndex": _b64(skin_idx, "<u2"),          # (V,4) u16
        "skinWeight": _b64(skin_w, "<f4"),           # (V,4) f32
        "frameCount": max(p["frameCount"] for p in people),
        "people": people,
    }


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="[preview-scene] %(message)s")
    p = argparse.ArgumentParser(description="Build the GUI 3D mesh-preview scene blob")
    p.add_argument("--ma-3d-dir", required=True, help="ma_3d output dir (parent of <seq>/)")
    p.add_argument("--seq-name", required=True)
    p.add_argument("--out", required=True, help="output .json path")
    p.add_argument("--model-npz", default=None,
                   help=f"SMPL-X neutral model npz (default {_DEFAULT_MODEL})")
    p.add_argument("--fps", type=int, default=30,
                   help="fallback frame rate; the ma_cap recorded rate wins when available")
    p.add_argument("--ma-cap-dir", default=None,
                   help="ma_cap output dir (parent of <seq>/) — source of the recorded fps")
    p.add_argument("--up-axis", default="auto",
                   help="World up axis ('z', 'y', '-y', ...). Pass the task "
                        "config's global.up_axis; 'auto' pose-detection is "
                        "the fallback and misreads floor-work sequences")
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    scene = build_scene(args.ma_3d_dir, args.seq_name, model_npz=args.model_npz,
                        fps=args.fps, ma_cap_dir=args.ma_cap_dir,
                        up_axis=args.up_axis)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(scene))
    tmp.replace(out)  # atomic: never leave a half-written blob for the GUI
    log.info("wrote %s (%.1f MB, %d people, %d frames) in %.1fs",
             out, out.stat().st_size / 1e6, len(scene["people"]),
             scene["frameCount"], time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
