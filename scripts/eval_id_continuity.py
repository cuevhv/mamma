#!/usr/bin/env python
"""Evaluate ma_masks subject-ID stability against depth-aware GT silhouettes.

GROUND TRUTH. For a ``data/mamma_multi/<seq>`` sequence, each subject's SMPL-X
parameters live in ``pred/params_NN.npz``. We forward SMPL-X to world-space
vertices, project into each camera with the sequence calibration, and rasterise
with the project's pyrender renderer. Occlusion is resolved with a z-buffer: a
subject's pixel is GT-visible only where its own depth equals the frontmost depth
across ALL subjects. This yields per-view, per-frame, per-subject VISIBLE masks
with known identities (the *visible masked area*, not the full projected body).

SCORING. For a ma_masks run's output masks, each predicted id is matched to a GT
subject per frame by mask IoU (Hungarian). We then report, per camera:
  * id_switches  — how often a predicted id's matched GT subject changes over time
                   (the metric the temporal-continuity fix targets),
  * idf1         — fraction of a predicted id's frames spent on its dominant GT id,
  * mean_iou     — mean visible-mask IoU of matched pairs.

REAL-WORLD PROTOCOL. Run ma_masks with several random >=4-camera subsets (and for
each of: baseline / +continuity / sam3_prompt), then point this script at each run
dir; aggregate `id_switches` across subsets. Lower switches with IoU unchanged =
the fix works.

Usage:
  python scripts/eval_id_continuity.py \
      --seq data/mamma_multi/260216_MultiMama_3_accidental_bump_000111_1 \
      --run output/ma_masks/<run>/mamma_multi/<seq> \
      --calib configs/examples/calib/260216_MultiMama.yaml \
      --smplx-models data/body_models/smplx_locked_head \
      [--cams IOI_01 IOI_07 ...] [--frame-stride 5] [--max-frames 200]

  # self-test the metric on synthetic masks (no GPU/models):
  python scripts/eval_id_continuity.py --self-test
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "optimization")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# ---------------------------------------------------------------------------
# Metric (no heavy deps) — match predicted ids to GT ids per frame, count switches
# ---------------------------------------------------------------------------
def _iou(a, b):
    a = a > 0
    b = b > 0
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u > 0 else 0.0


def match_frame(gt_masks: dict, pred_masks: dict, min_iou=0.1):
    """Return {pred_id: gt_id} for one frame via IoU Hungarian (>= min_iou)."""
    from scipy.optimize import linear_sum_assignment
    gids, pids = list(gt_masks), list(pred_masks)
    if not gids or not pids:
        return {}
    cost = np.ones((len(pids), len(gids)), dtype=np.float64)
    for i, p in enumerate(pids):
        for j, g in enumerate(gids):
            cost[i, j] = 1.0 - _iou(pred_masks[p], gt_masks[g])
    ri, ci = linear_sum_assignment(cost)
    out = {}
    for i, j in zip(ri, ci):
        if 1.0 - cost[i, j] >= min_iou:
            out[pids[i]] = (gids[j], 1.0 - cost[i, j])
    return out


def score_camera(gt_by_frame: dict, pred_by_frame: dict, min_iou=0.1):
    """gt_by_frame/pred_by_frame: {frame: {id: mask}}. Returns per-camera metrics."""
    frames = sorted(set(gt_by_frame) & set(pred_by_frame))
    per_pid_seq: dict = {}   # pid -> list of (frame, gid)
    ious = []
    for f in frames:
        m = match_frame(gt_by_frame[f], pred_by_frame[f], min_iou)
        for pid, (gid, iou) in m.items():
            per_pid_seq.setdefault(pid, []).append((f, gid))
            ious.append(iou)
    switches = 0
    idf1_num = idf1_den = 0
    for pid, seq in per_pid_seq.items():
        gids = [g for _, g in seq]
        switches += sum(1 for a, b in zip(gids, gids[1:]) if a != b)
        if gids:
            dom = max(set(gids), key=gids.count)
            idf1_num += gids.count(dom)
            idf1_den += len(gids)
    return {
        "frames_scored": len(frames),
        "id_switches": switches,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "idf1": (idf1_num / idf1_den) if idf1_den else 0.0,
    }


# ---------------------------------------------------------------------------
# Predicted masks (ma_masks PNGs)
# ---------------------------------------------------------------------------
def load_pred_masks(run_seq_dir, cam, frames):
    import cv2
    mdir = os.path.join(run_seq_dir, cam, "masks")
    out = {}
    for f in frames:
        d = {}
        for p in glob.glob(os.path.join(mdir, f"mask_{f:04d}_*.png")):
            pid = int(os.path.basename(p).split("_")[2].split(".")[0]) - 1
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                d[pid] = m > 127
        if d:
            out[f] = d
    return out


# ---------------------------------------------------------------------------
# GT visible silhouettes (SMPL-X -> project -> z-buffer)
# ---------------------------------------------------------------------------
def load_subjects(seq_dir, smplx_models_dir, device="cuda"):
    """Return (verts_world list[(F,V,3)], faces). One entry per pred/params_NN.npz."""
    import torch
    from utils_smplx import get_smplx_models, get_smplx_forward
    pfiles = sorted(glob.glob(os.path.join(seq_dir, "pred", "params_*.npz")))
    if not pfiles:
        raise FileNotFoundError(f"no pred/params_*.npz under {seq_dir}")
    first = np.load(pfiles[0], allow_pickle=True)
    num_betas = int(first["betas"].reshape(-1).shape[0])
    flat = bool(first["flat_hand_mean"])
    models = get_smplx_models(smplx_models_dir, num_betas=num_betas, flat_hand=flat,
                              n_people=1, device=device)[0]
    faces = np.asarray(models["neutral"].faces, dtype=np.int64)
    verts = []
    for pf in pfiles:
        d = np.load(pf, allow_pickle=True)
        F = d["poses"].shape[0]
        poses = torch.tensor(d["poses"], dtype=torch.float32, device=device)
        betas = torch.tensor(d["betas"].reshape(1, -1), dtype=torch.float32, device=device).repeat(F, 1)
        trans = torch.tensor(d["trans"], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = get_smplx_forward(poses, betas, trans, str(d["gender"]), models)
        verts.append(out.vertices.detach().cpu().numpy())
    return verts, faces


def gt_visible_masks_for_camera(verts_world, faces, cam, frames, depth_eps=0.05):
    """{frame: {subject_id: visible_mask}} for one camera, occlusion-resolved."""
    from utils.utils_camera import w2c
    from scene_debug.renderer import Renderer
    K = np.asarray(cam.intrinsics, dtype=np.float64)
    T = np.asarray(cam.T_cam_world, dtype=np.float64)
    W, H = int(cam.width), int(cam.height)
    rnd = Renderer(focal_length_px=float(K[0, 0]), img_w=W, img_h=H,
                   principal_p_x=float(K[0, 2]), principal_p_y=float(K[1, 2]), faces=faces)
    S = len(verts_world)
    out = {}
    for f in frames:
        vc = [w2c(verts_world[s][f], T) for s in range(S)]   # OpenCV camera-frame verts
        in_front = [v for v in vc if (v[:, 2] > 0).any()]    # skip subjects fully behind
        if not in_front:
            continue
        _, depth_all = rnd.render_front_view(vc, return_depth=True)
        masks = {}
        for s in range(S):
            if not (vc[s][:, 2] > 0).any():
                continue
            _, depth_s = rnd.render_front_view([vc[s]], return_depth=True)
            vis = (depth_s > 0) & (np.abs(depth_s - depth_all) < depth_eps)
            if vis.any():
                masks[s] = vis
        if masks:
            out[f] = masks
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def evaluate(seq_dir, run_seq_dir, calib_path, smplx_models_dir, cams=None,
             frame_stride=5, max_frames=None, depth_eps=0.05, min_iou=0.1):
    from capture.calibration import load_calibration
    calib = load_calibration(calib_path)
    verts, faces = load_subjects(seq_dir, smplx_models_dir)
    n_frames = min(v.shape[0] for v in verts)
    frames = list(range(0, n_frames, frame_stride))
    if max_frames:
        frames = frames[:max_frames]
    if cams is None:
        cams = sorted(c for c in calib.cameras
                      if os.path.isdir(os.path.join(run_seq_dir, c, "masks")))
    results = {}
    for cam in cams:
        if cam not in calib.cameras:
            print(f"  [skip] {cam}: not in calibration"); continue
        gt = gt_visible_masks_for_camera(verts, faces, calib.cameras[cam], frames, depth_eps)
        pred = load_pred_masks(run_seq_dir, cam, frames)
        results[cam] = score_camera(gt, pred, min_iou)
        r = results[cam]
        print(f"  {cam}: switches={r['id_switches']}  idf1={r['idf1']:.3f}  "
              f"meanIoU={r['mean_iou']:.3f}  frames={r['frames_scored']}")
    tot = sum(r["id_switches"] for r in results.values())
    miou = np.mean([r["mean_iou"] for r in results.values()]) if results else 0.0
    print(f"TOTAL id_switches={tot}  mean IoU={miou:.3f}  over {len(results)} cameras")
    return results


def _self_test():
    """Metric self-test (no GPU): a synthetic 1-frame swap -> 1 id_switch."""
    H, W = 20, 40
    def box(x1, x2):
        m = np.zeros((H, W), bool); m[5:15, x1:x2] = True; return m
    A, B = box(2, 12), box(28, 38)
    gt = {f: {0: A, 1: B} for f in range(6)}
    pred = {f: {0: A, 1: B} for f in range(6)}
    pred[3] = {0: B, 1: A}     # predicted ids swapped on frame 3
    r = score_camera(gt, pred)
    # pid0 gids 0,0,0,1,0,0 -> 2 switches; pid1 -> 2 switches => 4 total.
    assert r["id_switches"] == 4, r
    assert r["mean_iou"] > 0.99 and r["idf1"] < 0.9, r
    # a clean (no-swap) sequence must report zero switches
    r0 = score_camera(gt, {f: gt[f] for f in gt})
    assert r0["id_switches"] == 0 and r0["idf1"] == 1.0, r0
    print("self-test OK:", r, "| clean:", r0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seq")
    ap.add_argument("--run", help="ma_masks output seq dir (…/mamma_multi/<seq>)")
    ap.add_argument("--calib")
    ap.add_argument("--smplx-models", default="data/body_models/smplx_locked_head")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--depth-eps", type=float, default=0.05)
    args = ap.parse_args()
    if args.self_test:
        _self_test(); return
    if not (args.seq and args.run and args.calib):
        ap.error("--seq, --run and --calib are required (or use --self-test)")
    evaluate(args.seq, args.run, args.calib, args.smplx_models, cams=args.cams,
             frame_stride=args.frame_stride, max_frames=args.max_frames, depth_eps=args.depth_eps)


if __name__ == "__main__":
    main()
