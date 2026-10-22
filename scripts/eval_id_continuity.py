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
  * id_switches : how often a predicted id's matched GT subject changes over time
                   (the metric the temporal-continuity fix targets),
  * dom_id_frac : fraction of a predicted id's MATCHED frames spent on its
                   dominant GT id. NOT the MOT-literature IDF1 (it ignores
                   false negatives entirely and reports 1.0 on a total
                   identity collapse); renamed accordingly.
  * mean_iou    : mean visible-mask IoU of matched pairs.

REAL-WORLD PROTOCOL. Run ma_masks with several random >=4-camera subsets (and for
each of: baseline / +continuity / sam3_prompt), then point this script at each run
dir; aggregate `id_switches` across subsets. Lower switches with IoU unchanged =
the fix works.

Usage:
  python scripts/eval_id_continuity.py \
      --seq data/<dataset>/<seq> \
      --run output/ma_masks/<run>/<dataset>/<seq> \
      --calib configs/examples/calib/<calibration>.yaml \
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
# Metric (no heavy deps) - match predicted ids to GT ids per frame, count switches
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
    # Pre-mask sub-threshold pairs so the assignment cannot be forced through
    # them: filtering only after the solve lets the solver route through a
    # to-be-discarded cell and starve a valid match elsewhere.
    BLOCKED = 2.0
    cost[cost > 1.0 - min_iou] = BLOCKED
    ri, ci = linear_sum_assignment(cost)
    out = {}
    for i, j in zip(ri, ci):
        if cost[i, j] < BLOCKED and 1.0 - cost[i, j] >= min_iou:
            out[pids[i]] = (gids[j], 1.0 - cost[i, j])
    return out


def score_camera(gt_by_frame: dict, pred_by_frame: dict, min_iou=0.1,
                 records=None):
    """gt_by_frame/pred_by_frame: {frame: {id: mask}}. Returns per-camera metrics.

    Besides the original stability metrics (id_switches / dom_id_frac), reports the
    cleanup-quality metrics the mask-gate work is judged by:
      * id_accuracy: per-frame matches agreeing with the GLOBAL pred->gt
        mapping (matched-frame-count Hungarian over the whole camera, summed
        IoU as tie-break). Robust to the arbitrary id initialization;
        punishes mid-tracklet swaps.
      * dup_rate   : fraction of scored frames where two PREDICTED masks
        overlap at IoU > 0.5 (two ids on one person).
      * dup_inst_rate: duplicated pred instances / all pred instances, a
        magnitude companion to the per-frame incidence dup_rate.
      * gt_recall  : matched person-frames / GT-visible person-frames
        (coverage: how much of what is genuinely visible got a mask).
    """
    from scipy.optimize import linear_sum_assignment
    # Score over EVERY GT frame: a frame where the tracker produced nothing
    # is a total miss, not a frame to skip (dropping it from the denominators
    # would let total failure improve the metrics).
    frames = sorted(gt_by_frame)
    pred_by_frame = {f: pred_by_frame.get(f, {}) for f in frames}
    per_pid_seq: dict = {}   # pid -> list of (frame, gid)
    ious = []
    pair_iou_sum: dict = {}  # (pid, gid) -> summed IoU over all frames
    pair_count: dict = {}    # (pid, gid) -> matched-frame count
    gt_person_frames = 0
    matched_person_frames = 0
    dup_frames = 0
    dup_instances = 0
    pred_instances = 0
    for f in frames:
        gt_person_frames += len(gt_by_frame[f])
        m = match_frame(gt_by_frame[f], pred_by_frame[f], min_iou)
        if records is not None:
            records.append({
                "frame": int(f),
                "gt_ids": [int(g) for g in gt_by_frame[f]],
                # A blank PNG is an absent observation, not a predicted
                # instance. Keep historical per-camera metrics unchanged.
                "pred_ids": [int(p) for p, mask in pred_by_frame[f].items()
                             if np.any(mask)],
                "matches": [[int(p), int(g), float(iou)]
                            for p, (g, iou) in m.items()],
            })
        matched_person_frames += len(m)
        for pid, (gid, iou) in m.items():
            per_pid_seq.setdefault(pid, []).append((f, gid))
            ious.append(iou)
            pair_iou_sum[(pid, gid)] = pair_iou_sum.get((pid, gid), 0.0) + iou
            pair_count[(pid, gid)] = pair_count.get((pid, gid), 0) + 1
        preds = pred_by_frame[f]
        pids = list(preds)
        pred_instances += len(pids)
        dup_pids = {p for i, a in enumerate(pids) for b in pids[i + 1:]
                    if _iou(preds[a], preds[b]) > 0.5 for p in (a, b)}
        dup_instances += len(dup_pids)
        if dup_pids:
            dup_frames += 1
    switches = 0
    idf1_num = idf1_den = 0
    for pid, seq in per_pid_seq.items():
        gids = [g for _, g in seq]
        switches += sum(1 for a, b in zip(gids, gids[1:]) if a != b)
        if gids:
            dom = max(set(gids), key=gids.count)
            idf1_num += gids.count(dom)
            idf1_den += len(gids)
    # Global pred->gt mapping: Hungarian on matched-frame count (summed IoU
    # only as tie-break, scaled below 1 frame so it can never outvote a
    # count difference; count is what identity agreement is measured in,
    # and summed IoU lets a short unoccluded stretch outweigh a long
    # correct one).
    all_pids = sorted({p for p, _ in pair_count})
    all_gids = sorted({g for _, g in pair_count})
    global_map = {}
    if all_pids and all_gids:
        tie = 1.0 / (len(frames) + 1.0)
        cost = np.zeros((len(all_pids), len(all_gids)))
        for (p, g), n in pair_count.items():
            cost[all_pids.index(p), all_gids.index(g)] = \
                -(n + tie * pair_iou_sum[(p, g)])
        ri, ci = linear_sum_assignment(cost)
        global_map = {all_pids[i]: all_gids[j] for i, j in zip(ri, ci)
                      if cost[i, j] < 0}
    agree = total = 0
    for pid, seq in per_pid_seq.items():
        for _, gid in seq:
            total += 1
            agree += int(global_map.get(pid) == gid)
    return {
        "frames_scored": len(frames),
        "id_switches": switches,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "dom_id_frac": (idf1_num / idf1_den) if idf1_den else 0.0,
        "id_accuracy": (agree / total) if total else 0.0,
        "dup_rate": (dup_frames / len(frames)) if frames else 0.0,
        "dup_inst_rate": (dup_instances / pred_instances) if pred_instances else 0.0,
        "gt_recall": (matched_person_frames / gt_person_frames)
                     if gt_person_frames else 0.0,
    }


def score_sequence_records(records_by_camera):
    """Score identities with ONE pred->GT permutation for the entire rig.

    The historical per-camera ``id_accuracy`` deliberately remains unchanged.
    This companion catches a camera-wide permutation and counts missing masks
    in identity recall. Records come from ``score_camera(..., records=[])``;
    no masks or model dependencies are needed to rescore exported evidence.
    IDs in records and in the returned mapping are zero-based.
    """
    from collections import defaultdict
    from scipy.optimize import linear_sum_assignment

    counts, iou_sums = defaultdict(int), defaultdict(float)
    for records in records_by_camera.values():
        for record in records:
            for pid, gid, iou in record["matches"]:
                counts[pid, gid] += 1
                iou_sums[pid, gid] += iou
    pids = sorted({p for p, _ in counts})
    gids = sorted({g for _, g in counts})
    mapping = {}
    if pids and gids:
        # The TOTAL IoU contribution of any assignment is strictly below one
        # count, so the tie-break cannot trade away a matched observation.
        tie = 1.0 / (1.0 + sum(iou_sums.values()))
        weights = np.array([[counts[p, g] + tie * iou_sums[p, g]
                             for g in gids] for p in pids])
        ri, ci = linear_sum_assignment(-weights)
        mapping = {pids[i]: gids[j] for i, j in zip(ri, ci)
                   if counts[pids[i], gids[j]] > 0}

    def metrics(records):
        matched = correct = visible = predicted = 0
        for record in records:
            visible += len(record["gt_ids"])
            predicted += len(record["pred_ids"])
            for pid, gid, _ in record["matches"]:
                matched += 1
                correct += int(mapping.get(pid) == gid)
        return {
            "id_accuracy": correct / matched if matched else 0.0,
            "id_recall": correct / visible if visible else 0.0,
            "id_precision": correct / predicted if predicted else 0.0,
            "gt_recall": matched / visible if visible else 0.0,
            "correct_instances": correct, "matched_instances": matched,
            "gt_instances": visible, "pred_instances": predicted,
        }

    all_records = [r for records in records_by_camera.values() for r in records]
    return {"mapping": mapping, "overall": metrics(all_records),
            "cameras": {c: metrics(rs) for c, rs in records_by_camera.items()}}


# ---------------------------------------------------------------------------
# Predicted masks (ma_masks PNGs)
# ---------------------------------------------------------------------------
def load_pred_masks(run_seq_dir, cam, frames):
    """Predicted masks per frame from the packed ``masks.zip`` (ma_masks default)
    or, without one, the legacy loose ``mask_FFFF_PP.png`` files."""
    import cv2
    import zipfile
    mdir = os.path.join(run_seq_dir, cam, "masks")
    archive = os.path.join(mdir, "masks.zip")
    zf = zipfile.ZipFile(archive) if os.path.isfile(archive) else None
    names = zf.namelist() if zf is not None else [
        os.path.basename(p) for p in glob.glob(os.path.join(mdir, "mask_*.png"))]
    by_frame = {}
    for n in names:
        parts = n[:-4].split("_") if n.startswith("mask_") and n.endswith(".png") else []
        if len(parts) == 3:
            by_frame.setdefault(int(parts[1]), []).append((int(parts[2]) - 1, n))
    out = {}
    for f in frames:
        d = {}
        for pid, n in by_frame.get(f, []):
            if zf is not None:
                m = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_GRAYSCALE)
            else:
                m = cv2.imread(os.path.join(mdir, n), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                d[pid] = m > 127
        if d:
            out[f] = d
    if zf is not None:
        zf.close()
    if not out:
        print(f"  [warn] {cam}: no predicted masks found under {mdir}")
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


def gt_visible_masks_for_camera(verts_world, faces, cam, frames, depth_eps=0.005,
                                min_area_frac=0.0002, visibility_out=None):
    """{frame: {subject_id: visible_mask}} for one camera, occlusion-resolved.

    ``depth_eps`` is the z-buffer visibility tolerance in meters; large values
    double-assign contact pixels to both subjects and depress the IoU ceiling.
    Subjects whose visible area is below ``min_area_frac`` of the image are
    dropped: a sliver a few pixels wide can never reach min_iou against any
    real prediction, so admitting it builds an unreachable recall deficit.

    Optional visibility_out records visible/solo projected areas for an
    occlusion audit, including below-threshold subjects. It does not change
    masks or scoring. The denominator is the on-image solo silhouette, not
    the full body area outside the image.
    """
    from utils.utils_camera import w2c
    from scene_debug.renderer import Renderer
    K = np.asarray(cam.intrinsics, dtype=np.float64)
    T = np.asarray(cam.T_cam_world, dtype=np.float64)
    W, H = int(cam.width), int(cam.height)
    rnd = Renderer(focal_length_px=float(K[0, 0]), img_w=W, img_h=H,
                   principal_p_x=float(K[0, 2]), principal_p_y=float(K[1, 2]), faces=faces,
                   focal_length_px_y=float(K[1, 1]))
    S = len(verts_world)
    min_area = max(1, int(round(min_area_frac * W * H)))
    out = {}
    for f in frames:
        if visibility_out is not None:
            visibility_out[f] = {}
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
            if visibility_out is not None:
                visible_area, solo_area = int(vis.sum()), int((depth_s > 0).sum())
                visibility_out[f][s] = {
                    "visible_area": visible_area, "solo_area": solo_area,
                    "visible_fraction": visible_area / solo_area if solo_area else None,
                    "eligible": visible_area >= min_area,
                }
            if vis.sum() >= min_area:
                masks[s] = vis
        if masks:
            out[f] = masks
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def evaluate(seq_dir, run_seq_dir, calib_path, smplx_models_dir, cams=None,
             frame_stride=5, max_frames=None, depth_eps=0.005, min_iou=0.1,
             json_out=None, sequence_json_out=None, evidence_out=None):
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
    records_by_camera = {}
    for cam in cams:
        if cam not in calib.cameras:
            print(f"  [skip] {cam}: not in calibration"); continue
        gt = gt_visible_masks_for_camera(verts, faces, calib.cameras[cam], frames, depth_eps)
        pred = load_pred_masks(run_seq_dir, cam, frames)
        records_by_camera[cam] = []
        results[cam] = score_camera(gt, pred, min_iou, records=records_by_camera[cam])
        r = results[cam]
        print(f"  {cam}: id_acc={r['id_accuracy']:.3f}  dup={r['dup_rate']:.3f}  "
              f"recall={r['gt_recall']:.3f}  meanIoU={r['mean_iou']:.3f}  "
              f"switches={r['id_switches']}  frames={r['frames_scored']}")
    tot = sum(r["id_switches"] for r in results.values())
    miou = np.mean([r["mean_iou"] for r in results.values()]) if results else 0.0
    print(f"TOTAL id_switches={tot}  mean IoU={miou:.3f}  over {len(results)} cameras")
    shared = score_sequence_records(records_by_camera)
    print(f"SHARED mapping={shared['mapping']}  "
          f"id_acc={shared['overall']['id_accuracy']:.6f}  "
          f"id_recall={shared['overall']['id_recall']:.6f}")
    for path, payload in ((sequence_json_out, shared),
                          (evidence_out, records_by_camera)):
        if path:
            import json
            with open(path, "w") as f:
                json.dump(payload, f, indent=1)
    if json_out:
        import json
        with open(json_out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {json_out}")
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
    assert r["mean_iou"] > 0.99 and r["dom_id_frac"] < 0.9, r
    # global mapping: pid0->g0 dominates (5/6), so the swapped frame counts
    # against id_accuracy for both pids: 10/12 agree.
    assert abs(r["id_accuracy"] - 10 / 12) < 1e-9, r
    assert r["dup_rate"] == 0.0 and r["gt_recall"] == 1.0, r
    # a clean (no-swap) sequence must report zero switches
    r0 = score_camera(gt, {f: gt[f] for f in gt})
    assert r0["id_switches"] == 0 and r0["dom_id_frac"] == 1.0, r0
    assert r0["id_accuracy"] == 1.0 and r0["gt_recall"] == 1.0, r0
    # duplicates: both pred ids on the same box on one frame
    gt4 = {f: {0: A, 1: B} for f in range(4)}
    dup = {f: {0: A, 1: B} for f in range(4)}
    dup[2] = {0: A, 1: A}
    rd = score_camera(gt4, dup)
    assert rd["dup_rate"] == 0.25, rd
    print("self-test OK:", {k: round(v, 3) for k, v in r.items()},
          "| dup:", round(rd["dup_rate"], 3))


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
    ap.add_argument("--depth-eps", type=float, default=0.005)
    ap.add_argument("--json-out", default=None,
                    help="Write the per-camera metrics dict as JSON here")
    ap.add_argument("--sequence-json-out", default=None,
                    help="Write identity metrics using one mapping across ALL cameras")
    ap.add_argument("--evidence-out", default=None,
                    help="Write per-frame IoU matches and visible/predicted IDs for auditing")
    args = ap.parse_args()
    if args.self_test:
        _self_test(); return
    if not (args.seq and args.run and args.calib):
        ap.error("--seq, --run and --calib are required (or use --self-test)")
    evaluate(args.seq, args.run, args.calib, args.smplx_models, cams=args.cams,
             frame_stride=args.frame_stride, max_frames=args.max_frames,
             depth_eps=args.depth_eps, json_out=args.json_out,
             sequence_json_out=args.sequence_json_out, evidence_out=args.evidence_out)


if __name__ == "__main__":
    main()
