import os
import copy
import json
import shutil
import subprocess
import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm

import cv2
from ultralytics import YOLO
import open_clip

from core.logging import logger
from core.exceptions import CameraProcessingError
from core import geometry
from core.mask_store import MaskStore
from core.predictor_factory import build_video_predictor
from core.frame_source import ImageFileSource, VideoSource, frame_source_from_cam_data
from utils.drawing import show_box, show_mask_cv2
from PIL import Image
from utils.torch_math import cosine_knn

from sklearn.manifold import TSNE
import seaborn as sns
import platform

from utils.paths import string_path_to_windows

from typing import Protocol


class _VideoPredictor(Protocol):
    """Structural type for the SAM2-compatible video predictor API.

    Both the raw SAM2 predictor and _Sam3VideoAdapter satisfy this. The
    _Sam3PromptVideoAdapter (sam3_prompt mode) exposes a different
    session-based API and is accessed through _sam3_prompt_predictor(),
    not directly as self.predictor.
    """

    def init_state(self, video_path, **kwargs): ...
    def reset_state(self, inference_state): ...
    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, **kwargs): ...
    def propagate_in_video(self, inference_state, reverse: bool = False, **kwargs): ...


# Use imageio-ffmpeg's bundled, statically-linked ffmpeg binary instead of whatever
# happens to be on the system PATH. This avoids environment-specific ffmpeg issues
# (missing system package, OpenCL symbol clashes under Apptainer --nv, etc.) and
# makes video export reproducible across Linux / macOS / Windows / containers.
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def _sam_log(level: str, message: str):
    """Legacy wrapper — delegates to loguru.

    Maps level aliases to loguru's actual method names. Notably loguru exposes
    ``warning`` (not ``warn``); without this mapping ``_log_warn`` silently fell
    back to ``logger.info``, demoting every pipeline warning to INFO.
    """
    lvl = {"WARN": "warning", "WARNING": "warning", "FATAL": "critical"}.get(
        level.upper(), level.lower())
    getattr(logger, lvl, logger.info)(message)


DEFAULT_ASSIGNMENT_CONFIG = {
    "matching": {
        "start_frame": 10,
        "sample_interval": 50,
        "min_samples": 5,
        "yolo_person_conf": 0.5,
        "min_mask_area_ratio": 0.01,
        "yolo_dedup_enable": True,
        "yolo_dedup_iou": 0.75,
        "bootstrap_frame_offsets": [0],
        "bootstrap_min_score": 0.25,
        "feature_bank_max_size": 64,
        "online_bank_update_min_score": 0.35,
        "max_anchors_per_id": 0,  # <=0 means auto
        "min_anchor_frame_gap": 15,
        "anchor_score_threshold": 0.2,
        "anchor_fallback_to_best_any": True,
        "anchor_auto_max_cap": 32,
        "anchor_samples_per_anchor": 6,
        "anchor_enforce_cross_id_uniqueness": True,
        "anchor_conflict_iou": 0.65,
        # Seed/prompt frame choice: down-weight clip-boundary candidates
        # (see _seed_centrality_weight; length-scaled, inert on short clips).
        "seed_center_bias": 0.5,
        "clip_weight": 0.4,
        "epipolar_weight": 0.6,
        # Resolution-aware by default ("auto"): scaled from image diagonal.
        "epipolar_sigma_px": "auto",
        "max_epipolar_dist_px": "auto",
        "epipolar_sigma_diag_ratio": 0.075,
        "max_epipolar_dist_diag_ratio": 0.18,
        "clip_min_similarity": 0.0,
        "combined_min_score": 0.2,
        "allow_missing_ids": False,
    },
    "masks": {
        "merge_duplicate_tracklets": True,
        "merge_iou_threshold": 0.95,
        # Identity-continuity cleanup (validated production defaults):
        # the gate drops transient teleports/duplicate stretches; recovery
        # re-acquires a person lost to a takeover collapse (abstains without
        # takeover-scale jump evidence; SAM2 backends only, no-op elsewhere).
        "mask_continuity_gate": True,
        "recover_collapsed_tracklets": True,
        # Track-then-match seed prescreen: bound the seeded tracklets to the
        # top-M candidates per init subject (epipolar+CLIP shortlist) before
        # any tracking runs. "auto" (default) = engage only when detections >
        # N*M AND epipolar calibration is available — clean scenes are
        # untouched by construction; "off" = seed every detection (pre-v1.1.0
        # behaviour). Identity is still decided after tracking by the remap.
        # Validated on 444 archived ttm camera events (0 subject drops) +
        # GT-silhouette no-op check; see experimental_scripts/
        # seed_prescreen_validation.html (private).
        "seed_prescreen": "auto",
        "seed_candidates_per_subject": 3,
        "truncate_relatched_tracklets": True,
        "relatch_iou": 0.9,
        "relatch_min_run_frames": 5,
        "relatch_pre_frames": 5,
        "relatch_pre_area_px": 100,
        "discard_tiny_tracklets": True,
        "tiny_tracklet_min_area_ratio": 0.005,
        "tiny_tracklet_min_frame_ratio": 0.2,
        # Cross-camera identity for non-init cameras. "auto" (default) is
        # calibration- and backend-aware: track_then_match for SAM3-text detection
        # (sam3_prompt_light) always, and for YOLO backends (sam2/sam3) when epipolar
        # calibration (cam_int/cam_ext) is available — with geometry, track-then-match
        # takes every backend to ~100% cross-camera id-accuracy. Falls back to
        # match_then_track (legacy) for YOLO backends only when matching is CLIP-only
        # (no calibration). See _cross_camera_strategy(). Override with an explicit
        # strategy if desired.
        "cross_camera_strategy": "auto",
        # Seed selection for track-then-match seeds every detected person on the seed frame
        # and lets the cross-camera remap prune surplus by identity (CLIP + epipolar), rather
        # than pre-trimming to the expected subject count by raw detection confidence — which
        # would drop a partner-occluded real subject in favour of a cleaner-looking outsider.
        # max_seed_tracklets is a generous safety ceiling for pathological bystander-heavy
        # scenes (logged when hit).
        "max_seed_tracklets": 20,
    },
    "sam": {
        "propagate_reverse": True,
    },
    "exports": {
        "export_masked_outputs_mp4": True,
        "masked_outputs_fps": 30.0,
        "remove_masked_outputs_images": True,
        "export_prompt_overlays_mp4": True,
        "prompt_overlays_fps": 4.0,
        "remove_prompt_overlay_images": True,
        "export_prompt_similarity_mp4": True,
        "prompt_similarity_fps": 4.0,
        "remove_prompt_similarity_images": True,
    },
}

def get_device():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _sam_log("INFO", f"Using compute device: {device}")
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        _sam_log(
            "WARN",
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device


def _detect_centroid_outlier_runs(centroids, k=8.0, abs_floor=300.0,
                                  half_window=7, max_gap=4,
                                  bracket_max_gap=15):
    """Detect transient "teleport" excursions in a per-tracklet 2D mask centroid
    trajectory (numpy analog of the ma_3d translation-outlier gate).

    SAM2 occasionally re-locks a tracklet onto a neighbour for one or two frames,
    so its mask centroid jumps far and returns. This flags those frames.

    ``centroids``: dict ``{frame_idx: (x, y)}`` for the frames where the tracklet's
    mask is present and non-empty. A frame is flagged when its centroid deviates
    from the local temporal median by more than ``thr = max(abs_floor, k*MAD(step))``,
    where ``MAD(step)`` is the raw (unscaled) median absolute deviation of the
    frame-to-frame centroid step; ``k`` is calibrated against that raw scale.

    Only transient runs (frame-span <= ``max_gap``) are returned, and only when
    bracketed on both sides by present, non-flagged observations within
    ``bracket_max_gap`` frames of the run: distant brackets cannot certify a
    return. The bridge between the brackets is interpolated in frame space
    (time), where constant-velocity motion is linear, so genuine fast motion
    and dropout gaps are spared.

    Returns a list of ``(a, b)`` INCLUSIVE frame-index ranges whose masks should
    be dropped.
    """
    frames = sorted(centroids)
    K = len(frames)
    if K < 5:
        return []
    fr = np.asarray(frames)
    c = np.asarray([centroids[f] for f in frames], dtype=float)      # [K, 2]

    med = c.copy()
    for i in range(K):
        lo, hi = max(0, i - half_window), min(K, i + half_window + 1)
        med[i] = np.median(c[lo:hi], axis=0)
    dev = np.linalg.norm(c - med, axis=1)                            # [K]

    step = np.linalg.norm(np.diff(c, axis=0), axis=1)                # [K-1]
    mad = float(np.median(np.abs(step - np.median(step)))) if len(step) else 0.0
    thr = max(float(abs_floor), float(k) * mad)

    flagged = dev > thr
    runs = []
    i = 0
    while i < K:
        if flagged[i]:
            a = i
            while i < K and flagged[i]:
                i += 1
            b = i - 1
            transient = (
                (fr[b] - fr[a] + 1) <= max_gap
                and a - 1 >= 0 and b + 1 < K
                and not flagged[a - 1] and not flagged[b + 1]
                # brackets must be temporally CLOSE to the run: a "return"
                # across a long absence is not evidence of a transient.
                and (fr[a] - fr[a - 1]) <= bracket_max_gap
                and (fr[b + 1] - fr[b]) <= bracket_max_gap
            )
            if transient:
                c0, c1 = c[a - 1], c[b + 1]
                f0, f1 = float(fr[a - 1]), float(fr[b + 1])
                excursion = max(
                    float(np.linalg.norm(
                        c[j] - ((1.0 - t) * c0 + t * c1)))
                    for j in range(a, b + 1)
                    for t in [(float(fr[j]) - f0) / (f1 - f0)]
                )
                if excursion > thr:
                    runs.append((int(fr[a]), int(fr[b])))
        else:
            i += 1
    return runs


def _argmax_disjoint_masks(obj_ids, mask_logits):
    """Per-pixel argmax over per-object mask logits: each pixel belongs to at
    most one object (the highest-logit claimant above threshold).

    Used where fresh logits are in hand (recovery re-propagation), so the
    written masks are pixel-disjoint by construction and cannot leave an
    overlap seam against each other. Single-object frames pass through.
    """
    import torch as _torch
    if len(obj_ids) <= 1:
        return {int(o): (mask_logits[i] > 0.0).cpu().numpy()[0]
                if mask_logits[i].ndim == 3 else
                (mask_logits[i] > 0.0).cpu().numpy()
                for i, o in enumerate(obj_ids)}
    planes = _torch.stack(
        [mask_logits[i][0] if mask_logits[i].ndim == 3 else mask_logits[i]
         for i in range(len(obj_ids))])                     # [N, H, W]
    winner = planes.argmax(dim=0)
    positive = planes > 0.0
    return {int(o): (positive[i] & (winner == i)).cpu().numpy()
            for i, o in enumerate(obj_ids)}


def _mask2d(m):
    """Mask as 2D bool: writebacks store (H, W) but the tracker's original
    frames carry a leading singleton axis (1, H, W)."""
    m = np.asarray(m)
    return m.reshape(m.shape[-2], m.shape[-1])


def _collapse_flags(video_segments, sorted_frames, anchors, run_s, run_e,
                    margin=15):
    """Frames inside a re-propagated run still carrying a collapse signature.

    A flagged frame has a mask pair that describes ONE person wearing two
    ids rather than two people: the smaller mask lies almost entirely
    within a small dilation of the larger one (they tile the same body),
    and the pair is either a split body (union near a single person area,
    near-coincident centroids) or a scrap on the larger mask. A genuinely
    occluded neighbor pokes out beside the front person and fails the
    containment test, so ordinary close contact is never flagged. Person
    scale comes from the recovery anchor boxes. Returns
    (flags, n_scanned) with flags as [(frame_index_position, smaller_oid)].
    """
    import cv2
    box_area = float(np.median(
        [max(1.0, (b[2] - b[0]) * (b[3] - b[1])) for _, b in anchors]))
    person_mask_est = 0.6 * box_area
    body_len = float(np.sqrt(box_area))
    kernel = np.ones((31, 31), np.uint8)
    lo = max(0, run_s - margin)
    hi = min(len(sorted_frames) - 1, run_e + margin)
    flags = []
    for idx in range(lo, hi + 1):
        masks = {o: _mask2d(m)
                 for o, m in video_segments.frame(sorted_frames[idx]).items()
                 if m is not None and m.any()}
        stats = {}
        for oid, m in masks.items():
            ys, xs = np.nonzero(m)
            stats[oid] = (len(xs), xs.mean(), ys.mean())
        oids = sorted(stats)
        for i in range(len(oids)):
            for j in range(i + 1, len(oids)):
                a, b = stats[oids[i]], stats[oids[j]]
                small_oid, big_oid = ((oids[i], oids[j]) if a[0] <= b[0]
                                      else (oids[j], oids[i]))
                small, big = stats[small_oid][0], stats[big_oid][0]
                dist = float(np.hypot(a[1] - b[1], a[2] - b[2]))
                split_body = (small + big < 1.2 * person_mask_est
                              and dist < 0.6 * body_len)
                scrap = small < 0.25 * big and dist < body_len
                if not (split_body or scrap):
                    continue
                dil = cv2.dilate(masks[big_oid].astype(np.uint8), kernel) > 0
                cover = (masks[small_oid] & dil).sum() / float(small)
                if cover >= 0.85:
                    flags.append((idx, small_oid))
    return flags, hi - lo + 1


def _zero_flagged_masks(video_segments, sorted_frames, flags):
    """Blank the flagged (smaller) mask on each flagged frame; the larger
    mask keeps covering the person. Returns the number blanked."""
    zeroed = 0
    for idx, oid in flags:
        fidx = sorted_frames[idx]
        masks = video_segments.frame(fidx)
        m = masks.get(oid)
        if m is None or not m.any():
            continue
        masks[oid] = np.zeros_like(m)
        video_segments.set_frame(fidx, masks)
        zeroed += 1
    return zeroed


def _unify_failed_run_labels(video_segments, sorted_frames, run_s, run_e,
                             margin=15, jump_px=150.0, gap=10):
    """One id per continuity chain inside a failed recovery run.

    After scrap zeroing, the surviving person-sized mask can still alternate
    between tracklet ids frame to frame (either tracklet claims the person
    and the argmax keeps whichever is stronger that frame). Surviving masks
    are grouped into temporal continuity chains; each chain takes the id of
    the healthy boundary mask it connects to (entry preferred, exit
    otherwise) and chains connected to neither boundary are blanked, so a
    label can never alternate on one physical person. Returns
    (relabeled, blanked) frame counts.
    """
    lo = max(0, run_s - margin)
    hi = min(len(sorted_frames) - 1, run_e + margin)

    def centroids_at(idx):
        out = {}
        for oid, m in video_segments.frame(sorted_frames[idx]).items():
            if m is not None and m.any():
                ys, xs = np.nonzero(_mask2d(m))
                out[oid] = (float(xs.mean()), float(ys.mean()))
        return out

    def boundary(outward):
        for idx in outward:
            pts = centroids_at(idx)
            if pts:
                return pts
        return {}

    pre = boundary(range(lo - 1, max(-1, lo - 40), -1))
    post = boundary(range(hi + 1, min(len(sorted_frames), hi + 40)))

    chains = []          # {"members": [(idx, oid)], "last": (idx, x, y), "first": (x, y)}
    for idx in range(lo, hi + 1):
        for oid, (cx, cy) in centroids_at(idx).items():
            best = None
            for ch in chains:
                li, lx, ly = ch["last"]
                if idx - li <= 0 or idx - li > gap:
                    continue
                d = float(np.hypot(cx - lx, cy - ly))
                if d < jump_px and (best is None or d < best[0]):
                    best = (d, ch)
            if best is None:
                chains.append({"members": [(idx, oid)], "last": (idx, cx, cy),
                               "first": (cx, cy)})
            else:
                best[1]["members"].append((idx, oid))
                best[1]["last"] = (idx, cx, cy)

    def nearest_id(pts, xy):
        cand = [(float(np.hypot(xy[0] - p[0], xy[1] - p[1])), oid)
                for oid, p in pts.items()]
        cand.sort()
        return cand[0][1] if cand and cand[0][0] < 2 * jump_px else None

    relabeled = blanked = 0
    for ch in chains:
        target = nearest_id(pre, ch["first"])
        if target is None:
            target = nearest_id(post, ch["last"][1:])
        for idx, oid in ch["members"]:
            fidx = sorted_frames[idx]
            masks = video_segments.frame(fidx)
            m = masks.get(oid)
            if m is None or not m.any():
                continue
            if target is None:
                masks[oid] = np.zeros_like(m)
                blanked += 1
            elif oid != target:
                other = masks.get(target)
                if other is not None and other.any():
                    # two survivors converged on one frame: keep the larger
                    if other.sum() >= m.sum():
                        masks[oid] = np.zeros_like(m)
                    else:
                        masks[target] = m
                        masks[oid] = np.zeros_like(m)
                else:
                    masks[target] = m
                    masks[oid] = np.zeros_like(m)
                relabeled += 1
            else:
                continue
            video_segments.set_frame(fidx, masks)
    return relabeled, blanked


def _drop_single_frame_islands(video_segments):
    """Remove one-frame mask appearances bounded by absence on both sides.

    Nothing physical appears for exactly one frame at capture frame rates;
    such islands are label blips left at gate or recovery boundaries and
    read as id flicker downstream. Clip-end frames are left alone. Returns
    the number of masks dropped.
    """
    frames = video_segments.frames()
    presence = {}
    for f in frames:
        presence[f] = {oid for oid, m in video_segments.frame(f).items()
                       if m is not None and m.any()}
    dropped = 0
    for k in range(1, len(frames) - 1):
        f = frames[k]
        for oid in list(presence[f]):
            if oid in presence[frames[k - 1]] or oid in presence[frames[k + 1]]:
                continue
            masks = video_segments.frame(f)
            masks[oid] = np.zeros_like(masks[oid])
            video_segments.set_frame(f, masks)
            dropped += 1
    return dropped


def _atomic_np_save(path, obj):
    """np.save via a temp file + rename, so a kill mid-write can never leave a
    truncated cache that existence-based resume checks would trust."""
    tmp = path + ".tmp.npy"
    np.save(tmp, obj)
    os.replace(tmp, path)


def _resolve_auto_flag(value, n_frames, min_frames):
    """Resolve a bool-or-"auto" config flag against a clip length.

    "auto" enables the flag only when the clip has at least ``min_frames``
    frames (unknown length resolves to off). Strings other than "auto" are
    parsed as booleans; everything else is coerced with bool().
    """
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "auto":
            return bool(n_frames and int(n_frames) >= int(min_frames))
        return v in {"1", "true", "yes", "on"}
    return bool(value)


def _seed_centrality_weight(frame_idx, n_frames, bias=0.5):
    """Down-weight seed/prompt frame candidates near the clip boundaries.

    Propagation error grows with distance from the seed, so an endpoint seed
    doubles the worst-case propagation distance vs a central one. Linear
    weight: 1.0 at the clip center, ``1 - bias`` at either endpoint;
    ``bias=0`` disables. Multiplicative, so a genuinely better-separated
    off-center frame can still win.

    The full ``bias`` applies only from ~3000 frames up, fading linearly to
    zero at <=1000 frames: on short clips propagation distance barely
    matters, and overriding a better-separated frame costs recall.
    """
    if n_frames <= 1 or bias <= 0:
        return 1.0
    strength = float(bias) * min(1.0, max(0.0, (n_frames - 1000.0) / 2000.0))
    if strength <= 0:
        return 1.0
    x = abs(2.0 * float(frame_idx) / float(n_frames - 1) - 1.0)  # 0=center, 1=endpoint
    return 1.0 - strength * x


def _weighted_unit_mean(feats, weights):
    """Reliability-weighted mean direction of embedding vectors.

    Each embedding is normalized to unit length first, so its influence on
    the mean direction is set by its weight alone rather than by its raw
    norm. ``feats``: [N, D] tensor; ``weights``: [N] tensor summing to 1.
    Returns a unit vector [D].
    """
    feats = F.normalize(feats, dim=1)
    avg = (feats * weights.unsqueeze(1)).sum(dim=0)
    return F.normalize(avg.unsqueeze(0), dim=1).squeeze(0)


def _transient_area_spikes(areas, spike_ratio=3.0, neighbor_max_gap=15):
    """Frames where a tracklet's mask area transiently spikes.

    A 1-2 frame takeover onto a larger neighbour shows up as a sudden area jump
    that returns (SAM2 briefly segmenting the neighbour under this obj_id). Unlike
    the centroid test, this fires even when the two people are close together (a
    small centroid jump). ``areas``: ``{frame_idx: area>0}``. Returns the frames
    whose area exceeds ``spike_ratio`` times BOTH nearest present neighbours'
    areas, each within ``neighbor_max_gap`` frames: a sustained growth clears
    the trailing-neighbour test, and distant neighbours are no evidence of a
    transient. These are only *candidates*; the caller confirms each with a
    mask-IoU check.
    """
    frames = sorted(areas)
    if len(frames) < 3:
        return []
    out = []
    for i in range(1, len(frames) - 1):
        f = frames[i]
        if (f - frames[i - 1]) > neighbor_max_gap or (frames[i + 1] - f) > neighbor_max_gap:
            continue
        base = max(areas[frames[i - 1]], areas[frames[i + 1]])
        if base > 0 and areas[f] > spike_ratio * base:
            out.append(f)
    return out


class SegmentMultipleFrames:
    def __init__(
        self,
        sam_model_cfg,
        sam_checkpoint,
        yolo_checkpoint,
        output_path,
        labels_path=None,
        init_with_gt=False,
        assignment_config=None,
        sam_version="sam2",
        start_frame=None,
        end_frame=None,
        # Legacy aliases
        sam2_model_cfg=None,
        sam2_checkpoint=None,
    ):
        # Support legacy parameter names
        if sam2_model_cfg is not None:
            if sam_model_cfg is not None:
                raise ValueError("Cannot specify both sam_model_cfg and sam2_model_cfg")
            sam_model_cfg = sam2_model_cfg
        if sam2_checkpoint is not None:
            if sam_checkpoint is not None:
                raise ValueError("Cannot specify both sam_checkpoint and sam2_checkpoint")
            sam_checkpoint = sam2_checkpoint

        self.output_path = output_path
        self.labels_path = labels_path
        self.init_with_gt = init_with_gt
        self.sam_version = sam_version
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.assignment_config = copy.deepcopy(DEFAULT_ASSIGNMENT_CONFIG)
        self.expected_subjects = None
        self.subject_feature_bank = {}
        self.subject_crop_bank = {}
        # Cameras completed this run, usable as geometric references by the
        # track-then-match seed prescreen (see _prescreen_seed_candidates).
        self._prescreen_refs = []
        if assignment_config:
            self.set_assignment_config(assignment_config)
        self.device = get_device()
        tracking_overrides = self.assignment_config.get("sam3_tracking")
        self.predictor: _VideoPredictor = build_video_predictor(
            sam_version, sam_model_cfg, sam_checkpoint, device=self.device,
            tracking_overrides=tracking_overrides,
        )

        # sam3_prompt / sam3_prompt_light detect people with SAM3's own text
        # detector, so YOLO is never used — skip loading it (saves ~0.2 GB + the
        # checkpoint need). For sam3_prompt this also drops the redundant YOLO that
        # the multi-view bootstrap used to call.
        if sam_version in ("sam3_prompt", "sam3_prompt_light"):
            self.yolo_model = None
            if yolo_checkpoint:
                self._log_info(f"{sam_version}: YOLO not loaded (SAM3 text detection is used).")
        else:
            if not yolo_checkpoint or not os.path.isfile(yolo_checkpoint):
                raise FileNotFoundError(f"YOLO checkpoint file not found: {yolo_checkpoint}")
            self.yolo_model = YOLO(yolo_checkpoint, verbose=False)
            if platform.system() == "Windows":
                self.yolo_model.to('cpu')
            else:
                self.yolo_model.to(self.device)

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.clip_model.to(self.device)

        if sam_version in ("sam3", "sam3_prompt", "sam3_prompt_light"):
            self.img_mean_sam = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
            self.img_std_sam = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
        else:
            self.img_mean_sam = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            self.img_std_sam = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    @property
    def _is_sam3(self):
        return self.sam_version in ("sam3", "sam3_prompt", "sam3_prompt_light")

    @property
    def _frame_offset(self):
        """Offset to convert relative frame indices to absolute (global) indices."""
        return self.start_frame if self.start_frame is not None else 0


    def _log_info(self, message: str):
        _sam_log("INFO", message)

    def _log_warn(self, message: str):
        _sam_log("WARN", message)

    def _log_error(self, message: str):
        _sam_log("ERROR", message)

    def _id_color(self, obj_id):
        cmap = plt.get_cmap("tab10")
        rgba = cmap(int(obj_id) % 10)
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]))

    def _id_label(self, obj_id, score=None):
        if score is None:
            return f"id {int(obj_id):02d}"
        return f"id {int(obj_id):02d} s={float(score):.2f}"

    def _save_init_detection_overview(
        self,
        image,
        detections,
        frame_idx,
        out_init_dir,
        tag="auto",
        camera_name=None,
    ):
        """
        Save one overview image with all init detections, color-coded by object ID.
        """
        if detections is None or len(detections) == 0:
            return
        os.makedirs(out_init_dir, exist_ok=True)
        plt.figure(figsize=(12, 7))
        plt.imshow(image)
        title = f"Init detections ({tag}) frame {int(frame_idx)}"
        if camera_name is not None:
            title = f"{title} - {camera_name}"
        plt.title(title)
        for oid, (_crop, _feat, bbox, score) in enumerate(detections):
            color = self._id_color(oid)
            show_box(
                bbox,
                plt.gca(),
                color=color,
                linewidth=2.5,
                label=self._id_label(oid, score=score),
            )
        plt.axis("off")
        save_name = f"frame_{int(frame_idx):04d}_init_bboxes_{tag}.png"
        plt.savefig(os.path.join(out_init_dir, save_name))
        plt.close()

    def has_valid_detections(self, best_similarity):
        """Check if at least one object has a valid detection (frame_idx >= 0)."""
        for oid, (sim, frame_idx, img_bbx, bbx, target_img_bbx) in best_similarity.items():
            if frame_idx >= 0:  # Valid detection
                return True
        return False

    def _prune_sam_memory(self, inference_state, frame_idx, window):
        """Drop per-object non-conditioning memory outside a window of the frame
        being tracked, bounding stored state to O(window) instead of O(num_frames).

        SAM2 attends only to conditioning/anchor frames (kept here — we never touch
        ``cond_frame_outputs``) plus a bounded recent window (num_maskmem /
        max_obj_ptrs_in_encoder). Older non-conditioning memory is no longer read,
        so dropping it leaves results GT-equivalent (validated: identical GT, masks
        within ~0.995 IoU — only fp-level edge-pixel jitter from freeing memory
        mid-run). Opt-in via sam.prune_memory_state for very long clips.
        """
        per_obj = inference_state.get("output_dict_per_obj") if isinstance(inference_state, dict) else None
        for obj in (per_obj or {}).values():
            nc = obj.get("non_cond_frame_outputs")
            if nc:
                for j in [k for k in nc if abs(k - frame_idx) > window]:
                    nc.pop(j, None)

    def run_propagation(self, inference_state, cam_name="", image_size=None):
        # Run propagation and spill each frame's masks to disk as they are
        # produced. The previous code accumulated every frame's
        # full-resolution masks in an in-RAM dict (~5 GB/camera on long clips,
        # the user-reported host-RAM blow-up); MaskStore keeps the same content
        # bit-packed on disk so peak RAM is O(1 frame). Behaviour is unchanged.
        video_segments = MaskStore()
        sam_cfg = self.assignment_config.get("sam")
        if sam_cfg is None:
            sam_cfg = self.assignment_config.get("sam2", {})
            if sam_cfg:
                self._log_warn("Config key 'sam2' is deprecated; use 'sam' instead.")
        # Opt-in memory-state pruning: after each frame's mask is spilled
        # to disk, drop per-frame memory SAM can no longer attend to, bounding stored
        # state on long clips. GT-equivalent (anchors/cond kept; ~edge-pixel jitter).
        prune = bool(sam_cfg.get("prune_memory_state", False))
        keep_w = max(16, int(sam_cfg.get("memory_keep_window", 32)))
        self._log_info("Starting SAM propagation: forward pass.")
        with torch.inference_mode():
            for frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state, reverse=False):
                video_segments.set_frame(frame_idx, {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                })
                if prune:
                    self._prune_sam_memory(inference_state, frame_idx, keep_w)

            if bool(sam_cfg.get("propagate_reverse", True)):
                # run propagation backwards as the annotation can be in the middle of the video
                self._log_info("Starting SAM propagation: backward pass.")
                for frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state, reverse=True):
                    video_segments.set_frame(frame_idx, {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    })
                    if prune:
                        self._prune_sam_memory(inference_state, frame_idx, keep_w)

        # Unified post-processing: merge duplicate tracklets + discard tiny ones.
        # Infer image_size from propagated masks if not provided.
        if image_size is None:
            image_size = video_segments.image_size()

        # Opt-in collapse RECOVERY before the subtractive cleanup: when two
        # tracklets sat near-identical on one person for a sustained run, the
        # lost person is re-detected inside the run, the intruder tracklet is
        # re-seeded there with a box prompt, and only the run's sub-range is
        # re-propagated. Runs that survive recovery fall through to the
        # re-latch truncation / duplicate suppression below.
        if bool(self.assignment_config.get("masks", {}).get(
                "recover_collapsed_tracklets", False)):
            try:
                video_segments = self._recover_collapsed_runs(
                    inference_state, video_segments,
                    cam_name=cam_name or "SAM", image_size=image_size)
            except Exception as e:  # noqa: BLE001 — recovery must never kill the run
                import traceback
                self._log_warn(
                    f"[{cam_name or 'SAM'}] collapse recovery skipped "
                    f"(non-fatal): {e}\n{traceback.format_exc()}")

        detected_ids = video_segments.all_obj_ids()
        video_segments, detected_ids = self._postprocess_tracklets(
            cam_name or "SAM", video_segments, detected_ids, image_size=image_size,
        )
        dropped = _drop_single_frame_islands(video_segments)
        if dropped:
            self._log_info(f"[{cam_name or 'SAM'}] Dropped {dropped} "
                           f"single-frame mask island(s).")

        return video_segments



    def set_assignment_config(self, assignment_config):
        def _merge_dict(dst, src):
            for key, value in src.items():
                if isinstance(value, dict) and isinstance(dst.get(key), dict):
                    _merge_dict(dst[key], value)
                else:
                    dst[key] = value

        if not isinstance(assignment_config, dict):
            raise ValueError("assignment_config must be a dictionary")
        _merge_dict(self.assignment_config, assignment_config)

    def set_scene_constraints(self, expected_subjects=None):
        self.expected_subjects = expected_subjects

    def _setup_frame_access(self, frames, cam_data=None):
        """
        Configure per-camera frame access.

        Args:
            frames: FrameSource instance (ImageFileSource or VideoSource).
            cam_data: Optional cam_data dict (for backward compat fallback).
        """
        self._current_frames = frames
        self._current_sam_video_source = self._resolve_sam_video_source(frames, cam_data)

    def _read_frame_pil(self, frame_idx=None, file_path=None):
        """
        Read a frame as a PIL Image.

        Prefers the current FrameSource when available.
        Falls back to file_path for backward compatibility.
        """
        frames = getattr(self, '_current_frames', None)
        if frames is not None and frame_idx is not None:
            return frames.read_pil(frame_idx)
        if file_path is not None:
            return Image.open(file_path).convert("RGB")
        raise ValueError("Either frame_idx (with FrameSource) or file_path must be provided.")


    def _load_rgb_image_safe(self, img_path):
        try:
            with Image.open(img_path) as pil_img:
                return np.array(pil_img.convert("RGB"))
        except (OSError, IOError) as exc:
            self._log_warn(f"Failed to load image '{img_path}': {exc}")
            return None

    def _build_sanitized_video_dir(self, src_video_dir, frame_names, output_path):
        """
        Build a local sanitized copy of frames for SAM2 loading.
        If a frame is unreadable, replace it with the nearest valid frame to keep indices stable.
        """
        sanitized_dir = os.path.join(output_path, "_sam2_sanitized_frames")
        os.makedirs(sanitized_dir, exist_ok=True)

        replaced = 0
        prev_good = None
        next_cache = {}

        def _next_valid_from(start_idx):
            if start_idx in next_cache:
                return next_cache[start_idx]
            out = None
            for jj in range(start_idx, len(frame_names)):
                cand = os.path.join(src_video_dir, frame_names[jj])
                arr = self._load_rgb_image_safe(cand)
                if arr is not None:
                    out = arr
                    break
            next_cache[start_idx] = out
            return out

        for i, name in enumerate(frame_names):
            src = os.path.join(src_video_dir, name)
            dst = os.path.join(sanitized_dir, name)

            img_rgb = self._load_rgb_image_safe(src)
            if img_rgb is None:
                img_rgb = prev_good if prev_good is not None else _next_valid_from(i + 1)
                if img_rgb is None:
                    raise RuntimeError(f"Cannot recover unreadable frame '{src}' (no valid neighbors).")
                replaced += 1

            prev_good = img_rgb
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            ext = os.path.splitext(name)[1].lower()
            try:
                pil = Image.fromarray(img_rgb)
                if ext in [".jpg", ".jpeg"]:
                    pil.save(dst, quality=95)
                else:
                    pil.save(dst)
            except Exception:
                # Fallback to cv2 write if PIL save fails for any extension edge-case.
                ok = cv2.imwrite(dst, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError(f"Failed to write sanitized frame '{dst}'.")

        self._log_warn(
            f"Built sanitized frame directory for SAM2 at '{sanitized_dir}'. "
            f"Recovered/replaced unreadable frames: {replaced}."
        )
        return sanitized_dir, replaced

    def _sam_offload_kwargs(self, n_frames=None):
        """Resolve proactive SAM CPU-offload flags from the ``sam`` config.

        ``offload_video_to_cpu`` defaults ON: it only relocates the decoded-frame
        tensor to the host (identical masks, small overhead per SAM docstring) and
        is what keeps VRAM bounded on long / many-camera clips.
        ``offload_state_to_cpu`` defaults to ``"auto"``: enabled once a clip is long
        enough (``offload_state_min_frames``, default 1500) that the per-frame
        memory bank would otherwise dominate VRAM. Both accept explicit
        true/false to force the behaviour.

        Only applies to the init_state backends (sam2 and the sam3 tracker); the
        sam3_prompt session API bounds memory via its own defaults
        (max_cond_frames_in_attn=4) plus the disk-backed MaskStore.
        """
        sam_cfg = self.assignment_config.get("sam")
        if not sam_cfg:
            sam_cfg = self.assignment_config.get("sam2", {}) or {}
        video = bool(sam_cfg.get("offload_video_to_cpu", True))
        state = sam_cfg.get("offload_state_to_cpu", "auto")
        if isinstance(state, str):
            if state.strip().lower() == "auto":
                thr = int(sam_cfg.get("offload_state_min_frames", 1500))
                state = bool(n_frames is not None and n_frames >= thr)
            else:
                state = state.strip().lower() in {"1", "true", "yes", "on"}
        else:
            state = bool(state)
        return {"offload_video_to_cpu": video, "offload_state_to_cpu": state}

    def _init_state_with_fallback(self, video_dir, frame_names, output_path, context_name):
        """
        Robust SAM init_state with multiple fallback strategies.

        Accepts either a directory of frame images or an MP4 video file path.
        Both SAM2 and SAM3 support MP4 natively via decord.

        Fallback chain:
          1. Try init_state with the given path (directory or MP4).
          2. If it fails AND ``video_dir`` is a directory of frame images,
             retry with a sanitized local copy.

        The sanitized-frames recovery re-reads and rewrites individual frame
        files, so it only applies to a directory of images. For an MP4 file
        (video input) there are no per-frame files on disk — joining the mp4
        path with frame names would treat the file as a directory (ENOTDIR) —
        so the original init error is re-raised instead of attempting it.
        """
        from core.predictor_factory import set_lazy_frame_loading
        _sam_cfg = self.assignment_config.get("sam") or self.assignment_config.get("sam2", {}) or {}
        # Clip length for the length-gated ("auto") memory knobs. For an MP4
        # input there are no per-frame files, so probe the container header.
        n_frames = len(frame_names) if frame_names else None
        if n_frames is None and isinstance(video_dir, str) and os.path.isfile(video_dir):
            _cap = cv2.VideoCapture(video_dir)
            n_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            _cap.release()
        # Lazy frame loading (off = byte-identical eager path): bounds host RAM
        # to a small window instead of holding all decoded frames. "auto"
        # enables it from lazy_min_frames up; short clips keep the eager path.
        set_lazy_frame_loading(
            _resolve_auto_flag(_sam_cfg.get("lazy_frame_loading", False),
                               n_frames, _sam_cfg.get("lazy_min_frames", 1500)),
            int(_sam_cfg.get("lazy_frame_window", 8)),
            int(_sam_cfg.get("lazy_video_chunk", 64)))

        # Proactive CPU offload: SAM loads the whole decoded video
        # into VRAM at init_state, so VRAM grows with frame count. We pass the
        # offload flags up front (offload_video_to_cpu defaults ON — it only
        # relocates tensors to the host, masks are identical) so long / many-
        # camera clips don't OOM in the first place. The reactive retry below is
        # the backstop if even that is not enough or offload was disabled.
        offload = self._sam_offload_kwargs(n_frames)
        try:
            try:
                state = self.predictor.init_state(video_path=video_dir, **offload)
            except TypeError:
                # Backend's init_state doesn't accept offload kwargs — plain call.
                state = self.predictor.init_state(video_path=video_dir)
            return state, video_dir
        except torch.cuda.OutOfMemoryError as oom:
            # Proactive offload was not enough (or was disabled): force full CPU
            # offload and retry. Frames and per-frame state live on the host and
            # stream to the GPU as needed — bounded GPU memory, identical masks,
            # somewhat slower. Only triggered on an actual OOM.
            self._log_warn(
                f"[{context_name}] SAM init_state hit CUDA OOM; retrying with full CPU "
                f"offload (offload_video_to_cpu / offload_state_to_cpu). {oom}"
            )
            torch.cuda.empty_cache()
            try:
                try:
                    state = self.predictor.init_state(
                        video_path=video_dir,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                    )
                except TypeError:
                    # Backend's init_state doesn't accept offload kwargs — plain call.
                    state = self.predictor.init_state(video_path=video_dir)
                return state, video_dir
            except Exception as exc:
                init_error = exc
        except Exception as exc:
            init_error = exc

        # Only a directory of frame images can be sanitized. For an MP4 file
        # (or an empty frame list) there is nothing to sanitize, so surface
        # the real init error rather than reading the mp4 path as a directory.
        if not os.path.isdir(video_dir):
            raise init_error
        if frame_names is None or len(frame_names) == 0:
            raise RuntimeError(f"[{context_name}] Cannot build sanitized frames: empty frame list.")

        self._log_warn(
            f"[{context_name}] SAM init_state failed for '{video_dir}'. "
            f"Retrying with sanitized frames. Error: {init_error}"
        )
        sanitized_dir, replaced = self._build_sanitized_video_dir(video_dir, frame_names, output_path)
        try:
            try:
                state = self.predictor.init_state(video_path=sanitized_dir, **offload)
            except TypeError:
                state = self.predictor.init_state(video_path=sanitized_dir)
        except Exception:
            shutil.rmtree(sanitized_dir, ignore_errors=True)
            raise
        self._log_info(
            f"[{context_name}] SAM init_state succeeded with sanitized frames at '{sanitized_dir}' "
            f"(replaced={replaced})."
        )
        return state, sanitized_dir

    def _cleanup_sanitized_video_dir(self, runtime_video_dir, original_video_dir):
        if not runtime_video_dir:
            return
        if runtime_video_dir == original_video_dir:
            return
        # Defensive: only remove our own temporary sanitized directory pattern.
        if os.path.basename(runtime_video_dir) != "_sam2_sanitized_frames":
            return
        try:
            shutil.rmtree(runtime_video_dir, ignore_errors=False)
            self._log_info(f"Removed temporary sanitized frame directory: {runtime_video_dir}")
        except Exception as exc:
            self._log_warn(f"Failed to remove temporary sanitized frame directory '{runtime_video_dir}': {exc}")

    def _apply_frame_range(self, cam_data, frame_id):
        """
        Slice cam_data arrays for the selected frame range
        [self.start_frame : self.end_frame].

        For SAM, which needs a directory of only the selected frames, we copy
        (not symlink) the sliced images to a temp directory. This works on
        both Linux and Windows.

        Returns (cam_data, frame_id, temp_dir_or_None).
        frame_id is adjusted relative to the sliced range.
        """
        if self.start_frame is None and self.end_frame is None:
            return cam_data, frame_id, None

        img_paths = cam_data['img_abs_path']
        n_frames = len(img_paths)
        start = self.start_frame if self.start_frame is not None else 0
        end = self.end_frame if self.end_frame is not None else n_frames

        start = max(0, min(start, n_frames))
        end = max(start, min(end, n_frames))

        if start == 0 and end == n_frames:
            return cam_data, frame_id, None

        self._log_info(
            f"Applying frame range [{start}:{end}] ({end - start} of {n_frames} frames). "
            f"Note: all frame IDs below are relative to this range (add {start} for absolute)."
        )

        # Adjust frame_id relative to sliced range
        if frame_id is None:
            adjusted_frame_id = None
        else:
            if frame_id < start or frame_id >= end:
                clamped = max(start, min(frame_id, end - 1))
                self._log_warn(
                    f"init_frame {frame_id} is outside --start/--end range [{start}:{end}]. "
                    f"Clamping to {clamped}."
                )
                frame_id = clamped
            adjusted_frame_id = frame_id - start

        # Slice cam_data arrays
        sliced = {}
        for key, val in cam_data.items():
            if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == n_frames:
                sliced[key] = val[start:end]
            else:
                sliced[key] = val

        # Set frame filter so SAM only loads the sliced frames from the
        # original directory. No file copying needed — works for all backends
        # (SAM2, SAM3 tracker, and SAM3 prompt) via monkey-patched loaders.
        from core.predictor_factory import set_frame_filter
        sliced_paths = sliced['img_abs_path']
        frame_basenames = [os.path.basename(str(p)) for p in sliced_paths]
        set_frame_filter(frame_basenames)
        self._log_info(f"Frame filter set: {len(frame_basenames)} frames for SAM to load.")
        return sliced, adjusted_frame_id, None

    def _cleanup_frame_range_dir(self, _unused=None):
        """Clear the frame filter set by _apply_frame_range."""
        from core.predictor_factory import clear_frame_filter
        clear_frame_filter()

    def _export_image_dir_to_mp4(self, image_dir, fps=10.0, video_filename=None, remove_images=True):
        if not os.path.isdir(image_dir):
            return None

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        image_paths = [
            os.path.join(image_dir, p)
            for p in os.listdir(image_dir)
            if os.path.splitext(p)[1].lower() in valid_ext
        ]
        image_paths.sort()

        if len(image_paths) == 0:
            # Already exported or nothing to export.
            return None

        # Sanity-check the first image so we fail fast on unreadable inputs.
        first = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
        if first is None:
            self._log_warn(f"Cannot read first image in '{image_dir}'. Skipping mp4 export.")
            return None
        h, w = first.shape[:2]

        if video_filename is None:
            video_filename = f"{os.path.basename(image_dir)}.mp4"
        video_path = os.path.join(image_dir, video_filename)
        fps = max(0.1, float(fps))

        # Write a concat list so ffmpeg reads exactly the files we picked, in order,
        # regardless of naming scheme or mixed extensions. Avoids glob/pattern issues.
        concat_list_path = os.path.join(image_dir, "_ffmpeg_concat.txt")
        try:
            with open(concat_list_path, "w") as f:
                for p in image_paths:
                    # concat demuxer needs single-quoted paths with internal quotes escaped
                    escaped = os.path.abspath(p).replace("'", r"'\''")
                    f.write(f"file '{escaped}'\n")
                    f.write(f"duration {1.0 / fps}\n")
                # The concat demuxer requires the last file to be repeated without a duration
                # so its final frame is actually shown.
                last_escaped = os.path.abspath(image_paths[-1]).replace("'", r"'\''")
                f.write(f"file '{last_escaped}'\n")

            # libx264 + yuv420p + faststart = broadly browser-compatible H.264 in MP4.
            # -vf pad ensures even dimensions (libx264 with yuv420p requires even W/H).
            # Per-frame `duration 1/fps` lines in the concat list set pacing,
            # so we don't pass -vsync/-r explicitly (modern ffmpeg rejects their
            # combination). Default -fps_mode=auto produces a CFR output because
            # all source durations are identical.
            cmd = [
                FFMPEG_EXE, "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", concat_list_path,
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",
                video_path,
            ]

            try:
                subprocess.run(cmd, check=True)
            except FileNotFoundError:
                self._log_warn(f"ffmpeg binary not found at '{FFMPEG_EXE}'. Cannot export mp4.")
                return None
            except subprocess.CalledProcessError as e:
                self._log_warn(f"ffmpeg failed for '{video_path}' (exit {e.returncode}).")
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except Exception:
                        pass
                return None
        finally:
            if os.path.exists(concat_list_path):
                try:
                    os.remove(concat_list_path)
                except Exception:
                    pass

        # Verify ffmpeg actually produced something non-trivial.
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            self._log_warn(f"No readable frames found in '{image_dir}'. Skipping mp4 export.")
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass
            return None

        written = len(image_paths)

        if bool(remove_images):
            removed = 0
            for img_path in image_paths:
                try:
                    os.remove(img_path)
                    removed += 1
                except Exception:
                    pass
            self._log_info(
                f"Exported '{video_path}' ({written} frames @ {fps:.2f} FPS) and removed {removed} source images."
            )
        else:
            self._log_info(f"Exported '{video_path}' ({written} frames @ {fps:.2f} FPS).")
        return video_path

    def _get_matching_cfg(self):
        return self.assignment_config.get("matching", {})

    def _get_feature_bank_max_size(self):
        return int(self._get_matching_cfg().get("feature_bank_max_size", 64))

    def _encode_clip_batch(self, crops):
        if len(crops) == 0:
            return []
        batch = torch.stack([self.clip_preprocess(Image.fromarray(crop)) for crop in crops], dim=0).to(self.device)
        with torch.inference_mode():
            feats = self.clip_model.encode_image(batch)
        return [f.detach().cpu().float() for f in feats]

    def _append_to_subject_bank(self, oid, feature, crop=None):
        if feature is None:
            return False
        feat = feature.detach().cpu()
        feat = feat.float()
        if feat.ndim != 1:
            feat = feat.reshape(-1)

        if oid not in self.subject_feature_bank:
            self.subject_feature_bank[oid] = []
        if oid not in self.subject_crop_bank:
            self.subject_crop_bank[oid] = []

        bank = self.subject_feature_bank[oid]
        # Skip almost-duplicate embeddings to keep the bank compact.
        if len(bank) > 0:
            bank_tensor = torch.stack(bank, dim=0).to(feat.device)
            sims = F.cosine_similarity(
                F.normalize(feat.unsqueeze(0), dim=1),
                F.normalize(bank_tensor, dim=1),
            )
            if float(sims.max()) > 0.998:
                return False

        bank.append(feat)
        self.subject_crop_bank[oid].append(crop if crop is not None else None)

        max_size = self._get_feature_bank_max_size()
        while len(bank) > max_size:
            bank.pop(0)
            self.subject_crop_bank[oid].pop(0)
        return True

    def _get_subject_gallery(self, mask_data, oid):
        if oid in self.subject_feature_bank and len(self.subject_feature_bank[oid]) > 0:
            gallery = torch.stack(self.subject_feature_bank[oid], dim=0)
            crops = self.subject_crop_bank.get(oid, [])
            return gallery, crops

        # Not in the feature bank: fall back to per-frame mask_data. Callers may
        # pass {} (bank-only lookup), so a missing oid means "no gallery", not a
        # crash — the caller treats None as "skip this id".
        rec = mask_data.get(oid)
        gal = rec.get("features") if rec is not None else None
        if gal is None:
            return None, []
        if isinstance(gal, list):
            gallery = torch.stack(gal) if len(gal) > 0 else None
        else:
            gallery = gal if len(gal) > 0 else None
        crops = rec.get("img_bbx", [])
        return gallery, crops

    def initialize_subject_feature_bank(self, mask_data, expected_subjects=None):
        obj_ids = self._select_obj_ids(mask_data, expected_subjects=expected_subjects)
        self.subject_feature_bank = {}
        self.subject_crop_bank = {}

        for oid in obj_ids:
            gallery, crops = self._get_subject_gallery(mask_data, oid)
            if gallery is None:
                continue
            n = int(gallery.shape[0])
            for i in range(n):
                crop = crops[i] if i < len(crops) else None
                self._append_to_subject_bank(oid, gallery[i], crop=crop)
        self._log_info(
            f"Initialized subject feature bank sizes: "
            f"{ {oid: len(self.subject_feature_bank.get(oid, [])) for oid in obj_ids} }"
        )

    def save_subject_feature_bank(self, cache_path):
        payload = {}
        for oid, feats in self.subject_feature_bank.items():
            if len(feats) == 0:
                continue
            payload[int(oid)] = np.stack([f.detach().cpu().numpy() for f in feats], axis=0)
        # Atomic: parallel per-camera jobs may save concurrently; a torn read
        # by a starting sibling must be impossible.
        tmp = str(cache_path) + ".tmp.npy"
        np.save(tmp, payload, allow_pickle=True)
        os.replace(tmp, cache_path)
        self._log_info(f"Saved subject feature bank cache: {cache_path} (IDs={sorted(payload.keys())})")

    def load_subject_feature_bank(self, cache_path):
        if not os.path.exists(cache_path):
            return False
        try:
            payload = np.load(cache_path, allow_pickle=True)[()]
        except Exception as exc:
            self._log_warn(f"Failed to load feature bank cache '{cache_path}': {exc}")
            return False
        if not isinstance(payload, dict):
            self._log_warn(f"Invalid feature bank cache format at '{cache_path}' (expected dict payload).")
            return False
        self.subject_feature_bank = {}
        self.subject_crop_bank = {}
        loaded_ids = []
        for oid, arr in payload.items():
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            int_oid = int(oid)
            self.subject_feature_bank[int_oid] = [torch.from_numpy(v).float() for v in arr]
            self.subject_crop_bank[int_oid] = [None] * arr.shape[0]
            loaded_ids.append(int_oid)
        if len(loaded_ids) == 0:
            self._log_warn(f"Feature bank cache '{cache_path}' contains no usable entries.")
            return False
        self._log_info(f"Loaded subject feature bank cache: {cache_path} (IDs={sorted(loaded_ids)})")
        return True

    def bootstrap_feature_bank_from_multiview(self, source_cam_data, source_mask_data, target_cam_data_list, frame_id, expected_subjects=None):
        cfg = self._get_matching_cfg()
        yolo_person_conf = float(cfg.get("yolo_person_conf", 0.5))
        clip_weight = float(cfg.get("clip_weight", 0.4))
        epipolar_weight = float(cfg.get("epipolar_weight", 0.6))
        # Resolve image size from the first target camera
        target_image_size = None
        if len(target_cam_data_list) > 0:
            try:
                target_frames = frame_source_from_cam_data(target_cam_data_list[0])
                target_image_size = target_frames.image_size
            except Exception:
                pass
        epipolar_sigma_px, max_epipolar_dist_px, epipolar_source = self._resolve_epipolar_px_params(
            cfg, image_size=target_image_size
        )
        self._log_info(
            f"Bootstrap epipolar parameters: sigma_px={epipolar_sigma_px:.1f}, "
            f"max_dist_px={max_epipolar_dist_px:.1f} (source={epipolar_source})."
        )
        bootstrap_min_score = float(cfg.get("bootstrap_min_score", 0.25))
        offsets = cfg.get("bootstrap_frame_offsets", [0])
        if isinstance(offsets, int):
            offsets = [offsets]
        offsets = [int(o) for o in offsets]

        obj_ids = self._select_obj_ids(source_mask_data, expected_subjects=expected_subjects)
        if len(obj_ids) == 0 or len(target_cam_data_list) == 0:
            return

        if len(self.subject_feature_bank) == 0:
            self.initialize_subject_feature_bank(source_mask_data, expected_subjects=expected_subjects)

        total_added = 0
        target_views = 0
        for target_cam_data in target_cam_data_list:
            target_views += 1
            f_matrix = self._compute_fundamental_matrix(source_cam_data, target_cam_data)
            target_frames = frame_source_from_cam_data(target_cam_data)
            if len(target_frames) == 0:
                continue

            base_frame = frame_id if frame_id is not None else 0
            for offset in offsets:
                target_frame = int(np.clip(base_frame + offset, 0, len(target_frames) - 1))
                frame_img = target_frames.read_pil(target_frame)

                _, detections = self._collect_yolo_detections(frame_img, yolo_person_conf)
                if len(detections) == 0:
                    continue

                crops = [d[0] for d in detections]
                det_feats = torch.stack([d[1] for d in detections], dim=0)
                centers = [
                    np.array([(d[2][0] + d[2][2]) * 0.5, (d[2][1] + d[2][3]) * 0.5, 1.0], dtype=np.float64)
                    for d in detections
                ]
                m = det_feats.shape[0]
                k = len(obj_ids)
                s_clip = np.zeros((k, m), dtype=np.float32)
                s_epi = np.zeros((k, m), dtype=np.float32)
                has_epi = np.zeros((k, m), dtype=bool)

                for row, oid in enumerate(obj_ids):
                    gallery, _ = self._get_subject_gallery(source_mask_data, oid)
                    if gallery is not None and len(gallery) > 0:
                        _, sims = cosine_knn(det_feats, gallery, topk=1)
                        s_clip[row] = sims.squeeze(1).cpu().numpy()

                    ref_point = self._reference_point_from_mask_data(source_mask_data[oid], frame_id if frame_id is not None else 0)
                    if ref_point is None:
                        continue
                    for col, x2 in enumerate(centers):
                        epi_score, _ = self._epipolar_score(
                            f_matrix=f_matrix,
                            x1=ref_point,
                            x2=x2,
                            sigma_px=epipolar_sigma_px,
                            max_dist_px=max_epipolar_dist_px,
                        )
                        if epi_score is None:
                            continue
                        s_epi[row, col] = epi_score
                        has_epi[row, col] = True

                combined = clip_weight * s_clip + epipolar_weight * s_epi
                combined[~has_epi] = s_clip[~has_epi]

                from scipy.optimize import linear_sum_assignment
                cost = 1.0 - combined
                row_ind, col_ind = linear_sum_assignment(cost)
                for row, col in zip(row_ind, col_ind):
                    score = float(combined[row, col])
                    if score < bootstrap_min_score:
                        continue
                    oid = obj_ids[row]
                    added = self._append_to_subject_bank(oid, det_feats[col], crop=crops[col])
                    if added:
                        total_added += 1

        self._log_info(
            f"Completed multiview feature-bank bootstrap: added_features={total_added}, views_scanned={target_views}."
        )

    def _select_obj_ids(self, mask_data, expected_subjects=None):
        obj_ids = sorted(mask_data.keys())
        if expected_subjects is None or expected_subjects <= 0:
            expected_subjects = self.expected_subjects
        if expected_subjects is None or expected_subjects <= 0 or expected_subjects >= len(obj_ids):
            return obj_ids

        ranked = []
        for oid in obj_ids:
            masks = mask_data[oid].get("mask", [])
            ranked.append((oid, len(masks)))
        ranked.sort(key=lambda x: (-x[1], x[0]))
        selected = sorted([oid for oid, _ in ranked[:expected_subjects]])
        dropped = [oid for oid in obj_ids if oid not in selected]
        if dropped:
            self._log_warn(
                f"Trimming initialized IDs to expected_subjects={expected_subjects}. Dropped IDs: {dropped}"
            )
        return selected

    def _as_intrinsics(self, cam_int):
        return geometry.as_intrinsics(cam_int)

    def _as_extrinsics(self, cam_ext):
        return geometry.as_extrinsics(cam_ext)

    def _compute_fundamental_matrix(self, source_cam_data, target_cam_data):
        return geometry.compute_fundamental_matrix(source_cam_data, target_cam_data)

    def _has_epipolar_calibration(self, source_cam_data, target_cam_data):
        """Cheap precondition: do both cam_data carry the keys needed for epipolar
        geometry (cam_int/cam_ext)? Used to choose the cross-camera strategy without
        building the fundamental matrix. _compute_fundamental_matrix still does the
        full computation where the matrix is actually used (and may still return None
        for degenerate calibration, in which case the remap falls back to CLIP-only)."""
        return bool(
            source_cam_data is not None and target_cam_data is not None
            and "cam_int" in source_cam_data and "cam_int" in target_cam_data
            and "cam_ext" in source_cam_data and "cam_ext" in target_cam_data
        )

    def _mask_centroid_xy1(self, mask):
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None
        return np.array([float(xs.mean()), float(ys.mean()), 1.0], dtype=np.float64)

    def _epipolar_reference_set(self, cam_name, source_cam_data, target_cam_data,
                                mask_data):
        """Geometric references for the cross-camera identity remap: the init
        camera plus every completed camera (the same reference set the seed
        prescreen maintains). Each entry is ``(name, F, ref_point_fn)`` where
        ``ref_point_fn(oid, frame_idx)`` returns subject ``oid``'s position in
        that reference at that frame, or None. Scoring pools per-frame samples
        across all references (median), so one occluded or drifted reference
        cannot dominate the identity decision. References without usable
        calibration are skipped; with none, the remap is CLIP-only.
        """
        epi_refs = []
        init_f = self._compute_fundamental_matrix(source_cam_data, target_cam_data)
        if init_f is not None:
            epi_refs.append(("init", init_f,
                             lambda oid, fidx: self._reference_point_from_mask_data(
                                 mask_data.get(oid, {}), fidx)))
        for ref in self._prescreen_refs:
            if str(ref.get("cam_name")) == str(cam_name):
                continue
            ref_f = self._compute_fundamental_matrix(ref["cam_data"], target_cam_data)
            if ref_f is None:
                continue
            epi_refs.append((str(ref.get("cam_name", "?")), ref_f,
                             (lambda _r: lambda oid, fidx:
                              self._prescreen_ref_point(_r, oid, fidx))(ref)))
        return epi_refs

    def _reference_point_from_mask_data(self, obj_data, frame_idx):
        frames = np.asarray(obj_data.get("frame", []), dtype=np.int32)
        if frames.size == 0:
            return None
        nearest = int(np.argmin(np.abs(frames - int(frame_idx))))
        # Prefer the precomputed centroid (slim cache); fall back to the full
        # mask for legacy / full-dump caches. Identical value either way.
        cents = obj_data.get("centroid_xy")
        if cents is not None and nearest < len(cents) and cents[nearest] is not None:
            x, y = cents[nearest]
            return np.array([float(x), float(y), 1.0], dtype=np.float64)
        masks = obj_data.get("mask", [])
        if len(masks) == 0:
            return None
        mask = np.asarray(masks[nearest])
        return self._mask_centroid_xy1(mask)

    def _epipolar_score(self, f_matrix, x1, x2, sigma_px, max_dist_px):
        return geometry.epipolar_score(f_matrix, x1, x2, sigma_px, max_dist_px)

    def _is_auto_value(self, value):
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"auto", "adaptive", "default", ""}
        try:
            return float(value) <= 0.0
        except Exception:
            return False

    def _read_image_size(self, image_path):
        try:
            with Image.open(image_path) as pil_img:
                w, h = pil_img.size
            return int(w), int(h)
        except Exception:
            return None

    def _resolve_epipolar_px_params(self, cfg, sample_image_path=None, image_size=None):
        sigma_raw = cfg.get("epipolar_sigma_px", "auto")
        max_dist_raw = cfg.get("max_epipolar_dist_px", "auto")
        sigma_ratio = float(cfg.get("epipolar_sigma_diag_ratio", 0.075))
        max_dist_ratio = float(cfg.get("max_epipolar_dist_diag_ratio", 0.18))

        if image_size is None and sample_image_path is not None:
            image_size = self._read_image_size(sample_image_path)

        if image_size is None:
            # Safe fallback for legacy behavior around ~1080p footage.
            diag = float(np.sqrt(1920.0**2 + 1080.0**2))
            source = "fallback-1920x1080"
        else:
            w, h = image_size
            diag = float(np.sqrt(float(w) ** 2 + float(h) ** 2))
            source = f"{w}x{h}"

        if self._is_auto_value(sigma_raw):
            sigma_px = max(24.0, float(sigma_ratio) * diag)
        else:
            sigma_px = float(sigma_raw)

        if self._is_auto_value(max_dist_raw):
            max_dist_px = max(2.0 * sigma_px, float(max_dist_ratio) * diag)
        else:
            max_dist_px = float(max_dist_raw)

        return float(sigma_px), float(max_dist_px), source



    def _bbox_iou_xyxy(self, box_a, box_b):
        if box_a is None or box_b is None:
            return 0.0
        a = np.asarray(box_a, dtype=np.float32).reshape(-1)
        b = np.asarray(box_b, dtype=np.float32).reshape(-1)
        if a.shape[0] < 4 or b.shape[0] < 4:
            return 0.0
        ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)


    def _mean_pairwise_iou(self, boxes):
        """Compute the mean pairwise IoU among a list of [x1,y1,x2,y2] boxes."""
        if len(boxes) <= 1:
            return 0.0
        total, pairs = 0.0, 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                total += self._bbox_iou_xyxy(boxes[i], boxes[j])
                pairs += 1
        return total / pairs

    def _overlap_penalty(self, target_box, all_boxes, penalty_iou=0.3):
        """Compute a score penalty [0, 1] for a detection based on how much it
        overlaps with other detections on the same frame.

        Returns 1.0 (no penalty) when the target box doesn't overlap with
        anything, and decreases toward 0.0 as overlap increases.
        """
        if len(all_boxes) <= 1:
            return 1.0
        max_iou = 0.0
        for box in all_boxes:
            if box is target_box or np.array_equal(box, target_box):
                continue
            iou = self._bbox_iou_xyxy(target_box, box)
            if iou > max_iou:
                max_iou = iou
        if max_iou <= penalty_iou:
            return 1.0
        # Linear penalty: IoU of penalty_iou -> 1.0, IoU of 1.0 -> 0.0
        return max(0.0, 1.0 - (max_iou - penalty_iou) / (1.0 - penalty_iou))

    def _deduplicate_detections(self, detections, iou_threshold=0.75):
        # Greedy NMS-style dedup to avoid duplicated person boxes becoming duplicated IDs.
        if len(detections) <= 1:
            return detections
        order = sorted(range(len(detections)), key=lambda i: float(detections[i][3]), reverse=True)
        kept = []
        while order:
            i = order.pop(0)
            kept.append(i)
            survivors = []
            box_i = detections[i][2]
            for j in order:
                box_j = detections[j][2]
                iou = self._bbox_iou_xyxy(box_i, box_j)
                if iou < float(iou_threshold):
                    survivors.append(j)
            order = survivors
        return [detections[i] for i in kept]

    def _force_single_frame_anchors(self, best_similarity, anchor_matches, frames, cam_name):
        """For SAM3: re-detect all people on a single frame.

        SAM3's joint object consolidation requires all object prompts on the
        same conditioning frame. If anchors are spread across different frames,
        un-prompted objects get poisoned tracking memory.

        Strategy: scan sampled frames, pick the one where YOLO detects enough
        people to cover all IDs, and reassign all anchors there.
        """
        obj_ids = sorted(oid for oid, m in anchor_matches.items() if m)
        n_total = len(obj_ids)
        if n_total == 0:
            return best_similarity, anchor_matches

        # Check if all anchors are already on the same frame
        unique_frames = set()
        for matches in anchor_matches.values():
            for m in matches:
                unique_frames.add(int(m["frame_idx"]))
        if len(unique_frames) <= 1:
            return best_similarity, anchor_matches

        self._log_info(
            f"[{cam_name}] SAM3: anchors span {len(unique_frames)} frames "
            f"({sorted(unique_frames)}). Searching for a single frame with all {n_total} people."
        )

        from scipy.optimize import linear_sum_assignment

        cfg = self._get_matching_cfg()
        yolo_conf = float(cfg.get("yolo_person_conf", 0.5))
        n_frames = len(frames)
        sam3_redetect_samples = int(cfg.get("sam3_redetect_samples", 30))
        sample_count = min(sam3_redetect_samples, n_frames)
        candidate_frames = sorted(set(
            list(unique_frames) +
            [int(i) for i in np.linspace(0, n_frames - 1, num=sample_count, dtype=int)]
        ))

        # Score each candidate frame: how many people can be matched?
        best_result = None
        best_matched = -1
        best_total_score = -1.0

        for fidx in candidate_frames:
            frame_img = frames.read_pil(fidx)
            _, detections = self._collect_yolo_detections(
                frame_img, yolo_conf, include_clip_features=True
            )
            # A candidate frame with no detections can't anchor anything; skip it
            # (avoids torch.stack on an empty list — hit more often with SAM3
            # text detection than YOLO, but a latent crash for both).
            if len(detections) == 0:
                continue
            if len(detections) < n_total:
                # Not enough detections to cover all people — skip unless
                # it's better than what we have
                if len(detections) <= best_matched:
                    continue

            det_crops = [d[0] for d in detections]
            det_boxes = [d[2] for d in detections]
            det_feats = torch.stack(
                [d[1].detach().cpu().float().reshape(-1) for d in detections], dim=0
            )

            k = len(obj_ids)
            m = len(detections)
            s_clip = np.zeros((k, m), dtype=np.float32)
            for row, oid in enumerate(obj_ids):
                gallery, _ = self._get_subject_gallery({}, oid)
                if gallery is None:
                    continue
                _, sims = cosine_knn(det_feats, gallery, topk=1)
                s_clip[row] = sims.squeeze(1).cpu().numpy()

            cost = 1.0 - s_clip
            row_ind, col_ind = linear_sum_assignment(cost)

            # Count how many assignments have reasonable scores
            matched = 0
            total_score = 0.0
            for row, col in zip(row_ind, col_ind):
                score = float(s_clip[row, col])
                if score > 0.1:
                    matched += 1
                    total_score += score

            # Prefer: most people matched, then highest total score
            if (matched > best_matched) or (matched == best_matched and total_score > best_total_score):
                best_matched = matched
                best_total_score = total_score
                best_result = (fidx, detections, det_crops, det_boxes, det_feats,
                               s_clip, row_ind, col_ind)

            # Early exit if we found a frame with all people
            if matched >= n_total:
                break

        if best_result is None:
            self._log_warn(f"[{cam_name}] SAM3: no suitable single frame found. Keeping original anchors.")
            return best_similarity, anchor_matches

        fidx, detections, det_crops, det_boxes, det_feats, s_clip, row_ind, col_ind = best_result

        new_anchor_matches = {}
        new_best_similarity = dict(best_similarity)
        for row, col in zip(row_ind, col_ind):
            oid = obj_ids[row]
            score = float(s_clip[row, col])
            if score <= 0.05:
                continue
            new_anchor_matches[oid] = [{
                "score": score,
                "frame_idx": fidx,
                "img_bbx": det_crops[col],
                "bbox": np.asarray(det_boxes[col], dtype=np.float32),
                "target_img_bbx": det_crops[col],
            }]
            new_best_similarity[oid] = (
                score, fidx, det_crops[col],
                np.asarray(det_boxes[col], dtype=np.float32),
                det_crops[col],
            )

        # Any person not matched gets dropped (no anchor = no mask, but better
        # than poisoning other people's masks with a different-frame anchor)
        for oid in obj_ids:
            if oid not in new_anchor_matches:
                new_anchor_matches[oid] = []
                self._log_warn(
                    f"[{cam_name}] SAM3: person {oid} not detected on frame {fidx}. "
                    "This person will have no mask in this camera."
                )

        matched_count = len([oid for oid in obj_ids if new_anchor_matches.get(oid)])
        self._log_info(
            f"[{cam_name}] SAM3: all anchors on frame {fidx} "
            f"({matched_count}/{n_total} people matched, score={best_total_score:.2f})."
        )
        return new_best_similarity, new_anchor_matches

    def _anchors_conflict(self, a, b, conflict_iou):
        if a is None or b is None:
            return False
        if int(a.get("frame_idx", -1)) != int(b.get("frame_idx", -1)):
            return False
        box_a = a.get("bbox")
        box_b = b.get("bbox")
        if box_a is None or box_b is None:
            return False
        return self._bbox_iou_xyxy(box_a, box_b) >= float(conflict_iou)

    def _resolve_anchor_conflicts(
        self,
        anchor_matches,
        candidate_matches,
        max_anchors_per_id,
        min_anchor_frame_gap,
        conflict_iou=0.65,
    ):
        # Enforce cross-ID uniqueness of anchors in the same frame when bboxes overlap heavily.
        resolved = {oid: list(anchor_matches.get(oid, [])) for oid in anchor_matches.keys()}
        removed_conflicts = 0

        def _remove_conflicts_once():
            nonlocal removed_conflicts
            oids = sorted(resolved.keys())
            for i in range(len(oids)):
                oid_a = oids[i]
                for j in range(i + 1, len(oids)):
                    oid_b = oids[j]
                    matches_a = resolved.get(oid_a, [])
                    matches_b = resolved.get(oid_b, [])
                    for idx_a, ma in enumerate(matches_a):
                        for idx_b, mb in enumerate(matches_b):
                            if not self._anchors_conflict(ma, mb, conflict_iou):
                                continue
                            score_a = float(ma.get("score", -1.0))
                            score_b = float(mb.get("score", -1.0))
                            # Keep higher score; tie-break by lower object id.
                            drop_a = (score_a < score_b) or (score_a == score_b and oid_a > oid_b)
                            if drop_a:
                                resolved[oid_a].pop(idx_a)
                            else:
                                resolved[oid_b].pop(idx_b)
                            removed_conflicts += 1
                            return True
            return False

        guard = 0
        while _remove_conflicts_once():
            guard += 1
            if guard > 10000:
                break

        def _conflicts_with_other_ids(oid, cand):
            for other_oid, other_matches in resolved.items():
                if int(other_oid) == int(oid):
                    continue
                for other in other_matches:
                    if self._anchors_conflict(cand, other, conflict_iou):
                        return True
            return False

        # Refill dropped anchors from per-ID candidate pools while respecting gap and conflict constraints.
        for oid in sorted(resolved.keys()):
            selected = list(resolved.get(oid, []))
            seen_key = {(int(m.get("frame_idx", -1)), float(m.get("score", -1.0))) for m in selected}
            ranked_candidates = sorted(
                candidate_matches.get(oid, []),
                key=lambda x: float(x.get("score", -1.0)),
                reverse=True,
            )
            for cand in ranked_candidates:
                if len(selected) >= int(max_anchors_per_id):
                    break
                cand_frame = int(cand.get("frame_idx", -1))
                cand_key = (cand_frame, float(cand.get("score", -1.0)))
                if cand_key in seen_key:
                    continue
                if any(abs(cand_frame - int(s.get("frame_idx", -1))) < int(min_anchor_frame_gap) for s in selected):
                    continue
                if _conflicts_with_other_ids(oid, cand):
                    continue
                selected.append(cand)
                seen_key.add(cand_key)
            selected = sorted(selected, key=lambda x: int(x.get("frame_idx", -1)))
            resolved[oid] = selected

        return resolved, removed_conflicts

    def get_frames_from_far_bbx(self, obj_ids, save_masks, frame_idx):
        """Collect bbox metadata for visualization samples.

        No pruning is done here — tracklet-level pruning (merge duplicates,
        discard tiny) is handled by _postprocess_tracklets() after propagation,
        before masks are exported.
        """
        for obj_id in obj_ids:
            if not save_masks[obj_id]["mask"]:
                continue
            mask = save_masks[obj_id]["mask"][-1]
            bbox = np.array(cv2.boundingRect(mask))
            bbox[2:] += bbox[:2]
            save_masks[obj_id]["iou"].append(0.0)
            save_masks[obj_id]["bbox"].append(bbox.reshape(1, 2, 2))
        return save_masks


    def save_images_from_video(self, inference_state, video_segments, obj_ids, output_path, vis_frame_stride = 10):
        save_masks = {obj_id: {"img": [], "mask": [], "frame": [], "iou": [], "bbox": []} for obj_id in obj_ids}
        # SAM keeps the decoded frames here as a (N, 3, H, W) tensor (already on
        # the host when CPU offload is active). Index ONE frame at a time below
        # instead of unnormalizing+copying the whole video into an (N, H, W, 3)
        # uint8 array — that materialized ~3 redundant full-video copies per
        # camera on top of the mask RAM.
        images = inference_state['images']
        w_original = inference_state['video_width']
        h_original = inference_state['video_height']
        # ``images`` may be a tensor or a lazy AsyncVideoFrameLoader (CPU offload).
        n_frames = images.shape[0] if hasattr(images, "shape") else len(images)

        # Per-frame unnormalize: 1x3x1x1 stats -> 3x1x1 for a single (3, H, W) frame.
        std_chw = self.img_std_sam[0]
        mean_chw = self.img_mean_sam[0]

        def _frame_uint8(frame_idx):
            frame = images[frame_idx]
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu()
            else:
                frame = torch.as_tensor(np.asarray(frame))
            frame = torch.clamp(frame.float() * std_chw + mean_chw, 0, 1)  # unnormalize
            return (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # (H, W, 3)
        export_cfg = self.assignment_config.get("exports", {})
        skip_overlays = bool(export_cfg.get("skip_masked_outputs", False))

        mask_folder = os.path.join(output_path, "masks")
        os.makedirs(mask_folder, exist_ok=True)
        masked_outputs = None
        if not skip_overlays:
            masked_outputs = os.path.join(output_path, "masked_outputs")
            os.makedirs(masked_outputs, exist_ok=True)
        self._log_info(
            f"Saving propagated masks{'' if skip_overlays else ' and overlays'} to '{output_path}' "
            f"(frames={n_frames}, obj_ids={list(obj_ids)}, vis_stride={vis_frame_stride})."
        )
        from concurrent.futures import ThreadPoolExecutor
        import threading
        save_masks_lock = threading.Lock()

        def _overlay_half_color(obj_id):
            # Reproduces show_mask_cv2's per-pixel addend exactly: the colored
            # mask it built is (bool_mask * cmap_color * 255).astype(uint8)
            # * 0.5 — i.e. a CONSTANT half-valued vector wherever the mask is
            # set. uint8 integers + exact halves are exactly representable in
            # float32, so accumulating in float32 over the mask ROI is
            # bit-identical to the old full-frame float64 path at a fraction
            # of the cost (no per-object full-frame temporaries).
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("tab10")
            c = (np.array(cmap(0 if obj_id is None else obj_id)[:3]) * 255).astype(np.uint8)
            return c.astype(np.float32) * 0.5

        def _process_frame(frame_idx):
            if frame_idx not in video_segments:
                return
            image = _frame_uint8(frame_idx)
            image = cv2.resize(image, (w_original, h_original))
            masked_img = image.astype(np.float32) if not skip_overlays else None
            frame_samples = []

            for out_obj_id, out_mask in video_segments[frame_idx].items():
                mask = out_mask[0] if out_mask.ndim == 3 else out_mask
                mask_bw = (mask * 255).astype(np.uint8)
                if frame_idx % vis_frame_stride == 0:
                    frame_samples.append((out_obj_id, image, mask_bw, frame_idx))
                if not skip_overlays:
                    if mask.dtype == np.bool_:
                        x, y, bw, bh = cv2.boundingRect(mask_bw)
                        if bw and bh:
                            roi = masked_img[y:y + bh, x:x + bw]
                            roi[mask[y:y + bh, x:x + bw]] += _overlay_half_color(out_obj_id)
                    else:  # non-boolean masks: keep the original exact path
                        colored_mask = show_mask_cv2(out_mask, obj_id=out_obj_id)[:, :, :3]
                        masked_img = masked_img + colored_mask * 0.5
                _png = os.path.join(mask_folder, f"mask_{frame_idx:04d}_{(int(out_obj_id)+1):02d}.png")
                if not cv2.imwrite(_png, mask_bw):
                    # The mask PNGs are the step's primary output; a silent
                    # drop (disk full, NFS error) must fail the run, not
                    # produce a partially-written masks/ dir marked complete.
                    raise IOError(f"failed to write mask PNG: {_png}")

            if not skip_overlays:
                cv2.imwrite(os.path.join(masked_outputs, f"frame_{frame_idx:04d}.jpg"),
                            cv2.resize(np.clip(masked_img, 0, 255).astype(np.uint8)[:, :, ::-1], (max(1, w_original//4), max(1, h_original//4))))

            if frame_samples:
                with save_masks_lock:
                    for out_obj_id, img, mask_bw, fidx in frame_samples:
                        save_masks[out_obj_id]["img"].append(img)
                        save_masks[out_obj_id]["mask"].append(mask_bw)
                        save_masks[out_obj_id]["frame"].append(fidx)

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(tqdm.tqdm(
                    executor.map(_process_frame, range(n_frames)),
                    total=n_frames,
                ))
        finally:
            # Drop the spilled store and release cached CUDA blocks before the next
            # camera — in a finally so the temp dir is removed even if frame
            # processing raises (close() is idempotent).
            if hasattr(video_segments, "close"):
                video_segments.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Run bbox pruning on sampled frames
        for frame_idx in range(0, n_frames, vis_frame_stride):
            save_masks = self.get_frames_from_far_bbx(obj_ids, save_masks, frame_idx)

        # save_masks = self.remove_masks_by_area(save_masks)
        saved_counts = {int(oid): len(data["frame"]) for oid, data in save_masks.items()}
        self._log_info(f"Completed mask export. Saved sampled frames per ID: {saved_counts}")

        if not skip_overlays and bool(export_cfg.get("export_masked_outputs_mp4", True)):
            self._export_image_dir_to_mp4(
                masked_outputs,
                fps=float(export_cfg.get("masked_outputs_fps", 30.0)),
                video_filename="masked_outputs.mp4",
                remove_images=bool(export_cfg.get("remove_masked_outputs_images", True)),
            )
        return save_masks


    def remove_masks_by_area(self, save_masks):
        total_area = []
        for obj_id, data in save_masks.items():
            masks = np.array(data["mask"])
            area = masks.sum(axis=(-1,-2))
            total_area += area.tolist()

        # remove masks that do not feet a std deviation
        std_area = np.std(total_area)
        mean_area = np.mean(total_area)
        self._log_info(f"Mask-area pruning stats: mean={mean_area:.2f}, std={std_area:.2f}")
        for obj_id, data in save_masks.items():
            masks = np.array(data["mask"])
            area = masks.sum(axis=(-1,-2))
            self._log_info(f"Object {obj_id}: sampled area count={len(area)}, frames={len(data['frame'])}")
            del_idx = []
            for i, a in enumerate(area):
                if a < (mean_area - std_area) or a > (mean_area + std_area):
                    del_idx.append(i)
            # remove the mask
            if del_idx:
                data["mask"] = np.delete(masks, del_idx, axis=0)
                data["img"] = np.delete(data["img"], del_idx, axis=0)
                data["frame"] = np.delete(data["frame"], del_idx, axis=0)
                data["iou"] = np.delete(data["iou"], del_idx, axis=0)
                data["bbox"] = np.delete(data["bbox"], del_idx, axis=0)
                self._log_warn(f"Removed outlier masks for object {obj_id}: idx={del_idx}")
        return save_masks


    def _resolve_sam_video_source(self, frames, cam_data=None):
        """
        Determine the video source for SAM's init_state.

        For VideoSource with full video: returns the MP4 path (native decoding).
        For VideoSource with frame subset: extracts JPEG frames to a temp dir
            (SAM can't index into a partial MP4).
        For ImageFileSource: returns the directory containing the frame images.

        Args:
            frames: FrameSource instance.
            cam_data: Optional cam_data dict (for backward compat).

        Returns:
            str: Path to MP4 file or frame directory for SAM.
        """
        if isinstance(frames, VideoSource):
            vp = frames.video_path
            reader = frames.reader
            import cv2
            cap = cv2.VideoCapture(vp)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if len(frames) < total:
                # Frame subset — extract JPEG frames for SAM only
                range_start = int(reader.start)
                range_end = int(reader.end) if reader.end is not None else (range_start + len(frames))
                sam_frame_dir = os.path.join(
                    self.output_path,
                    frames.cam_name,
                    f"_sam_frames_{range_start:06d}_{range_end:06d}",
                )
                os.makedirs(sam_frame_dir, exist_ok=True)
                self._log_info(
                    f"[{frames.cam_name}] Extracting {len(frames)} frames for SAM "
                    f"(frame range [{reader.start}:{reader.end}])."
                )
                for i in range(len(frames)):
                    fpath = os.path.join(sam_frame_dir, frames.frame_names[i])
                    if not os.path.isfile(fpath):
                        bgr = reader.read_bgr(i)
                        cv2.imwrite(fpath, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return sam_frame_dir
            return vp

        if isinstance(frames, ImageFileSource):
            img_dir = os.path.dirname(frames.paths[0])
            # If frames is a subset of the directory (e.g., ma_cap was run with
            # --start/--end), set the frame filter so SAM only loads matching
            # files.  Without this, SAM loads ALL images in the directory and
            # its frame indices no longer match the frames object.
            all_images = [
                f for f in os.listdir(img_dir)
                if os.path.splitext(f)[-1].lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
                }
            ]
            frame_basenames = {os.path.basename(p) for p in frames.paths}
            if len(frame_basenames) < len(all_images):
                from core.predictor_factory import set_frame_filter
                set_frame_filter(list(frame_basenames))
                self._log_info(
                    f"Frame filter auto-set: {len(frame_basenames)} of "
                    f"{len(all_images)} images in directory "
                    f"(subset from upstream step)."
                )
            return img_dir

        # Fallback
        if cam_data is not None and 'video_path' in cam_data:
            return str(cam_data['video_path'])
        if cam_data is not None and 'img_abs_path' in cam_data:
            paths = cam_data['img_abs_path'].tolist()
            if paths:
                return os.path.dirname(paths[0])
        raise ValueError("Cannot resolve SAM video source from the provided frame source.")

    def process_multi_video_auto(self, cam_data, frame_id, masks=None, reference_cam_data=None, expected_subjects=None, interactive=False):
        # Apply frame range if configured
        _temp_dir = None
        if self.start_frame is not None or self.end_frame is not None:
            cam_data, frame_id, _temp_dir = self._apply_frame_range(cam_data, frame_id)

        # Build unified frame source
        frames = frame_source_from_cam_data(cam_data)
        self._setup_frame_access(frames, cam_data)
        cam_name = frames.cam_name
        out_path = os.path.join(self.output_path, cam_name)
        os.makedirs(out_path, exist_ok=True)

        # Process the first video
        if masks is None:
            found_key = None
            if self.init_with_gt:
                bbox_candidate_keys = ['bboxes', 'bbox', 'gt_bboxes']
                for key in bbox_candidate_keys:
                    if key in cam_data:
                        found_key = key
                        break

            if self.sam_version == "sam3_prompt":
                self._log_info(f"[{cam_name}] Initializing first view using SAM3 text prompt 'person'.")
                masks = self.process_first_video_sam3_prompt(
                    frames,
                    frame_id,
                    out_path,
                    expected_subjects=expected_subjects,
                )
            elif interactive:
                self._log_info(f"[{cam_name}] Initializing first view using interactive GUI.")
                masks = self.process_first_video_interactive(
                    frames,
                    frame_id,
                    out_path,
                    expected_subjects=expected_subjects,
                )
            elif self.init_with_gt and found_key is not None:
                self._log_info(f"[{cam_name}] Initializing first view using GT bboxes from key '{found_key}'.")
                gt_bboxes = cam_data[found_key]
                bbox_mask_candidate_keys = ['bboxes_mask', 'bbox_mask', 'gt_bboxes_mask', 'is_body_in_img']
                found_mask_key = None
                for key in bbox_mask_candidate_keys:
                    if key in cam_data:
                        found_mask_key = key
                        break
                if found_mask_key is None:
                    raise ValueError(f"init_with_gt_bbox is True but no bbox mask key found in cam_data. Tried keys: {bbox_mask_candidate_keys}")
                gt_bboxes_mask = cam_data[found_mask_key]
                masks = self.process_first_video_with_gt_bbox(
                    frames,
                    frame_id,
                    out_path,
                    gt_bboxes,
                    gt_bboxes_mask,
                    expected_subjects=expected_subjects,
                )
            else:
                self._log_info(f"[{cam_name}] Initializing first view using automatic person detection.")
                masks = self.process_first_video_auto(
                    frames,
                    frame_id,
                    out_path,
                    expected_subjects=expected_subjects,
                )
            if masks is None:
                self._log_warn(f"[{cam_name}] Initialization skipped: no valid person detections found.")
                return None
            self.save_picked_masks(masks, out_path, render_viz=self._debug_crop_summary())
            self.compute_clip_features(masks)
            if len(self.subject_feature_bank) == 0:
                self.initialize_subject_feature_bank(masks, expected_subjects=expected_subjects)
            self._finalize_mask_cache(masks, out_path)

        # Process new video
        else:
            first_key = sorted(masks.keys())[0] if len(masks) > 0 else None
            if first_key is not None and 'features' not in masks[first_key]:
                self._log_info(f"[{cam_name}] Input masks do not contain CLIP features. Computing features first.")
                self.compute_clip_features(masks)

            out_path = os.path.join(self.output_path, cam_name)
            os.makedirs(out_path, exist_ok=True)

            if os.path.exists(os.path.join(out_path, "masks.npy")):
                self._log_info(f"[{cam_name}] Reusing existing masks cache: {os.path.join(out_path, 'masks.npy')}")
                return masks

            if self.sam_version == "sam3_prompt":
                new_mask_data = self.process_new_video_sam3_prompt(
                    frames, masks, out_path,
                    source_cam_data=reference_cam_data,
                    target_cam_data=cam_data,
                    expected_subjects=expected_subjects,
                )
            else:
                new_mask_data = self.process_new_video(
                    frames,
                    masks,
                    out_path,
                    source_cam_data=reference_cam_data,
                    target_cam_data=cam_data,
                    expected_subjects=expected_subjects,
                )
            self.save_picked_masks(new_mask_data, out_path, render_viz=self._debug_crop_summary())
            self.compute_clip_features(new_mask_data)
            if self._debug_crop_summary():
                try:
                    self.plot_tsne(new_mask_data, out_path)
                except Exception as exc:
                    self._log_warn(f"Skipping t-SNE visualization due to runtime error: {exc}")
            self._finalize_mask_cache(new_mask_data, out_path)

        # But still return the mask data from the first view
        # Alternatively, we could return the new_mask_data to be used in the next view
        self._cleanup_frame_range_dir(_temp_dir)
        return masks

    def process_multi_video_using_gt_bboxes(self, cam_data):
        # Apply frame range if configured
        _temp_dir = None
        if self.start_frame is not None or self.end_frame is not None:
            cam_data, _dummy_frame_id, _temp_dir = self._apply_frame_range(cam_data, 0)

        # Build unified frame source
        frames = frame_source_from_cam_data(cam_data)
        self._setup_frame_access(frames, cam_data)
        cam_name = frames.cam_name
        out_path = os.path.join(self.output_path, cam_name)
        os.makedirs(out_path, exist_ok=True)

        npy_file_path = os.path.join(out_path, "masks.npy")
        if os.path.exists(npy_file_path):
            self._log_info(f"Reusing existing masks cache: {npy_file_path}")
            masks = np.load(npy_file_path, allow_pickle=True)[()]
            self._cleanup_frame_range_dir(_temp_dir)
            return masks

        gt_bboxes = cam_data['bboxes']
        gt_bboxes_mask = cam_data['bboxes_mask']

        sam_source = self._current_sam_video_source
        frame_names = frames.frame_names
        inference_state, _runtime_video_dir = self._init_state_with_fallback(
            video_dir=sam_source,
            frame_names=frame_names,
            output_path=out_path,
            context_name=cam_name,
        )
        self.predictor.reset_state(inference_state)

        for frame_id, (bbox_frame, bbox_mask_frame) in enumerate(zip(gt_bboxes, gt_bboxes_mask)):
            for obj_id, (bbox, bbox_mask) in enumerate(zip(bbox_frame, bbox_mask_frame)):
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_id,
                    obj_id=obj_id,
                    box=bbox,
                )

        self._log_info(f"[{cam_name}] Starting propagation from dense GT prompts.")
        video_segments = self.run_propagation(inference_state)

        obj_ids_range = range(gt_bboxes.shape[1])
        output_path = os.path.join(self.output_path, cam_name)
        os.makedirs(output_path, exist_ok=True)
        masks = self.save_images_from_video(inference_state, video_segments, obj_ids_range, output_path)
        _atomic_np_save(os.path.join(output_path, "masks.npy"), masks)
        self._log_info(f"[{cam_name}] Saved mask cache: {os.path.join(output_path, 'masks.npy')}")
        self._cleanup_sanitized_video_dir(_runtime_video_dir, sam_source)
        self._cleanup_frame_range_dir(_temp_dir)
        return masks


    def process_first_video_auto(self, frames, frame_id, output_path, expected_subjects=None):
        """Process the init camera using automatic YOLO person detection.

        Args:
            frames: FrameSource providing access to video/image frames.
            frame_id: Initial frame index for detection.
            output_path: Directory for output masks and visualizations.
            expected_subjects: Expected number of people (auto-detected if None).
        """
        if os.path.exists(os.path.join(output_path, "masks.npy")):
            self._log_info(
                f"masks.npy already exists in {output_path}, skipping processing "
                "(init anchors/prompts are not recomputed)."
            )
            save_masks = np.load(os.path.join(output_path, "masks.npy"), allow_pickle=True)[()]
            return save_masks

        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        self._log_info(f"Starting automatic init-camera processing for output '{output_path}'.")

        n_frames = len(frames)
        auto_select = frame_id is None

        if frame_id is not None and frame_id >= n_frames:
            self._log_warn(f"Requested init frame {frame_id} is out of range; using frame {n_frames - 1}.")
            frame_id = n_frames - 1

        cfg = self.assignment_config.get("matching", {})
        yolo_conf = float(cfg.get("yolo_person_conf", 0.5))
        sample_count = min(30, n_frames)
        sampled = list(np.linspace(0, n_frames - 1, num=sample_count, dtype=int))

        if auto_select:
            # Auto-select: score all candidate frames, pick the one with
            # the most detections at highest total confidence.
            self._log_info(f"Auto-selecting best init frame from {sample_count} candidates.")
            best_score = -1
            best_frame_id = 0
            best_img = None
            best_dets = []
            for cand_frame in sampled:
                cand_img = frames.read_pil(int(cand_frame))
                img, dets = self._collect_yolo_detections(cand_img, yolo_conf, include_clip_features=False)
                if not dets:
                    continue
                # Score: number of detections * mean confidence
                n_det = len(dets)
                mean_conf = sum(float(d[3]) for d in dets) / n_det
                score = n_det * mean_conf
                # Separation factor: crowded frames make bad seeds.
                score *= 1.0 - self._mean_pairwise_iou([d[2] for d in dets])
                # Centrality factor: endpoint seeds double the worst-case
                # propagation distance.
                score *= _seed_centrality_weight(
                    int(cand_frame), n_frames,
                    float(cfg.get("seed_center_bias", 0.0)))
                # Prefer frames with expected_subjects detections
                if expected_subjects and n_det == expected_subjects:
                    score *= 1.5  # bonus for exact match
                if score > best_score:
                    best_score = score
                    best_frame_id = int(cand_frame)
                    best_img = img
                    best_dets = dets

            if best_dets:
                frame_id = best_frame_id
                selected_img = best_img
                detections = best_dets
                self._log_info(
                    f"Auto-selected init frame {frame_id}: {len(detections)} people detected "
                    f"(score={best_score:.2f})."
                )
            else:
                frame_id = 0
                selected_img = None
                detections = []
        else:
            # Manual frame_id: try it first, fall back to sampled frames with lower thresholds
            conf_thresholds = (yolo_conf, 0.4, 0.25, 0.15)
            candidate_frames = [frame_id] + sampled
            seen = set()
            candidate_frames = [f for f in candidate_frames if not (f in seen or seen.add(f))]

            selected_img = None
            detections = []
            for conf in conf_thresholds:
                for cand_frame in candidate_frames:
                    cand_img = frames.read_pil(cand_frame)
                    img, dets = self._collect_yolo_detections(cand_img, conf, include_clip_features=False)
                    if dets:
                        selected_img = img
                        detections = dets
                        frame_id = int(cand_frame)
                        self._log_info(
                            f"Selected init frame {frame_id}: detected {len(dets)} person boxes "
                            f"(YOLO conf threshold={conf:.2f})."
                        )
                        break
                if detections:
                    break

        if not detections or selected_img is None:
            self._log_warn(
                "No person bboxes detected in sampled frames. "
                "Skipping this camera without saving masks."
            )
            return None

        if expected_subjects is None or expected_subjects <= 0:
            expected_subjects = self.expected_subjects
        if (expected_subjects is None or expected_subjects <= 0) and len(detections) > 0:
            expected_subjects = len(detections)
            self.expected_subjects = expected_subjects
            self._log_info(f"Inferred scene subject count from init detections: N={expected_subjects}.")
        if expected_subjects is not None and expected_subjects > 0 and len(detections) > expected_subjects:
            total_candidates = len(detections)
            detections = sorted(detections, key=lambda x: float(x[3]), reverse=True)[:expected_subjects]
            self._log_info(
                f"Applied subject-count cap during init: kept top {expected_subjects}/{total_candidates} detections."
            )

        # Candidate-frame search skipped CLIP encoding for speed; encode now for the chosen detections.
        if len(detections) > 0 and any(d[1] is None for d in detections):
            det_crops = [d[0] for d in detections]
            det_feats = self._encode_clip_batch(det_crops)
            detections = [
                (crop, feat, bbox, score)
                for (crop, _, bbox, score), feat in zip(detections, det_feats)
            ]

        cam_name = frames.cam_name

        self._log_info(f"[{cam_name}] Preparing init prompts from frame {frame_id}.")
        ann_obj_id = len(detections)
        out_init_dir = os.path.join(self.output_path, "initialize")
        self._save_init_detection_overview(
            image=selected_img,
            detections=detections,
            frame_idx=frame_id,
            out_init_dir=out_init_dir,
            tag="auto",
            camera_name=cam_name,
        )

        init_best_similarity = {
            oid: (float(score), int(frame_id), crop, np.asarray(bbox, dtype=np.float32), crop)
            for oid, (crop, _, bbox, score) in enumerate(detections)
        }
        init_anchor_matches = {
            oid: [
                {
                    "score": float(score),
                    "frame_idx": int(frame_id),
                    "img_bbx": crop,
                    "bbox": np.asarray(bbox, dtype=np.float32),
                    "target_img_bbx": crop,
                }
            ]
            for oid, (crop, _, bbox, score) in enumerate(detections)
        }

        # A single init-frame prompt per ID. Multi-frame init anchors were
        # removed: anchor association is unreliable on similar-looking people
        # and each anchor conditions only its own frame, so they poison the
        # tracker's identity memory instead of stabilizing it (measured: raw
        # tracking on identical costumes was ~2x cleaner without them).
        self._log_info("Using a single init-frame prompt per ID.")

        # Initialize the SAM predictor with the detected bboxes
        self._log_info(f"[{cam_name}] Initializing SAM state and injecting init anchor prompts.")
        sam_source = self._current_sam_video_source
        frame_names = frames.frame_names
        inference_state, runtime_video_dir = self._init_state_with_fallback(
            video_dir=sam_source,
            frame_names=frame_names,
            output_path=output_path,
            context_name=cam_name,
        )
        self.predictor.reset_state(inference_state)
        self.show_best_similarity_and_add_bboxes(
            init_best_similarity,
            frames=frames,
            video_dir=runtime_video_dir,
            inference_state=inference_state,
            frame_names=frame_names,
            output_path=output_path,
            anchor_matches=init_anchor_matches,
        )
        self._log_info(f"[{cam_name}] Starting propagation from the init prompts.")
        video_segments = self.run_propagation(inference_state)

        obj_ids = range(ann_obj_id)
        os.makedirs(output_path, exist_ok=True)
        save_masks = self.save_images_from_video(inference_state, video_segments, obj_ids, output_path)
        _atomic_np_save(os.path.join(output_path, "masks.npy"), save_masks)
        self._log_info(f"[{cam_name}] Saved mask cache: {os.path.join(output_path, 'masks.npy')}")
        self._cleanup_sanitized_video_dir(runtime_video_dir, sam_source)
        return save_masks

    def process_first_video_interactive(self, frames, frame_id, output_path, expected_subjects=None):
        """Process the init camera using interactive GUI point prompts.

        Opens a tkinter GUI where the user clicks on people to track.
        Each click group (red/blue) becomes a separate person ID.
        Left-click = positive point, right-click = remove point.

        Args:
            frames: FrameSource providing access to video/image frames.
            frame_id: Frame to display in the GUI. None = use frame 0.
            output_path: Directory for output masks and visualizations.
            expected_subjects: Expected number of people (informational).
        """
        from utils.gui import show_images_gui

        if os.path.exists(os.path.join(output_path, "masks.npy")):
            self._log_info(f"masks.npy already exists in {output_path}, skipping.")
            return np.load(os.path.join(output_path, "masks.npy"), allow_pickle=True)[()]

        os.makedirs(output_path, exist_ok=True)
        cam_name = frames.cam_name
        n_frames = len(frames)

        # Auto-select frame if None
        if frame_id is None:
            frame_id = 0

        # Init SAM state
        sam_source = self._current_sam_video_source
        frame_names = frames.frame_names
        inference_state, runtime_video_dir = self._init_state_with_fallback(
            video_dir=sam_source,
            frame_names=frame_names,
            output_path=output_path,
            context_name=cam_name,
        )
        self.predictor.reset_state(inference_state)

        # Show GUI — user clicks on people
        self._log_info(
            f"[{cam_name}] Opening interactive GUI. "
            "Left-click to mark people (red = person 1, blue = person 2). "
            "Right-click to remove a point. Close window when done."
        )
        if expected_subjects:
            self._log_info(f"[{cam_name}] Expecting {expected_subjects} people.")

        # clicks: {person_id (int): {frame_idx: [(x, y), ...]}}
        clicks = show_images_gui(inference_state, sam_version=self.sam_version)

        # Convert GUI coordinates (network resolution) to original image resolution
        w_original = inference_state['video_width']
        h_original = inference_state['video_height']
        _, _, h_network, w_network = inference_state['images'].shape

        prompts = {}
        for obj_id, frame_clicks in sorted(clicks.items()):
            for click_frame, coords in frame_clicks.items():
                if not coords:
                    continue
                original_coords = [
                    (int(x * w_original / w_network), int(y * h_original / h_network))
                    for x, y in coords
                ]
                points = np.array(original_coords, dtype=np.float32)
                labels = np.ones(len(points), dtype=np.int32)  # all positive

                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=int(click_frame),
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
                prompts[obj_id] = (points, labels, int(click_frame))
                self._log_info(
                    f"[{cam_name}] Person {obj_id}: {len(points)} point(s) on frame {click_frame}."
                )

        if not prompts:
            self._log_warn(f"[{cam_name}] No clicks provided. Skipping.")
            self._cleanup_sanitized_video_dir(runtime_video_dir, sam_source)
            return None

        n_people = len(prompts)
        self._log_info(f"[{cam_name}] {n_people} people marked. Starting propagation.")

        # Save click visualization
        person_colors = [
            "#FF0000", "#0066FF", "#00CC00", "#FF9900", "#CC00CC",
            "#00CCCC", "#FFCC00", "#FF66CC", "#6633FF", "#99CC00",
        ]
        for obj_id, (points, labels, click_frame) in prompts.items():
            color = person_colors[obj_id % len(person_colors)]
            plt.figure(figsize=(9, 6))
            plt.title(f"Person {obj_id} — frame {click_frame}")
            plt.imshow(frames.read_pil(click_frame))
            for x, y in points:
                plt.scatter(x, y, c=color, s=50, zorder=5, edgecolors="white", linewidths=1)
            plt.savefig(os.path.join(output_path, f"interactive_person_{obj_id}.png"))
            plt.close()

        # Propagate
        video_segments = self.run_propagation(inference_state)

        obj_ids = sorted(prompts.keys())
        save_masks = self.save_images_from_video(inference_state, video_segments, obj_ids, output_path)
        _atomic_np_save(os.path.join(output_path, "masks.npy"), save_masks)
        self._log_info(f"[{cam_name}] Saved mask cache: {os.path.join(output_path, 'masks.npy')}")
        self._cleanup_sanitized_video_dir(runtime_video_dir, sam_source)
        return save_masks

    # ── SAM3 prompt helpers ──────────────────────────────────────────────────

    def _sam3_prompt_predictor(self):
        """Return the SAM3 prompt predictor, or None if not available."""
        from core.predictor_factory import _Sam3PromptVideoAdapter
        if isinstance(self.predictor, _Sam3PromptVideoAdapter):
            return self.predictor
        self._log_error("sam3_prompt mode requires _Sam3PromptVideoAdapter predictor.")
        return None

    def _sam3_prompt_min_mask_area(self, frames):
        """Return the minimum mask area used to reject tiny SAM3 prompt outputs.

        Very small masks are usually low-value signal for frame selection:
        background people, partial bodies, or noise. The threshold is scaled by
        image size via ``min_mask_area_ratio``.
        """
        cfg = self._get_matching_cfg()
        min_mask_ratio = float(cfg.get("min_mask_area_ratio", 0.01))
        if frames is not None and len(frames) > 0:
            img_w, img_h = frames.image_size
            return int(img_w * img_h * min_mask_ratio)
        return 1000

    def _sam3_prompt_candidate_frames(self, n_frames, frame_id=None):
        """Return candidate frames for selecting a SAM3 text-prompt initialization frame.

        The search stays cheap on purpose: include the configured start frame,
        optionally include the user-provided frame, and fill the rest with
        evenly spaced samples through the clip.
        """
        if n_frames <= 0:
            return []
        cfg = self._get_matching_cfg()
        start_idx = min(int(cfg.get("start_frame", 0)), n_frames - 1)
        max_candidates = int(cfg.get("sam3_redetect_samples", 30))

        candidates = [start_idx]
        if frame_id is not None:
            candidates.append(int(frame_id))
        if n_frames > 1:
            # Scale gently with clip length while keeping the search bounded.
            sample_count = min(max_candidates, max(6, n_frames // 20), n_frames)
            candidates.extend(
                int(i) for i in np.linspace(start_idx, n_frames - 1, num=sample_count, dtype=int)
            )
        return sorted(set(fidx for fidx in candidates if 0 <= fidx < n_frames))

    def _sam3_prompt_mask_area(self, mask):
        """Return the foreground area of a SAM3 mask in pixels."""
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[0]
        return int((mask > 0).sum())

    def _sam3_prompt_size_weight(self, mask_area, image_area):
        """Return a soft confidence weight based on relative mask size.

        Small clean crops are not necessarily useful for remapping: background
        outsiders often have low overlap IoU precisely because they are far away
        and isolated. This weight counterbalances that by favoring tracks with
        more substantial visible area, without hard-rejecting smaller subjects.
        """
        if image_area <= 0:
            return 1.0
        cfg = self._get_matching_cfg()
        min_ratio = float(cfg.get("min_mask_area_ratio", 0.01))
        good_ratio = max(min_ratio * 4.0, 0.02)
        area_ratio = float(mask_area) / float(image_area)
        if area_ratio <= min_ratio:
            return 0.1
        if area_ratio >= good_ratio:
            return 1.0
        t = (area_ratio - min_ratio) / max(good_ratio - min_ratio, 1e-8)
        return 0.1 + 0.9 * t

    def _sam3_prompt_score_result(self, result, min_area, expected_subjects=None):
        """Score a SAM3 prompt result using the largest valid mask areas.

        We prefer frames with several substantial person masks over frames with
        many tiny detections. The top masks are softly capped using a
        median-based threshold so one person very close to the camera does not
        dominate the score by itself.
        """
        top_k = int(expected_subjects) if expected_subjects is not None and expected_subjects > 0 else 4
        valid_areas = []
        for mask in result.get("masks", {}).values():
            area = self._sam3_prompt_mask_area(mask)
            if area >= min_area:
                valid_areas.append(area)
        valid_areas.sort(reverse=True)

        if not valid_areas:
            return {
                "score": (0.0, 0.0, 0, float("-inf")),
                "valid_count": 0,
                "valid_areas": [],
            }

        clipped_top = valid_areas[:top_k]
        clip_cap = max(float(np.median(clipped_top)) * 1.5, float(min_area))
        balanced_area_score = float(sum(min(area, clip_cap) for area in clipped_top))
        spread_bonus = float(np.median(clipped_top))
        return {
            "score": (balanced_area_score, spread_bonus, len(valid_areas), -len(result.get("obj_ids", []))),
            "valid_count": len(valid_areas),
            "valid_areas": valid_areas,
        }

    def _sam3_prompt_select_frame(self, predictor, cam_name, n_frames, frame_id=None,
                                  expected_subjects=None, min_area=1000):
        """Pick a prompt frame by scoring SAM3 text-prompt outputs on sampled frames.

        SAM3 is queried on each candidate frame, the session is reset between
        trials, and only after the best frame is identified do we re-apply the
        prompt for the actual propagation pass.
        """
        candidate_frames = self._sam3_prompt_candidate_frames(n_frames, frame_id=frame_id)
        if not candidate_frames:
            return None, None

        self._log_info(
            f"[{cam_name}] Selecting SAM3 prompt frame from {len(candidate_frames)} candidate(s) "
            f"(min_area={min_area}px)."
        )

        best_frame = None
        best_result = None
        best_score = None
        for cand in candidate_frames:
            predictor.reset_session()
            result = predictor.add_text_prompt(frame_idx=cand, text="person")
            score_info = self._sam3_prompt_score_result(
                result, min_area=min_area, expected_subjects=expected_subjects,
            )
            # Centrality: scale the primary (area) term so endpoint prompt
            # frames only win when clearly better.
            _w = _seed_centrality_weight(
                int(cand), n_frames,
                float(self.assignment_config.get("matching", {}).get("seed_center_bias", 0.0)))
            score_info["score"] = (score_info["score"][0] * _w,) + tuple(score_info["score"][1:])
            self._log_info(
                f"[{cam_name}]   frame {cand}: {len(result.get('obj_ids', []))} ids, "
                f"{score_info['valid_count']} valid masks, "
                f"areas={score_info['valid_areas'][:4]}"
            )
            if best_score is None or score_info["score"] > best_score:
                best_score = score_info["score"]
                best_frame = cand
                best_result = result

        return best_frame, best_result

    def _postprocess_tracklets(self, cam_name, video_segments, detected_ids, image_size=None):
        """Unified post-propagation cleanup for all SAM versions.

        1. Merge duplicate tracklets (near-identical masks = same person tracked twice)
        2. Truncate re-latched tracklets (a track that lost its person mid-sequence
           and re-acquired onto another person)
        3. Discard tiny tracklets (noise, not real people)

        Controlled by the ``masks:`` config section.
        """
        mask_cfg = self.assignment_config.get("masks", {})

        # Step 1: merge duplicates
        if bool(mask_cfg.get("merge_duplicate_tracklets", True)):
            merge_iou = float(mask_cfg.get("merge_iou_threshold", 0.95))
            video_segments, detected_ids = self._merge_duplicate_tracklets(
                cam_name, video_segments, detected_ids, iou_thresh=merge_iou,
            )

        # Step 2: truncate re-latched tracklets (partial-duration duplicates the
        # whole-clip mean of Step 1 cannot see)
        if bool(mask_cfg.get("truncate_relatched_tracklets", True)):
            video_segments = self._truncate_relatched_tracklets(
                cam_name, video_segments, detected_ids,
                iou_thresh=float(mask_cfg.get("relatch_iou", 0.9)),
                min_run_frames=int(mask_cfg.get("relatch_min_run_frames", 5)),
                pre_frames=int(mask_cfg.get("relatch_pre_frames", 5)),
                pre_area_px=int(mask_cfg.get("relatch_pre_area_px", 100)),
                # The mask-continuity gate also resolves sustained ambiguous
                # duplicates here (run-boundary jump); off by default.
                resolve_by_jump=bool(mask_cfg.get("mask_continuity_gate", False)),
                min_jump_ratio=float(mask_cfg.get("relatch_min_jump_ratio", 0.025)),
                image_size=image_size,
            )

        # Step 3: discard tiny tracklets
        if bool(mask_cfg.get("discard_tiny_tracklets", True)):
            min_area_ratio = float(mask_cfg.get("tiny_tracklet_min_area_ratio", 0.005))
            min_frame_ratio = float(mask_cfg.get("tiny_tracklet_min_frame_ratio", 0.2))
            if image_size:
                min_area = int(image_size[0] * image_size[1] * min_area_ratio)
            else:
                min_area = 1000

            for obj_id in list(detected_ids):
                # Stream per-frame areas (one unpacked mask at a time) instead of
                # holding every frame's masks in RAM.
                real_frames = 0
                total_frames = 0
                for fidx in video_segments.frames():
                    area = video_segments.obj_area(fidx, obj_id)
                    if area is None:
                        continue
                    total_frames += 1
                    if area >= min_area:
                        real_frames += 1
                if total_frames > 0 and real_frames / total_frames < min_frame_ratio:
                    self._log_warn(
                        f"[{cam_name}] Discarding tracklet {obj_id}: mask too small on "
                        f"{total_frames - real_frames}/{total_frames} frames "
                        f"(min_area={min_area}px, min_frame_ratio={min_frame_ratio})."
                    )
                    detected_ids = [d for d in detected_ids if d != obj_id]
                    video_segments.discard_obj(obj_id)

        # Step 4: mask-continuity gate (opt-in). Drop per-frame masks where a
        # tracklet's centroid teleports for one or two frames and returns (SAM2
        # transiently re-locking onto a neighbour) — a transient the whole-clip
        # merge and the >=5-frame re-latch truncation above cannot see.
        if bool(mask_cfg.get("mask_continuity_gate", False)):
            video_segments = self._gate_mask_continuity(
                cam_name, video_segments, detected_ids, image_size=image_size,
                k=float(mask_cfg.get("mask_gate_k", 8.0)),
                floor_ratio=float(mask_cfg.get("mask_gate_floor_ratio", 0.12)),
                half_window=int(mask_cfg.get("mask_gate_half_window", 7)),
                max_gap=int(mask_cfg.get("mask_gate_max_gap", 4)),
                dup_iou=float(mask_cfg.get("mask_gate_dup_iou", 0.8)),
                spike_ratio=float(mask_cfg.get("mask_gate_spike_ratio", 3.0)),
            )

        return video_segments, detected_ids

    def _gate_mask_continuity(self, cam_name, video_segments, detected_ids,
                              image_size=None, k=8.0, floor_ratio=0.12,
                              half_window=7, max_gap=4, dup_iou=0.8, spike_ratio=3.0):
        """Drop per-frame masks that transiently teleport or duplicate (opt-in Step 4).

        Two complementary per-tracklet detectors, per camera:
          A. Centroid teleport: the mask centroid jumps far from its local median
             and returns (``_detect_centroid_outlier_runs``). Catches far jumps and
             swaps, but is blind to a duplicate onto a NEARBY neighbour (small
             centroid jump).
          B. Transient IoU-duplicate: the mask area spikes for a frame or two and
             the spiked mask nearly coincides (IoU > ``dup_iou``) with another
             tracklet — SAM2 briefly grabbing that neighbour. Catches the
             near-neighbour case independent of distance.
        Flagged frames are zeroed, so the person is left un-segmented there and
        ma_3d triangulates from the clean cameras only — strictly better than a
        teleported/duplicate mask. No-op when off; best-effort (never fatal).
        """
        # Resolution-agnostic threshold: the floor is a fraction of THIS camera's
        # image diagonal (and the MAD term is measured from the data), so the gate
        # behaves identically at any resolution and for portrait/landscape (W^2+H^2
        # is orientation-invariant). Recover the size from the mask store if the
        # caller did not supply it, so we never fall back to a fixed pixel guess.
        if not image_size:
            image_size = video_segments.image_size()
        if image_size:
            diag = float((image_size[0] ** 2 + image_size[1] ** 2) ** 0.5)
        else:
            diag = 2000.0
        abs_floor = float(floor_ratio) * diag

        # Best-effort: a bug in the gate must never abort the ma_masks run (the
        # masks are already produced; the gate only cleans transient outliers).
        try:
            def _flat(m):
                arr = np.asarray(m)
                return arr[0] if arr.ndim == 3 else arr

            # One pass over frames: per-frame centroid + area per obj. Masks may
            # carry a leading channel dim ([1,H,W]); drop it as _compute_centroids does.
            centroids = {oid: {} for oid in detected_ids}
            areas = {oid: {} for oid in detected_ids}
            for fidx in video_segments.frames():
                masks = video_segments.frame(fidx)
                for oid in detected_ids:
                    m = masks.get(oid)
                    if m is None:
                        continue
                    arr = _flat(m)
                    a = int(arr.sum())
                    if a <= 0:
                        continue
                    areas[oid][fidx] = a
                    c = self._mask_centroid_xy1(arr)
                    if c is not None:
                        centroids[oid][fidx] = (float(c[0]), float(c[1]))

            # Frames to zero, per obj (union of both detectors).
            drop = {oid: set() for oid in detected_ids}

            # Detector A: centroid teleport (far jumps + swaps).
            for oid in detected_ids:
                for a, b in _detect_centroid_outlier_runs(
                    centroids[oid], k=k, abs_floor=abs_floor,
                    half_window=half_window, max_gap=max_gap,
                ):
                    drop[oid].update(range(a, b + 1))

            # Detector B: transient IoU-duplicate (near-neighbour takeover). Only
            # the rare area-spike frames are IoU-checked, so this stays cheap.
            for oid in detected_ids:
                for f in _transient_area_spikes(areas[oid], spike_ratio=spike_ratio):
                    if f in drop[oid]:
                        continue
                    masks = video_segments.frame(f)
                    m = masks.get(oid)
                    if m is None:
                        continue
                    arr = _flat(m)
                    for other in detected_ids:
                        if other == oid:
                            continue
                        mo = masks.get(other)
                        if mo is None:
                            continue
                        bo = _flat(mo)
                        inter = int(np.logical_and(arr, bo).sum())
                        if inter == 0:
                            continue
                        union = int(np.logical_or(arr, bo).sum())
                        if union > 0 and inter / union > dup_iou:
                            drop[oid].add(f)
                            break

            # Zero all flagged frames.
            for oid in detected_ids:
                if not drop[oid]:
                    continue
                n_zeroed = 0
                for fidx in sorted(drop[oid]):
                    masks = video_segments.frame(fidx)
                    m = masks.get(oid)
                    if m is None or not m.any():
                        continue
                    masks[oid] = np.zeros_like(m)
                    video_segments.set_frame(fidx, masks)
                    n_zeroed += 1
                frs = ", ".join(str(f) for f in sorted(drop[oid]))
                self._log_info(
                    f"[{cam_name}] mask-continuity gate: tracklet {oid} teleport/duplicate "
                    f"on frames [{frs}] (floor={abs_floor:.0f}px) — zeroed {n_zeroed} "
                    f"frame(s); the person is left un-segmented there."
                )
        except Exception as e:
            self._log_warn(f"[{cam_name}] mask-continuity gate skipped (non-fatal): {e}")
        return video_segments

    def _merge_duplicate_tracklets(self, cam_name, video_segments, detected_ids,
                                    iou_thresh=0.95, sample_count=20):
        """Merge tracklet IDs that cover the same person.

        When SAM assigns multiple IDs to one person (e.g., upper body + lower
        body, or a re-detection after brief loss), this merges them so the
        tiny-mask filter doesn't discard both partial tracklets.

        For each pair of IDs, computes average mask IoU across sampled frames.
        If IoU > iou_thresh, the smaller tracklet is absorbed into the larger one.
        """
        if len(detected_ids) < 2:
            return video_segments, detected_ids

        sorted_frames = video_segments.frames()
        n_sample = min(sample_count, len(sorted_frames))
        if n_sample == 0:
            return video_segments, detected_ids
        sample_idxs = [sorted_frames[int(i)] for i in
                       np.linspace(0, len(sorted_frames) - 1, n_sample, dtype=int)]

        # Load only the sampled frames once (≤ sample_count frames in RAM) for
        # the area + IoU statistics, instead of re-reading from the store.
        sampled = {fidx: video_segments.frame(fidx) for fidx in sample_idxs}

        # Compute average mask area per ID (for deciding which to keep)
        id_avg_area = {}
        for oid in detected_ids:
            areas = []
            for fidx in sample_idxs:
                mask = sampled[fidx].get(oid)
                if mask is not None:
                    m = mask[0] if mask.ndim == 3 else mask
                    areas.append(float(m.sum()))
            id_avg_area[oid] = np.mean(areas) if areas else 0.0

        # Compute pairwise IoU
        merge_pairs = []
        ids = list(detected_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                oid_a, oid_b = ids[i], ids[j]
                ious = []
                for fidx in sample_idxs:
                    mask_a = sampled[fidx].get(oid_a)
                    mask_b = sampled[fidx].get(oid_b)
                    if mask_a is None or mask_b is None:
                        continue
                    ma = (mask_a[0] if mask_a.ndim == 3 else mask_a) > 0
                    mb = (mask_b[0] if mask_b.ndim == 3 else mask_b) > 0
                    intersection = float((ma & mb).sum())
                    union = float((ma | mb).sum())
                    if union > 0:
                        ious.append(intersection / union)
                if ious and np.mean(ious) > iou_thresh:
                    merge_pairs.append((oid_a, oid_b, np.mean(ious)))

        if not merge_pairs:
            return video_segments, detected_ids

        del sampled  # free the sampled frames before the per-frame merge rewrites

        # Merge: absorb smaller into larger
        absorbed = set()
        for oid_a, oid_b, avg_iou in merge_pairs:
            if oid_a in absorbed or oid_b in absorbed:
                continue
            # Keep the one with larger average area
            if id_avg_area[oid_a] >= id_avg_area[oid_b]:
                keep, drop = oid_a, oid_b
            else:
                keep, drop = oid_b, oid_a

            self._log_info(
                f"[{cam_name}] Merging duplicate tracklets: {drop} -> {keep} "
                f"(avg IoU={avg_iou:.3f}, areas: {keep}={id_avg_area[keep]:.0f}, "
                f"{drop}={id_avg_area[drop]:.0f})"
            )

            # Transfer the dropped ID's masks where only it exists, keep the
            # larger mask where both exist, then remove the dropped ID. Done as a
            # streaming per-frame rewrite in the store.
            video_segments.merge_obj(drop=drop, keep=keep)
            absorbed.add(drop)

        # Update detected_ids
        detected_ids = [oid for oid in detected_ids if oid not in absorbed]

        if absorbed:
            self._log_info(
                f"[{cam_name}] Merged {len(absorbed)} duplicate tracklet(s). "
                f"Remaining IDs: {detected_ids}"
            )

        return video_segments, detected_ids

    def _resolve_relatch_culprit_by_jump(self, centroid_series, oid_a, oid_b,
                                         run_s, pre_frames,
                                         jump_ratio=2.0, min_jump_px=60.0):
        """Pick the intruder of a SUSTAINED near-identical run when neither
        tracklet was absent just before it (so the area/absence test is
        ambiguous). At the run start the intruder's centroid jumps onto the
        other tracklet — a discontinuity from its OWN pre-run trajectory — while
        the real person's centroid moves continuously. Returns the tracklet that
        jumped, or None when neither jump is clearly larger (both continuous,
        both jumped, or missing data), keeping the caller's abstain behaviour.

        ``min_jump_px`` is an absolute-pixels floor; the caller derives it from
        the image diagonal (``relatch_min_jump_ratio``) so the resolver
        triggers identically at any resolution. The 60.0 default only backstops
        direct calls that pass no floor.

        area-based inflation does NOT work here: during such a run both masks are
        near-identical, so both areas inflate together; the boundary jump is the
        signal that distinguishes them.
        """
        def jump(oid):
            seq = centroid_series[oid]
            ref = None
            for p in range(run_s - 1, max(-1, run_s - 1 - pre_frames), -1):
                if 0 <= p < len(seq) and seq[p] is not None:
                    ref = seq[p]
                    break
            cur = seq[run_s] if 0 <= run_s < len(seq) else None
            if ref is None or cur is None:
                return None
            return float(np.hypot(cur[0] - ref[0], cur[1] - ref[1]))

        ja, jb = jump(oid_a), jump(oid_b)
        if ja is None or jb is None:
            return None
        (hi, lo, culprit) = (ja, jb, oid_a) if ja >= jb else (jb, ja, oid_b)
        if hi >= min_jump_px and hi >= jump_ratio * max(lo, 1.0):
            return culprit
        return None

    def _state_frame_rgb(self, inference_state, frame_idx):
        """Reconstruct frame ``frame_idx`` as an RGB uint8 array at video
        resolution from the predictor's own frame store (SAM2-compatible
        backends keep ``images`` + ``video_width/height`` in the state), so
        recovery needs no separate FrameSource plumbing."""
        frame = inference_state["images"][frame_idx]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu()
        else:
            frame = torch.as_tensor(np.asarray(frame))
        frame = torch.clamp(
            frame.float() * self.img_std_sam[0] + self.img_mean_sam[0], 0, 1)
        img = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        h = int(inference_state.get("video_height") or img.shape[0])
        w = int(inference_state.get("video_width") or img.shape[1])
        if (img.shape[0], img.shape[1]) != (h, w):
            img = cv2.resize(img, (w, h))
        return img

    def _find_recovery_prompts(self, inference_state, video_segments, sorted_frames,
                               run_s, run_e, require_ids=None, min_dets=2,
                               det_stride=30, det_conf=0.5, covered_iou=0.30,
                               anchor_gap=120):
        """Inside a frame range, find frames where the detector sees a person
        NOT covered by any tracked mask: the lost person.

        Two callers: collapse recovery (``require_ids`` = the near-identical
        pair, at least one must be present so the merged blob anchors the
        coverage test; ``min_dets=2``) and lost-tracklet re-acquisition
        (``require_ids=None``: the tracklet in question is absent by
        definition; ``min_dets=1``). Sweeps at ``det_stride`` and returns a
        LIST of ``(pos, box_xyxy)`` anchors spaced >= ``anchor_gap``
        positions apart (ascending). Empty when no uncovered person is ever
        detected (e.g. fully occluded or genuinely out of frame).
        """
        step = max(1, int(det_stride))
        positions, left, right = [], run_s, run_e
        while left <= right:
            positions.append(left)
            if right != left:
                positions.append(right)
            left += step
            right -= step
        found = {}
        for pos in positions:
            fidx = sorted_frames[pos]
            img = self._state_frame_rgb(inference_state, fidx)
            _img, dets = self._collect_yolo_detections(
                Image.fromarray(img), det_conf, include_clip_features=False)
            if len(dets) < min_dets:
                continue
            masks = video_segments.frame(fidx)
            # Coverage union over EVERY tracked id on this frame: a correctly
            # tracked third person is covered by their own mask and can never
            # be mistaken for the lost person (review finding: 3+-person).
            union = None
            required_seen = require_ids is None
            for oid, m in masks.items():
                if m is None:
                    continue
                mm = (m[0] if m.ndim == 3 else m) > 0
                union = mm if union is None else (union | mm)
                required_seen = required_seen or oid in require_ids
            if union is None or not required_seen:
                continue
            best = None
            for d in dets:
                x1, y1, x2, y2 = [int(round(float(v))) for v in d[2]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(union.shape[1], x2), min(union.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                cover = float(union[y1:y2, x1:x2].sum()) / float((x2 - x1) * (y2 - y1))
                if cover < covered_iou and (best is None or cover < best[0]):
                    best = (cover, [x1, y1, x2, y2])
            if best is not None:
                found[pos] = best[1]
        if not found:
            return []
        # Greedy min-gap selection over positions, ascending: DENSE anchors.
        # One prompt per run drifts back onto the neighbour within ~100 frames
        # under partial occlusion (Stage A validation, runs 704-1632 and
        # 2102-3357: separated dancers stayed unmasked); anchors every
        # ~anchor_gap frames re-correct the propagation at each cond frame.
        chosen = []
        for pos in sorted(found):
            if not chosen or pos - chosen[-1][0] >= anchor_gap:
                chosen.append((pos, found[pos]))
        return chosen

    def _recover_collapsed_runs(self, inference_state, video_segments,
                                cam_name="SAM", image_size=None):
        """Re-acquire a person lost to a tracklet collapse (opt-in Step 0 of
        post-processing; ``masks.recover_collapsed_tracklets``).

        For each sustained near-identical run: re-detect people inside the
        run, add a box prompt for the intruder tracklet on an uncovered
        detection (SAM2-compatible predictors accept prompts on arbitrary
        frames for existing ids), and re-propagate ONLY the run's sub-range
        outward from the recovery frame: unlike a passive init anchor, which
        holds for exactly one frame, the recency memory then follows the
        recovered person. Iterates until no sustained runs remain or the budget is
        spent; surviving runs fall through to truncation / suppression.
        """
        mask_cfg = self.assignment_config.get("masks", {})
        match_cfg = self.assignment_config.get("matching", {})
        min_len = int(mask_cfg.get("recover_min_run_frames", 30))
        max_rounds = int(mask_cfg.get("recover_max_rounds", 2))
        max_prompts = int(mask_cfg.get("recover_max_prompts", 24))
        det_stride = int(mask_cfg.get("recover_detect_stride", 30))
        covered_iou = float(mask_cfg.get("recover_covered_iou", 0.30))
        anchor_gap = int(mask_cfg.get("recover_anchor_gap", 120))
        # Per-run anchor cap: measured on salsa round 3, saturating one run
        # with anchors both starved the other runs (a region a single prompt
        # had rescued got nothing and was zeroed) and did not hold anyway
        # (drift-back between anchors bridges the near-identical scan across
        # the 1-frame anchor dips, so suppression deletes the whole run,
        # anchors included). A few anchors per run, budget spread across
        # runs, is strictly better.
        per_run_cap = int(mask_cfg.get("recover_max_anchors_per_run", 3))
        det_conf = float(match_cfg.get("yolo_person_conf", 0.5))

        size = image_size or video_segments.image_size()
        diag = float((size[0] ** 2 + size[1] ** 2) ** 0.5) if size else 2000.0
        min_jump_px = float(mask_cfg.get("relatch_min_jump_ratio", 0.025)) * diag

        detected_ids = video_segments.all_obj_ids()
        if len(detected_ids) < 2:
            return video_segments

        prompts_used = 0
        for round_i in range(max_rounds):
            scan = self._scan_near_identical_runs(video_segments, detected_ids)
            runs = [r for r in (scan["runs"] if scan else [])
                    if r[3] - r[2] + 1 >= min_len]
            if not runs:
                break
            progressed = False
            for oid_a, oid_b, run_s, run_e in runs:
                if prompts_used >= max_prompts:
                    break
                sorted_frames = scan["sorted_frames"]
                anchors = self._find_recovery_prompts(
                    inference_state, video_segments, sorted_frames,
                    run_s, run_e, require_ids=(oid_a, oid_b), min_dets=2,
                    det_stride=det_stride, det_conf=det_conf,
                    covered_iou=covered_iou, anchor_gap=anchor_gap)
                if not anchors:
                    self._log_info(
                        f"[{cam_name}] Collapse recovery: no uncovered person "
                        f"detected inside run {sorted_frames[run_s]}-"
                        f"{sorted_frames[run_e]} ({oid_a}/{oid_b}); leaving it "
                        "to truncation/suppression.")
                    continue
                # Evenly-spaced subset (endpoints first) under the per-run cap
                # and the remaining global budget.
                cap = min(max(1, per_run_cap), max(1, max_prompts - prompts_used))
                if len(anchors) > cap:
                    idx = np.linspace(0, len(anchors) - 1, cap, dtype=int)
                    anchors = [anchors[i] for i in sorted(set(idx.tolist()))]
                # The tracklet to re-seed = the intruder (its person is the
                # uncovered one). Boundary-continuity evidence picks it; when
                # symmetric (gradual collapse) fall back to whichever id has
                # the LARGER average boundary jump (with one camera the
                # assignment stays ambiguous either way and the cross-camera
                # remap re-decides identity downstream).
                culprits, _keeper, detail = self._pick_duplicate_suppression(
                    scan["centroid_series"], oid_a, oid_b, run_s, run_e,
                    pre_frames=5, min_jump_px=min_jump_px)

                def _avg(oid):
                    js = self._duplicate_boundary_jump(
                        scan["centroid_series"], oid, run_s, 5, -1)
                    je = self._duplicate_boundary_jump(
                        scan["centroid_series"], oid, run_e, 5, +1)
                    vals = [v for v in (js, je) if v is not None]
                    return (sum(vals) / len(vals)) if vals else -1.0

                ja, jb = _avg(oid_a), _avg(oid_b)
                if max(ja, jb) < min_jump_px and len(detected_ids) > 2:
                    # No takeover-scale boundary jump, and MORE than two
                    # people tracked: a wrong re-seed here could leak a
                    # third person's identity into the pair, so abstain.
                    # With exactly two tracked people the exemption below
                    # applies: a verified maskless person plus a stacked
                    # pair is a certain collapse (far-from-seed drift shows
                    # no jump by construction), and the worst wrong pick is
                    # a repairable swap, never a leak onto a bystander.
                    self._log_info(
                        f"[{cam_name}] Collapse recovery: run "
                        f"{sorted_frames[run_s]}-{sorted_frames[run_e]} has "
                        f"no takeover jump (max {max(ja, jb):.0f}px < "
                        f"{min_jump_px:.0f}px) with {len(detected_ids)} "
                        "people tracked; treating as genuine convergence, "
                        "not re-seeding.")
                    continue
                if len(culprits) == 1:
                    lost = culprits[0]
                else:
                    lost = oid_a if ja > jb else oid_b
                anchor_frames = [sorted_frames[p] for p, _ in anchors]
                self._log_info(
                    f"[{cam_name}] Collapse recovery: re-seeding tracklet {lost} "
                    f"with {len(anchors)} anchor(s) at frames {anchor_frames} "
                    f"({detail}) and re-propagating frames "
                    f"{sorted_frames[run_s]}-{sorted_frames[run_e]}.")
                placed = self._apply_recovery_prompts(
                    inference_state, video_segments, sorted_frames,
                    lost, anchors, run_s, run_e, cam_name)
                if placed == 0:
                    continue
                prompts_used += placed
                progressed = True
            if not progressed:
                break

        return video_segments

    def _apply_recovery_prompts(self, inference_state, video_segments,
                                sorted_frames, lost, anchors, run_s, run_e,
                                cam_name):
        """Shared tail of both recovery triggers: make the prompts CONDITIONING
        frames and re-propagate the run. Returns how many prompts were placed.

        SAM2 semantics require the prompt to become a conditioning frame, or
        the first re-propagated step recomputes the prompt frame without the
        click and destroys it (a correction on an already-tracked frame is
        non-cond under the default add_all_frames_to_correct_as_cond=False).
        Clearing the lost object's tracked flags inside the run makes the
        prompt an init cond frame; its non-cond memory is cleared over the
        WHOLE video, not just the run. Memory attention draws on recent
        non-cond frames and object pointers from before the run boundary,
        and entries recorded while the tracklet sat on the wrong person
        out-vote a single fresh prompt within a frame or two. Conditioning
        frames (the original seed and earlier recovery anchors) survive and
        carry the identity."""
        try:
            obj_idx = self.predictor._obj_id_to_idx(inference_state, lost)
            tracked = inference_state["frames_tracked_per_obj"][obj_idx]
            ncond = inference_state["output_dict_per_obj"][obj_idx][
                "non_cond_frame_outputs"]
            for f in range(sorted_frames[run_s], sorted_frames[run_e] + 1):
                tracked.pop(f, None)
            ncond.clear()
        except (AttributeError, KeyError, TypeError) as e:
            # Non-SAM2-shaped state (sam3 tracker): the prompt still applies;
            # re-propagation is then best-effort.
            self._log_info(
                f"[{cam_name}] Recovery: could not reset the run's memory for "
                f"tracklet {lost} ({e}); proceeding with the prompt only.")
        placed = 0
        for p, box in anchors:
            try:
                self.predictor.add_new_points_or_box(
                    inference_state, frame_idx=sorted_frames[p],
                    obj_id=lost, box=np.asarray(box, dtype=np.float32))
                placed += 1
            except Exception as e:  # noqa: BLE001
                self._log_warn(
                    f"[{cam_name}] Recovery prompt failed at frame "
                    f"{sorted_frames[p]}: {e}")
        if placed == 0:
            return 0
        # Re-propagate once across the run from the FIRST anchor; the later
        # anchors are cond frames along the way and re-correct drift as the
        # pass reaches them. A margin past the run bounds sweeps the argmax
        # writeback through the seam frames just OUTSIDE the detected run
        # (near-identical stretches sitting under the scan threshold between
        # two recovery rounds would otherwise keep their overlap).
        margin = 15
        fidx = sorted_frames[anchors[0][0]]
        span_fw = sorted_frames[run_e] - fidx + margin
        span_bw = fidx - sorted_frames[run_s] + margin
        with torch.inference_mode():
            for rev, span in ((False, span_fw), (True, span_bw)):
                if span <= 0:
                    continue
                for f2, oids2, logits2 in self.predictor.propagate_in_video(
                        inference_state, start_frame_idx=fidx,
                        max_frame_num_to_track=span, reverse=rev):
                    video_segments.set_frame(
                        f2, _argmax_disjoint_masks(oids2, logits2))
        flags, scanned = _collapse_flags(
            video_segments, sorted_frames, anchors, run_s, run_e, margin)
        if scanned and len({i for i, _o in flags}) > 0.25 * scanned:
            zeroed = _zero_flagged_masks(video_segments, sorted_frames, flags)
            relabeled, blanked = _unify_failed_run_labels(
                video_segments, sorted_frames, run_s, run_e, margin)
            self._log_info(
                f"[{cam_name}] Recovery did not hold for tracklet {lost} in "
                f"frames {sorted_frames[run_s]}-{sorted_frames[run_e]}; "
                f"zeroed {zeroed} misplaced mask(s), unified {relabeled} "
                f"alternating label(s), blanked {blanked} unanchored one(s): "
                f"absent rather than on the wrong person.")
        return placed


    def _scan_near_identical_runs(self, video_segments, detected_ids,
                                  iou_thresh=0.9, min_run_frames=5,
                                  pre_frames=5, sample_count=20):
        """Find sustained near-identical (IoU >= ``iou_thresh``) runs between
        tracklet pairs: the shared detector behind re-latch truncation,
        duplicate suppression, and collapse recovery.

        Cheap prescreen on sampled frames first: only pairs that peak at
        >= iou_thresh anywhere get the full per-frame streaming pass, so
        clean sequences pay nothing beyond the sampling cost. Returns None
        when nothing is flagged, else a dict with ``sorted_frames``,
        ``runs`` as ``[(oid_a, oid_b, run_s, run_e)]`` (run bounds are
        POSITIONS into sorted_frames, inclusive, brief dips bridged),
        ``area_series`` and ``centroid_series`` for the involved ids.
        """
        if len(detected_ids) < 2:
            return None
        sorted_frames = video_segments.frames()
        n_sample = min(sample_count, len(sorted_frames))
        if n_sample == 0:
            return None
        sample_idxs = [sorted_frames[int(i)] for i in
                       np.linspace(0, len(sorted_frames) - 1, n_sample, dtype=int)]

        def _pair_iou(masks, oid_a, oid_b):
            mask_a, mask_b = masks.get(oid_a), masks.get(oid_b)
            if mask_a is None or mask_b is None:
                return 0.0
            ma = (mask_a[0] if mask_a.ndim == 3 else mask_a) > 0
            mb = (mask_b[0] if mask_b.ndim == 3 else mask_b) > 0
            union = float((ma | mb).sum())
            return float((ma & mb).sum()) / union if union > 0 else 0.0

        ids = list(detected_ids)
        flagged = []
        for fidx in sample_idxs:
            masks = video_segments.frame(fidx)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pair = (ids[i], ids[j])
                    if pair in flagged:
                        continue
                    if _pair_iou(masks, *pair) >= iou_thresh:
                        flagged.append(pair)
        if not flagged:
            return None

        # One streaming pass over all frames: per-frame IoU series for every
        # flagged pair plus area/centroid series for the involved IDs.
        involved = sorted({oid for pair in flagged for oid in pair})
        iou_series = {pair: [] for pair in flagged}
        area_series = {oid: [] for oid in involved}
        centroid_series = {oid: [] for oid in involved}
        for fidx in sorted_frames:
            masks = video_segments.frame(fidx)
            for oid in involved:
                mask = masks.get(oid)
                if mask is None:
                    area_series[oid].append(0)
                    centroid_series[oid].append(None)
                else:
                    m = (mask[0] if mask.ndim == 3 else mask) > 0
                    area_series[oid].append(int(m.sum()))
                    ys, xs = np.where(m)
                    centroid_series[oid].append(
                        (float(xs.mean()), float(ys.mean())) if xs.size else None)
            for pair in flagged:
                iou_series[pair].append(_pair_iou(masks, *pair))

        all_runs = []
        for oid_a, oid_b in flagged:
            series = iou_series[(oid_a, oid_b)]
            # Maximal runs of positions with IoU >= iou_thresh.
            runs, start = [], None
            for pos, v in enumerate(series + [0.0]):
                if v >= iou_thresh and start is None:
                    start = pos
                elif v < iou_thresh and start is not None:
                    if pos - start >= min_run_frames:
                        runs.append((start, pos - 1))
                    start = None
            # Bridge brief dips: runs separated by fewer than pre_frames
            # positions are one takeover event — such a gap is too short to be
            # a genuine recovery, and too short to re-establish pre-run
            # evidence for the follow-up run (which would otherwise be skipped
            # as ambiguous, leaving its frames poisoned).
            bridged = []
            for run in runs:
                if bridged and run[0] - bridged[-1][1] - 1 < pre_frames:
                    bridged[-1] = (bridged[-1][0], run[1])
                else:
                    bridged.append(run)
            all_runs += [(oid_a, oid_b, s, e) for s, e in bridged]
        return {"sorted_frames": sorted_frames, "runs": all_runs,
                "area_series": area_series, "centroid_series": centroid_series}

    def _duplicate_boundary_jump(self, centroid_series, oid, pos, pre_frames, step):
        """Centroid discontinuity of ``oid`` at a run boundary: distance from
        its centroid AT position ``pos`` to its nearest present centroid up
        to ``pre_frames`` positions away in the ``step`` direction (-1 looks
        before the run start, +1 after the run end). None when either side
        has no centroid."""
        seq = centroid_series[oid]
        cur = seq[pos] if 0 <= pos < len(seq) else None
        if cur is None:
            return None
        for k in range(1, pre_frames + 1):
            p = pos + step * k
            if 0 <= p < len(seq) and seq[p] is not None:
                ref = seq[p]
                return float(np.hypot(cur[0] - ref[0], cur[1] - ref[1]))
        return None

    def _pick_duplicate_suppression(self, centroid_series, oid_a, oid_b,
                                    run_s, run_e, pre_frames, min_jump_px,
                                    ratio=1.25):
        """A near-identical run is about to be exported as TWO identical
        masks on one person; decide which tracklet(s) to zero after the
        strict jump resolver abstained. Combines the discontinuity evidence
        at BOTH boundaries (the intruder jumps onto the victim at the run
        start and back off at the run end; the victim's own tracklet stays
        continuous). Returns ``(culprits, keeper, detail)``: one culprit
        when a side is clearly less continuous, else BOTH (a gradual
        full-contact convergence carries no identity information, so the
        honest output is no mask: downstream triangulates from clean
        cameras; duplicates would poison ma_2d/ma_3d)."""
        def boundary_jumps(oid):
            return (self._duplicate_boundary_jump(centroid_series, oid, run_s, pre_frames, -1),
                    self._duplicate_boundary_jump(centroid_series, oid, run_e, pre_frames, +1))
        (jsa, jea), (jsb, jeb) = boundary_jumps(oid_a), boundary_jumps(oid_b)
        # Compare only boundaries measured for BOTH tracklets; averaging a
        # one-boundary mean against a two-boundary mean would bias the pick.
        pairs = [(x, y) for x, y in ((jsa, jsb), (jea, jeb))
                 if x is not None and y is not None]
        ja = sum(p[0] for p in pairs) / len(pairs) if pairs else None
        jb = sum(p[1] for p in pairs) / len(pairs) if pairs else None
        detail = f"boundary jumps {oid_a}:{'-' if ja is None else f'{ja:.0f}px'} " \
                 f"{oid_b}:{'-' if jb is None else f'{jb:.0f}px'}"
        if ja is not None and jb is not None:
            hi, lo, culprit = (ja, jb, oid_a) if ja >= jb else (jb, ja, oid_b)
            if hi >= 0.5 * min_jump_px and hi >= ratio * max(lo, 1.0):
                return [culprit], (oid_b if culprit == oid_a else oid_a), detail
        return [oid_a, oid_b], None, detail

    def _truncate_relatched_tracklets(self, cam_name, video_segments, detected_ids,
                                      iou_thresh=0.9, min_run_frames=5,
                                      pre_frames=5, pre_area_px=100,
                                      sample_count=20, resolve_by_jump=False,
                                      min_jump_ratio=0.025, image_size=None):
        """Zero the frames on which a tracklet re-latched onto another person.

        When a tracked person leaves the scene, the tracker's memory can
        re-acquire "its" object on the most similar visible person a few frames
        later: the tracklet is a real, distinct person early on and a
        near-perfect duplicate (IoU ~1) of another tracklet from the takeover
        frame onward. _merge_duplicate_tracklets cannot catch this: its
        whole-clip mean IoU is diluted by the legitimate early segment
        (measured 0.74 on a real re-latch vs the 0.95 merge threshold).

        Runs come from _scan_near_identical_runs. The culprit is the tracklet
        that was empty/tiny (area <= pre_area_px) on every one of the
        pre_frames frames just before the run while the other stayed
        substantial; with ``resolve_by_jump`` (the mask-continuity-gate
        family, off by default) an ambiguous run is next resolved by the
        run-boundary centroid jump, and failing that the run is
        duplicate-SUPPRESSED (_pick_duplicate_suppression): the weaker-
        continuity tracklet, or both when symmetric, is zeroed so two
        identical masks are never exported. With everything off (the
        committed default) ambiguous runs are logged and left untouched:
        a logged miss is recoverable, a wrong truncation is not.
        """
        scan = self._scan_near_identical_runs(
            video_segments, detected_ids, iou_thresh=iou_thresh,
            min_run_frames=min_run_frames, pre_frames=pre_frames,
            sample_count=sample_count)
        if not scan:
            return video_segments
        sorted_frames = scan["sorted_frames"]
        area_series = scan["area_series"]
        centroid_series = scan["centroid_series"]

        # Resolution-agnostic floor for the opt-in run-boundary jump resolver:
        # a fraction of the image diagonal, the same convention as the mask-
        # continuity gate (0.025 * diag is ~64 px at 2056x1504, matching the
        # previously fixed 60 px at that scale).
        min_jump_px = 60.0
        if resolve_by_jump:
            size = image_size or video_segments.image_size()
            diag = float((size[0] ** 2 + size[1] ** 2) ** 0.5) if size else 2000.0
            min_jump_px = float(min_jump_ratio) * diag

        for oid_a, oid_b, run_s, run_e in scan["runs"]:
            f_s, f_e = sorted_frames[run_s], sorted_frames[run_e]
            if run_s < pre_frames:
                # No pre-run evidence exists at the clip start. Historically
                # left to the duplicate merge, but a partial-duration run
                # never reaches the merge's whole-clip mean-IoU bar (observed:
                # frames 0-464 shipped as duplicates). With
                # the gate on, suppress using END-boundary evidence only
                # (_pick_duplicate_suppression handles a missing start jump).
                if not resolve_by_jump:
                    self._log_warn(
                        f"[{cam_name}] Tracklets {oid_a}/{oid_b} are near-identical "
                        f"(IoU>={iou_thresh}) from frame {f_s} to {f_e} but the run starts "
                        f"at the clip start — leaving it to the duplicate merge, not truncating."
                    )
                    continue
                culprits, keeper, detail = self._pick_duplicate_suppression(
                    centroid_series, oid_a, oid_b, run_s, run_e,
                    pre_frames, min_jump_px)
                reason = "duplicate-suppressed"
                self._log_info(
                    f"[{cam_name}] Duplicate suppression (clip-start run): tracklets "
                    f"{oid_a}/{oid_b} are near-identical (IoU>={iou_thresh}) on frames "
                    f"{f_s}-{f_e} ({detail}) — zeroing "
                    f"{'tracklet ' + str(culprits[0]) if len(culprits) == 1 else 'BOTH tracklets'} "
                    f"there so identical duplicate masks are never exported."
                )
                for culprit in culprits:
                    n_zeroed = 0
                    for pos in range(run_s, run_e + 1):
                        fidx = sorted_frames[pos]
                        masks = video_segments.frame(fidx)
                        mask = masks.get(culprit)
                        if mask is None or not mask.any():
                            continue
                        masks[culprit] = np.zeros_like(mask)
                        video_segments.set_frame(fidx, masks)
                        n_zeroed += 1
                    self._log_info(
                        f"[{cam_name}] Re-latch truncation ({reason}): tracklet {culprit} "
                        f"zeroed on {n_zeroed} frame(s) of the near-identical run {f_s}-{f_e}"
                        f"{'' if keeper is None else f' (kept tracklet {keeper})'}. "
                        "Frames outside the run are untouched."
                    )
                continue
            pre_a = area_series[oid_a][run_s - pre_frames:run_s]
            pre_b = area_series[oid_b][run_s - pre_frames:run_s]
            a_gone = max(pre_a) <= pre_area_px
            b_gone = max(pre_b) <= pre_area_px
            a_solid = min(pre_a) > 10 * pre_area_px
            b_solid = min(pre_b) > 10 * pre_area_px
            if a_gone and b_solid:
                culprits, keeper, reason = [oid_a], oid_b, "absent-then-reacquired"
            elif b_gone and a_solid:
                culprits, keeper, reason = [oid_b], oid_a, "absent-then-reacquired"
            else:
                # Neither was clearly absent (both present as distinct people
                # before the run). Opt-in (with the mask-continuity gate):
                # first the strict run-boundary jump resolver; if it abstains,
                # duplicate suppression (never export two identical masks).
                # Off by default -> unchanged / warn only.
                culprit = (self._resolve_relatch_culprit_by_jump(
                               centroid_series, oid_a, oid_b, run_s, pre_frames,
                               min_jump_px=min_jump_px)
                           if resolve_by_jump else None)
                if culprit is not None:
                    keeper = oid_b if culprit == oid_a else oid_a
                    culprits, reason = [culprit], "jump-resolved"
                    self._log_info(
                        f"[{cam_name}] Re-latch (jump-resolved): tracklet {culprit} jumped onto "
                        f"{keeper} at the start of the near-identical run {f_s}-{f_e} "
                        f"(IoU>={iou_thresh}) while {keeper} stayed continuous."
                    )
                elif resolve_by_jump:
                    culprits, keeper, detail = self._pick_duplicate_suppression(
                        centroid_series, oid_a, oid_b, run_s, run_e,
                        pre_frames, min_jump_px)
                    reason = "duplicate-suppressed"
                    self._log_info(
                        f"[{cam_name}] Duplicate suppression: tracklets {oid_a}/{oid_b} are "
                        f"near-identical (IoU>={iou_thresh}) on frames {f_s}-{f_e} with no "
                        f"clear intruder ({detail}) — zeroing "
                        f"{'tracklet ' + str(culprits[0]) if len(culprits) == 1 else 'BOTH tracklets'} "
                        f"there so identical duplicate masks are never exported."
                    )
                else:
                    self._log_warn(
                        f"[{cam_name}] Tracklets {oid_a}/{oid_b} are near-identical "
                        f"(IoU>={iou_thresh}) from frame {f_s} to {f_e}, but neither was "
                        f"clearly absent nor clearly jumped at the run start (pre-run areas "
                        f"{oid_a}: {pre_a}, {oid_b}: {pre_b}) — not truncating."
                    )
                    continue
            for culprit in culprits:
                n_zeroed = 0
                for pos in range(run_s, run_e + 1):
                    fidx = sorted_frames[pos]
                    masks = video_segments.frame(fidx)
                    mask = masks.get(culprit)
                    if mask is None or not mask.any():
                        continue
                    masks[culprit] = np.zeros_like(mask)
                    video_segments.set_frame(fidx, masks)
                    n_zeroed += 1
                self._log_info(
                    f"[{cam_name}] Re-latch truncation ({reason}): tracklet {culprit} "
                    f"zeroed on {n_zeroed} frame(s) of the near-identical run {f_s}-{f_e}"
                    f"{'' if keeper is None else f' (kept tracklet {keeper})'}. "
                    "Frames outside the run are untouched."
                )
        return video_segments

    def _sam3_prompt_detect_and_propagate(self, cam_name, sam_source, n_frames,
                                          frame_id=None, expected_subjects=None, frames=None):
        """Run SAM3 text prompt 'person' with prompt-frame selection and propagation.

        Flow:
        1. Start a SAM3 session on the current video or frame directory.
        2. Evaluate a small set of candidate prompt frames.
        3. Re-apply the text prompt on the selected frame.
        4. Propagate masks through the clip.
        5. Remove tracks that remain tiny on most frames.

        Returns:
            (detected_ids, video_segments, best_frame) or (None, None, None) on failure.
        """
        predictor = self._sam3_prompt_predictor()
        if predictor is None:
            return None, None, None

        # Honor the same offload contract as every other backend: without it
        # the session uploads the whole fp16 video tensor to VRAM (~30 GB at
        # 4.9k frames), the dominant term behind sam3_prompt long-clip OOMs.
        _offload = bool((self.assignment_config.get("sam") or {})
                        .get("offload_video_to_cpu", True))
        self._log_info(
            f"[{cam_name}] Starting SAM3 prompt session on '{sam_source}' "
            f"(offload_video_to_cpu={_offload}).")
        predictor.start_session(sam_source, offload_video_to_cpu=_offload)

        min_area = self._sam3_prompt_min_mask_area(frames)
        best_frame, _ = self._sam3_prompt_select_frame(
            predictor,
            cam_name,
            n_frames,
            frame_id=frame_id,
            expected_subjects=expected_subjects,
            min_area=min_area,
        )
        if best_frame is None:
            self._log_warn(f"[{cam_name}] SAM3 text prompt found no usable candidate frames.")
            predictor.close_session()
            return None, None, None

        predictor.reset_session()
        self._log_info(f"[{cam_name}] Adding text prompt 'person' on frame {best_frame}.")
        best_result = predictor.add_text_prompt(frame_idx=best_frame, text="person")
        detected_ids = best_result["obj_ids"]
        if not detected_ids:
            self._log_warn(f"[{cam_name}] SAM3 text prompt detected no people on selected frame {best_frame}.")
            predictor.close_session()
            return None, None, None
        self._log_info(
            f"[{cam_name}] SAM3 detected {len(detected_ids)} people on frame {best_frame} "
            f"(IDs: {detected_ids})."
        )

        # Propagate — stream each frame's masks straight into the disk-backed
        # store instead of returning one big in-RAM dict.
        # Optionally bound SAM3's per-frame output cache (cached_frame_outputs),
        # which otherwise grows ~5x faster than the sam3 tracker path and OOMs on
        # long clips. Opt-in via sam3.yaml: sam.sam3_prune_cache_window (frames).
        sam_cfg = self.assignment_config.get("sam") or {}
        prune_w = sam_cfg.get("sam3_prune_cache_window", None)
        prune_w = int(prune_w) if prune_w not in (None, "", 0) else None
        self._log_info(
            f"[{cam_name}] Propagating SAM3 text prompt through video"
            + (f" (cache prune window={prune_w}, keep prompt frame {best_frame})." if prune_w else ".")
        )
        video_segments = MaskStore()
        try:
            predictor.propagate(sink=video_segments.set_frame,
                                prune_cache_window=prune_w, keep_frames=(best_frame,))
            self._log_info(f"[{cam_name}] Propagation complete. {len(video_segments)} frame segments.")
            predictor.close_session()

            if not video_segments:
                self._log_warn(f"[{cam_name}] Propagation returned empty segments.")
                video_segments.close()
                return None, None, None

            # Unified post-processing: merge duplicates + discard tiny tracklets
            image_size = frames.image_size if frames is not None and len(frames) > 0 else None
            video_segments, detected_ids = self._postprocess_tracklets(
                cam_name, video_segments, detected_ids, image_size=image_size,
            )
        except Exception:
            # Don't leak the spilled store if propagation/post-processing raises
            # (close() is idempotent).
            video_segments.close()
            raise

        return detected_ids, video_segments, best_frame

    def _build_masks_from_remap(self, frames, video_segments, id_remap, output_path):
        """Build save_masks dict, write per-frame mask PNGs and overlay images.

        Args:
            frames: FrameSource for reading frame images.
            video_segments: {frame_idx: {sam3_id: mask_array}} from propagation.
            id_remap: {sam3_id: output_id} mapping.
            output_path: Directory for mask PNGs and overlays.

        Returns:
            save_masks dict compatible with the rest of the pipeline.
        """
        n_frames = len(frames)
        export_cfg = self.assignment_config.get("exports", {})
        skip_overlays = bool(export_cfg.get("skip_masked_outputs", False))

        vis_frame_stride = max(1, n_frames // 5)
        mask_folder = os.path.join(output_path, "masks")
        os.makedirs(mask_folder, exist_ok=True)
        masked_outputs = None
        if not skip_overlays:
            masked_outputs = os.path.join(output_path, "masked_outputs")
            os.makedirs(masked_outputs, exist_ok=True)

        out_ids = sorted(set(id_remap.values()))
        save_masks = {}
        for out_id in out_ids:
            save_masks[out_id] = {"img": [], "mask": [], "frame": [], "iou": [], "bbox": []}

        self._log_info(
            f"Saving masks{'' if skip_overlays else ' and overlays'} to '{output_path}' "
            f"(frames={len(video_segments)}, obj_ids={out_ids})."
        )

        from concurrent.futures import ThreadPoolExecutor
        import threading

        # Thread-safe collection for save_masks samples
        save_masks_lock = threading.Lock()

        def _process_one_frame(frame_idx):
            """Process a single frame: read, overlay, write masks."""
            if frame_idx < 0 or frame_idx >= len(frames):
                self._log_warn(
                    f"Skipping SAM3 overlay export for frame {frame_idx}: "
                    f"available frame range is [0, {len(frames)})"
                )
                return

            try:
                frame_rgb = frames.read_rgb(frame_idx)
            except Exception as exc:
                self._log_warn(f"Failed to read frame {frame_idx} while building masks: {exc}")
                return

            masked_img = frame_rgb.copy().astype(np.float32) if not skip_overlays else None
            frame_samples = []  # collect samples for save_masks

            for sam3_id, mask in video_segments[frame_idx].items():
                if sam3_id not in id_remap:
                    continue
                out_id = id_remap[sam3_id]

                if mask.ndim == 3:
                    mask = mask[0]
                mask_bool = mask > 0
                mask_uint8 = mask_bool.astype(np.uint8) * 255

                # Colored overlay
                if not skip_overlays:
                    colored = show_mask_cv2(mask_bool[np.newaxis], obj_id=out_id)[:, :, :3]
                    masked_img = masked_img + colored * 0.5

                # Per-frame mask PNG (primary output; fail loudly on a bad write)
                _png = os.path.join(mask_folder, f"mask_{frame_idx:04d}_{(out_id + 1):02d}.png")
                if not cv2.imwrite(_png, mask_uint8):
                    raise IOError(f"failed to write mask PNG: {_png}")

                # Visualization samples at regular intervals
                if frame_idx % vis_frame_stride == 0:
                    ys, xs = np.where(mask_bool)
                    bbox = (np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
                            if len(xs) > 0 else np.zeros(4, dtype=np.float32))
                    frame_samples.append((out_id, frame_rgb, mask_uint8, frame_idx, bbox))

            # Save overlay frame (downscaled for disk space)
            if not skip_overlays:
                h, w = frame_rgb.shape[:2]
                overlay = np.clip(masked_img, 0, 255).astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(
                    os.path.join(masked_outputs, f"frame_{frame_idx:04d}.jpg"),
                    cv2.resize(overlay, (max(1, w // 4), max(1, h // 4))),
                )

            # Collect samples thread-safely
            if frame_samples:
                with save_masks_lock:
                    for out_id, fr, mu, fi, bb in frame_samples:
                        save_masks[out_id]["img"].append(fr)
                        save_masks[out_id]["mask"].append(mu)
                        save_masks[out_id]["frame"].append(fi)
                        save_masks[out_id]["iou"].append(1.0)
                        save_masks[out_id]["bbox"].append(bb)

        sorted_frames = sorted(video_segments.keys())
        with ThreadPoolExecutor(max_workers=6) as executor:
            list(tqdm.tqdm(
                executor.map(_process_one_frame, sorted_frames),
                total=len(sorted_frames), desc="building masks",
            ))

        _atomic_np_save(os.path.join(output_path, "masks.npy"), save_masks)
        self._log_info(f"Saved mask cache: {os.path.join(output_path, 'masks.npy')}")

        # Export overlay images to MP4 and clean up
        if not skip_overlays and bool(export_cfg.get("export_masked_outputs_mp4", True)):
            self._export_image_dir_to_mp4(
                masked_outputs,
                fps=float(export_cfg.get("masked_outputs_fps", 30.0)),
                video_filename="masked_outputs.mp4",
                remove_images=bool(export_cfg.get("remove_masked_outputs_images", True)),
            )

        return save_masks

    def _sam3_prompt_save_init_overview(self, frames, video_segments, id_remap,
                                        best_frame, output_path, cam_name):
        """Save an overview image showing all detected people on the init frame."""
        try:
            init_dir = os.path.join(output_path, "initialize")
            os.makedirs(init_dir, exist_ok=True)

            frame_rgb = frames.read_rgb(best_frame)
            n_people = len(id_remap)
            fig, axes = plt.subplots(1, n_people + 1, figsize=(4 * (n_people + 1), 4))
            if n_people == 0:
                return

            # Full frame with all masks overlaid
            overlay = frame_rgb.copy().astype(np.float32)
            for sam3_id, out_id in sorted(id_remap.items(), key=lambda x: x[1]):
                mask = video_segments.get(best_frame, {}).get(sam3_id)
                if mask is None:
                    continue
                if mask.ndim == 3:
                    mask = mask[0]
                colored = show_mask_cv2(mask[np.newaxis] > 0, obj_id=out_id)[:, :, :3]
                overlay = overlay + colored * 0.5
                # Draw ID label at mask centroid
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    axes[0].text(cx, cy, str(out_id), color='white', fontsize=14,
                                fontweight='bold', ha='center', va='center',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor=self._id_color(out_id), alpha=0.8))

            axes[0].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
            axes[0].set_title(f"[{cam_name}] frame {best_frame} — {n_people} people")
            axes[0].axis("off")

            # Per-person crops
            for idx, (sam3_id, out_id) in enumerate(sorted(id_remap.items(), key=lambda x: x[1])):
                ax = axes[idx + 1]
                mask = video_segments.get(best_frame, {}).get(sam3_id)
                if mask is None:
                    ax.text(0.5, 0.5, "No mask", ha="center", va="center")
                    ax.set_title(f"Person {out_id}")
                    ax.axis("off")
                    continue
                if mask.ndim == 3:
                    mask = mask[0]
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                    crop = frame_rgb[y1:y2, x1:x2]
                    ax.imshow(crop)
                else:
                    ax.text(0.5, 0.5, "Empty mask", ha="center", va="center")
                ax.set_title(f"Person {out_id}")
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(init_dir, "sam3_prompt_init_overview.png"), dpi=150)
            plt.close()
            self._log_info(f"[{cam_name}] Saved init overview to '{init_dir}'.")
        except Exception as exc:
            self._log_warn(f"[{cam_name}] Failed to save init overview: {exc}")

    # ── SAM3 prompt: init camera ───────────────────────────────────────────

    def process_first_video_sam3_prompt(self, frames, frame_id, output_path, expected_subjects=None):
        """Process the init camera using SAM3 text prompt 'person'.

        No YOLO needed. SAM3 auto-detects all person instances and tracks them.
        IDs are remapped to 0-based consecutive integers.
        """
        if os.path.exists(os.path.join(output_path, "masks.npy")):
            self._log_info(f"masks.npy already exists in {output_path}, skipping.")
            return np.load(os.path.join(output_path, "masks.npy"), allow_pickle=True)[()]

        os.makedirs(output_path, exist_ok=True)
        cam_name = frames.cam_name
        sam_source = self._current_sam_video_source

        detected_ids, video_segments, best_frame = self._sam3_prompt_detect_and_propagate(
            cam_name, sam_source, len(frames),
            frame_id=frame_id, expected_subjects=expected_subjects, frames=frames,
        )
        if detected_ids is None:
            return None

        # Remap SAM3 IDs to 0-based consecutive IDs
        sorted_ids = sorted(detected_ids)
        id_remap = {old: new for new, old in enumerate(sorted_ids)}

        if expected_subjects is None or expected_subjects <= 0:
            self.expected_subjects = len(sorted_ids)

        # Save initialization overview showing all detected people on the init frame
        self._sam3_prompt_save_init_overview(
            frames, video_segments, id_remap, best_frame, output_path, cam_name,
        )

        save_masks = self._build_masks_from_remap(frames, video_segments, id_remap, output_path)
        if hasattr(video_segments, "close"):
            video_segments.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return save_masks

    # ── SAM3 prompt: non-init camera ─────────────────────────────────────────

    def process_new_video_sam3_prompt(self, frames, mask_data, output_path,
                                      source_cam_data=None, target_cam_data=None,
                                      expected_subjects=None):
        """Process a non-init camera using SAM3 text prompt 'person'.

        Uses text="person" to auto-detect and track all people (no YOLO, no
        anchors). Then remaps SAM3's auto-assigned IDs to the init camera's IDs
        using CLIP feature matching + Hungarian assignment.
        """
        mask_filepath = os.path.join(output_path, "masks.npy")
        if os.path.exists(mask_filepath):
            self._log_info(f"Reusing existing masks cache: {mask_filepath}")
            return np.load(mask_filepath, allow_pickle=True)[()]

        os.makedirs(output_path, exist_ok=True)
        cam_name = frames.cam_name
        sam_source = self._current_sam_video_source

        # Step 1: Detect and propagate with text prompt "person"
        detected_ids, video_segments, best_frame = self._sam3_prompt_detect_and_propagate(
            cam_name, sam_source, len(frames), expected_subjects=expected_subjects, frames=frames,
        )
        if detected_ids is None:
            # Same contract as track-then-match: an empty truthful result, not
            # the init camera's mask_data (which would poison this camera's
            # masks.npy resume cache).
            self._log_warn(f"[{cam_name}] SAM3 text-detect found no people; writing empty masks for this camera.")
            return {}

        # Save pre-remap visualization (SAM3's raw IDs before CLIP matching)
        try:
            pre_remap_dir = os.path.join(output_path, "pre_remap")
            os.makedirs(pre_remap_dir, exist_ok=True)
            # Save a few sample frames with SAM3's original IDs
            sample_frames = sorted(video_segments.keys())
            n_samples = min(8, len(sample_frames))
            sample_indices = [sample_frames[int(i)] for i in np.linspace(0, len(sample_frames)-1, n_samples, dtype=int)]

            person_colors = [
                [1,0,0], [0,0.4,1], [0,0.8,0], [1,0.6,0], [0.8,0,0.8],
                [0,0.8,0.8], [1,0.8,0], [1,0.4,0.8], [0.4,0.2,1], [0.6,0.8,0],
            ]
            for fidx in sample_indices:
                frame_rgb = frames.read_rgb(fidx)
                overlay = frame_rgb.copy().astype(np.float32)
                for sam3_id, mask in video_segments[fidx].items():
                    if mask.ndim == 3:
                        mask = mask[0]
                    if mask.sum() == 0:
                        continue
                    color = person_colors[sam3_id % len(person_colors)]
                    for c in range(3):
                        overlay[:,:,c] = np.where(mask > 0,
                            overlay[:,:,c] * 0.5 + color[c] * 255 * 0.5,
                            overlay[:,:,c])
                    ys, xs = np.where(mask > 0)
                    if len(xs) > 0:
                        cv2.putText(overlay, str(sam3_id),
                            (int(xs.mean())-10, int(ys.mean())+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
                cv2.imwrite(
                    os.path.join(pre_remap_dir, f"pre_remap_frame_{fidx:04d}.png"),
                    np.clip(overlay, 0, 255).astype(np.uint8)[:,:,::-1],
                )
            self._log_info(f"[{cam_name}] Saved {n_samples} pre-remap visualizations to '{pre_remap_dir}'.")
        except Exception as exc:
            self._log_warn(f"[{cam_name}] Failed to save pre-remap visualization: {exc}")

        # Step 2: Match SAM3 IDs to init camera IDs via CLIP + epipolar
        id_remap = self._remap_tracklets_to_init(
            cam_name, frames, mask_data, detected_ids, video_segments, best_frame,
            source_cam_data=source_cam_data, target_cam_data=target_cam_data,
        )

        # Step 3: Build and save masks with remapped IDs
        save_masks = self._build_masks_from_remap(frames, video_segments, id_remap, output_path)
        if hasattr(video_segments, "close"):
            video_segments.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Note: we intentionally do NOT update the feature bank after non-init
        # cameras in sam3_prompt mode. The remap can be wrong (especially for
        # similar-looking people), and adding wrong features poisons the bank
        # for subsequent cameras. The init camera's features are sufficient.

        return save_masks


    def _remap_tracklets_to_init(self, cam_name, frames, mask_data,
                                 local_ids, video_segments, best_frame,
                                 source_cam_data=None, target_cam_data=None):
        """Match per-camera tracklet IDs to init-camera IDs using CLIP + epipolar.

        Backend-neutral (sam2 / sam3 / sam3_prompt_light / sam3_prompt). ``local_ids``
        are the per-camera local tracklet ids (any id space); this consumes those
        tracklets + the init-camera CLIP gallery and returns {local_id: init_id}.

        Samples person crops across multiple frames, builds weighted CLIP
        features, aggregates epipolar consistency across the same frame set,
        and uses Hungarian assignment to find the optimal ID mapping.

        Args:
            cam_name: Camera name for logging.
            frames: FrameSource for this camera.
            mask_data: Init camera mask data (for CLIP gallery + mask centroids).
            local_ids: List of per-camera local tracklet IDs (any id space).
            video_segments: {frame_idx: {local_id: mask}} from propagation.
            best_frame: Selected prompt frame for this camera.
            source_cam_data: Init camera's cam_data (for epipolar geometry).
            target_cam_data: This camera's cam_data (for epipolar geometry).

        Returns:
            Dict mapping local_id -> init_camera_id.
        """
        from scipy.optimize import linear_sum_assignment

        init_obj_ids = sorted(mask_data.keys())

        # ── Multi-frame CLIP feature extraction ────────────────────────────
        # Sample crops from multiple frames for more robust matching.
        n_frames = len(video_segments)
        cfg = self._get_matching_cfg()
        remap_clip_interval = int(cfg.get("remap_clip_interval", 10))
        n_clip_frames = max(3, n_frames // max(1, remap_clip_interval))
        sample_frame_idxs = sorted(set(
            [best_frame] +
            [int(i) for i in np.linspace(0, max(video_segments.keys()), num=n_clip_frames, dtype=int)]
        ))

        # Collect per-SAM3-ID features across frames, weighted by mask clarity.
        # Frames where a person overlaps less with others contribute more to the
        # averaged CLIP feature, reducing contamination from occluding people.
        local_all_feats = {sid: [] for sid in local_ids}   # (feat, weight) pairs
        local_centroids_by_frame = {}
        local_valid_ids = []
        image_area = float(frames.image_size[0] * frames.image_size[1]) if len(frames) > 0 else 0.0

        for fidx in sample_frame_idxs:
            if fidx not in video_segments:
                continue
            try:
                frame_rgb = frames.read_rgb(fidx)
            except Exception as exc:
                self._log_warn(f"Failed to read frame {fidx} for CLIP remap: {exc}")
                continue

            # Pre-compute bounding boxes for this frame (for overlap weighting).
            # SAM3 masks are non-overlapping at pixel level, but bounding boxes
            # can overlap heavily when people stand close together. The CLIP crop
            # uses the bbox, so bbox IoU determines how much of another person
            # "leaks" into this person's crop.
            frame_bboxes = {}
            frame_mask_areas = {}
            frame_centroids = {}
            for sid in local_ids:
                mask = video_segments[fidx].get(sid)
                if mask is None:
                    continue
                if mask.ndim == 3:
                    mask = mask[0]
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                frame_bboxes[sid] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                frame_mask_areas[sid] = self._sam3_prompt_mask_area(mask)
                centroid = self._mask_centroid_xy1(mask)
                if centroid is not None:
                    frame_centroids[sid] = centroid

            if frame_centroids:
                local_centroids_by_frame[fidx] = frame_centroids

            crops_this_frame = []
            sids_this_frame = []
            weights_this_frame = []
            for sid in local_ids:
                if sid not in frame_bboxes:
                    continue
                x1, y1, x2, y2 = frame_bboxes[sid]
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Compute max bbox IoU with any other person on this frame
                max_iou = 0.0
                for other_sid, other_box in frame_bboxes.items():
                    if other_sid == sid:
                        continue
                    iou = self._bbox_iou_xyxy(
                        [x1, y1, x2, y2], list(other_box)
                    )
                    max_iou = max(max_iou, iou)
                overlap_ratio = max_iou

                # Prefer crops that are both clean (low overlap) and prominent
                # enough to represent the tracked subject reliably.
                overlap_weight = max(0.1, 1.0 - overlap_ratio)
                size_weight = self._sam3_prompt_size_weight(
                    frame_mask_areas.get(sid, 0), image_area,
                )
                weight = overlap_weight * size_weight

                crops_this_frame.append(crop)
                sids_this_frame.append(sid)
                weights_this_frame.append(weight)

            if crops_this_frame:
                feats = self._encode_clip_batch(crops_this_frame)
                for sid, feat, w in zip(sids_this_frame, feats, weights_this_frame):
                    local_all_feats[sid].append((feat.reshape(-1), w))

        # Weighted average of CLIP features across frames for each person
        local_avg_feats = []
        for sid in local_ids:
            feat_weight_pairs = local_all_feats[sid]
            if not feat_weight_pairs:
                continue
            feats = torch.stack([f for f, _ in feat_weight_pairs])
            weights = torch.tensor([w for _, w in feat_weight_pairs], dtype=torch.float32)
            weight_sum = weights.sum()
            if weight_sum > 1e-8:
                weights = weights / weight_sum
            else:
                weights = torch.ones_like(weights) / len(weights)
            avg_feat = _weighted_unit_mean(feats, weights)
            local_avg_feats.append(avg_feat)
            local_valid_ids.append(sid)

        if not local_avg_feats:
            # No appearance evidence at all (frame reads failed or every crop
            # was empty). Guessing an identity mapping here would export masks
            # under arbitrary person ids; fail the camera so a re-run retries.
            raise CameraProcessingError(
                cam_name, "no CLIP features could be built for any tracklet; "
                          "refusing to assign identities without evidence")

        self._log_info(
            f"[{cam_name}] Multi-frame CLIP: {len(sample_frame_idxs)} frames sampled, "
            f"{len(local_valid_ids)} people with features."
        )

        # ── CLIP similarity ────────────────────────────────────────────────
        local_feats = torch.stack(local_avg_feats, dim=0)
        k, m = len(init_obj_ids), len(local_valid_ids)
        s_clip = np.zeros((k, m), dtype=np.float32)
        for row, oid in enumerate(init_obj_ids):
            gallery, _ = self._get_subject_gallery(mask_data, oid)
            if gallery is None:
                continue
            _, sims = cosine_knn(local_feats, gallery, topk=1)
            s_clip[row] = sims.squeeze(1).cpu().numpy()

        # ── Epipolar scoring (if calibration available) ────────────────────
        cfg = self._get_matching_cfg()
        clip_weight = float(cfg.get("clip_weight", 0.4))
        epipolar_weight = float(cfg.get("epipolar_weight", 0.6))
        clip_min_similarity = float(cfg.get("clip_min_similarity", 0.0))
        combined_min_score = float(cfg.get("combined_min_score", 0.2))
        image_size = frames.image_size if len(frames) > 0 else None
        epipolar_sigma_px, max_epipolar_dist_px, _ = self._resolve_epipolar_px_params(
            cfg, image_size=image_size
        )

        epi_refs = self._epipolar_reference_set(
            cam_name, source_cam_data, target_cam_data, mask_data)

        s_epi = np.zeros((k, m), dtype=np.float32)
        has_epi = np.zeros((k, m), dtype=bool)
        has_any_epi = False

        if epi_refs:
            self._log_info(
                f"[{cam_name}] Using epipolar geometry for ID remapping "
                f"({len(epi_refs)} reference camera(s): "
                f"{', '.join(r[0] for r in epi_refs)}).")
            has_any_epi = True
            for row, oid in enumerate(init_obj_ids):
                for col in range(m):
                    sid = local_valid_ids[col]
                    epi_scores = []
                    for fidx in sample_frame_idxs:
                        target_centroid = local_centroids_by_frame.get(fidx, {}).get(sid)
                        if target_centroid is None:
                            continue
                        for _ref_name, ref_f, ref_point_fn in epi_refs:
                            ref_point = ref_point_fn(oid, fidx)
                            if ref_point is None:
                                continue
                            epi_score, epi_dist = self._epipolar_score(
                                f_matrix=ref_f,
                                x1=ref_point,
                                x2=target_centroid,
                                sigma_px=epipolar_sigma_px,
                                max_dist_px=max_epipolar_dist_px,
                            )
                            if epi_score is not None:
                                epi_scores.append(epi_score)
                    if epi_scores:
                        s_epi[row, col] = float(np.median(epi_scores))
                        has_epi[row, col] = True
        else:
            if not getattr(self, "_clip_only_warned", False):
                self._log_warn(
                    f"[{cam_name}] No camera calibration (cam_int/cam_ext) — cross-camera ID "
                    "remap is CLIP-only (appearance), which is markedly less accurate. Provide "
                    "calibration to enable epipolar geometry."
                )
                self._clip_only_warned = True
            else:
                self._log_info(f"[{cam_name}] No camera calibration — using CLIP only for ID remapping.")

        # ── Combined score + Hungarian assignment ──────────────────────────
        if has_any_epi:
            combined = clip_weight * s_clip + epipolar_weight * s_epi
            # Fall back to CLIP only when epipolar could not be evaluated at all.
            # A true epipolar score of 0 should remain a penalty.
            combined[~has_epi] = s_clip[~has_epi]
        else:
            combined = s_clip

        # Matches with very weak appearance evidence are still possible, but
        # downweighted before Hungarian so stronger candidates win more easily.
        combined[s_clip < clip_min_similarity] *= 0.5

        row_ind, col_ind = linear_sum_assignment(1.0 - combined)

        id_remap = {}
        remap_scores = {}
        for row, col in zip(row_ind, col_ind):
            init_id = init_obj_ids[row]
            local_id = local_valid_ids[col]
            clip_s = float(s_clip[row, col])
            epi_s = float(s_epi[row, col]) if has_any_epi and has_epi[row, col] else 0.0
            comb_s = float(combined[row, col])
            if has_any_epi and not has_epi[row, col]:
                # This cell fell back to the raw CLIP score, which is not on
                # the same scale as the blended cells it competed against.
                self._log_warn(
                    f"[{cam_name}] Remap pair tracklet {local_id} -> {init_id} decided "
                    "without epipolar coverage (CLIP-only score against blended rivals)."
                )
            if comb_s < combined_min_score:
                self._log_info(
                    f"[{cam_name}] Rejecting weak remap candidate: tracklet {local_id} -> {init_id} "
                    f"(combined={comb_s:.3f}, CLIP={clip_s:.3f}"
                    + (f", epipolar={epi_s:.3f})" if has_any_epi and has_epi[row, col] else ")")
                )
                continue
            self._log_info(
                f"[{cam_name}] ID remap: tracklet {local_id} -> {init_id} "
                f"(combined={comb_s:.3f}, CLIP={clip_s:.3f}"
                + (f", epipolar={epi_s:.3f})" if has_any_epi and has_epi[row, col] else ")")
            )
            id_remap[local_id] = init_id
            remap_scores[local_id] = comb_s

        # Drop unmatched SAM3 IDs — only track people from the init camera.
        # Extra people detected in this camera but not in the init camera are ignored.
        n_dropped = sum(1 for sid in local_valid_ids if sid not in id_remap)
        if n_dropped > 0:
            self._log_info(
                f"[{cam_name}] Dropped {n_dropped} extra person(s) not in init camera."
            )

        # Enrich the subject feature bank with this camera's confidently
        # remapped appearance (the occlusion-weighted average feature), so
        # later cameras match against evidence from every completed view,
        # not just the init camera and the one-off bootstrap sweep.
        enrich_min = float(cfg.get("online_bank_update_min_score", 0.35))
        col_by_sid = {sid: i for i, sid in enumerate(local_valid_ids)}
        for local_id, init_id in id_remap.items():
            if remap_scores.get(local_id, 0.0) < enrich_min:
                continue
            col = col_by_sid.get(local_id)
            if col is not None:
                self._append_to_subject_bank(init_id, local_avg_feats[col])

        # Store scores so callers can decide whether to trust this remap
        self._last_remap_scores = remap_scores
        return id_remap

    def process_first_video_with_gt_bbox(self, frames, frame_id, output_path, gt_bboxes, gt_bboxes_mask, expected_subjects=None):
        """Process the init camera using ground-truth bounding boxes.

        Args:
            frames: FrameSource providing access to video/image frames.
            frame_id: Initial frame index.
            output_path: Directory for output masks.
            gt_bboxes: Ground-truth bounding boxes (Frames, Subjects, 4).
            gt_bboxes_mask: Visibility mask (Frames, Subjects).
            expected_subjects: Expected number of people (auto-detected if None).
        """
        # Use a frame_id that gt_bboxes_mask are True for all subjects
        try:
            mask_arr = np.asarray(gt_bboxes_mask)
            n_frames = len(frames)

            # Align frame axis with frames length even if input is (subjects, frames)
            if mask_arr.ndim >= 2:
                if mask_arr.shape[0] == n_frames:
                    mask_frames = mask_arr
                elif mask_arr.shape[1] == n_frames:
                    mask_frames = np.swapaxes(mask_arr, 0, 1)
                else:
                    mask_frames = mask_arr
                    self._log_warn(
                        f"GT bbox visibility shape {mask_arr.shape} does not match n_frames={n_frames}; using input as-is."
                    )
            else:
                mask_frames = mask_arr

            if mask_frames.ndim >= 2 and mask_frames.shape[0] > 0:
                fully_visible = np.where(mask_frames.all(axis=1))[0]
                if len(fully_visible) > 0:
                    if frame_id in fully_visible:
                        chosen = frame_id
                    else:
                        # pick the fully-visible frame closest to requested frame_id
                        closest_idx = int(fully_visible[np.argmin(np.abs(fully_visible - frame_id))])
                        chosen = closest_idx
                        self._log_info(f"Overriding requested init frame {frame_id} with GT-visible frame {chosen}.")
                    frame_id = int(chosen)
                else:
                    visible_counts = mask_frames.sum(axis=1)
                    best_visible = visible_counts.max()
                    if best_visible > 0:
                        candidates = np.where(visible_counts == best_visible)[0]
                        chosen = int(candidates[np.argmin(np.abs(candidates - frame_id))])
                        if frame_id != chosen:
                            self._log_info(
                                f"No frame has all GT boxes; using frame {chosen} with max visible subjects ({best_visible})."
                            )
                        frame_id = chosen
                    else:
                        self._log_warn("No GT bbox visibility flags are True; keeping requested init frame.")
            else:
                self._log_warn("GT bbox visibility array has unexpected shape; keeping requested init frame.")
        except Exception as exc:
            self._log_warn(f"Failed GT-based init-frame selection ({exc}); keeping requested frame.")

        if os.path.exists(os.path.join(output_path, "masks.npy")):
            self._log_info(f"Reusing existing masks cache: {os.path.join(output_path, 'masks.npy')}")
            save_masks = np.load(os.path.join(output_path, "masks.npy"), allow_pickle=True)[()]
            return save_masks

        os.makedirs(self.output_path, exist_ok=True)
        self._log_info(f"Starting GT-bbox init-camera processing for output '{output_path}'.")

        # Get the bboxes from i-th frame
        if frame_id >= len(frames):
            self._log_warn(f"Requested init frame {frame_id} is out of range; using frame {len(frames) - 1}.")
            frame_id = len(frames) - 1
        cam_name = frames.cam_name

        self._log_info(f"[{cam_name}] Using GT bboxes from frame {frame_id} for initialization.")
        img = frames.read_pil(frame_id)
        imgs_bbx = []
        init_vis_detections = []
        gt_bboxes = gt_bboxes[frame_id]  # (num_objs, 4)
        if expected_subjects is None or expected_subjects <= 0:
            try:
                visible_count = int(np.asarray(gt_bboxes_mask[frame_id]).astype(bool).sum())
                if visible_count > 0:
                    expected_subjects = visible_count
                    self.expected_subjects = expected_subjects
                    self._log_info(f"Inferred scene subject count from GT visibility: N={expected_subjects}.")
            except Exception:
                pass

        # Get the bounding boxes of person class and add to SAM2 predictor
        ann_obj_id = 0
        for src_obj_id, box in enumerate(gt_bboxes):
            if src_obj_id >= len(gt_bboxes_mask[frame_id]):
                continue
            if gt_bboxes_mask[frame_id][src_obj_id]:  # only use bbox if visible at this frame
                if expected_subjects is not None and expected_subjects > 0 and ann_obj_id >= expected_subjects:
                    break
                bbx = box
                img_bbx = np.asarray(img)
                img_bbx = img_bbx[int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2])]
                bbox = np.array([bbx[0], bbx[1], bbx[2], bbx[3]])
                imgs_bbx.append((img_bbx, frame_id, bbox))
                init_vis_detections.append((img_bbx, None, bbox, None))
                ann_obj_id += 1

        out_init_dir = os.path.join(self.output_path, "initialize")
        self._save_init_detection_overview(
            image=np.asarray(img),
            detections=init_vis_detections,
            frame_idx=frame_id,
            out_init_dir=out_init_dir,
            tag="gt",
            camera_name=cam_name,
        )

        # Initialize the SAM predictor with the detected bboxes
        self._log_info(f"[{cam_name}] Initializing SAM state with GT prompts from frame {frame_id}.")
        sam_source = self._current_sam_video_source
        frame_names = frames.frame_names
        inference_state, _runtime_video_dir = self._init_state_with_fallback(
            video_dir=sam_source,
            frame_names=frame_names,
            output_path=output_path,
            context_name=cam_name,
        )
        self.predictor.reset_state(inference_state)
        for i, (img_bbx, frame_idx, bbox) in enumerate(imgs_bbx):
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=i,
                box=bbox,
            )

        self._log_info(f"[{cam_name}] Starting propagation from GT prompts.")
        video_segments = self.run_propagation(inference_state)

        # render the segmentation results every few frames
        obj_ids = range(ann_obj_id)
        os.makedirs(output_path, exist_ok=True)
        save_masks = self.save_images_from_video(inference_state, video_segments, obj_ids, output_path)
        _atomic_np_save(os.path.join(output_path, "masks.npy"), save_masks)
        self._log_info(f"[{cam_name}] Saved mask cache: {os.path.join(output_path, 'masks.npy')}")
        self._cleanup_sanitized_video_dir(_runtime_video_dir, sam_source)
        return save_masks

    def _cross_camera_strategy(self, calibration_available=None):
        """Resolve the cross-camera identity strategy for non-init cameras.

        'auto' (default) is calibration- and backend-aware, per the validation
        campaign (verified with epipolar geometry active):
          * SAM3-text detection (sam3_prompt_light) -> always 'track_then_match'.
          * YOLO detection (sam2, sam3) -> 'track_then_match' WHEN epipolar
            calibration (cam_int/cam_ext) is available: the remap's dominant,
            backend-INDEPENDENT geometry term then drives every backend to ~100%
            cross-camera id-accuracy (e.g. sam3 legacy 82.5% -> ttm 99.9% on
            pass_clap; sam2 ~100% either way). Without calibration the remap falls
            back to CLIP-only, where YOLO appearance crops are unreliable and
            track-then-match regresses sam2 / is flat for sam3 -> keep the legacy
            'match_then_track' path for the YOLO backends in that case only.
        Explicit 'track_then_match' / 'match_then_track' override 'auto'.
        (sam3_prompt uses its own track-then-match path and never consults this.)

        Args:
            calibration_available: True if epipolar geometry can be computed for
                this camera pair (cam_int/cam_ext present). None/False steers the
                YOLO 'auto' fallback to legacy match-then-track.
        """
        val = str(self.assignment_config.get("masks", {})
                  .get("cross_camera_strategy", "auto")).strip().lower()
        if val == "auto":
            if self.sam_version in ("sam3_prompt_light", "sam3_prompt"):
                val = "track_then_match"
            elif calibration_available:
                # YOLO backends reach sam3_prompt parity under track-then-match once
                # the epipolar geometry term is available (it is backend-independent).
                val = "track_then_match"
            else:
                # CLIP-only: appearance-only remap is unreliable for YOLO crops.
                val = "match_then_track"
        if val not in ("track_then_match", "match_then_track"):
            self._log_warn(f"Unknown cross_camera_strategy '{val}'; using 'match_then_track'.")
            val = "match_then_track"
        return val

    def _select_seed_frame(self, frames, frame_id=None, expected_subjects=None):
        """Pick a clean seed frame for per-camera tracking and return its detections.

        Backend-neutral (YOLO for sam2/sam3, SAM3-text for sam3_prompt_light via the
        _collect_yolo_detections dispatch). Scores candidate frames by
        ``n_det * mean_conf * (1 - mean_pairwise_iou)`` (×1.5 when
        n_det == expected_subjects), mirroring the init-camera auto-selector, then
        returns every detection (the cross-camera remap prunes surplus by identity).

        Returns (best_frame:int, detections:list[(crop, feat, bbox, score)]) or
        (None, []) when nothing is detected.
        """
        n_frames = len(frames)
        if n_frames == 0:
            return None, []
        cfg = self.assignment_config.get("matching", {})
        yolo_conf = float(cfg.get("yolo_person_conf", 0.5))
        sample_count = min(30, n_frames)
        sampled = [int(f) for f in np.linspace(0, n_frames - 1, num=sample_count, dtype=int)]
        candidate_frames = []
        if frame_id is not None:
            candidate_frames.append(min(int(frame_id), n_frames - 1))
        candidate_frames += sampled
        seen = set()
        candidate_frames = [f for f in candidate_frames if not (f in seen or seen.add(f))]

        best_score, best_frame, best_dets = -1.0, None, []
        for cand in candidate_frames:
            _img, dets = self._collect_yolo_detections(
                frames.read_pil(cand), yolo_conf, include_clip_features=False)
            if not dets:
                continue
            n_det = len(dets)
            score = sum(float(d[3]) for d in dets)  # == n_det * mean_conf (n_det >= 1 here)
            # Prefer frames where the subjects don't overlap (cleaner local seeds).
            score *= 1.0 - self._mean_pairwise_iou([d[2] for d in dets])
            # Down-weight clip-boundary candidates (mirrors the init selector).
            score *= _seed_centrality_weight(
                int(cand), n_frames, float(cfg.get("seed_center_bias", 0.0)))
            # Prefer frames with exactly the expected subject count: these are the
            # cleanest (all real subjects, no splits/spurious). Rewarding n_det > expected
            # instead steers toward over-detecting frames (splits) and regresses tracking
            # (measured); seeding all detections already handles genuine extra people
            # (outsiders) via the remap without needing to bias frame choice.
            if expected_subjects and n_det == expected_subjects:
                score *= 1.5
            if score > best_score:
                best_score, best_frame, best_dets = score, cand, dets

        if best_frame is None:  # fallback: lower thresholds
            for conf in (0.4, 0.25, 0.15):
                for cand in candidate_frames:
                    _img, dets = self._collect_yolo_detections(
                        frames.read_pil(cand), conf, include_clip_features=False)
                    if dets:
                        best_frame, best_dets = cand, dets
                        break
                if best_frame is not None:
                    break
        if best_frame is None or not best_dets:
            return None, []

        # Seed EVERY detection and let _remap_tracklets_to_init prune surplus by identity —
        # a partner-occluded real subject must not be dropped for a cleaner outsider that
        # merely detects at higher confidence. A generous max_seed_tracklets ceiling only
        # bounds pathological many-bystander scenes, and is logged (never a silent truncation).
        max_seed = int(self.assignment_config.get("masks", {}).get("max_seed_tracklets", 20) or 0)
        if max_seed > 0 and len(best_dets) > max_seed:
            self._log_warn(
                f"Seed cap: {len(best_dets)} detections exceed max_seed_tracklets={max_seed}; "
                f"keeping the {max_seed} highest-confidence detections (bystander-heavy frame?). "
                "Raise masks.max_seed_tracklets if real subjects are being lost."
            )
            best_dets = sorted(best_dets, key=lambda x: float(x[3]), reverse=True)[:max_seed]
        return best_frame, best_dets

    def _register_prescreen_reference(self, cam_data, output_path, obj_ids, mask_data=None):
        """Record a completed camera as a geometric reference for the seed
        prescreen of later cameras. Its per-frame mask PNGs give exact
        same-frame subject positions; the slim mask cache is the fallback."""
        try:
            self._prescreen_refs.append({
                "cam_name": str(cam_data.get("cam_name", "?")) if isinstance(cam_data, dict) else "?",
                "cam_data": cam_data,
                "mask_dir": os.path.join(output_path, "masks"),
                "obj_ids": [int(o) for o in obj_ids],
                "mask_data": mask_data,
            })
        except Exception as exc:  # noqa: BLE001 — a bad ref must not fail the camera
            self._log_warn(f"Could not register prescreen reference: {exc}")

    def _prescreen_ref_point(self, ref, oid, frame_idx):
        """Subject ``oid``'s image position in a reference camera at
        ``frame_idx``: exact-frame from the per-frame mask PNG when present,
        else the nearest sampled centroid from the slim mask cache."""
        png = os.path.join(ref["mask_dir"], f"mask_{int(frame_idx):04d}_{int(oid) + 1:02d}.png")
        if os.path.isfile(png):
            m = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.max() > 0:
                return self._mask_centroid_xy1(m > 0)
        md = ref.get("mask_data")
        if md is not None and oid in md:
            return self._reference_point_from_mask_data(md[oid], frame_idx)
        return None

    def _prescreen_seed_candidates(self, cam_name, frames, detections, seed_frame,
                                   mask_data, source_cam_data, target_cam_data):
        """Bound the track-then-match seed set to the strongest candidates per
        init subject (top-M per subject, union) BEFORE any tracking runs.

        Propagation cost and GPU memory scale with the number of seeded
        tracklets, so a bystander-heavy view (12+ detections for a 1-subject
        scene) can exhaust the GPU even with memory pruning on. Shortlisting
        reuses the same epipolar+CLIP scoring the cross-camera remap already
        trusts for the final identity decision, asked a strictly easier
        question: not "which candidate IS subject s" but "which M candidates
        could be". Identity is still decided after tracking, by the remap,
        with multi-frame evidence. Per-subject slots (never a global top-K)
        mean an occluded subject only competes with candidates for its own
        identity, never with brighter bystanders matching someone else.
        References are the init camera plus every camera already completed in
        this run: multiple epipolar lines intersect, collapsing the
        depth-ambiguity of a single reference view.

        Engages only when len(detections) > N*M and epipolar calibration is
        available; otherwise returns ``detections`` unchanged, so clean scenes
        stay byte-identical. Every dropped candidate is logged with its
        scores. Any internal error falls back to the unfiltered detections
        (current validated behaviour), loudly.
        """
        mask_cfg = self.assignment_config.get("masks", {}) or {}
        mode = str(mask_cfg.get("seed_prescreen", "off") or "off").strip().lower()
        if mode in ("off", "false", "0", "no", "disabled", "none"):
            return detections
        explicit = mode in ("on", "true", "1", "yes", "enabled")

        subjects = sorted(int(o) for o in (mask_data or {}).keys())
        n_subjects = len(subjects)
        m_per = max(2, int(mask_cfg.get("seed_candidates_per_subject", 3) or 3))
        k = len(detections)
        if n_subjects == 0 or k <= n_subjects * m_per:
            return detections

        try:
            # ── References: init camera + every camera completed this run ──
            refs = []
            if isinstance(source_cam_data, dict):
                init_name = str(source_cam_data.get("cam_name", "") or "")
                refs.append({
                    "cam_name": init_name or "init",
                    "cam_data": source_cam_data,
                    "mask_dir": os.path.join(self.output_path, init_name, "masks"),
                    "obj_ids": subjects,
                    "mask_data": mask_data,
                })
            refs.extend(self._prescreen_refs)

            usable = []
            for ref in refs:
                f_matrix = self._compute_fundamental_matrix(ref["cam_data"], target_cam_data)
                if f_matrix is not None:
                    usable.append((ref, f_matrix))
            if not usable:
                log = self._log_warn if explicit else self._log_info
                log(f"[{cam_name}] Seed prescreen: no epipolar calibration available — "
                    f"appearance-only shortlisting is unreliable, keeping all {k} detections.")
                return detections

            cfg = self._get_matching_cfg()
            clip_weight = float(cfg.get("clip_weight", 0.4))
            epipolar_weight = float(cfg.get("epipolar_weight", 0.6))
            image_size = frames.image_size if len(frames) > 0 else None
            sigma_px, max_dist_px, _ = self._resolve_epipolar_px_params(cfg, image_size=image_size)

            # Candidate centroids (bbox centers — well within the epipolar
            # sigma of a mask centroid) and CLIP features (one batched forward).
            cand_pts = []
            for _crop, _feat, bbox, _score in detections:
                x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox).reshape(-1)[:4]]
                cand_pts.append(np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, 1.0], dtype=np.float64))
            feats = self._encode_clip_batch([d[0] for d in detections])
            cand_feats = F.normalize(torch.stack([f.reshape(-1) for f in feats], dim=0), dim=1)

            s_clip = np.zeros((n_subjects, k), dtype=np.float32)
            for row, oid in enumerate(subjects):
                gallery, _ = self._get_subject_gallery(mask_data, oid)
                if gallery is None:
                    continue
                _, sims = cosine_knn(cand_feats, gallery, topk=1)
                s_clip[row] = sims.squeeze(1).cpu().numpy()

            epi_sum = np.zeros((n_subjects, k), dtype=np.float64)
            epi_cnt = np.zeros((n_subjects,), dtype=np.int32)
            for ref, f_matrix in usable:
                for row, oid in enumerate(subjects):
                    if oid not in ref.get("obj_ids", subjects):
                        continue
                    ref_pt = self._prescreen_ref_point(ref, oid, seed_frame)
                    if ref_pt is None:
                        continue
                    scores_this_ref = np.zeros((k,), dtype=np.float64)
                    ok = True
                    for col in range(k):
                        score, _dist = self._epipolar_score(
                            f_matrix=f_matrix, x1=ref_pt, x2=cand_pts[col],
                            sigma_px=sigma_px, max_dist_px=max_dist_px)
                        if score is None:
                            ok = False
                            break
                        scores_this_ref[col] = score
                    if ok:
                        epi_sum[row] += scores_this_ref
                        epi_cnt[row] += 1

            fused = np.zeros((n_subjects, k), dtype=np.float64)
            for row in range(n_subjects):
                if epi_cnt[row] > 0:
                    fused[row] = (epipolar_weight * (epi_sum[row] / epi_cnt[row])
                                  + clip_weight * s_clip[row])
                else:
                    fused[row] = s_clip[row]

            keep = set()
            for row in range(n_subjects):
                for col in np.argsort(-fused[row])[:m_per]:
                    keep.add(int(col))
            # Never track fewer candidates than there are subjects.
            if len(keep) < min(n_subjects, k):
                for col in np.argsort(-fused.max(axis=0)):
                    keep.add(int(col))
                    if len(keep) >= min(n_subjects, k):
                        break

            dropped = [c for c in range(k) if c not in keep]
            if not dropped:
                return detections
            n_ref_names = [r["cam_name"] for r, _ in usable]
            self._log_info(
                f"[{cam_name}] Seed prescreen: keeping {len(keep)} of {k} detections "
                f"(top-{m_per} per subject x {n_subjects} subject(s); references: "
                f"{', '.join(n_ref_names)}).")
            for col in dropped:
                row = int(np.argmax(fused[:, col]))
                epi_part = (epi_sum[row, col] / epi_cnt[row]) if epi_cnt[row] > 0 else float("nan")
                self._log_info(
                    f"[{cam_name}] Seed prescreen: dropping detection {col} "
                    f"(bbox center {cand_pts[col][0]:.0f},{cand_pts[col][1]:.0f}): best fused score "
                    f"{fused[row, col]:.3f} vs subject {subjects[row]} (epi {epi_part:.3f}, "
                    f"clip {s_clip[row, col]:.3f}) — outside every subject's top-{m_per}.")
            return [detections[i] for i in sorted(keep)]
        except Exception as exc:  # noqa: BLE001 — fall back to validated behaviour, loudly
            self._log_warn(
                f"[{cam_name}] Seed prescreen failed ({type(exc).__name__}: {exc}) — "
                f"keeping all {k} detections (pre-prescreen behaviour).")
            return detections

    def _detect_and_track_local(self, frames, output_path, frame_id=None, expected_subjects=None,
                                mask_data=None, source_cam_data=None, target_cam_data=None):
        """Detect people on one seed frame, seed each as an arbitrary LOCAL obj id,
        propagate forward+reverse, and post-process. Backend-neutral track step for
        the track-then-match path (sam2 / sam3 / sam3_prompt_light) — no cross-camera
        identity is committed here.

        Returns (local_ids, video_segments, best_frame) or (None, None, None).
        """
        cam_name = frames.cam_name
        sam_source = self._current_sam_video_source

        best_frame, detections = self._select_seed_frame(
            frames, frame_id=frame_id, expected_subjects=expected_subjects)
        if not detections:
            self._log_warn(f"[{cam_name}] No person detections for track-then-match seeding.")
            return None, None, None
        if mask_data is not None:
            detections = self._prescreen_seed_candidates(
                cam_name, frames, detections, best_frame,
                mask_data, source_cam_data, target_cam_data)
        self._log_info(
            f"[{cam_name}] Seeding {len(detections)} local tracklets on frame {best_frame} "
            "(track-then-match)."
        )

        # Seed all detections on the single best_frame with arbitrary local ids 0..k-1.
        # One conditioning frame inherently satisfies SAM3's joint-consolidation rule.
        best_similarity = {
            oid: (float(score), int(best_frame), crop, np.asarray(bbox, dtype=np.float32), crop)
            for oid, (crop, _feat, bbox, score) in enumerate(detections)
        }
        anchor_matches = {
            oid: [{
                "score": float(score),
                "frame_idx": int(best_frame),
                "img_bbx": crop,
                "bbox": np.asarray(bbox, dtype=np.float32),
                "target_img_bbx": crop,
            }]
            for oid, (crop, _feat, bbox, score) in enumerate(detections)
        }

        inference_state, runtime_video_dir = self._init_state_with_fallback(
            video_dir=sam_source,
            frame_names=frames.frame_names,
            output_path=output_path,
            context_name=cam_name,
        )
        self.predictor.reset_state(inference_state)
        try:
            self.show_best_similarity_and_add_bboxes(
                best_similarity,
                frames=frames,
                video_dir=runtime_video_dir,
                inference_state=inference_state,
                frame_names=frames.frame_names,
                output_path=output_path,
                anchor_matches=anchor_matches,
            )
            video_segments = self.run_propagation(inference_state, cam_name=cam_name)
        except Exception as exc:
            self._log_error(
                f"[{cam_name}] track-then-match propagation failed: {exc}.")
            self._cleanup_sanitized_video_dir(runtime_video_dir, sam_source)
            # Never degrade to a silent skip: the caller must not cache anything
            # for this camera (a transient failure would otherwise become a
            # sticky masks.npy that resume trusts forever).
            raise CameraProcessingError(cam_name, f"track-then-match propagation failed: {exc}") from exc

        self._cleanup_sanitized_video_dir(runtime_video_dir, sam_source)
        local_ids = video_segments.all_obj_ids()
        if not local_ids:
            self._log_warn(f"[{cam_name}] track-then-match produced no tracklets.")
            if hasattr(video_segments, "close"):
                video_segments.close()
            return None, None, None
        return local_ids, video_segments, best_frame

    def _process_new_video_track_then_match(self, frames, mask_data, output_path,
                                            source_cam_data=None, target_cam_data=None,
                                            expected_subjects=None):
        """Track-then-match cross-camera identity. Propagate this camera
        independently into clean local tracklets, then remap WHOLE tracklets to the
        init-camera ids (multi-frame, occlusion-weighted CLIP + epipolar). Never
        updates the feature bank — gallery stays init + multiview-bootstrap."""
        cam_name = frames.cam_name
        local_ids, video_segments, best_frame = self._detect_and_track_local(
            frames, output_path, expected_subjects=expected_subjects,
            mask_data=mask_data, source_cam_data=source_cam_data,
            target_cam_data=target_cam_data)
        if local_ids is None:
            # No tracklets in this camera (nothing detected/tracked). Return a
            # truthful empty result — never the init camera's mask_data, which
            # the caller would cache as this camera's masks.npy and resume
            # would then trust forever.
            self._log_warn(f"[{cam_name}] no tracklets; writing empty masks for this camera.")
            return {}

        id_remap = self._remap_tracklets_to_init(
            cam_name, frames, mask_data, local_ids, video_segments, best_frame,
            source_cam_data=source_cam_data, target_cam_data=target_cam_data,
        )
        save_masks = self._build_masks_from_remap(frames, video_segments, id_remap, output_path)
        # This camera's per-frame masks are now on disk — register it as a
        # geometric reference for later cameras' seed prescreen.
        self._register_prescreen_reference(
            target_cam_data if isinstance(target_cam_data, dict) else {},
            output_path, obj_ids=sorted(set(id_remap.values())), mask_data=save_masks)
        if hasattr(video_segments, "close"):
            video_segments.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return save_masks

    def process_new_video(self, frames, mask_data, output_path, source_cam_data=None, target_cam_data=None, expected_subjects=None):
        """Process a non-init camera. Dispatches on the resolved cross-camera strategy
        (config 'cross_camera_strategy', default 'auto'): 'track_then_match' (propagate
        independently, then remap whole tracklets) or the legacy 'match_then_track'
        (commit identity at seed time). See _cross_camera_strategy()."""
        mask_filepath = os.path.join(output_path, "masks.npy")
        if os.path.exists(mask_filepath):
            self._log_info(f"Reusing existing masks cache: {mask_filepath}")
            return np.load(mask_filepath, allow_pickle=True)[()]

        os.makedirs(output_path, exist_ok=True)

        # Support legacy str (video_dir) path for backward compat
        if isinstance(frames, str):
            video_dir = frames
            frame_names = [
                p for p in os.listdir(video_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
            ]
            try:
                frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            except ValueError:
                frame_names.sort()
            frames = ImageFileSource(
                [os.path.join(video_dir, fn) for fn in frame_names],
                camera_name=os.path.basename(output_path),
            )

        # Epipolar geometry is the remap's dominant, backend-independent signal and
        # it decides the 'auto' strategy for YOLO backends. Check availability cheaply
        # (key presence — the remap builds the actual matrix where it's used) and warn
        # loudly on the CLIP-only fallback: a silent INFO-level fallback here previously
        # masked a large cross-camera accuracy loss.
        calib_ok = self._has_epipolar_calibration(source_cam_data, target_cam_data)
        if not calib_ok and not getattr(self, "_clip_only_warned", False):
            self._log_warn(
                "No camera calibration (cam_int/cam_ext) available for cross-camera ID "
                "matching — falling back to CLIP-only (appearance) matching. This is "
                "markedly less accurate for cross-camera identity, especially for the YOLO "
                "backends (sam2/sam3). Provide calibration (e.g. run with --ma_cap_dir so "
                "raw/IOI_*.npz is discovered) to enable epipolar geometry."
            )
            self._clip_only_warned = True

        strategy = self._cross_camera_strategy(calibration_available=calib_ok)
        cam_label = frames.cam_name  # frames is always a FrameSource here (str converted above)
        self._log_info(
            f"[{cam_label}] cross-camera strategy: {strategy} "
            f"(epipolar geometry {'available' if calib_ok else 'UNAVAILABLE — CLIP-only'})."
        )
        if strategy == "track_then_match":
            return self._process_new_video_track_then_match(
                frames, mask_data, output_path, source_cam_data, target_cam_data, expected_subjects)
        return self._process_new_video_match_then_track(
            frames, mask_data, output_path, source_cam_data, target_cam_data, expected_subjects)

    def _process_new_video_match_then_track(self, frames, mask_data, output_path, source_cam_data=None, target_cam_data=None, expected_subjects=None):
        """Legacy cross-camera identity: match init ids to per-frame anchor detections
        and seed the tracker with them before propagation (match-then-track)."""
        frame_names = frames.frame_names
        cam_name = frames.cam_name
        sam_source = self._current_sam_video_source
        self._log_info(f"Starting camera processing for '{cam_name}' with {len(frames)} frames.")

        inference_state, runtime_video_dir = self._init_state_with_fallback(
            video_dir=sam_source,
            frame_names=frame_names,
            output_path=output_path,
            context_name=cam_name,
        )
        self.predictor.reset_state(inference_state)

        # Match existing mask_data objects to detections in this new camera
        best_similarity, anchor_matches = self.detect_bbx_and_similarity(
            mask_data,
            frames,
            source_cam_data=source_cam_data,
            target_cam_data=target_cam_data,
            expected_subjects=expected_subjects,
        )

        if not self.has_valid_detections(best_similarity):
            self._log_error(
                f"No valid detections found for any ID in '{cam_name}'. "
                "Skipping propagation. Check matching.start_frame/samples and detection thresholds."
            )
            self._cleanup_sanitized_video_dir(runtime_video_dir, sam_source)
            # Never the init camera's mask_data: the caller caches the return
            # value as THIS camera's masks.npy and resume would trust it forever.
            return {}

        # SAM3 requires all object prompts on the same frame — its joint
        # consolidation poisons tracking memory for objects without prompts
        # on a given conditioning frame.
        if self._is_sam3:
            best_similarity, anchor_matches = self._force_single_frame_anchors(
                best_similarity, anchor_matches, frames, cam_name,
            )

        try:
            self.show_best_similarity_and_add_bboxes(
                best_similarity,
                frames=frames,
                video_dir=runtime_video_dir,
                inference_state=inference_state,
                frame_names=frame_names,
                output_path=output_path,
                anchor_matches=anchor_matches,
            )
            self._save_anchor_report(output_path, anchor_matches, n_frames=len(frames))
            self._save_anchor_visualizations(output_path, anchor_matches, n_frames=len(frames))
        except Exception as e:
            self._cleanup_sanitized_video_dir(runtime_video_dir, sam_source)
            # Infrastructure failure, not an empty camera: surface it so the
            # camera is recorded as failed instead of cached as done.
            raise CameraProcessingError(
                cam_name, f"anchor prompt injection failed: {e}") from e

        video_segments = self.run_propagation(inference_state)

        self._log_info(f"Propagation completed for '{cam_name}'. Produced {len(video_segments)} frame segments.")
        obj_ids = sorted(best_similarity.keys())
        new_mask_data = self.save_images_from_video(inference_state, video_segments, obj_ids, output_path)
        _atomic_np_save(os.path.join(output_path, "masks.npy"), new_mask_data)
        self._log_info(f"Saved mask cache: {os.path.join(output_path, 'masks.npy')}")
        self._cleanup_sanitized_video_dir(runtime_video_dir, sam_source)
        return new_mask_data


    def _select_anchor_matches(self, candidates, max_anchors, min_frame_gap):
        if len(candidates) == 0:
            return []
        if max_anchors <= 1:
            top = max(candidates, key=lambda x: float(x["score"]))
            return [top]

        min_frame_gap = max(0, int(min_frame_gap))
        by_time = sorted(candidates, key=lambda x: int(x["frame_idx"]))
        first_frame = int(by_time[0]["frame_idx"])
        last_frame = int(by_time[-1]["frame_idx"])

        # Pass 1: coverage-aware picks, one representative per temporal bin.
        bin_edges = np.linspace(first_frame, last_frame + 1, num=max_anchors + 1, dtype=int)
        selected = []
        for b in range(max_anchors):
            lo, hi = int(bin_edges[b]), int(bin_edges[b + 1])
            if b == max_anchors - 1:
                in_bin = [c for c in by_time if int(c["frame_idx"]) >= lo and int(c["frame_idx"]) <= hi]
            else:
                in_bin = [c for c in by_time if int(c["frame_idx"]) >= lo and int(c["frame_idx"]) < hi]
            if len(in_bin) == 0:
                continue
            in_bin = sorted(in_bin, key=lambda x: float(x["score"]), reverse=True)
            for cand in in_bin:
                frame_idx = int(cand["frame_idx"])
                if any(abs(frame_idx - int(sel["frame_idx"])) < min_frame_gap for sel in selected):
                    continue
                selected.append(cand)
                break

        # Pass 2: fill remaining slots by score while respecting minimum gap.
        if len(selected) < max_anchors:
            by_score = sorted(candidates, key=lambda x: float(x["score"]), reverse=True)
            for cand in by_score:
                if len(selected) >= max_anchors:
                    break
                frame_idx = int(cand["frame_idx"])
                if any(abs(frame_idx - int(sel["frame_idx"])) < min_frame_gap for sel in selected):
                    continue
                selected.append(cand)

        if len(selected) == 0:
            selected = [max(candidates, key=lambda x: float(x["score"]))]
        selected = sorted(selected, key=lambda x: int(x["frame_idx"]))
        return selected

    def _resolve_max_anchors_per_id(self, n_frames, requested_max, min_frame_gap, auto_cap):
        requested_max = int(requested_max)
        if requested_max > 0:
            return requested_max
        min_frame_gap = max(1, int(min_frame_gap))
        auto_cap = max(1, int(auto_cap))
        auto_count = int(np.ceil(float(n_frames) / float(min_frame_gap * 1.5)))
        auto_count = max(3, auto_count)
        return min(auto_cap, auto_count)


    def _save_anchor_report(self, output_path, anchor_matches, n_frames):
        report = {
            "n_frames": int(n_frames),
            "per_id": {},
        }
        for oid, matches in anchor_matches.items():
            report["per_id"][str(int(oid))] = [
                {
                    "frame_idx": int(m["frame_idx"]),
                    "score": float(m["score"]),
                }
                for m in matches
            ]
        report_path = os.path.join(output_path, "anchor_report.json")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as exc:
            self._log_warn(f"Failed to save anchor report '{report_path}': {exc}")

    def _save_anchor_visualizations(self, output_path, anchor_matches, n_frames):
        vis_dir = os.path.join(output_path, "anchor_visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 1) Global timeline view: frame index vs object id.
        try:
            plt.figure(figsize=(12, 3 + 0.8 * max(1, len(anchor_matches))))
            y_labels = sorted(anchor_matches.keys())
            y_positions = {oid: idx for idx, oid in enumerate(y_labels)}
            for oid in y_labels:
                matches = anchor_matches.get(oid, [])
                if len(matches) == 0:
                    continue
                xs = [int(m["frame_idx"]) for m in matches]
                ys = [y_positions[oid]] * len(matches)
                sizes = [max(20.0, 180.0 * float(m["score"])) for m in matches]
                plt.scatter(xs, ys, s=sizes, alpha=0.75, color=[self._id_color(oid)], label=f"id {oid}")
                for x, y, m in zip(xs, ys, matches):
                    plt.text(x, y + 0.08, f'{float(m["score"]):.2f}', fontsize=8, ha="center")
            plt.xlim(0, max(1, int(n_frames) - 1))
            plt.yticks(list(range(len(y_labels))), [str(oid) for oid in y_labels])
            plt.xlabel("Frame Index")
            plt.ylabel("Object ID")
            plt.title("Selected Anchors Timeline")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "anchors_timeline.png"))
            plt.close()
        except Exception as exc:
            self._log_warn(f"Failed to save anchor timeline visualization in '{vis_dir}': {exc}")

        # 2) Per-ID anchor crop sheets.
        for oid in sorted(anchor_matches.keys()):
            matches = anchor_matches.get(oid, [])
            if len(matches) == 0:
                continue
            cols = min(5, max(1, len(matches)))
            rows = (len(matches) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes = axes.flatten()
            for idx, match in enumerate(matches):
                ax = axes[idx]
                crop = match.get("img_bbx")
                if crop is None:
                    ax.text(0.5, 0.5, "No crop", ha="center", va="center")
                    ax.axis("off")
                    continue
                ax.imshow(crop)
                ax.set_title(
                    f'id={int(oid):02d} f={int(match["frame_idx"])} s={float(match["score"]):.2f}',
                    fontsize=9,
                    color=self._id_color(oid),
                )
                ax.axis("off")
            for ax in axes[len(matches):]:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"anchors_id_{int(oid):02d}.png"))
            plt.close()

    # Combined multi-view assignment (CLIP + epipolar geometry)
    def detect_bbx_and_similarity(
        self,
        mask_data,
        frames,
        source_cam_data=None,
        target_cam_data=None,
        expected_subjects=None,
        start_frame=None,
        samples=None,
    ):
        from scipy.optimize import linear_sum_assignment  # Hungarian

        cfg = self.assignment_config.get("matching", {})
        if start_frame is None:
            start_frame = int(cfg.get("start_frame", 10))
        if samples is None:
            sample_interval = int(cfg.get("sample_interval", 50))
            min_samples = int(cfg.get("min_samples", 5))
            available = len(frames) - int(cfg.get("start_frame", 10))
            samples = max(min_samples, available // max(1, sample_interval))
        yolo_person_conf = float(cfg.get("yolo_person_conf", 0.5))
        clip_weight = float(cfg.get("clip_weight", 0.4))
        epipolar_weight = float(cfg.get("epipolar_weight", 0.6))
        # Resolve image size for epipolar parameters from FrameSource
        image_size = frames.image_size if len(frames) > 0 else None
        epipolar_sigma_px, max_epipolar_dist_px, epipolar_source = self._resolve_epipolar_px_params(
            cfg, image_size=image_size
        )
        clip_min_similarity = float(cfg.get("clip_min_similarity", 0.0))
        combined_min_score = float(cfg.get("combined_min_score", 0.2))
        allow_missing_ids = bool(cfg.get("allow_missing_ids", False))
        online_bank_update_min_score = float(cfg.get("online_bank_update_min_score", 0.35))
        max_anchors_per_id = int(cfg.get("max_anchors_per_id", 3))
        min_anchor_frame_gap = int(cfg.get("min_anchor_frame_gap", 25))
        anchor_score_threshold = float(cfg.get("anchor_score_threshold", combined_min_score))
        anchor_fallback_to_best_any = bool(cfg.get("anchor_fallback_to_best_any", True))
        anchor_auto_max_cap = int(cfg.get("anchor_auto_max_cap", 20))
        anchor_samples_per_anchor = int(cfg.get("anchor_samples_per_anchor", 4))
        anchor_enforce_cross_id_uniqueness = bool(cfg.get("anchor_enforce_cross_id_uniqueness", True))
        anchor_conflict_iou = float(cfg.get("anchor_conflict_iou", 0.65))

        obj_ids = self._select_obj_ids(mask_data, expected_subjects=expected_subjects)
        best_similarity = {oid: (0.0, -1, None, None, None) for oid in obj_ids}
        best_any = {oid: (-1.0, -1, None, None, None) for oid in obj_ids}
        candidate_matches = {oid: [] for oid in obj_ids}
        if not obj_ids:
            return best_similarity, {oid: [] for oid in obj_ids}

        galleries = {}
        gallery_crops = {}
        for oid in obj_ids:
            gallery, crops = self._get_subject_gallery(mask_data, oid)
            galleries[oid] = gallery
            gallery_crops[oid] = crops

        n = len(frames)
        if start_frame >= n:
            self._log_warn(f"matching.start_frame={start_frame} is out of range (n_frames={n}); using 0.")
            start_frame = 0
        max_anchors_per_id = self._resolve_max_anchors_per_id(
            n_frames=n,
            requested_max=max_anchors_per_id,
            min_frame_gap=min_anchor_frame_gap,
            auto_cap=anchor_auto_max_cap,
        )
        if self._is_sam3 and max_anchors_per_id > 1:
            self._log_info(
                f"Capping max_anchors_per_id from {max_anchors_per_id} to 1 for SAM3 "
                "(multi-frame anchors cause poisoned tracking memory in joint consolidation)."
            )
            max_anchors_per_id = 1
        target_samples = max(int(samples), max_anchors_per_id * max(1, anchor_samples_per_anchor))
        samples = max(1, min(int(target_samples), n - start_frame))
        indices = np.linspace(start_frame, n - 1, num=samples, dtype=int)
        self._log_info(
            f"Matching sample plan: sampled_frames={len(indices)}, anchors_per_id={max_anchors_per_id}, "
            f"min_anchor_gap={min_anchor_frame_gap}"
        )
        self._log_info(
            f"Epipolar parameters: sigma_px={epipolar_sigma_px:.1f}, "
            f"max_dist_px={max_epipolar_dist_px:.1f} (source={epipolar_source})."
        )

        f_matrix = self._compute_fundamental_matrix(source_cam_data, target_cam_data)
        if f_matrix is None:
            self._log_warn("Epipolar term disabled for this camera pair (missing/invalid intrinsics or extrinsics).")
        else:
            self._log_info("Epipolar term enabled for this camera pair.")

        for frame_idx in indices:
            frame_img = frames.read_pil(frame_idx)
            centers, imgs_bbx, feats = [], [], []
            _, detections = self._collect_yolo_detections(
                frame_img,
                conf_thresh=yolo_person_conf,
                include_clip_features=True,
            )
            for crop, feat, det_box, _score in detections:
                if feat is None:
                    continue
                imgs_bbx.append(crop)
                det_box = np.asarray(det_box, dtype=np.float32)
                centers.append(
                    np.array(
                        [(det_box[0] + det_box[2]) * 0.5, (det_box[1] + det_box[3]) * 0.5, 1.0],
                        dtype=np.float64,
                    )
                )
                feats.append((crop, det_box, feat))

            if not feats:
                continue

            det_boxes = [x[1] for x in feats]
            det_feats = torch.stack([x[2].detach().cpu() for x in feats], dim=0)
            m = det_feats.shape[0]
            k = len(obj_ids)
            s_clip = np.zeros((k, m), dtype=np.float32)
            s_epi = np.zeros((k, m), dtype=np.float32)
            has_epi = np.zeros((k, m), dtype=bool)
            nn_idx = [[None] * m for _ in range(k)]

            for row, oid in enumerate(obj_ids):
                gallery = galleries[oid]
                if gallery is not None and len(gallery) > 0:
                    idxs, sims = cosine_knn(det_feats, gallery, topk=1)
                    s_clip[row] = sims.squeeze(1).cpu().numpy()
                    nn_idx[row] = idxs.squeeze(1).tolist()

                ref_point = self._reference_point_from_mask_data(mask_data[oid], frame_idx)
                if ref_point is None:
                    continue
                for col, x2 in enumerate(centers):
                    epi_score, epi_dist = self._epipolar_score(
                        f_matrix=f_matrix,
                        x1=ref_point,
                        x2=x2,
                        sigma_px=epipolar_sigma_px,
                        max_dist_px=max_epipolar_dist_px,
                    )
                    if epi_score is None:
                        continue
                    s_epi[row, col] = epi_score
                    has_epi[row, col] = True

            combined = clip_weight * s_clip + epipolar_weight * s_epi
            combined[~has_epi] = s_clip[~has_epi]
            combined[s_clip < clip_min_similarity] *= 0.5

            for row, oid in enumerate(obj_ids):
                for col in range(m):
                    raw_score = float(combined[row, col])
                    penalty = self._overlap_penalty(det_boxes[col], det_boxes)
                    score = raw_score * penalty
                    if score > best_any[oid][0]:
                        idx_match = nn_idx[row][col]
                        target_img_bbx = None
                        if idx_match is not None:
                            target_bank = gallery_crops.get(oid, [])
                            if idx_match < len(target_bank):
                                target_img_bbx = target_bank[idx_match]
                        best_any[oid] = (score, int(frame_idx), imgs_bbx[col], det_boxes[col], target_img_bbx)

            if k == 0 or m == 0:
                continue
            cost = 1.0 - combined
            row_ind, col_ind = linear_sum_assignment(cost)

            for row, col in zip(row_ind, col_ind):
                raw_score = float(combined[row, col])
                if raw_score < combined_min_score:
                    continue
                oid = obj_ids[row]

                # Penalize detections on crowded frames where boxes overlap.
                penalty = self._overlap_penalty(det_boxes[col], det_boxes)
                score = raw_score * penalty

                if score > best_similarity[oid][0]:
                    idx_match = nn_idx[row][col]
                    target_img_bbx = None
                    if idx_match is not None:
                        target_bank = gallery_crops.get(oid, [])
                        if idx_match < len(target_bank):
                            target_img_bbx = target_bank[idx_match]
                    best_similarity[oid] = (score, int(frame_idx), imgs_bbx[col], det_boxes[col], target_img_bbx)

                if score >= anchor_score_threshold:
                    idx_match = nn_idx[row][col]
                    target_img_bbx = None
                    if idx_match is not None:
                        target_bank = gallery_crops.get(oid, [])
                        if idx_match < len(target_bank):
                            target_img_bbx = target_bank[idx_match]
                    candidate_matches[oid].append(
                        {
                            "score": score,
                            "frame_idx": int(frame_idx),
                            "img_bbx": imgs_bbx[col],
                            "bbox": det_boxes[col],
                            "target_img_bbx": target_img_bbx,
                            "feature": det_feats[col],
                        }
                    )

        if not allow_missing_ids:
            for oid in obj_ids:
                if best_similarity[oid][1] < 0 and best_any[oid][1] >= 0:
                    best_similarity[oid] = best_any[oid]

        anchor_matches = {}
        for oid in obj_ids:
            selected = self._select_anchor_matches(
                candidate_matches.get(oid, []),
                max_anchors=max_anchors_per_id,
                min_frame_gap=min_anchor_frame_gap,
            )
            if len(selected) == 0 and best_similarity[oid][1] >= 0:
                score, frame_idx, img_bbx, bbox, target_img_bbx = best_similarity[oid]
                selected = [
                    {
                        "score": score,
                        "frame_idx": int(frame_idx),
                        "img_bbx": img_bbx,
                        "bbox": bbox,
                        "target_img_bbx": target_img_bbx,
                        "feature": None,
                    }
                ]
            if len(selected) == 0 and anchor_fallback_to_best_any and best_any[oid][1] >= 0:
                score, frame_idx, img_bbx, bbox, target_img_bbx = best_any[oid]
                selected = [
                    {
                        "score": score,
                        "frame_idx": int(frame_idx),
                        "img_bbx": img_bbx,
                        "bbox": bbox,
                        "target_img_bbx": target_img_bbx,
                        "feature": None,
                    }
                ]
            anchor_matches[oid] = selected

        if anchor_enforce_cross_id_uniqueness and len(anchor_matches) > 1:
            anchor_matches, removed_conflicts = self._resolve_anchor_conflicts(
                anchor_matches=anchor_matches,
                candidate_matches=candidate_matches,
                max_anchors_per_id=max_anchors_per_id,
                min_anchor_frame_gap=min_anchor_frame_gap,
                conflict_iou=anchor_conflict_iou,
            )
            if removed_conflicts > 0:
                self._log_warn(
                    f"Resolved {removed_conflicts} conflicting anchor assignments "
                    f"(cross-ID overlap IoU >= {anchor_conflict_iou:.2f})."
                )

        # Keep best_similarity consistent with final anchor set to avoid re-introducing
        # dropped/conflicting prompts in downstream fallback logic.
        for oid in obj_ids:
            selected = anchor_matches.get(oid, [])
            if len(selected) > 0:
                best = max(selected, key=lambda x: float(x.get("score", -1.0)))
                best_similarity[oid] = (
                    float(best.get("score", 0.0)),
                    int(best.get("frame_idx", -1)),
                    best.get("img_bbx"),
                    best.get("bbox"),
                    best.get("target_img_bbx"),
                )
            else:
                best_similarity[oid] = (0.0, -1, None, None, None)

        # Online bank enrichment with high-confidence selected anchors.
        for oid in obj_ids:
            for match in anchor_matches.get(oid, []):
                if float(match["score"]) < online_bank_update_min_score:
                    continue
                feat = match.get("feature")
                if feat is None:
                    continue
                self._append_to_subject_bank(oid, feat, crop=match.get("img_bbx"))

        objects_with_detections = [oid for oid, (_, frame_idx, _, _, _) in best_similarity.items() if frame_idx >= 0]
        objects_without_detections = [oid for oid in obj_ids if oid not in objects_with_detections]
        if objects_without_detections:
            self._log_warn(f"No matched detection selected for IDs: {objects_without_detections}")
        if objects_with_detections:
            self._log_info(f"Matched IDs in this view: {objects_with_detections}")
        else:
            self._log_warn("No detections found for any initialized IDs in sampled frames.")

        anchor_counts = {oid: len(anchor_matches.get(oid, [])) for oid in obj_ids}
        self._log_info(f"Selected temporal anchors per ID: {anchor_counts}")
        return best_similarity, anchor_matches


    # From Hanz's original implementation
    def show_best_similarity_and_add_bboxes(
        self,
        best_similarity,
        frames=None,
        video_dir=None,
        inference_state=None,
        frame_names=None,
        output_path=None,
        anchor_matches=None,
    ):
        vis_root = os.path.join(output_path, "anchor_visualizations")
        similarity_dir = os.path.join(vis_root, "prompt_similarity_panels")
        overlay_dir = os.path.join(vis_root, "prompt_frame_overlays")
        os.makedirs(similarity_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
        self._log_info(
            f"Saving anchor prompt visualizations under '{vis_root}' "
            f"(similarity panels: '{os.path.basename(similarity_dir)}', overlays: '{os.path.basename(overlay_dir)}')."
        )
        frame_prompt_boxes = {}

        # Add one or multiple anchor boxes per object ID before propagation.
        for ann_obj_id in sorted(best_similarity.keys()):
            if anchor_matches is not None:
                matches = anchor_matches.get(ann_obj_id, [])
            else:
                matches = []

            if len(matches) == 0:
                matches = [best_similarity[ann_obj_id]]
                matches = [
                    {
                        "score": matches[0][0],
                        "frame_idx": int(matches[0][1]),
                        "img_bbx": matches[0][2],
                        "bbox": matches[0][3],
                        "target_img_bbx": matches[0][4],
                    }
                ]

            for anchor_idx, match in enumerate(matches):
                similarity = float(match["score"])
                ann_frame_idx = int(match["frame_idx"])
                img_bbx = match["img_bbx"]
                bbx = match["bbox"]
                target_img_bbx = match.get("target_img_bbx")
                id_color = self._id_color(ann_obj_id)
                id_label = self._id_label(ann_obj_id, score=similarity)

                # Skip invalid detections
                if ann_frame_idx < 0 or bbx is None:
                    if anchor_idx == 0:
                        self._log_warn(f"Skipping object ID {ann_obj_id}: no valid detection candidate.")
                    continue

                # plot img_bbx and target_img_bbx
                plt.figure(figsize=(9, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(img_bbx)
                plt.title(f"{id_label} frame {ann_frame_idx}")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                if target_img_bbx is not None:
                    plt.imshow(target_img_bbx)
                    plt.title(f"target object from bag of features")
                else:
                    plt.text(0.5, 0.5, "No CLIP gallery match", ha="center", va="center")
                    plt.title("target object from bag of features")
                plt.axis("off")
                plt.savefig(
                    os.path.join(
                        similarity_dir,
                        f"similarity_id_{ann_obj_id:02d}_frame_{ann_frame_idx:04d}_anchor_{anchor_idx:02d}.png",
                    )
                )
                plt.close()

                box = np.array(bbx, dtype=np.float32)
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box=box,
                )
                frame_prompt_boxes.setdefault(int(ann_frame_idx), []).append(
                    {
                        "obj_id": int(ann_obj_id),
                        "anchor_idx": int(anchor_idx),
                        "box": box,
                        "score": similarity,
                    }
                )

                plt.figure(figsize=(9, 6))
                plt.title(f"frame {ann_frame_idx}")
                if frames is not None and ann_frame_idx < len(frames):
                    pil_img = frames.read_pil(ann_frame_idx)
                else:
                    frame_path = os.path.join(video_dir, frame_names[ann_frame_idx])
                    if platform.system() == "Windows":
                        frame_path = string_path_to_windows(frame_path)
                    pil_img = Image.open(frame_path).convert("RGB")
                plt.imshow(pil_img)
                show_box(
                    box,
                    plt.gca(),
                    color=id_color,
                    linewidth=2.5,
                    label=self._id_label(ann_obj_id),
                )
                plt.savefig(
                    os.path.join(
                        overlay_dir,
                        f"annotated_frame_{ann_frame_idx:04d}_person_id_{ann_obj_id:02d}_anchor_{anchor_idx:02d}.png",
                    )
                )
                plt.close()

        # Save one combined overlay per frame containing all prompt boxes for easier QA.
        for frame_idx in sorted(frame_prompt_boxes.keys()):
            prompts = frame_prompt_boxes.get(frame_idx, [])
            if len(prompts) == 0:
                continue

            if frames is not None and frame_idx < len(frames):
                pil_img = frames.read_pil(frame_idx)
            else:
                frame_path = os.path.join(video_dir, frame_names[frame_idx])
                if platform.system() == "Windows":
                    frame_path = string_path_to_windows(frame_path)
                pil_img = Image.open(frame_path).convert("RGB")

            plt.figure(figsize=(9, 6))
            plt.title(f"frame {frame_idx} all prompt boxes")
            plt.imshow(pil_img)
            for prompt in prompts:
                obj_id = int(prompt["obj_id"])
                show_box(
                    prompt["box"],
                    plt.gca(),
                    color=self._id_color(obj_id),
                    linewidth=2.5,
                    label=self._id_label(obj_id, score=prompt.get("score")),
                )
            plt.savefig(os.path.join(overlay_dir, f"annotated_frame_{frame_idx:04d}_all_prompts.png"))
            plt.close()
        if len(frame_prompt_boxes) > 0:
            self._log_info(
                f"Saved combined prompt overlay images for {len(frame_prompt_boxes)} frame(s) in '{overlay_dir}'."
            )

        export_cfg = self.assignment_config.get("exports", {})
        if bool(export_cfg.get("export_prompt_overlays_mp4", True)):
            self._export_image_dir_to_mp4(
                overlay_dir,
                fps=float(export_cfg.get("prompt_overlays_fps", 4.0)),
                video_filename="prompt_frame_overlays.mp4",
                remove_images=bool(export_cfg.get("remove_prompt_overlay_images", True)),
            )
        if bool(export_cfg.get("export_prompt_similarity_mp4", True)):
            self._export_image_dir_to_mp4(
                similarity_dir,
                fps=float(export_cfg.get("prompt_similarity_fps", 4.0)),
                video_filename="prompt_similarity_panels.mp4",
                remove_images=bool(export_cfg.get("remove_prompt_similarity_images", True)),
            )


    def compute_clip_features(self, mask_data):
        for mask_id, data in mask_data.items():
            # Skip if features were already loaded (slim masks.npy resume).
            existing = data.get("features")
            if existing is not None and hasattr(existing, "shape") and len(existing) > 0:
                continue
            # get the bounding boxes
            crops = data.get("img_bbx", [])
            features = self._encode_clip_batch(crops)
            if len(features) != 0:
                features = torch.stack(features, dim=0)

            # img_bbx = data["img_bbx"]
            # features = self.id_extractor(img_bbx)
            data["features"] = features


    def plot_tsne(self, mask_data, seq_name):
        if len(mask_data) > 1:
            features = []
            labels = []
            frames = []
            for mask_id, data in mask_data.items():
                feat = data.get("features", None)
                if feat is None:
                    continue
                if isinstance(feat, list):
                    if len(feat) == 0:
                        continue
                    feat_tensor = torch.stack(feat, dim=0).detach().cpu()
                else:
                    if not hasattr(feat, "shape"):
                        continue
                    if len(feat) == 0:
                        continue
                    feat_tensor = feat.detach().cpu()

                frame_source = data.get("img_bbx_frame", data.get("frame", []))
                frame_arr = np.asarray(frame_source)
                n_feat = int(feat_tensor.shape[0])
                if frame_arr.shape[0] < n_feat:
                    # In rare cases fallback frame list may be shorter than features.
                    # Keep visualization stable by padding with last known frame.
                    if frame_arr.shape[0] == 0:
                        frame_arr = np.zeros((n_feat,), dtype=np.int32)
                    else:
                        pad = np.full((n_feat - frame_arr.shape[0],), int(frame_arr[-1]), dtype=frame_arr.dtype)
                        frame_arr = np.concatenate([frame_arr, pad], axis=0)
                elif frame_arr.shape[0] > n_feat:
                    frame_arr = frame_arr[:n_feat]

                features.append(feat_tensor)
                labels.append(np.array([mask_id] * n_feat))
                frames.append(frame_arr)
            if len(features) == 0:
                self._log_warn("Skipping t-SNE: no valid features available.")
                return
            features = torch.cat(features).cpu().numpy()
            labels = np.concatenate(labels)
            frames = np.concatenate(frames)
            if features.shape[0] != frames.shape[0]:
                n = min(features.shape[0], frames.shape[0], labels.shape[0])
                if n <= 0:
                    self._log_warn("Skipping t-SNE: empty aligned arrays after sanity checks.")
                    return
                self._log_warn(
                    f"t-SNE input mismatch (features={features.shape[0]}, frames={frames.shape[0]}, labels={labels.shape[0]}). "
                    f"Truncating to {n}."
                )
                features = features[:n]
                labels = labels[:n]
                frames = frames[:n]
            tsne = TSNE(n_components=2, perplexity=min(30, len(features)//2), max_iter=1000, random_state=42)
            tsne_features = tsne.fit_transform(features)
            plt.figure(figsize=(10, 10))
            # label each point with their corresponding frame number on top of the scatter plot
            for i, txt in enumerate(frames):
                plt.annotate(txt, (tsne_features[i, 0], tsne_features[i, 1]), fontsize=8)
            sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=labels, palette="Set1", legend="full")
            plt.title("t-SNE of features")
            plt.savefig(os.path.join(seq_name, "features_tsne.png"))
            plt.close()


    def _debug_crop_summary(self):
        """Whether to render the per-person crop-summary / t-SNE debug PNGs.

        Off by default: these matplotlib renders are pure
        debugging aids and cost wall time; the underlying img_bbx crops that feed
        cross-camera matching are always built regardless. Enable with
        ``exports.debug_crop_summary: true`` (or ``--debug_crop_summary``).
        """
        return bool(self.assignment_config.get("exports", {}).get("debug_crop_summary", False))

    def _debug_full_masks_npy(self):
        """Whether masks.npy should keep the full-res frames + masks.

        Off by default: masks.npy is slimmed to what matching/resume need
        (centroids + crops + features), ~10x smaller. The full-res masks remain
        available as the per-frame mask PNGs. Enable with
        ``exports.debug_full_masks_npy`` / ``--debug_full_masks_npy``.
        """
        return bool(self.assignment_config.get("exports", {}).get("debug_full_masks_npy", False))

    def _compute_centroids(self, mask_data):
        """Precompute each person's per-sampled-frame mask centroid (xy).

        The cross-camera epipolar score only needs the centroid of each sampled
        mask, not the full-res mask. Storing the centroid lets us drop the heavy
        masks from the in-memory dict and from masks.npy while keeping matching
        byte-identical. Aligned index-for-index with ``frame``.
        """
        for data in mask_data.values():
            if not isinstance(data, dict) or "centroid_xy" in data:
                continue
            cents = []
            for m in data.get("mask", []):
                arr = np.asarray(m)
                arr = arr[0] if arr.ndim == 3 else arr
                c = self._mask_centroid_xy1(arr)
                cents.append(None if c is None else [float(c[0]), float(c[1])])
            data["centroid_xy"] = cents

    def _slim_mask_record(self, mask_data):
        """Drop the full-res frames + masks from an in-memory mask dict.

        After img_bbx + features + centroids are computed, the full ``img`` and
        ``mask`` arrays are never read again — yet the init camera's dict is held
        resident for the whole multi-camera run and is the bulk of masks.npy.
        Drop them; matching keeps using img_bbx / features / centroid_xy. The
        full-res masks remain on disk as the per-frame mask PNGs.
        """
        for data in mask_data.values():
            if isinstance(data, dict):
                data.pop("img", None)
                data.pop("mask", None)

    def _finalize_mask_cache(self, mask_data, out_path):
        """Precompute centroids and (unless full-dump) slim the dict + re-save.

        masks.npy was written full by the inner segmentation method (a crash-safe
        completion marker); here we replace it with the slim record once the
        matching essentials exist. With ``debug_full_masks_npy`` we keep the full
        cache instead.
        """
        self._compute_centroids(mask_data)
        if self._debug_full_masks_npy():
            return  # keep the full masks.npy written by the inner method
        self._slim_mask_record(mask_data)
        try:
            _atomic_np_save(os.path.join(out_path, "masks.npy"), mask_data)
        except Exception as exc:  # noqa: BLE001
            self._log_warn(f"Failed to re-save slim mask cache at '{out_path}': {exc}")

    def save_picked_masks(self, mask_data, output_folder, render_viz=True):
        if render_viz:
            self._log_info(f"Generating per-ID mask crop summaries in '{output_folder}'.")
        if len(mask_data) == 0:
            if render_viz:
                self._log_warn("No mask data available to summarize.")
            return

        first_key = sorted(mask_data.keys())[0]
        keys = list(mask_data[first_key].keys())
        self._log_info(f"Mask-data fields: {keys}")

        def _bbox_from_mask(mask_arr):
            m = np.asarray(mask_arr)
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            ys, xs = np.where(m > 0)
            if xs.size == 0 or ys.size == 0:
                return None
            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max()) + 1
            y2 = int(ys.max()) + 1
            return x1, y1, x2, y2

        def _sanitize_bbox(bbx_raw, img_shape, mask_arr=None):
            h, w = int(img_shape[0]), int(img_shape[1])
            arr = np.asarray(bbx_raw).squeeze()
            x1 = y1 = x2 = y2 = None

            try:
                if arr.shape == (2, 2):
                    x1, y1 = float(arr[0][0]), float(arr[0][1])
                    x2, y2 = float(arr[1][0]), float(arr[1][1])
                elif arr.ndim == 1 and arr.shape[0] >= 4:
                    a0, a1, a2, a3 = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
                    # Prefer xyxy; fall back to xywh if needed.
                    if a2 > a0 and a3 > a1:
                        x1, y1, x2, y2 = a0, a1, a2, a3
                    else:
                        x1, y1, x2, y2 = a0, a1, a0 + max(0.0, a2), a1 + max(0.0, a3)
            except Exception:
                x1 = y1 = x2 = y2 = None

            if x1 is None or y1 is None or x2 is None or y2 is None:
                fallback = _bbox_from_mask(mask_arr) if mask_arr is not None else None
                if fallback is None:
                    return None
                x1, y1, x2, y2 = fallback

            if not np.isfinite([x1, y1, x2, y2]).all():
                fallback = _bbox_from_mask(mask_arr) if mask_arr is not None else None
                if fallback is None:
                    return None
                x1, y1, x2, y2 = fallback

            x1 = int(np.clip(np.floor(x1), 0, max(0, w - 1)))
            y1 = int(np.clip(np.floor(y1), 0, max(0, h - 1)))
            x2 = int(np.clip(np.ceil(x2), 0, w))
            y2 = int(np.clip(np.ceil(y2), 0, h))
            if x2 <= x1 or y2 <= y1:
                fallback = _bbox_from_mask(mask_arr) if mask_arr is not None else None
                if fallback is None:
                    return None
                fx1, fy1, fx2, fy2 = fallback
                x1 = int(np.clip(fx1, 0, max(0, w - 1)))
                y1 = int(np.clip(fy1, 0, max(0, h - 1)))
                x2 = int(np.clip(fx2, 0, w))
                y2 = int(np.clip(fy2, 0, h))
                if x2 <= x1 or y2 <= y1:
                    return None
            return x1, y1, x2, y2

        # Build per-person crops (img_bbx) — ALWAYS, because img_bbx feeds the
        # cross-camera CLIP gallery (matching). The matplotlib crop-summary PNG
        # is pure debugging viz and is only rendered when render_viz is set
        # (exports.debug_crop_summary); the img_bbx output is identical either
        # way, so cross-camera matching is unaffected by the flag.
        for obj_id, data in mask_data.items():
            # Slim cache (resume): full frames were dropped but img_bbx is already
            # present — nothing to rebuild or render for this person.
            if "img" not in data:
                data.setdefault("img_bbx", [])
                data.setdefault("img_bbx_frame", [])
                continue
            imgs = data["img"]
            masks = data["mask"]
            frames = data["frame"]
            bboxs = data["bbox"]
            data["img_bbx"] = []
            data["img_bbx_frame"] = []
            axes = None
            if render_viz:
                cols = 3
                rows = max(1, (len(frames) + cols - 1) // cols)
                fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
                axes = axes.flatten()
            skipped = 0
            for idx, (img, mask, frame_idx, bbx) in enumerate(zip(imgs, masks, frames, bboxs)):
                ax = axes[idx] if render_viz else None
                coords = _sanitize_bbox(bbx, img.shape, mask_arr=mask)
                if coords is None:
                    skipped += 1
                    if render_viz:
                        ax.text(0.5, 0.5, "Invalid/empty bbox", ha="center", va="center")
                        ax.set_title(f"Object {obj_id} Frame {frame_idx}")
                        ax.axis("off")
                    continue
                x1, y1, x2, y2 = coords
                img_bbx = img[y1:y2, x1:x2]
                if img_bbx.size == 0:
                    skipped += 1
                    if render_viz:
                        ax.text(0.5, 0.5, "Empty crop", ha="center", va="center")
                        ax.set_title(f"Object {obj_id} Frame {frame_idx}")
                        ax.axis("off")
                    continue

                data["img_bbx"].append(img_bbx)
                data["img_bbx_frame"].append(int(frame_idx))
                if render_viz:
                    ax.imshow(img_bbx)
                    ax.set_title(f"Object {obj_id} Frame {frame_idx}")
                    ax.axis("off")

            if render_viz:
                for ax in axes[len(frames):]:
                    ax.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f"person_{obj_id:02d}_crop_summary.png"))
                plt.close()
            if skipped > 0:
                self._log_warn(
                    f"Object {obj_id}: skipped {skipped} invalid/empty bbox crops while generating summaries."
                )

    def _collect_sam3_text_detections(self, img, conf_thresh, include_clip_features=True):
        """SAM3 open-vocabulary 'person' detection — the sam3_prompt_light drop-in
        for YOLO. Returns (PIL image, [(crop, feature, bbox_xyxy, score)]), the
        same contract as _collect_yolo_detections, so anchors/matching/tracking
        are reused unchanged.
        """
        cfg = self._get_matching_cfg()
        np_img = np.asarray(img.convert("RGB"))
        h, w = np_img.shape[:2]
        dets = self.predictor.text_detect(np_img, text="person", conf_thresh=conf_thresh)
        crops, boxes_xyxy, scores = [], [], []
        for bbx, sc in dets:
            x1, y1, x2, y2 = (int(np.clip(bbx[0], 0, w - 1)), int(np.clip(bbx[1], 0, h - 1)),
                              int(np.clip(bbx[2], 0, w)), int(np.clip(bbx[3], 0, h)))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = np_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            boxes_xyxy.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            scores.append(float(sc))
        if len(crops) > 1 and bool(cfg.get("yolo_dedup_enable", True)):
            raw = [(c, None, b, s) for c, b, s in zip(crops, boxes_xyxy, scores)]
            deduped = self._deduplicate_detections(raw, iou_threshold=float(cfg.get("yolo_dedup_iou", 0.75)))
            crops = [d[0] for d in deduped]
            boxes_xyxy = [d[2] for d in deduped]
            scores = [float(d[3]) for d in deduped]
        detections = []
        if len(crops) > 0 and include_clip_features:
            feats = self._encode_clip_batch(crops)
            detections = [(c, f, b, s) for c, f, b, s in zip(crops, feats, boxes_xyxy, scores)]
        elif len(crops) > 0:
            detections = [(c, None, b, s) for c, b, s in zip(crops, boxes_xyxy, scores)]
        return img, detections

    def _collect_yolo_detections(self, img_or_path, conf_thresh: float, include_clip_features: bool = True):
        """Run YOLO person detection on a frame.

        Args:
            img_or_path: PIL Image or file path string.
            conf_thresh: Minimum YOLO confidence for person class.
            include_clip_features: If True, compute CLIP features for each detection.

        Returns:
            (PIL image, list of (crop, feature, bbox, score))
        """
        cfg = self._get_matching_cfg()
        yolo_dedup_enable = bool(cfg.get("yolo_dedup_enable", True))
        yolo_dedup_iou = float(cfg.get("yolo_dedup_iou", 0.75))
        if isinstance(img_or_path, Image.Image):
            img = img_or_path.convert("RGB")
        else:
            with Image.open(img_or_path) as pil_img:
                img = pil_img.copy()
        # sam3_prompt / sam3_prompt_light: detect people with SAM3's own
        # open-vocabulary text detector ("person") instead of YOLO — same
        # (img, [(crop,feat,bbox,score)]) contract, so all downstream
        # anchor/bootstrap/matching code is unchanged. For sam3_prompt this
        # removes the redundant YOLO detection in the multi-view bootstrap (the
        # only place sam3_prompt called YOLO); the detector is already loaded.
        if self.sam_version in ("sam3_prompt", "sam3_prompt_light"):
            return self._collect_sam3_text_detections(img, conf_thresh, include_clip_features)
        out = self.yolo_model(img, verbose=False)
        detections = []
        np_img = np.asarray(img)
        crops = []
        boxes_xyxy = []
        scores = []
        for box in out[0].boxes:
            if int(box.cls) == 0 and float(box.conf) > conf_thresh:
                bbx = box.xyxy.cpu().numpy()[0]
                crop = np_img[int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2])]
                if crop.size == 0:
                    continue
                crops.append(crop)
                boxes_xyxy.append(np.array([bbx[0], bbx[1], bbx[2], bbx[3]], dtype=np.float32))
                scores.append(float(box.conf))
        if len(crops) > 1 and yolo_dedup_enable:
            raw = [(c, None, b, s) for c, b, s in zip(crops, boxes_xyxy, scores)]
            deduped = self._deduplicate_detections(raw, iou_threshold=yolo_dedup_iou)
            if len(deduped) < len(raw):
                self._log_info(
                    f"YOLO person-box dedup removed {len(raw) - len(deduped)} overlapping detections "
                    f"(iou_thr={yolo_dedup_iou:.2f})."
                )
            crops = [d[0] for d in deduped]
            boxes_xyxy = [d[2] for d in deduped]
            scores = [float(d[3]) for d in deduped]
        if len(crops) > 0 and include_clip_features:
            feats = self._encode_clip_batch(crops)
            for crop, feat, bbox, score in zip(crops, feats, boxes_xyxy, scores):
                detections.append((crop, feat, bbox, score))
        elif len(crops) > 0:
            for crop, bbox, score in zip(crops, boxes_xyxy, scores):
                detections.append((crop, None, bbox, score))
        return img, detections
