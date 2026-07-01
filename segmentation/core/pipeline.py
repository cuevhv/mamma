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
        "sample_interval": 5,
        "min_samples": 5,
        "yolo_person_conf": 0.5,
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
        "init_anchor_enable": True,
        "init_anchor_samples": 80,
        "init_anchor_score_threshold": 0.2,
        "init_anchor_max_per_id": 0,  # <=0 means auto
        "clip_weight": 0.35,
        "epipolar_weight": 0.65,
        # Resolution-aware by default ("auto"): scaled from image diagonal.
        "epipolar_sigma_px": "auto",
        "max_epipolar_dist_px": "auto",
        "epipolar_sigma_diag_ratio": 0.055,
        "max_epipolar_dist_diag_ratio": 0.12,
        "clip_min_similarity": 0.0,
        "combined_min_score": 0.2,
        "allow_missing_ids": False,
    },
    "masks": {
        "merge_duplicate_tracklets": True,
        "merge_iou_threshold": 0.95,
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
        # the multi-view bootstrap used to call. (issue #14)
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

    def _abs_frame(self, relative_idx):
        """Convert a relative frame index to absolute for logging."""
        return int(relative_idx) + self._frame_offset

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
        mid-run). Opt-in via sam.prune_memory_state for very long clips. (issue #14)
        """
        per_obj = inference_state.get("output_dict_per_obj") if isinstance(inference_state, dict) else None
        for obj in (per_obj or {}).values():
            nc = obj.get("non_cond_frame_outputs")
            if nc:
                for j in [k for k in nc if abs(k - frame_idx) > window]:
                    nc.pop(j, None)

    def run_propagation(self, inference_state, cam_name="", image_size=None):
        # Run propagation and spill each frame's masks to disk as they are
        # produced (issue #14). The previous code accumulated every frame's
        # full-resolution masks in an in-RAM dict (~5 GB/camera on long clips,
        # the user-reported host-RAM blow-up); MaskStore keeps the same content
        # bit-packed on disk so peak RAM is O(1 frame). Behaviour is unchanged.
        video_segments = MaskStore()
        sam_cfg = self.assignment_config.get("sam")
        if sam_cfg is None:
            sam_cfg = self.assignment_config.get("sam2", {})
            if sam_cfg:
                self._log_warn("Config key 'sam2' is deprecated; use 'sam' instead.")
        # Opt-in memory-state pruning (issue #14): after each frame's mask is spilled
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

        detected_ids = video_segments.all_obj_ids()
        video_segments, detected_ids = self._postprocess_tracklets(
            cam_name or "SAM", video_segments, detected_ids, image_size=image_size,
        )

        return video_segments


    def set_output_path(self, output_path):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

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

    def _read_frame_rgb(self, frame_idx=None, file_path=None):
        """Read a frame as an RGB numpy array. See _read_frame_pil for arguments."""
        return np.array(self._read_frame_pil(frame_idx=frame_idx, file_path=file_path))

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
        """Resolve proactive SAM CPU-offload flags from the ``sam`` config (issue #14).

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
        # Opt-in float16 frame storage (default off → byte-identical). Halves the
        # decoded-frame tensor (CPU or GPU) when enabled via sam.fp16_frames.
        from core.predictor_factory import set_frame_storage_fp16, set_lazy_frame_loading
        _sam_cfg = self.assignment_config.get("sam") or self.assignment_config.get("sam2", {}) or {}
        set_frame_storage_fp16(bool(_sam_cfg.get("fp16_frames", False)))
        # Opt-in lazy frame loading (default off → byte-identical): bounds host RAM
        # to a small LRU window instead of holding all decoded frames. (issue #14)
        set_lazy_frame_loading(bool(_sam_cfg.get("lazy_frame_loading", False)),
                               int(_sam_cfg.get("lazy_frame_window", 8)))

        # Proactive CPU offload (issue #14): SAM loads the whole decoded video
        # into VRAM at init_state, so VRAM grows with frame count. We pass the
        # offload flags up front (offload_video_to_cpu defaults ON — it only
        # relocates tensors to the host, masks are identical) so long / many-
        # camera clips don't OOM in the first place. The reactive retry below is
        # the backstop if even that is not enough or offload was disabled.
        n_frames = len(frame_names) if frame_names else None
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
        np.save(cache_path, payload, allow_pickle=True)
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
        clip_weight = float(cfg.get("clip_weight", 0.35))
        epipolar_weight = float(cfg.get("epipolar_weight", 0.65))
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
        cam_int = np.asarray(cam_int)
        if cam_int.ndim == 2 and cam_int.shape[0] >= 3 and cam_int.shape[1] >= 3:
            return cam_int[:3, :3].astype(np.float64)
        if cam_int.ndim >= 3:
            mat = cam_int.reshape(-1, cam_int.shape[-2], cam_int.shape[-1])[0]
            return mat[:3, :3].astype(np.float64)
        return None

    def _as_extrinsics(self, cam_ext):
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
            return self._as_extrinsics(mat)
        return None

    def _compute_fundamental_matrix(self, source_cam_data, target_cam_data):
        if source_cam_data is None or target_cam_data is None:
            return None
        if "cam_int" not in source_cam_data or "cam_int" not in target_cam_data:
            return None
        if "cam_ext" not in source_cam_data or "cam_ext" not in target_cam_data:
            return None

        k1 = self._as_intrinsics(source_cam_data["cam_int"])
        k2 = self._as_intrinsics(target_cam_data["cam_int"])
        e1 = self._as_extrinsics(source_cam_data["cam_ext"])
        e2 = self._as_extrinsics(target_cam_data["cam_ext"])
        if k1 is None or k2 is None or e1 is None or e2 is None:
            return None

        r1 = e1[:3, :3]
        t1 = e1[:3, 3]
        r2 = e2[:3, :3]
        t2 = e2[:3, 3]

        r = r2 @ r1.T
        t = t2 - (r @ t1)
        tx = np.array(
            [
                [0.0, -t[2], t[1]],
                [t[2], 0.0, -t[0]],
                [-t[1], t[0], 0.0],
            ],
            dtype=np.float64,
        )
        e = tx @ r
        try:
            f = np.linalg.inv(k2).T @ e @ np.linalg.inv(k1)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(f).all():
            return None
        return f

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
        score = float(np.exp(-0.5 * ((dist / sigma) ** 2)))
        return score, dist

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
        sigma_ratio = float(cfg.get("epipolar_sigma_diag_ratio", 0.055))
        max_dist_ratio = float(cfg.get("max_epipolar_dist_diag_ratio", 0.12))

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

    def _max_pairwise_iou(self, boxes):
        """Compute the maximum pairwise IoU among a list of [x1,y1,x2,y2] boxes."""
        if len(boxes) <= 1:
            return 0.0
        max_iou = 0.0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = self._bbox_iou_xyxy(boxes[i], boxes[j])
                if iou > max_iou:
                    max_iou = iou
        return max_iou

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
        # camera on top of the mask RAM (issue #14).
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

        def _process_frame(frame_idx):
            if frame_idx not in video_segments:
                return
            image = _frame_uint8(frame_idx)
            image = cv2.resize(image, (w_original, h_original))
            masked_img = image.copy() if not skip_overlays else None
            frame_samples = []

            for out_obj_id, out_mask in video_segments[frame_idx].items():
                mask = out_mask[0] if out_mask.ndim == 3 else out_mask
                mask_bw = (mask * 255).astype(np.uint8)
                if frame_idx % vis_frame_stride == 0:
                    frame_samples.append((out_obj_id, image, mask_bw, frame_idx))
                if not skip_overlays:
                    colored_mask = show_mask_cv2(out_mask, obj_id=out_obj_id)[:, :, :3]
                    masked_img = masked_img + colored_mask * 0.5
                cv2.imwrite(os.path.join(mask_folder, f"mask_{frame_idx:04d}_{(int(out_obj_id)+1):02d}.png"), mask_bw)

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
            # processing raises (close() is idempotent). (issue #14 cleanup)
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
        np.save(os.path.join(output_path, "masks.npy"), masks)
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

        cfg = self.assignment_config.get("matching", {})
        init_anchor_enable = bool(cfg.get("init_anchor_enable", True))
        if self._is_sam3 and init_anchor_enable:
            self._log_info(
                "Multi-frame anchors automatically disabled for SAM3 — "
                "SAM3 consolidates all objects jointly per conditioning frame, "
                "so partial-object anchor frames produce poisoned tracking memory."
            )
            init_anchor_enable = False
        if init_anchor_enable and n_frames > 1:
            init_best_similarity, init_anchor_matches = self._build_init_anchor_matches(
                frames=frames,
                init_frame_idx=frame_id,
                init_detections=detections,
            )
        else:
            self._log_info("Init multi-anchor prompting disabled; using a single init-frame prompt per ID.")

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
        self._save_anchor_report(output_path, init_anchor_matches, n_frames=n_frames)
        self._save_anchor_visualizations(output_path, init_anchor_matches, n_frames=n_frames)

        self._log_info(f"[{cam_name}] Starting propagation from init anchors.")
        video_segments = self.run_propagation(inference_state)

        obj_ids = range(ann_obj_id)
        os.makedirs(output_path, exist_ok=True)
        save_masks = self.save_images_from_video(inference_state, video_segments, obj_ids, output_path)
        np.save(os.path.join(output_path, "masks.npy"), save_masks)
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
        np.save(os.path.join(output_path, "masks.npy"), save_masks)
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
        min_mask_ratio = float(cfg.get("min_mask_area_ratio", 0.005))
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
        min_ratio = float(cfg.get("min_mask_area_ratio", 0.005))
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
        2. Discard tiny tracklets (noise, not real people)

        Controlled by the ``masks:`` config section.
        """
        mask_cfg = self.assignment_config.get("masks", {})

        # Step 1: merge duplicates
        if bool(mask_cfg.get("merge_duplicate_tracklets", True)):
            merge_iou = float(mask_cfg.get("merge_iou_threshold", 0.95))
            video_segments, detected_ids = self._merge_duplicate_tracklets(
                cam_name, video_segments, detected_ids, iou_thresh=merge_iou,
            )

        # Step 2: discard tiny tracklets
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

        return video_segments, detected_ids

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

        self._log_info(f"[{cam_name}] Starting SAM3 prompt session on '{sam_source}'.")
        predictor.start_session(sam_source)

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
        # store instead of returning one big in-RAM dict (issue #14).
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
            # (close() is idempotent). (issue #14 cleanup)
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

                # Per-frame mask PNG
                cv2.imwrite(
                    os.path.join(mask_folder, f"mask_{frame_idx:04d}_{(out_id + 1):02d}.png"),
                    mask_uint8,
                )

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

        np.save(os.path.join(output_path, "masks.npy"), save_masks)
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
            return mask_data

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

    def _sam3_prompt_update_feature_bank(self, cam_name, frames, video_segments, id_remap):
        """Add clean crops from this camera to the per-person feature bank.

        Samples frames, selects crops with low bbox overlap, encodes with CLIP,
        and appends to self.subject_feature_bank. This enriches the gallery for
        subsequent cameras with new viewpoints.
        """
        n_frames = len(video_segments)
        sample_count = min(5, n_frames)
        sample_idxs = [int(i) for i in np.linspace(
            0, max(video_segments.keys()), num=sample_count, dtype=int
        )]
        added = 0

        for fidx in sample_idxs:
            if fidx not in video_segments:
                continue
            try:
                frame_rgb = frames.read_rgb(fidx)
            except Exception as exc:
                self._log_warn(f"Failed to read frame {fidx} for feature bank update: {exc}")
                continue

            # Compute bboxes for overlap check
            bboxes = {}
            for sid, mask in video_segments[fidx].items():
                if sid not in id_remap:
                    continue
                if mask.ndim == 3:
                    mask = mask[0]
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                bboxes[sid] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

            for sid, (x1, y1, x2, y2) in bboxes.items():
                # Skip crops with high bbox overlap
                max_iou = max(
                    (self._bbox_iou_xyxy([x1,y1,x2,y2], list(b))
                     for other, b in bboxes.items() if other != sid),
                    default=0.0,
                )
                if max_iou > 0.3:
                    continue

                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                mapped_id = id_remap[sid]
                feats = self._encode_clip_batch([crop])
                if feats:
                    if self._append_to_subject_bank(mapped_id, feats[0], crop=crop):
                        added += 1

        if added > 0:
            self._log_info(f"[{cam_name}] Updated feature bank: +{added} clean crops from this view.")

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
            avg_feat = (feats * weights.unsqueeze(1)).sum(dim=0)
            avg_feat = F.normalize(avg_feat.unsqueeze(0), dim=1).squeeze(0)
            local_avg_feats.append(avg_feat)
            local_valid_ids.append(sid)

        if not local_avg_feats:
            self._log_warn(f"[{cam_name}] No valid crops for CLIP matching.")
            return {sid: sid for sid in local_ids}

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
        clip_weight = float(cfg.get("clip_weight", 0.35))
        epipolar_weight = float(cfg.get("epipolar_weight", 0.65))
        clip_min_similarity = float(cfg.get("clip_min_similarity", 0.0))
        combined_min_score = float(cfg.get("combined_min_score", 0.2))
        image_size = frames.image_size if len(frames) > 0 else None
        epipolar_sigma_px, max_epipolar_dist_px, _ = self._resolve_epipolar_px_params(
            cfg, image_size=image_size
        )

        f_matrix = self._compute_fundamental_matrix(source_cam_data, target_cam_data)
        s_epi = np.zeros((k, m), dtype=np.float32)
        has_epi = np.zeros((k, m), dtype=bool)
        has_any_epi = False

        if f_matrix is not None:
            self._log_info(f"[{cam_name}] Using epipolar geometry for ID remapping.")
            has_any_epi = True
            for row, oid in enumerate(init_obj_ids):
                for col in range(m):
                    sid = local_valid_ids[col]
                    epi_scores = []
                    for fidx in sample_frame_idxs:
                        ref_point = self._reference_point_from_mask_data(mask_data[oid], fidx)
                        target_centroid = local_centroids_by_frame.get(fidx, {}).get(sid)
                        if ref_point is None or target_centroid is None:
                            continue
                        epi_score, epi_dist = self._epipolar_score(
                            f_matrix=f_matrix,
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
        np.save(os.path.join(output_path, "masks.npy"), save_masks)
        self._log_info(f"[{cam_name}] Saved mask cache: {os.path.join(output_path, 'masks.npy')}")
        self._cleanup_sanitized_video_dir(_runtime_video_dir, sam_source)
        return save_masks

    def _cross_camera_strategy(self, calibration_available=None):
        """Resolve the cross-camera identity strategy for non-init cameras.

        'auto' (default) is calibration- and backend-aware, per the validation
        campaign (re-verified 2026-06-29 with epipolar geometry active):
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
        ``n_det * mean_conf`` (×1.5 when n_det == expected_subjects), mirroring the
        init-camera auto-selector, then returns every detection (the cross-camera
        remap prunes surplus by identity).

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

    def _detect_and_track_local(self, frames, output_path, frame_id=None, expected_subjects=None):
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
                f"[{cam_name}] track-then-match propagation failed: {exc}. Skipping camera.")
            self._cleanup_sanitized_video_dir(runtime_video_dir, sam_source)
            return None, None, None

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
            frames, output_path, expected_subjects=expected_subjects)
        if local_ids is None:
            return mask_data

        id_remap = self._remap_tracklets_to_init(
            cam_name, frames, mask_data, local_ids, video_segments, best_frame,
            source_cam_data=source_cam_data, target_cam_data=target_cam_data,
        )
        save_masks = self._build_masks_from_remap(frames, video_segments, id_remap, output_path)
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
            return mask_data

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
            self._log_error(f"Failed while injecting anchor prompts into SAM2 state: {e}. Skipping propagation.")
            self._cleanup_sanitized_video_dir(runtime_video_dir, sam_source)
            return mask_data

        video_segments = self.run_propagation(inference_state)

        self._log_info(f"Propagation completed for '{cam_name}'. Produced {len(video_segments)} frame segments.")
        obj_ids = sorted(best_similarity.keys())
        new_mask_data = self.save_images_from_video(inference_state, video_segments, obj_ids, output_path)
        np.save(os.path.join(output_path, "masks.npy"), new_mask_data)
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

    def _build_init_anchor_matches(self, frames, init_frame_idx, init_detections):
        """
        Build temporal anchor prompts for the init camera by matching detections
        from sampled frames to init IDs using CLIP-only Hungarian assignment.
        """
        if len(init_detections) == 0:
            return {}, {}

        cfg = self.assignment_config.get("matching", {})
        yolo_person_conf = float(cfg.get("yolo_person_conf", 0.5))
        sample_interval = int(cfg.get("sample_interval", 5))
        min_samples = int(cfg.get("min_samples", 5))
        init_anchor_samples = max(min_samples, len(frames) // max(1, sample_interval))
        init_anchor_score_threshold = float(cfg.get("init_anchor_score_threshold", 0.25))
        init_anchor_max_per_id = int(cfg.get("init_anchor_max_per_id", 0))
        if init_anchor_max_per_id <= 0:
            init_anchor_max_per_id = int(cfg.get("max_anchors_per_id", 0))
        min_anchor_frame_gap = int(cfg.get("min_anchor_frame_gap", 25))
        anchor_auto_max_cap = int(cfg.get("anchor_auto_max_cap", 20))
        anchor_samples_per_anchor = int(cfg.get("anchor_samples_per_anchor", 4))
        anchor_enforce_cross_id_uniqueness = bool(cfg.get("anchor_enforce_cross_id_uniqueness", True))
        anchor_conflict_iou = float(cfg.get("anchor_conflict_iou", 0.65))

        n_frames = len(frames)
        max_anchors_per_id = self._resolve_max_anchors_per_id(
            n_frames=n_frames,
            requested_max=init_anchor_max_per_id,
            min_frame_gap=min_anchor_frame_gap,
            auto_cap=anchor_auto_max_cap,
        )

        target_samples = max(init_anchor_samples, max_anchors_per_id * max(1, anchor_samples_per_anchor))
        sample_count = max(1, min(int(target_samples), n_frames))
        sampled_indices = np.linspace(0, n_frames - 1, num=sample_count, dtype=int)
        sampled_indices = sorted(set([int(init_frame_idx)] + [int(i) for i in sampled_indices.tolist()]))

        obj_ids = list(range(len(init_detections)))
        candidate_matches = {oid: [] for oid in obj_ids}
        rolling_bank = {}
        max_bank_size = self._get_feature_bank_max_size()

        # Seed each ID with its init-frame detection.
        for oid, (crop, feat, bbox, score) in enumerate(init_detections):
            if feat is None:
                continue
            feat_cpu = feat.detach().cpu().float().reshape(-1)
            rolling_bank[oid] = [feat_cpu]
            candidate_matches[oid].append(
                {
                    "score": max(1.0, float(score)),
                    "frame_idx": int(init_frame_idx),
                    "img_bbx": crop,
                    "bbox": np.asarray(bbox, dtype=np.float32),
                    "target_img_bbx": crop,
                    "feature": feat_cpu,
                }
            )

        self._log_info(
            f"Init-anchor sampling: frames={len(sampled_indices)}, anchors_per_id={max_anchors_per_id}, "
            f"min_gap={min_anchor_frame_gap}, score_threshold={init_anchor_score_threshold:.2f}"
        )

        from scipy.optimize import linear_sum_assignment

        for frame_idx in sampled_indices:
            if int(frame_idx) == int(init_frame_idx):
                continue
            frame_img = frames.read_pil(int(frame_idx))

            _, detections = self._collect_yolo_detections(
                frame_img,
                conf_thresh=yolo_person_conf,
                include_clip_features=True,
            )
            if len(detections) == 0:
                continue

            det_crops = [d[0] for d in detections]
            det_boxes = [d[2] for d in detections]
            det_feats = torch.stack([d[1].detach().cpu().float().reshape(-1) for d in detections], dim=0)

            k = len(obj_ids)
            m = det_feats.shape[0]
            if k == 0 or m == 0:
                continue

            s_clip = np.zeros((k, m), dtype=np.float32)
            for row, oid in enumerate(obj_ids):
                bank = rolling_bank.get(oid, [])
                if len(bank) == 0:
                    continue
                gallery = torch.stack(bank, dim=0)
                _, sims = cosine_knn(det_feats, gallery, topk=1)
                s_clip[row] = sims.squeeze(1).cpu().numpy()

            cost = 1.0 - s_clip
            row_ind, col_ind = linear_sum_assignment(cost)
            for row, col in zip(row_ind, col_ind):
                oid = obj_ids[row]
                raw_score = float(s_clip[row, col])
                if raw_score < init_anchor_score_threshold:
                    continue

                # Penalize detections that overlap heavily with other detections
                # on the same frame — SAM struggles with overlapping box prompts.
                penalty = self._overlap_penalty(det_boxes[col], det_boxes)
                score = raw_score * penalty

                feat = det_feats[col]
                candidate_matches[oid].append(
                    {
                        "score": score,
                        "frame_idx": int(frame_idx),
                        "img_bbx": det_crops[col],
                        "bbox": np.asarray(det_boxes[col], dtype=np.float32),
                        "target_img_bbx": det_crops[col],
                        "feature": feat,
                    }
                )

                bank = rolling_bank.setdefault(oid, [])
                should_add = True
                if len(bank) > 0:
                    bank_tensor = torch.stack(bank, dim=0)
                    sims = F.cosine_similarity(
                        F.normalize(feat.unsqueeze(0), dim=1),
                        F.normalize(bank_tensor, dim=1),
                    )
                    should_add = float(sims.max()) <= 0.998
                if should_add:
                    bank.append(feat)
                    while len(bank) > max_bank_size:
                        bank.pop(0)

        anchor_matches = {}
        best_similarity = {}
        for oid in obj_ids:
            selected = self._select_anchor_matches(
                candidate_matches.get(oid, []),
                max_anchors=max_anchors_per_id,
                min_frame_gap=min_anchor_frame_gap,
            )

            # Ensure init frame is always one of the prompts.
            init_candidates = [m for m in candidate_matches.get(oid, []) if int(m["frame_idx"]) == int(init_frame_idx)]
            if init_candidates:
                init_match = max(init_candidates, key=lambda x: float(x["score"]))
                if not any(int(m["frame_idx"]) == int(init_frame_idx) for m in selected):
                    selected.append(init_match)
                    selected = sorted(selected, key=lambda x: float(x["score"]), reverse=True)
                    selected = selected[:max(1, max_anchors_per_id)]
                    selected = sorted(selected, key=lambda x: int(x["frame_idx"]))

            if len(selected) == 0 and init_candidates:
                selected = [init_candidates[0]]

            anchor_matches[oid] = selected

            if len(selected) > 0:
                best = max(selected, key=lambda x: float(x["score"]))
                best_similarity[oid] = (
                    float(best["score"]),
                    int(best["frame_idx"]),
                    best["img_bbx"],
                    best["bbox"],
                    best.get("target_img_bbx"),
                )
            else:
                best_similarity[oid] = (0.0, -1, None, None, None)

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
                    f"Resolved {removed_conflicts} conflicting init-anchor assignments "
                    f"(cross-ID overlap IoU >= {anchor_conflict_iou:.2f})."
                )
            for oid in obj_ids:
                selected = anchor_matches.get(oid, [])
                if len(selected) > 0:
                    best = max(selected, key=lambda x: float(x["score"]))
                    best_similarity[oid] = (
                        float(best["score"]),
                        int(best["frame_idx"]),
                        best["img_bbx"],
                        best["bbox"],
                        best.get("target_img_bbx"),
                    )
                else:
                    best_similarity[oid] = (0.0, -1, None, None, None)

        anchor_counts = {oid: len(anchor_matches.get(oid, [])) for oid in obj_ids}
        self._log_info(f"Selected init temporal anchors per ID: {anchor_counts}")
        return best_similarity, anchor_matches

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
            sample_interval = int(cfg.get("sample_interval", 5))
            min_samples = int(cfg.get("min_samples", 5))
            available = len(frames) - int(cfg.get("start_frame", 10))
            samples = max(min_samples, available // max(1, sample_interval))
        yolo_person_conf = float(cfg.get("yolo_person_conf", 0.5))
        clip_weight = float(cfg.get("clip_weight", 0.35))
        epipolar_weight = float(cfg.get("epipolar_weight", 0.65))
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

        Off by default (issue #14 follow-up): these matplotlib renders are pure
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
        byte-identical. Aligned index-for-index with ``frame``. (issue #14)
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
        cache instead. (issue #14)
        """
        self._compute_centroids(mask_data)
        if self._debug_full_masks_npy():
            return  # keep the full masks.npy written by the inner method
        self._slim_mask_record(mask_data)
        try:
            np.save(os.path.join(out_path, "masks.npy"), mask_data)
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
        # way, so cross-camera matching is unaffected by the flag. (issue #14 follow-up)
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
        are reused unchanged. (issue #14 follow-up)
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
        # only place sam3_prompt called YOLO); the detector is already loaded. (issue #14)
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
