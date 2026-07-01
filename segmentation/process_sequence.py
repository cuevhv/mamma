#!/usr/bin/env python
"""Sequence processing entry point.

For local/container use, prefer ``run_ma_masks.py`` instead.

Usage:
    python process_sequence.py --ma_cap_dir /path --seq_name my_seq --out output
"""
import os
import sys
import glob
import argparse
import platform
from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np

from core.logging import logger
from utils.paths import string_path_to_windows
from core.video_reader import cam_data_from_video

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# Note: the YOLO checkpoint path is no longer a module-level constant; it
# flows through `process_seq(yolo_checkpoint=...)` so the MAMMA inference
# runner can inject `MAMMA_YOLO_CHECKPOINT` from .env via argparse. The
# previous `YOLO_CKPT = "weights/yolo12x.pt"` was a relative path that
# only resolved when this script was invoked with cwd=segmentation/, which
# was fragile and undocumented.
SAM2_CKPT = os.environ.get("SAM2_CHECKPOINT", "weights/sam2.1_hiera_large.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM3_CKPT = "facebook/sam3"
SAM3_CFG = None

INIT_FRAME_ID = None  # None = auto-select best init frame per camera
DEFAULT_ASSIGNMENT_CFG = "configs/sam2.yaml"

if platform.system() == "Windows":
    SAM2_CKPT = string_path_to_windows(SAM2_CKPT)


def _log(level: str, message: str):
    """Legacy wrapper — delegates to loguru.

    Maps level aliases to loguru's method names (loguru exposes ``warning``, not
    ``warn``) so ``_log("WARN", ...)`` isn't silently demoted to INFO.
    """
    lvl = {"WARN": "warning", "WARNING": "warning", "FATAL": "critical"}.get(
        level.upper(), level.lower())
    getattr(logger, lvl, logger.info)(message)


# ---------------------------------------------------------------------------
# Camera discovery helpers
# ---------------------------------------------------------------------------

def normalize_cam_name(name: str) -> str:
    """Strip .npz / .mp4 extensions and whitespace."""
    name = name.strip()
    for ext in (".npz", ".mp4"):
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
    return name


def normalize_cam_names(cam_names):
    """Parse camera name list (supports comma-separated or array)."""
    if not cam_names:
        return None
    if len(cam_names) == 1 and "," in cam_names[0]:
        cam_names = cam_names[0].split(",")
    cleaned = [c.strip() for c in cam_names if c and c.strip()]
    return cleaned or None


def resolve_cam_npz_paths(data_folder, cam_names):
    resolved = []
    for cam_name in cam_names:
        target = cam_name if cam_name.endswith(".npz") else f"{cam_name}.npz"
        preferred = os.path.join(data_folder, "gt", target)
        if os.path.isfile(preferred):
            resolved.append(preferred)
            continue
        matches = sorted(glob.glob(os.path.join(data_folder, "**", target), recursive=True))
        if matches:
            resolved.append(matches[0])
        else:
            _log("WARN", f"Camera '{cam_name}' not found under '{data_folder}'. Skipping.")
    return resolved


def find_ioi_npz_paths(data_folder: str, cam_names=None):
    cam_names = normalize_cam_names(cam_names)
    if cam_names:
        ioi_npz_paths = resolve_cam_npz_paths(data_folder, cam_names)
        _log("INFO", f"Resolved {len(ioi_npz_paths)} IOI files from explicit camera list.")
        return ioi_npz_paths

    ioi_npz_paths = sorted(glob.glob(os.path.join(data_folder, "gt", "IOI_*.npz")))
    if len(ioi_npz_paths) <= 0:
        ioi_npz_paths = sorted(glob.glob(os.path.join(data_folder, "**", "IOI_*.npz"), recursive=True))
    _log("INFO", f"Discovered {len(ioi_npz_paths)} IOI files under '{data_folder}'.")
    return ioi_npz_paths


def reorder_ioi_paths(ioi_paths, cam_init):
    """Put the IOI npz that matches cam_init first."""
    if not cam_init:
        return ioi_paths
    target = cam_init if cam_init.endswith('.npz') else f"{cam_init}.npz"
    primary = [p for p in ioi_paths if p.lower().endswith(target.lower())]
    if not primary:
        _log("WARN", f"Requested --cam_init '{cam_init}' not found in IOI list. Keeping default order.")
        return ioi_paths
    remainder = [p for p in ioi_paths if p not in primary]
    _log("INFO", f"Using init camera file first: {primary[0]}")
    return primary + remainder


def find_video_files(videos_dir: str, cam_names=None) -> list:
    """Discover MP4 video files, optionally filtered by cam_names."""
    cam_names_set = None
    if cam_names:
        cam_names_set = {normalize_cam_name(c) for c in cam_names}

    video_files = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))
    if not video_files:
        video_files = sorted(glob.glob(os.path.join(videos_dir, "**", "*.mp4"), recursive=True))

    if cam_names_set:
        video_files = [
            v for v in video_files
            if os.path.splitext(os.path.basename(v))[0] in cam_names_set
        ]
    _log("INFO", f"Discovered {len(video_files)} video files under '{videos_dir}'.")
    return video_files


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def find_image_cam_dirs(images_root_dir: str, cam_names=None) -> list:
    """Discover camera subdirectories containing image frames.

    Expected structure: images_root_dir/<cam_name>/<frame>.jpg

    Args:
        images_root_dir: Root directory containing one subdirectory per camera.
        cam_names: Optional list of camera names to filter by.

    Returns:
        Sorted list of camera directory paths.
    """
    cam_names_set = None
    if cam_names:
        cam_names_set = {normalize_cam_name(c) for c in cam_names}

    cam_dirs = sorted(
        d for d in glob.glob(os.path.join(images_root_dir, "*"))
        if os.path.isdir(d)
    )

    if cam_names_set:
        cam_dirs = [d for d in cam_dirs if os.path.basename(d) in cam_names_set]

    # Filter to dirs that actually contain images
    cam_dirs = [
        d for d in cam_dirs
        if any(os.path.splitext(f)[1] in IMAGE_EXTENSIONS for f in os.listdir(d))
    ]

    _log("INFO", f"Discovered {len(cam_dirs)} camera image directories under '{images_root_dir}'.")
    return cam_dirs


def cam_data_from_image_dir(cam_dir: str, start: int = None, end: int = None) -> Dict[str, Any]:
    """Build a cam_data dict from a directory of image frames.

    Args:
        cam_dir: Directory containing image files (jpg/png).
        start: First frame index (0-based, inclusive). None = 0.
        end: Last frame index (0-based, exclusive). None = all frames.

    Returns:
        Dict compatible with process_multi_video_auto / FrameSource factory.
    """
    image_files = sorted(
        f for f in os.listdir(cam_dir)
        if os.path.splitext(f)[1] in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No image files found in '{cam_dir}'")

    # Apply frame range
    if start is not None or end is not None:
        s = max(0, start or 0)
        e = min(len(image_files), end or len(image_files))
        image_files = image_files[s:e]

    cam_name = os.path.basename(cam_dir)
    abs_paths = [os.path.join(os.path.abspath(cam_dir), f) for f in image_files]

    _log("INFO", f"ImageDir: '{cam_name}' — {len(abs_paths)} frames"
         + (f" (range [{start}:{end}])" if start is not None or end is not None else ""))

    return {
        'cam_name': np.array(cam_name),
        'img_abs_path': np.array(abs_paths),
    }


def load_npz_as_dict(npz_path: str) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def load_assignment_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not os.path.isfile(config_path):
        _log("WARN", f"Assignment config not found at '{config_path}'. Using defaults.")
        return {}
    try:
        import yaml
    except ImportError:
        _log("WARN", "PyYAML unavailable; cannot load assignment config. Using defaults.")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Assignment config must be a dict, got {type(loaded)}")
    return loaded


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_seq(
    data_folder,
    init_frame,
    out_path,
    cam_init=None,
    use_gt_bbox=False,
    init_with_gt=False,
    skip_collage=False,
    cam_names=None,
    assignment_config=None,
    expected_subjects_override=None,
    sam_version="sam2",
    sam_checkpoint=None,
    yolo_checkpoint=None,
    start_frame=None,
    end_frame=None,
    videos_dir=None,
    images_root_dir=None,
    interactive=False,
    undistort=False,
    calibration_path=None,
):
    if not yolo_checkpoint and sam_version not in ("sam3_prompt", "sam3_prompt_light"):
        raise ValueError(
            "process_seq requires yolo_checkpoint (except for sam3_prompt / "
            "sam3_prompt_light, which detect via SAM3 text). Callers must supply a "
            "path to the YOLOv12-X weights — see --yolo-checkpoint on run_ma_masks.py. "
            "The MAMMA inference runner injects this from MAMMA_YOLO_CHECKPOINT / .env.local."
        )
    from core.pipeline import SegmentMultipleFrames
    from core.logging import enable_file_logging

    cam_names = normalize_cam_names(cam_names)
    seq_out_path = os.path.join(out_path, os.path.basename(data_folder))

    # Save all logs to a file in the output directory
    os.makedirs(seq_out_path, exist_ok=True)
    enable_file_logging(os.path.join(seq_out_path, "run.log"))

    # Copy config file to output for reproducibility
    if isinstance(assignment_config, dict) and assignment_config:
        import yaml
        cfg_copy = os.path.join(seq_out_path, "config.yaml")
        try:
            with open(cfg_copy, "w") as f:
                yaml.dump(assignment_config, f, default_flow_style=False)
            _log("INFO", f"Saved config to '{cfg_copy}'.")
        except Exception:
            pass

    # --- Discover cameras ---
    if images_root_dir:
        # Image directory mode: each subdirectory is a camera
        cam_dirs = find_image_cam_dirs(images_root_dir, cam_names=cam_names)
        if not cam_dirs:
            _log("ERROR", f"No camera image directories found under '{images_root_dir}'.")
            return

        if cam_init is None:
            cam_init = cam_names[0] if cam_names else os.path.basename(cam_dirs[0])
            _log("WARN",
                 f"No --cam_init specified. Using '{cam_init}' (first available camera). "
                 "If not all people are visible in this camera, specify a different one with --cam_init.")
        os.makedirs(seq_out_path, exist_ok=True)

        # Reorder so cam_init is first
        primary = [d for d in cam_dirs if os.path.basename(d).lower() == cam_init.lower()]
        remainder = [d for d in cam_dirs if d not in primary]
        if primary:
            _log("INFO", f"Using init camera image dir first: {primary[0]}")
        else:
            _log("WARN", f"--cam_init '{cam_init}' not found among image dirs. Using first available.")
        cam_dirs = primary + remainder

        cam_data_list = [cam_data_from_image_dir(d, start=start_frame, end=end_frame) for d in cam_dirs]
        effective_start_frame = None
        effective_end_frame = None

    elif videos_dir:
        video_files = find_video_files(videos_dir, cam_names=cam_names)
        if not video_files:
            _log("ERROR", f"No video files found under '{videos_dir}'.")
            return

        if cam_init is None:
            cam_init = cam_names[0] if cam_names else os.path.splitext(os.path.basename(video_files[0]))[0]
            _log("WARN",
                 f"No --cam_init specified. Using '{cam_init}' (first available camera). "
                 "If not all people are visible in this camera, specify a different one with --cam_init.")
        os.makedirs(seq_out_path, exist_ok=True)

        # Reorder so cam_init is first
        target = f"{cam_init}.mp4"
        primary = [v for v in video_files if os.path.basename(v).lower() == target.lower()]
        remainder = [v for v in video_files if v not in primary]
        if primary:
            _log("INFO", f"Using init camera video first: {primary[0]}")
        else:
            _log("WARN", f"--cam_init '{cam_init}' not found among videos. Using first available.")
        video_files = primary + remainder

        cam_data_list = [cam_data_from_video(vf, start=start_frame, end=end_frame) for vf in video_files]
        effective_start_frame = None
        effective_end_frame = None

        # If NPZ files are also available (--ma_cap_dir), inject calibration
        # data (cam_int, cam_ext) into the video-based cam_data dicts.
        # This enables epipolar geometry for cross-camera matching.
        npz_paths = find_ioi_npz_paths(data_folder, cam_names=cam_names)
        if npz_paths:
            _log("INFO", f"Found {len(npz_paths)} NPZ files — injecting calibration into video cam_data.")
            npz_by_cam = {}
            for p in npz_paths:
                cam = os.path.basename(p).split(".")[0]
                npz_by_cam[cam] = p

            for cd in cam_data_list:
                cam = str(cd['cam_name'])
                if cam in npz_by_cam:
                    npz_data = load_npz_as_dict(npz_by_cam[cam])
                    for key in ('cam_int', 'cam_ext', 'cam_img_h', 'cam_img_w'):
                        if key in npz_data:
                            cd[key] = npz_data[key]
                    _log("INFO", f"  {cam}: calibration injected from NPZ.")
                else:
                    _log("WARN", f"  {cam}: no matching NPZ file — epipolar will be unavailable.")

    else:
        ioi_npz_paths = find_ioi_npz_paths(data_folder, cam_names=cam_names)
        if not ioi_npz_paths:
            _log("ERROR", f"No IOI files found under '{data_folder}'.")
            return

        if cam_init is None:
            cam_init = cam_names[0] if cam_names else os.path.basename(ioi_npz_paths[0]).split(".")[0]
            _log("WARN",
                 f"No --cam_init specified. Using '{cam_init}' (first available camera). "
                 "If not all people are visible in this camera, specify a different one with --cam_init.")
        os.makedirs(seq_out_path, exist_ok=True)
        ioi_npz_paths = reorder_ioi_paths(ioi_npz_paths, cam_init)

        cam_data_list = []
        for npz_path in ioi_npz_paths:
            _log("INFO", f"Loading camera file: {npz_path}")
            cam_data_list.append(load_npz_as_dict(npz_path))

        effective_start_frame = start_frame
        effective_end_frame = end_frame

    # Optional calibration handling (--calibration). Two independent effects, both
    # backed by the same loaded calibration:
    #   1. Inject epipolar calibration (cam_int/cam_ext) into every cam_data that
    #      lacks it, so cross-camera epipolar geometry works in video / image modes
    #      WITHOUT needing --ma_cap_dir to discover NPZs. NPZ-sourced cam_data already
    #      carries these, so we only fill gaps — never overwrite. cam.intrinsics /
    #      cam.T_cam_world match the NPZ cam_int / cam_ext exactly (same convention,
    #      verified), so the injected values are equivalent to the NPZ path.
    #   2. --undistort: attach the per-camera Camera so frame_source_from_cam_data
    #      undistorts frames transparently.
    if calibration_path or undistort:
        # Calibration file is optional: it's the source for epipolar injection
        # (standalone modes) and the preferred --undistort distortion source.
        # Without it, --undistort falls back to the per-camera distortion carried
        # in the ma_cap NPZ (chained --ma_cap_dir mode).
        calib_cams = {}
        if calibration_path:
            # capture/ lives in the superproject; this script is invoked
            # with cwd=segmentation/, so push the repo root onto sys.path.
            _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _repo_root not in sys.path:
                sys.path.insert(0, _repo_root)
            from capture import load_calibration  # noqa: E402
            calib_cams = load_calibration(calibration_path).cameras

            # 1. Epipolar calibration — fill only when absent (don't clobber NPZ).
            n_inj = 0
            for cd in cam_data_list:
                cam = calib_cams.get(str(cd.get('cam_name', '')))
                if cam is None:
                    continue
                if 'cam_int' in cd and 'cam_ext' in cd:
                    continue
                if not hasattr(cam, 'intrinsics') or not hasattr(cam, 'T_cam_world'):
                    continue
                cd['cam_int'] = np.asarray(cam.intrinsics, dtype=np.float64)
                cd['cam_ext'] = np.asarray(cam.T_cam_world, dtype=np.float64)
                if getattr(cam, 'width', None) is not None:
                    cd.setdefault('cam_img_w', int(cam.width))
                if getattr(cam, 'height', None) is not None:
                    cd.setdefault('cam_img_h', int(cam.height))
                n_inj += 1
            if n_inj:
                _log("INFO", f"--calibration: injected epipolar calibration (cam_int/cam_ext) "
                             f"into {n_inj} of {len(cam_data_list)} camera(s).")
            # Cameras still lacking cam_int/cam_ext can't use epipolar geometry (not covered
            # by the calibration file and no NPZ calibration) — they fall back to CLIP-only.
            uncovered = [str(cd.get('cam_name', '')) for cd in cam_data_list
                         if 'cam_int' not in cd or 'cam_ext' not in cd]
            if uncovered:
                _log("WARN", f"--calibration: no usable epipolar calibration for camera(s) "
                             f"{uncovered} — they will use CLIP-only cross-camera matching.")

        # 2. Undistortion (only when explicitly requested). With a calibration
        # file, attach the calib Camera (preferred). Without one, just flag the
        # camera: frame_source_from_cam_data builds the Camera from the NPZ's
        # distortion fields and no-ops where none are present.
        if undistort:
            for cd in cam_data_list:
                cam_name = str(cd.get('cam_name', ''))
                cam = calib_cams.get(cam_name)
                if cam is not None:
                    cd['_undistort_camera'] = cam
                cd['_undistort'] = True
            n_explicit = sum(1 for cd in cam_data_list if cd.get('_undistort_camera') is not None)
            if calibration_path:
                _log("INFO", f"--undistort: enabled for {len(cam_data_list)} cameras "
                             f"({n_explicit} from --calibration, rest from NPZ distortion where present)")
            else:
                _log("INFO", f"--undistort: no --calibration; using per-camera NPZ distortion "
                             f"for {len(cam_data_list)} cameras where present")

    n_cameras = len(cam_data_list)
    _log("INFO", f"Sequence: {data_folder} -> {seq_out_path}")
    _log("INFO", f"Init camera: {cam_init}, {n_cameras} cameras total")

    # --- Build pipeline ---
    cfg = SAM2_CFG if sam_version == "sam2" else SAM3_CFG
    ckpt = sam_checkpoint or (SAM2_CKPT if sam_version == "sam2" else SAM3_CKPT)
    if sam_version == "sam3_prompt":
        cfg = SAM3_CFG
        ckpt = sam_checkpoint or SAM3_CKPT

    pipeline = SegmentMultipleFrames(
        sam_model_cfg=cfg,
        sam_checkpoint=ckpt,
        yolo_checkpoint=yolo_checkpoint,
        output_path=seq_out_path,
        init_with_gt=init_with_gt,
        assignment_config=assignment_config,
        sam_version=sam_version,
        start_frame=effective_start_frame,
        end_frame=effective_end_frame,
    )
    feature_bank_cache = os.path.join(seq_out_path, "subject_feature_bank.npy")
    loaded_feature_bank_cache = pipeline.load_subject_feature_bank(feature_bank_cache)

    # --- Resolve expected subjects ---
    expected_subjects = _resolve_expected_subjects(
        assignment_config, expected_subjects_override, loaded_feature_bank_cache, pipeline
    )

    # --- Process init camera ---
    init_cam_data = cam_data_list[0]
    _log("INFO", f"[1/{n_cameras}] Processing init camera: {cam_init}")

    if use_gt_bbox:
        masks = pipeline.process_multi_video_using_gt_bboxes(cam_data=init_cam_data)
    else:
        masks = pipeline.process_multi_video_auto(
            cam_data=init_cam_data, frame_id=init_frame, masks=None,
            reference_cam_data=init_cam_data, expected_subjects=expected_subjects,
            interactive=interactive,
        )

    if masks is None:
        _log("ERROR", "Init-camera segmentation failed. Stopping.")
        return

    if expected_subjects is None:
        expected_subjects = pipeline.expected_subjects or len(masks)
        pipeline.set_scene_constraints(expected_subjects=expected_subjects)
        _log("INFO", f"Inferred N={expected_subjects} people in the scene (from init camera).")

    _log("INFO", f"[1/{n_cameras}] Init camera done.")

    # --- Bootstrap feature bank ---
    remaining = cam_data_list[1:]
    bootstrap_multiview_bank = True
    if isinstance(assignment_config, dict):
        bootstrap_multiview_bank = bool(assignment_config.get("scene", {}).get("bootstrap_multiview_bank", True))

    if not use_gt_bbox and bootstrap_multiview_bank and remaining and not loaded_feature_bank_cache:
        _log("INFO", "Bootstrapping feature bank from remaining views.")
        pipeline.bootstrap_feature_bank_from_multiview(
            source_cam_data=init_cam_data, source_mask_data=masks,
            target_cam_data_list=remaining, frame_id=init_frame,
            expected_subjects=expected_subjects,
        )
    if not use_gt_bbox:
        pipeline.save_subject_feature_bank(feature_bank_cache)

    # --- Process remaining cameras ---
    for idx, cam_data in enumerate(remaining, start=2):
        cam_label = str(cam_data.get('cam_name', f'cam_{idx}'))
        _log("INFO", f"[{idx}/{n_cameras}] Processing camera: {cam_label}")
        if use_gt_bbox:
            masks = pipeline.process_multi_video_using_gt_bboxes(cam_data=cam_data)
        else:
            masks = pipeline.process_multi_video_auto(
                cam_data=cam_data, frame_id=init_frame, masks=masks,
                reference_cam_data=init_cam_data, expected_subjects=expected_subjects,
            )
        if not use_gt_bbox and masks is not None:
            pipeline.save_subject_feature_bank(feature_bank_cache)
        _log("INFO", f"[{idx}/{n_cameras}] Done: {cam_label}")

    # --- Cross-camera consistency check (debug viz) ---
    # Stacks the per-person crop-summary PNGs across cameras; those PNGs are only
    # produced when debug_crop_summary is on, so gate this on the same flag to
    # avoid empty work by default. (issue #14)
    _debug_viz = bool(isinstance(assignment_config, dict)
                      and assignment_config.get("exports", {}).get("debug_crop_summary", False))
    if _debug_viz:
        from utils.visualization import generate_cross_camera_summary
        generate_cross_camera_summary(seq_out_path, log_fn=_log)

    # --- Collage video ---
    if not skip_collage:
        _generate_collage(seq_out_path)


def _resolve_expected_subjects(assignment_config, override, loaded_cache, pipeline):
    """Determine expected subject count from CLI, config, cache, or auto-detect."""
    if override is not None:
        n = int(override)
        _log("INFO", f"Expecting N={n} people in the scene (from --expected_subjects)")
        pipeline.set_scene_constraints(expected_subjects=n)
        return n

    cfg_expected = None
    if isinstance(assignment_config, dict):
        cfg_expected = assignment_config.get("scene", {}).get("expected_subjects")

    if cfg_expected is not None:
        n = int(cfg_expected)
        _log("INFO", f"Expecting N={n} people in the scene (from config)")
        pipeline.set_scene_constraints(expected_subjects=n)
        return n

    if loaded_cache and len(pipeline.subject_feature_bank) > 0:
        n = len(pipeline.subject_feature_bank)
        _log("INFO", f"Expecting N={n} people in the scene (from feature-bank cache)")
        pipeline.set_scene_constraints(expected_subjects=n)
        return n

    _log("INFO", "N will be inferred from detections at init camera.")
    return None


def _generate_collage(seq_out_path):
    """Generate collage video from mask outputs."""
    try:
        from utils.post_video_from_imgs import find_cameras, process_collage_body
        seq_dir = Path(seq_out_path)
        cameras = find_cameras(seq_dir)
        if cameras:
            process_collage_body(seq_dir, seq_dir.name, cameras, 30, 256, 0, 0, True)
        else:
            _log("WARN", f"No camera folders found for collage in '{seq_dir.name}'.")
    except Exception as exc:
        _log("WARN", f"Collage generation failed: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    _log("INFO", "CLI arguments:")
    for k, v in vars(args).items():
        _log("INFO", f"  {k}: {v}")

    if not os.path.isdir(args.dataset_dir):
        _log("ERROR", f"Dataset directory does not exist: {args.dataset_dir}")
        return

    dataset_name = os.path.basename(args.dataset_dir)
    out_dir = os.path.join(args.out, dataset_name)
    cam_names = normalize_cam_names(args.cam_names)

    if not args.seq_name:
        _log("ERROR", "--seq_name is required.")
        return
    data_folder = os.path.join(args.dataset_dir, args.seq_name)
    if not os.path.isdir(data_folder):
        _log("ERROR", f"Sequence folder does not exist: {data_folder}")
        return

    assignment_config = load_assignment_config(args.cfg)
    if assignment_config:
        _log("INFO", f"Loaded config: {args.cfg}")

    process_seq(
        data_folder=data_folder,
        init_frame=args.init_frame,
        out_path=out_dir,
        cam_init=args.cam_init,
        use_gt_bbox=args.use_gt_bbox,
        init_with_gt=args.init_with_gt,
        cam_names=cam_names,
        assignment_config=assignment_config,
        expected_subjects_override=args.expected_subjects,
        sam_version=args.sam_version,
        sam_checkpoint=args.sam_checkpoint,
        yolo_checkpoint=args.yolo_checkpoint,
        start_frame=args.start,
        end_frame=args.end,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a single sequence")
    parser.add_argument('--ma_cap_dir', '--dataset_dir', type=str, required=True, dest='dataset_dir')
    parser.add_argument('--seq_name', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--init_frame', type=int, default=INIT_FRAME_ID,
                        help='Init frame for detection. Default: auto-select best frame.')
    parser.add_argument('--cam_init', type=str, default=None)
    parser.add_argument('--cam_names', nargs='*', default=None)
    parser.add_argument('--use_gt_bbox', action='store_true')
    parser.add_argument('--init_with_gt', action='store_true')
    parser.add_argument('--expected_subjects', type=int, default=None)
    parser.add_argument('--cfg', '--assignment_cfg', type=str, default=DEFAULT_ASSIGNMENT_CFG, dest='cfg')
    parser.add_argument('--sam_version', default='sam3_prompt', choices=['sam2', 'sam3', 'sam3_prompt'])
    parser.add_argument('--sam_checkpoint', default=None)
    parser.add_argument('--yolo-checkpoint', '--yolo_checkpoint',
                        dest='yolo_checkpoint', required=True,
                        help='Path to the YOLOv12-X person-detection weights '
                             '(.pt file). Required.')
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    main(parser.parse_args())
