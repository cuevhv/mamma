#!/usr/bin/env python
"""Entry point for SAM-based video segmentation (ma_masks step).

Supports three input modes:
  1. NPZ mode (default): reads IOI_*.npz files from --ma_cap_dir/--seq_name
  2. Video mode (--videos_dir): reads MP4 files directly
  3. Image mode (--images_root_dir): reads image frames from camera subdirectories

Both modes support SAM2 (default) and SAM3 backends via --sam_version.

Examples:
    # NPZ mode (standard pipeline)
    python run_ma_masks.py \\
        --ma_cap_dir /path/to/dataset --seq_name my_sequence \\
        --out output --cam_init IOI_09

    # Video mode (MP4 input)
    python run_ma_masks.py \\
        --ma_cap_dir /path/to/dataset --seq_name my_sequence \\
        --out output --cam_init IOI_09 \\
        --videos_dir /path/to/videos/

    # Image mode (cam_01/*.jpg, cam_02/*.jpg, ...)
    python run_ma_masks.py \\
        --ma_cap_dir /path/to/dataset --seq_name my_sequence \\
        --out output --cam_init cam_01 \\
        --images_root_dir /path/to/frames/

    # SAM3 backend with frame range
    python run_ma_masks.py \\
        --ma_cap_dir /path/to/dataset --seq_name my_sequence \\
        --out output --sam_version sam3 --start 10 --end 50
"""
import argparse
import os

# Force matplotlib's non-interactive backend before any import below pulls
# in ``matplotlib.pyplot`` (core.pipeline does, at module load). Without
# this the interactive default (TkAgg) is used, and Tk objects garbage-
# collected off the main thread during SAM2 propagation abort the process
# with ``Tcl_AsyncDelete: async handler deleted by the wrong thread``. The
# inference engine also pins this (see inference/engines.py._child_env);
# setting it here keeps direct ``python run_ma_masks.py`` runs safe too.
os.environ.setdefault("MPLBACKEND", "Agg")

from core.logging import logger
from process_sequence import (
    process_seq,
    load_assignment_config,
    normalize_cam_names,
    INIT_FRAME_ID,
    DEFAULT_ASSIGNMENT_CFG,
)


def main():
    parser = argparse.ArgumentParser(
        description="MA Masks: SAM-based video segmentation with multi-camera ID matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Input modes:
  By default, the pipeline reads IOI_*.npz files from <ma_cap_dir>/<seq_name>/gt/.
  Use --videos_dir to read MP4 video files instead (frames are extracted automatically).

SAM backends:
  --sam_version sam2   SAM2 (default, faster propagation, requires anchors for drift)
  --sam_version sam3   SAM3 (better tracking quality, auto-downloads from HuggingFace)
""",
    )

    # --- Required arguments ---
    parser.add_argument('--ma_cap_dir', default=None,
                        help='Root dataset directory containing the sequence '
                             '(NPZ chained mode). Optional when --videos_dir '
                             'or --images_root_dir is set (frames come from '
                             'there; calibration may be supplied separately).')
    parser.add_argument('--dataset_name', default=None,
                        help='Dataset sublevel for the output layout '
                             '(<out>/<dataset_name>/<seq_name>/). Defaults to '
                             'basename(--ma_cap_dir) when --ma_cap_dir is set, '
                             'else empty (no dataset sublevel). The runner '
                             'passes this so output paths match the layout '
                             'downstream steps expect.')
    parser.add_argument('--seq_name', required=True,
                        help='Sequence name (subfolder of ma_cap_dir)')
    parser.add_argument('--out', required=True,
                        help='Output directory for masks, overlays, and videos')

    # --- Input mode ---
    parser.add_argument('--videos_dir', default=None,
                        help='Directory containing MP4 video files (one per camera). '
                             'Video filenames should match camera names (e.g., IOI_09.mp4).')
    parser.add_argument('--images_root_dir', default=None,
                        help='Directory containing camera subdirs, each with image frames. '
                             'Expected structure: images_root_dir/<cam_name>/<frame>.jpg. '
                             'Subdirectory names are used as camera names.')
    parser.add_argument('--calibration', default=None,
                        help='Calibration file (yaml/xcp/json). Injects camera intrinsics/'
                             'extrinsics (cam_int/cam_ext) into cam_data, enabling cross-camera '
                             'epipolar geometry for ID matching in --videos_dir / --images_root_dir '
                             'modes (NPZ/--ma_cap_dir modes already carry calibration). Also the '
                             'preferred distortion source for --undistort; if omitted, --undistort '
                             'falls back to the per-camera distortion in the ma_cap NPZ. Existing '
                             'NPZ calibration is not overwritten.')
    parser.add_argument('--undistort', action='store_true',
                        help='Undistort frames (any supported lens model) before SAM / YOLO. '
                             'Coefficients come from --calibration when given, else from the '
                             'per-camera NPZ (chained --ma_cap_dir mode). No-op for cameras with '
                             'no distortion data. Default off.')

    # --- SAM backend ---
    parser.add_argument('--sam_version', default='sam2',
                        choices=['sam2', 'sam3', 'sam3_prompt', 'sam3_prompt_light'],
                        help='SAM backend: sam2 (default), sam3 (tracker API + YOLO), '
                             'sam3_prompt (text prompt "person" via the full video '
                             'predictor — heaviest VRAM), or sam3_prompt_light (text prompt '
                             '"person" via SAM3 detector + lean tracker — no YOLO, lower VRAM '
                             'than sam3_prompt and no long-clip OOM)')
    parser.add_argument('--sam_checkpoint', default=None,
                        help='Override SAM checkpoint path or HuggingFace model ID')
    parser.add_argument('--yolo-checkpoint', '--yolo_checkpoint',
                        dest='yolo_checkpoint', default=None,
                        help='Path to the YOLOv12-X person-detection weights '
                             '(.pt file). Required for all backends except '
                             'sam3_prompt_light (which detects via SAM3 text). '
                             'The MAMMA inference runner injects this from '
                             'MAMMA_YOLO_CHECKPOINT / .env.local.')

    # --- Frame range ---
    parser.add_argument('--start', type=int, default=None,
                        help='First frame index to process (0-based, inclusive)')
    parser.add_argument('--end', type=int, default=None,
                        help='Last frame index to process (0-based, exclusive)')
    parser.add_argument('--init_frame', type=int, default=INIT_FRAME_ID,
                        help='Frame index for person detection initialization. '
                             'Default: auto-select the best frame per camera (most people detected).')

    # --- Camera selection ---
    parser.add_argument('--cam_init', default=None,
                        help='Camera to use for ID initialization (e.g., IOI_09)')
    parser.add_argument('--cam_names', nargs='*', default=None,
                        help='List of camera names to process (default: all cameras)')

    # --- Detection & tracking ---
    parser.add_argument('--use_gt_bbox', action='store_true',
                        help='Use ground-truth bounding boxes from NPZ (requires GT data)')
    parser.add_argument('--init_with_gt', action='store_true',
                        help='Initialize first view with GT bboxes, auto-detect rest')
    parser.add_argument('--expected_subjects', type=int, default=None,
                        help='Number of people N in the scene (auto-detected if not set)')
    parser.add_argument('--interactive', action='store_true',
                        help='Use interactive GUI to click on people instead of YOLO auto-detection. '
                             'Opens a tkinter window on the init camera. Requires a display.')
    parser.add_argument('--cross-camera-strategy', '--cross_camera_strategy',
                        dest='cross_camera_strategy', default=None,
                        choices=['auto', 'track_then_match', 'match_then_track'],
                        help='Cross-camera identity for non-init cameras. '
                             'auto (default): calibration- and backend-aware — '
                             'track_then_match for SAM3-text detection '
                             '(sam3_prompt_light), and for YOLO backends (sam2/sam3) '
                             'when epipolar calibration (cam_int/cam_ext) is available; '
                             'otherwise match_then_track (the CLIP-only appearance remap '
                             'is unreliable for YOLO crops). track_then_match: force '
                             'per-camera track then whole-tracklet remap. '
                             'match_then_track: legacy — commit identity at seed time. '
                             '(sam3_prompt is always track-then-match regardless.)')
    parser.add_argument('--legacy-cross-camera', '--legacy_cross_camera',
                        dest='legacy_cross_camera', action='store_true',
                        help='Shortcut for --cross-camera-strategy match_then_track.')

    # --- Configuration ---
    parser.add_argument('--cfg', '--assignment_cfg', default=None,
                        dest='cfg',
                        help='YAML config file. Default: auto-selected based on --sam_version '
                             '(sam2 -> configs/sam2.yaml, sam3/sam3_prompt -> configs/sam3.yaml)')
    parser.add_argument('--skip_collage', action='store_true',
                        help='Skip collage video generation at the end')
    parser.add_argument('--skip_masked_outputs', action='store_true',
                        help='Skip overlay visualization images and MP4 (saves time/disk)')
    parser.add_argument('--lazy-frames', '--lazy_frames', dest='lazy_frames', action='store_true',
                        help='Bound host RAM by loading frames on demand (a small window) '
                             'instead of all at once — needed for long videos that would '
                             'otherwise run out of RAM. Sets sam.lazy_frame_loading. mp4 input '
                             'streams via cv2 (near-identical, ~+24%% wall); image/frame input '
                             'is byte-identical. Off by default (short clips do not need it).')
    parser.add_argument('--prune-memory', '--prune_memory', dest='prune_memory', action='store_true',
                        help='Also bound SAM2 per-frame memory state on very long clips '
                             '(sets sam.prune_memory_state; GT-equivalent, edge-pixel).')
    parser.add_argument('--debug_crop_summary', action='store_true',
                        help='Render per-person crop-summary + t-SNE + cross-camera '
                             'debug PNGs (off by default; pure viz, costs wall time). '
                             'The mask PNGs and masks.npy cache are produced either way.')
    parser.add_argument('--debug_full_masks_npy', action='store_true',
                        help='Keep full-res frames + masks in masks.npy (~10x bigger). '
                             'Off by default: masks.npy is slimmed to what matching/resume '
                             'need (centroids + crops + features); full masks remain as the '
                             'per-frame mask PNGs.')

    args = parser.parse_args()

    # YOLO is required only for the YOLO-based backends; sam3_prompt and
    # sam3_prompt_light detect via SAM3 text.
    if args.sam_version not in ("sam3_prompt", "sam3_prompt_light") and not args.yolo_checkpoint:
        logger.error(f"--yolo_checkpoint is required for --sam_version {args.sam_version}.")
        return

    # Auto-select config based on sam_version if not specified
    if args.cfg is None:
        if args.sam_version in ("sam3", "sam3_prompt", "sam3_prompt_light"):
            args.cfg = "configs/sam3.yaml"
        else:
            args.cfg = "configs/sam2.yaml"

    if not os.path.isfile(args.cfg):
        logger.error(f"Config file not found: {args.cfg}")
        return

    logger.info("run_ma_masks entry point")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    if args.ma_cap_dir:
        data_folder = os.path.join(args.ma_cap_dir, args.seq_name)
        if not os.path.isdir(data_folder):
            logger.error(f"Sequence folder does not exist: {data_folder}")
            return
        # Default dataset_name = basename(ma_cap_dir); --dataset_name overrides.
        dataset_name = args.dataset_name or os.path.basename(args.ma_cap_dir)
    else:
        # Standalone videos/images mode: no ma_cap NPZ tree provided.
        if not args.videos_dir and not args.images_root_dir:
            logger.error(
                "Either --ma_cap_dir or one of --videos_dir/--images_root_dir is required."
            )
            return
        # data_folder is only used for the basename in seq_out_path
        # and (when ma_cap_dir is set) for resolving relative frame-source paths.
        data_folder = args.seq_name
        # Use the runner-supplied dataset_name (matches the layout ma_2d
        # expects: <mask_path>/<seq>/<cam>/masks/). Empty when neither
        # source provides one (e.g. ad-hoc standalone CLI usage).
        dataset_name = args.dataset_name or ""

    # Resolve videos_dir: if relative AND ma_cap_dir is set, check under data_folder first
    videos_dir = args.videos_dir
    if videos_dir and not os.path.isabs(videos_dir) and args.ma_cap_dir:
        candidate = os.path.join(data_folder, videos_dir)
        if os.path.isdir(candidate):
            videos_dir = candidate

    if videos_dir and not os.path.isdir(videos_dir):
        logger.error(f"Videos directory does not exist: {videos_dir}")
        return

    # Resolve images_root_dir: if relative AND ma_cap_dir is set, check under data_folder first
    images_root_dir = args.images_root_dir
    if images_root_dir and not os.path.isabs(images_root_dir) and args.ma_cap_dir:
        candidate = os.path.join(data_folder, images_root_dir)
        if os.path.isdir(candidate):
            images_root_dir = candidate

    if images_root_dir and not os.path.isdir(images_root_dir):
        logger.error(f"Images root directory does not exist: {images_root_dir}")
        return

    if videos_dir and images_root_dir:
        logger.error("Cannot use both --videos_dir and --images_root_dir. Pick one.")
        return

    out_dir = os.path.join(args.out, dataset_name) if dataset_name else args.out

    cam_names = normalize_cam_names(args.cam_names)

    # Warn if cam_init is specified but not included in cam_names
    if args.cam_init and cam_names:
        from process_sequence import normalize_cam_name
        init_normalized = normalize_cam_name(args.cam_init)
        names_normalized = [normalize_cam_name(c) for c in cam_names]
        if init_normalized not in names_normalized:
            logger.error(
                f"--cam_init '{args.cam_init}' is not in --cam_names {args.cam_names}. "
                "The init camera must be included in --cam_names, otherwise there are no "
                "masks to match against. Either add it to --cam_names or omit --cam_names "
                "to process all cameras."
            )
            return

    assignment_config = load_assignment_config(args.cfg)
    if assignment_config:
        logger.info(f"Loaded assignment config: {args.cfg}")

    # Inject CLI flag into assignment config so the pipeline can read it
    if args.skip_masked_outputs:
        if assignment_config is None:
            assignment_config = {}
        assignment_config.setdefault("exports", {})["skip_masked_outputs"] = True
        # Collage requires masked_outputs videos, so skip it too
        args.skip_collage = True

    if args.lazy_frames:
        if assignment_config is None:
            assignment_config = {}
        assignment_config.setdefault("sam", {})["lazy_frame_loading"] = True

    if args.prune_memory:
        if assignment_config is None:
            assignment_config = {}
        assignment_config.setdefault("sam", {})["prune_memory_state"] = True

    if args.debug_crop_summary:
        if assignment_config is None:
            assignment_config = {}
        assignment_config.setdefault("exports", {})["debug_crop_summary"] = True

    if args.debug_full_masks_npy:
        if assignment_config is None:
            assignment_config = {}
        assignment_config.setdefault("exports", {})["debug_full_masks_npy"] = True

    if args.legacy_cross_camera and args.cross_camera_strategy not in (None, "match_then_track"):
        logger.warning(
            f"Both --legacy-cross-camera and --cross-camera-strategy "
            f"{args.cross_camera_strategy!r} given; --legacy-cross-camera wins "
            "(using match_then_track)."
        )
    strategy = "match_then_track" if args.legacy_cross_camera else args.cross_camera_strategy
    if strategy:
        if assignment_config is None:
            assignment_config = {}
        assignment_config.setdefault("masks", {})["cross_camera_strategy"] = strategy
        if args.sam_version == "sam3_prompt" and strategy == "match_then_track":
            logger.info("--cross-camera-strategy ignored for sam3_prompt "
                        "(always track-then-match by construction).")

    process_seq(
        data_folder=data_folder,
        init_frame=args.init_frame,
        out_path=out_dir,
        cam_init=args.cam_init,
        use_gt_bbox=args.use_gt_bbox,
        init_with_gt=args.init_with_gt,
        skip_collage=args.skip_collage,
        cam_names=cam_names,
        assignment_config=assignment_config,
        expected_subjects_override=args.expected_subjects,
        sam_version=args.sam_version,
        sam_checkpoint=args.sam_checkpoint,
        yolo_checkpoint=args.yolo_checkpoint,
        start_frame=args.start,
        end_frame=args.end,
        videos_dir=videos_dir,
        images_root_dir=images_root_dir,
        interactive=args.interactive,
        undistort=args.undistort,
        calibration_path=args.calibration,
    )


if __name__ == '__main__':
    main()
