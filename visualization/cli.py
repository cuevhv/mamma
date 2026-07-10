"""``python -m visualization`` CLI: render Rerun + overlay videos for one sequence.

Polished replacement for the upstream ``run_ma_vis.py``. Differences:

* Replaces the implicit ``dataset_name`` dispatch (which set fps and a
  display-resize factor for the Rerun rig) with explicit ``--fps`` and
  ``--rerun-display-scale`` flags.
* The vendored SMPL-X face connectivity at
  ``visualization/assets/smplx_faces.npy`` is the default; override with
  ``--faces``.

For drop-in compatibility with the upstream / inference runner, every
flag that the runner emits with an underscore (``--seq_name``,
``--ma_cap_dir``, ``--rerun_light``, ...) is accepted in addition to
the dash form (``--seq-name``, etc.).

Output layout under ``--out-path/<seq_name>/``::

    scene.rrd
    overlay/<cam>.mp4
    preview.mp4
"""
from __future__ import annotations

import argparse
import logging
import sys

from .pipeline import run_visualization


def _resolve_fps(args) -> int:
    """Explicit ``--fps`` wins; else the capture's recorded rate from
    ``<ma_cap_dir>/<seq>/gt/global.npz``; else 30. The frame rate is a fact
    about the footage — asking for it only invites wrong-speed playback."""
    if args.fps is not None:
        return args.fps
    if args.ma_cap_dir and args.seq_name:
        import os

        import numpy as np
        p = os.path.join(args.ma_cap_dir, args.seq_name, "gt", "global.npz")
        try:
            with np.load(p, allow_pickle=True) as z:
                if "fps" in z.files and int(z["fps"]) > 0:
                    fps = int(z["fps"])
                    sys.stderr.write(f"fps: {fps} (from capture global.npz)\n")
                    return fps
        except (OSError, ValueError):
            pass
    return 30


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m visualization",
        description="Render the MAMMA visualization for one sequence.",
    )
    # Required inputs / outputs (dash + underscore forms both accepted).
    p.add_argument("--seq-name", "--seq_name", required=True,
                   help="Sequence name to visualize.")
    p.add_argument("--ma-cap-dir", "--ma_cap_dir", default=None,
                   help="Path to ma_cap output (contains <seq>/gt/<cam>.npz). "
                        "Omit and use --calibration for standalone mode.")
    p.add_argument("--ma-3d-dir", "--ma_3d_dir", required=True,
                   help="Path to ma_3d output (contains <seq>/verts_joints_body_id*.npz).")
    p.add_argument("--ma-2d-dir", "--ma_2d_dir", default=None,
                   help="Path to ma_2d output (required unless --rerun-light).")
    p.add_argument("--out-path", "--out_path", required=True,
                   help="Output root. Writes <out>/<seq>/scene.rrd and friends.")
    # Standalone-mode inputs (alternative to --ma-cap-dir).
    p.add_argument("--videos-dir", "--videos_dir", default=None,
                   help="Standalone mode: directory of <cam_name>.mp4 files. "
                        "Requires --calibration.")
    p.add_argument("--images-root-dir", "--images_root_dir", default=None,
                   help="Standalone mode: directory of <cam_name>/*.jpg "
                        "subdirectories. Requires --calibration.")
    p.add_argument("--calibration", default=None,
                   help="Calibration file (yaml/xcp/json). Required when "
                        "--ma-cap-dir is omitted.")
    p.add_argument("--cam-names", "--cam_names", nargs="+", default=None,
                   help="Camera names to include in standalone mode. Required "
                        "with --calibration so synthesis knows which cameras to write.")
    p.add_argument("--undistort", action="store_true",
                   help="Undistort overlay-background frames before compositing the "
                        "mesh, using each camera's distortion model — OpenCV "
                        "radtan/opencv_brown (k1,k2,p1,p2[,k3]) or Vicon "
                        "vicon_radial_2. Coefficients come from the per-camera NPZs "
                        "under --ma-cap-dir or from --calibration. Default off.")
    p.add_argument("--start-frame", "--start_frame", "--start", type=int,
                   default=None, dest="start_frame",
                   help="Standalone mode: first source-video frame to read "
                        "(0-based, inclusive). Aligns overlay backgrounds with "
                        "ma_3d's meshes when those were optimised for a sub-range. "
                        "In chained mode (--ma-cap-dir), the range is read from "
                        "the per-camera NPZ instead.")
    p.add_argument("--end-frame", "--end_frame", "--end", type=int,
                   default=None, dest="end_frame",
                   help="Standalone mode: last source-video frame to read "
                        "(0-based, exclusive). See --start-frame.")

    p.add_argument("--cam-names-2d-keypoints", "--cam_names_2d_keypoints",
                   nargs="+", default=None,
                   help="Subset of cameras whose 2D landmarks to log into the .rrd.")
    p.add_argument("--cam-names-overlay", "--cam_names_overlay",
                   nargs="+", default=None,
                   help="Cameras to render as overlay mp4s + preview. Pass 'all' "
                        "(or '*') to render every camera; default is the first 4.")

    p.add_argument("--up-axis", "--up_axis", default="z",
                   choices=["x", "y", "z", "-x", "-y", "-z"],
                   help="Signed world up axis (default: z).")
    p.add_argument("--fps", type=int, default=None,
                   help="FPS for both the Rerun timeline and overlay videos. "
                        "Default: the capture's recorded rate (global.npz, "
                        "ma_cap mode), else 30.")
    p.add_argument("--rerun-display-scale", "--rerun_display_scale",
                   type=float, default=1.0,
                   help="Downscale factor applied to camera images shown in "
                        "the Rerun viewer (default 1.0 = full res). The "
                        "old dataset names mapped to: bedlam_lab=0.1, harmony4d=0.2. "
                        "NOTE: ignored while --rerun-images is on (the default); "
                        "scale is derived per-camera from --rerun-image-long-edge "
                        "instead. Pass --no-rerun-images to fall back to this flag.")

    p.add_argument("--rerun-images", "--rerun_images",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Log a JPEG backdrop image stream onto each camera's "
                        "image entity in the Rerun scene (so 2D landmarks "
                        "render over the actual capture frames instead of "
                        "a black background). Use --no-rerun-images to disable.")
    p.add_argument("--rerun-image-long-edge", "--rerun_image_long_edge",
                   type=int, default=480,
                   help="Long-edge target (px) for the per-camera JPEG "
                        "backdrop logged into Rerun. Drives the per-camera "
                        "Pinhole resolution + 2D landmark scaling so all "
                        "three layers stay aligned. Default 480.")
    p.add_argument("--rerun-image-jpeg-quality", "--rerun_image_jpeg_quality",
                   type=int, default=75,
                   help="JPEG quality (1..100) for the per-camera backdrop. "
                        "Lower = smaller .rrd. Default 75.")
    p.add_argument("--rerun-image-num-workers", "--rerun_image_num_workers",
                   type=int, default=None,
                   help="Thread-pool size for parallel per-camera image "
                        "decode+encode. Default = min(num_cameras, 4). "
                        "Bump on big machines with many cameras; drop to 1 "
                        "for deterministic single-threaded behaviour.")

    p.add_argument("--rerun-video", "--rerun_video",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Log each camera's backdrop as a re-encoded H.264 video "
                        "(rr.AssetVideo) instead of a per-frame JPEG stream. "
                        "~10-15x smaller .rrd, and viewer-compatible for any "
                        "source codec (HEVC/AV1 are re-encoded to H.264). **On by "
                        "default**; use --no-rerun-video for the legacy per-frame "
                        "JPEG backdrop (--rerun-images). Re-encode is skipped when "
                        "the source is already H.264 at the requested framing, and "
                        "cached otherwise. Needs an ffmpeg binary (system or the "
                        "bundled imageio-ffmpeg); auto-falls back to JPEG if none.")
    p.add_argument("--rerun-video-long-edge", "--rerun_video_long_edge",
                   type=int, default=720,
                   help="Long-edge target (px) for the H.264 backdrop under "
                        "--rerun-video. Higher than the JPEG default since video "
                        "is cheap on size; keep modest at high camera counts so "
                        "the browser can decode all streams. Default 720.")
    p.add_argument("--rerun-video-crf", "--rerun_video_crf",
                   type=int, default=20,
                   help="H.264 quality for the --rerun-video backdrop (libx264 "
                        "CRF, lower = better/larger). Default 20.")

    p.add_argument("--rerun-light", "--rerun_light", action="store_true",
                   help="Skip 2D landmark loading + projection logging.")
    p.add_argument("--skip-overlay", "--skip_overlay", action="store_true",
                   help="Skip pyrender overlay video rendering.")

    # ``--resolution`` is the upstream name; ``--overlay-resolution`` is the new one.
    p.add_argument("--overlay-resolution", "--overlay_resolution", "--resolution",
                   type=int, default=1280,
                   help="Long-side target resolution for overlay videos. "
                        "<=0 keeps the source camera resolution.")
    p.add_argument("--overlay-max-frames", "--overlay_max_frames",
                   type=int, default=None,
                   help="Optional cap on frames per camera.")
    p.add_argument("--overlay-num-workers", "--overlay_num_workers",
                   type=int, default=None,
                   help="Parallelism for overlay rendering. Default: auto "
                        "(min(#cameras, 4), with serial retry for any camera "
                        "whose worker fails). 1 = single process.")
    # ``--overlay_imgs_pth`` is the upstream name.
    p.add_argument("--overlay-image-prefix", "--overlay_image_prefix", "--overlay_imgs_pth",
                   default="",
                   help="Prefix to prepend to camera image paths "
                        "(useful when remounting datasets).")

    p.add_argument("--max-preview-cams", "--max_preview_cams",
                   type=int, default=4,
                   help="Max number of tiles in the preview collage. "
                        "0 = include every rendered overlay (full contact sheet).")
    p.add_argument("--faces", default=None,
                   help="Override path to SMPL-X face connectivity (.npy). "
                        "Default uses the vendored asset.")

    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v for INFO, -vv for DEBUG (default WARNING).")
    return p


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    if not args.rerun_light and args.ma_2d_dir is None:
        sys.stderr.write("error: --ma-2d-dir is required unless --rerun-light is set\n")
        sys.exit(2)

    if args.rerun_image_long_edge <= 0:
        sys.stderr.write("error: --rerun-image-long-edge must be positive\n")
        sys.exit(2)
    if not 1 <= args.rerun_image_jpeg_quality <= 100:
        sys.stderr.write(
            "error: --rerun-image-jpeg-quality must be in [1, 100]\n"
        )
        sys.exit(2)

    # Standalone-mode dispatch: build MultiViewCameras directly from
    # --calibration + frame source. No NPZ round-trip — keeps the
    # ma_vis output dir free of ma_cap-named scaffolding files.
    cameras = None
    if args.ma_cap_dir is None:
        if not args.calibration:
            sys.stderr.write(
                "error: --calibration is required when --ma-cap-dir is omitted.\n"
            )
            sys.exit(2)
        if not (args.videos_dir or args.images_root_dir):
            sys.stderr.write(
                "error: --videos-dir or --images-root-dir is required when "
                "--ma-cap-dir is omitted.\n"
            )
            sys.exit(2)
        if not args.cam_names:
            sys.stderr.write(
                "error: --cam-names is required in standalone mode.\n"
            )
            sys.exit(2)
        from .cameras import MultiViewCameras
        cameras = MultiViewCameras.from_calibration(
            args.calibration,
            cam_names=args.cam_names,
            videos_dir=args.videos_dir,
            images_root_dir=args.images_root_dir,
            frame_start=args.start_frame,
            frame_end=args.end_frame,
        )
        sys.stderr.write(
            f"loaded {len(cameras)} cameras from {args.calibration} "
            f"(source={'videos' if args.videos_dir else 'images'})\n"
        )

    rrd_path = run_visualization(
        seq_name=args.seq_name,
        ma_cap_dir=args.ma_cap_dir,
        cameras=cameras,
        ma_3d_dir=args.ma_3d_dir,
        ma_2d_dir=args.ma_2d_dir,
        out_path=args.out_path,
        cam_names_2d_keypoints=args.cam_names_2d_keypoints,
        cam_names_overlay=args.cam_names_overlay,
        up_axis=args.up_axis,
        fps=_resolve_fps(args),
        rerun_display_scale=args.rerun_display_scale,
        skip_overlay=args.skip_overlay,
        rerun_light=args.rerun_light,
        overlay_resolution=args.overlay_resolution,
        overlay_max_frames=args.overlay_max_frames,
        overlay_num_workers=args.overlay_num_workers,
        overlay_image_prefix=args.overlay_image_prefix,
        max_preview_cams=args.max_preview_cams,
        faces_path=args.faces,
        undistort=args.undistort,
        rerun_images=args.rerun_images,
        rerun_image_long_edge=args.rerun_image_long_edge,
        rerun_image_jpeg_quality=args.rerun_image_jpeg_quality,
        rerun_image_num_workers=args.rerun_image_num_workers,
        rerun_video=args.rerun_video,
        rerun_video_long_edge=args.rerun_video_long_edge,
        rerun_video_crf=args.rerun_video_crf,
    )
    print(f"wrote {rrd_path}")


if __name__ == "__main__":
    main()
