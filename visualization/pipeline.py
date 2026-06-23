"""Top-level visualization orchestrator: ``run_visualization(...)``.

Produces three artifacts under ``<out_dir>/<seq_name>/``:

* ``scene.rrd`` — Rerun log: cameras, ground plane, predicted meshes per
  frame, and (unless ``rerun_light``) projected 2D landmarks.
* ``overlay/<cam_name>.mp4`` — per-camera SMPL-X overlay videos
  (skipped when ``skip_overlay=True``).
* ``preview.mp4`` — preview-grid collage of the first ``max_preview_cams``
  overlay videos.

Faithful port of the upstream ``run_ma_vis.py::execute_rerun``.
Differences:

* No ``dataset_name`` dispatch -- ``fps`` and ``rerun_display_scale`` are
  explicit. Older callers used ``"bedlam_lab"`` (fps=30, scale=0.1) or
  ``"harmony4d"`` (fps=20, scale=0.2); pick whatever matches your data.
* No silent ``mm`` -> ``m`` heuristic.
* The SMPL-X face connectivity is loaded from a vendored asset by
  default (``visualization/assets/smplx_faces.npy``) -- override via
  ``faces_path`` if you have a custom topology.
"""
from __future__ import annotations

import glob
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .cameras import Camera, MultiViewCameras
from .collage import make_preview_collage
from .motion import load_landmarks, load_predicted_vertices
from .overlay import _default_palette, render_overlay_videos
from .rerun_log import RerunSceneLogger, compute_floor_height, ffmpeg_available

log = logging.getLogger(__name__)

_VENDORED_FACES = Path(__file__).parent / "assets" / "smplx_faces.npy"
_UP_AXIS_MAP = {"x": 0, "y": 1, "z": 2}


def run_visualization(
    *,
    seq_name: str,
    ma_3d_dir,
    out_path,
    ma_cap_dir=None,
    cameras: Optional["MultiViewCameras"] = None,
    ma_2d_dir=None,
    cam_names_2d_keypoints: Optional[Sequence[str]] = None,
    cam_names_overlay: Optional[Sequence[str]] = None,
    up_axis: str = "z",
    fps: int = 30,
    rerun_display_scale: float = 1.0,
    skip_overlay: bool = False,
    rerun_light: bool = False,
    overlay_resolution: Optional[int] = 1280,
    overlay_max_frames: Optional[int] = None,
    overlay_num_workers: int = 1,
    overlay_image_prefix: str = "",
    max_preview_cams: int = 4,
    faces_path=None,
    colors_rgb: Optional[Sequence[Tuple[float, float, float]]] = None,
    undistort: bool = False,
    rerun_images: bool = True,
    rerun_image_long_edge: int = 480,
    rerun_image_jpeg_quality: int = 75,
    rerun_image_num_workers: Optional[int] = None,
    rerun_video: bool = True,
    rerun_video_long_edge: int = 720,
    rerun_video_crf: int = 20,
) -> Path:
    """Run the full visualization pipeline. Returns the path to ``scene.rrd``.

    Args mirror the upstream ``--`` flags one-for-one. Raises
    ``FileNotFoundError`` if any required input dir is missing.
    """
    t_total = time.perf_counter()

    seq_name = str(seq_name)
    ma_3d_dir = Path(ma_3d_dir)
    out_path = Path(out_path)
    if up_axis not in _UP_AXIS_MAP:
        raise ValueError(f"up_axis must be x/y/z, got {up_axis!r}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if rerun_display_scale <= 0:
        raise ValueError(f"rerun_display_scale must be positive, got {rerun_display_scale}")
    up_axis_idx = _UP_AXIS_MAP[up_axis]

    # Camera source: either an in-memory MultiViewCameras (built by
    # cli.py for standalone calibration+videos workflows) or a gt_dir
    # of NPZs (chained pipeline). Exactly one must be provided.
    if cameras is None:
        if ma_cap_dir is None:
            raise ValueError(
                "run_visualization needs either 'cameras' (in-memory) or "
                "'ma_cap_dir' (NPZ dir); got neither"
            )
        ma_cap_dir = Path(ma_cap_dir)
        gt_dir = ma_cap_dir / seq_name / "gt"
        log.info("seq=%s gt=%s", seq_name, gt_dir)
        cameras = MultiViewCameras.load(gt_dir)
    else:
        log.info("seq=%s cameras from in-memory MultiViewCameras", seq_name)
    motion_dir = ma_3d_dir / seq_name
    rrd_path = out_path / seq_name / "scene.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("motion=%s rrd=%s", motion_dir, rrd_path)
    log.info("loaded %d cameras: %s", len(cameras), cameras.names)

    motions = load_predicted_vertices(motion_dir)
    log.info(
        "loaded %d people, frame range %d..%d",
        len(motions),
        min((m.vertices.shape[0] for m in motions), default=0),
        max((m.vertices.shape[0] for m in motions), default=0),
    )

    landmarks_by_cam = {}
    if not rerun_light:
        if ma_2d_dir is None:
            raise ValueError("ma_2d_dir is required unless rerun_light=True")
        landmarks_by_cam = load_landmarks(
            Path(ma_2d_dir) / seq_name, cam_names=cam_names_2d_keypoints
        )
        log.info("loaded landmarks for %d cameras", len(landmarks_by_cam))

    faces = np.load(faces_path or _VENDORED_FACES, allow_pickle=True)
    if colors_rgb is None:
        colors_rgb = _default_palette(max(10, len(motions)))

    # ---- Rerun scene log ----------------------------------------------
    # The H.264 video backdrop (rerun_video) is the default; it needs an ffmpeg
    # binary, so when none is reachable we transparently fall back to the legacy
    # per-frame JPEG path rather than failing the run.
    use_video = bool(rerun_video) and ffmpeg_available()
    if rerun_video and not use_video:
        log.warning("--rerun-video is on but no ffmpeg binary was found "
                    "(system or imageio-ffmpeg); falling back to the JPEG backdrop")
    # image_long_edge drives a per-camera display scale so Pinhole, 2D landmarks,
    # and the backdrop all land on the same downscaled pixel grid. The video path
    # uses its own (higher) long-edge default since H.264 makes resolution cheap.
    if use_video:
        image_long_edge = int(rerun_video_long_edge)
    elif rerun_images:
        image_long_edge = int(rerun_image_long_edge)
    else:
        image_long_edge = None
    t = time.perf_counter()
    with RerunSceneLogger(
        rrd_path=str(rrd_path),
        fps=fps,
        display_scale=rerun_display_scale,
        image_long_edge=image_long_edge,
        # Per-sequence app id so the viewer never reuses another sequence's cached
        # blueprint/layout when this .rrd is opened natively.
        app_id=f"MAMMA · {seq_name}",
    ) as logger:
        logger.log_cameras(cameras)
        floor = compute_floor_height(motions, up_axis=up_axis_idx)
        logger.log_ground(floor_height=floor, up_axis=up_axis_idx)
        log.info("logged rig + ground (floor=%.4f) in %.2fs", floor, time.perf_counter() - t)

        t = time.perf_counter()
        logger.log_meshes(motions, faces, colors_rgb)
        log.info("logged meshes in %.2fs", time.perf_counter() - t)

        if not rerun_light and landmarks_by_cam:
            t = time.perf_counter()
            logger.log_landmark_projections(cameras, landmarks_by_cam, colors_rgb)
            log.info("logged landmark projections in %.2fs", time.perf_counter() - t)

        if use_video:
            t = time.perf_counter()
            logger.log_camera_video_streams(
                cameras,
                crf=rerun_video_crf,
                num_workers=rerun_image_num_workers,
            )
            log.info("logged camera video streams in %.2fs", time.perf_counter() - t)
        elif rerun_images:
            t = time.perf_counter()
            logger.log_camera_image_streams(
                cameras,
                jpeg_quality=rerun_image_jpeg_quality,
                num_workers=rerun_image_num_workers,
            )
            log.info("logged camera image streams in %.2fs", time.perf_counter() - t)

    # ---- Overlay videos -----------------------------------------------
    overlay_dir = out_path / seq_name / "overlay"
    preview_path = out_path / seq_name / "preview.mp4"
    overlay_paths: List[Path] = []

    if skip_overlay:
        log.info("--skip-overlay: not rendering overlay videos")
    else:
        overlay_cams = _select_overlay_cameras(cameras.cameras, cam_names_overlay)
        if not overlay_cams:
            log.info("no renderable cameras for overlay; skipping")
        else:
            t = time.perf_counter()
            results = render_overlay_videos(
                cameras=overlay_cams,
                motions=motions,
                faces=faces,
                out_dir=overlay_dir,
                fps=fps,
                resolution=overlay_resolution,
                max_frames=overlay_max_frames,
                num_workers=overlay_num_workers,
                image_prefix=overlay_image_prefix,
                colors_rgb=colors_rgb,
                undistort=undistort,
            )
            overlay_paths = [r.video_path for r in results if r.video_path is not None]
            log.info(
                "overlay rendered: %d/%d videos in %.2fs",
                len(overlay_paths), len(overlay_cams), time.perf_counter() - t,
            )

    # ---- Preview collage ----------------------------------------------
    if not overlay_paths:
        # Pick up any pre-existing overlays so reruns can still produce a preview.
        overlay_paths = [Path(p) for p in sorted(glob.glob(str(overlay_dir / "*.mp4")))]
    overlay_paths = [p for p in overlay_paths if p.name.lower() != "preview.mp4"]

    if overlay_paths:
        t = time.perf_counter()
        ok = make_preview_collage(
            overlay_paths, preview_path,
            cam_names=cam_names_overlay, max_videos=max_preview_cams,
        )
        log.info(
            "preview collage %s in %.2fs",
            "wrote" if ok else "skipped (no readable frames)", time.perf_counter() - t,
        )
    else:
        log.info("no overlay videos -> skipping preview collage")

    log.info("visualization pipeline finished in %.2fs", time.perf_counter() - t_total)
    return rrd_path


def _select_overlay_cameras(
    cameras: Sequence[Camera], requested: Optional[Sequence[str]]
) -> List[Camera]:
    """Resolve the subset of cameras to render overlays for.

    Defaults to the first 4 by sorted name if nothing was requested.
    Missing requested names are warned about and dropped (matches upstream).
    """
    available = sorted(cameras, key=lambda c: c.name)
    if not requested:
        return available[:4]
    by_name = {c.name: c for c in available}
    selected = [by_name[n] for n in requested if n in by_name]
    missing = [n for n in requested if n not in by_name]
    if missing:
        log.warning("overlay cameras not found: %s", missing)
    return selected
