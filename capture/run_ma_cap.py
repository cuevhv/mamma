"""Drop-in replacement for ``capture-hall-toolkit/capture.py``'s
GT-export job, for the publishable local pipeline.

CLI surface (matches what ``inference/steps/ma_cap.py`` emits and what
the apptainer ``ma_cap`` step.py originally invoked)::

    python capture/run_ma_cap.py \\
        --json     <capture.json> \\
        --seq_name <seq>           \\
        --cam_names <cam01> <cam02> ... \\
        --out      <out>/ma_cap/<tag>/<dataset>

Writes the NPZ contract that downstream steps consume::

    <out>/<seq_name>/gt/global.npz
    <out>/<seq_name>/gt/<cam_name>.npz   (one per camera)

This is the minimum subset of the upstream ``capture.py`` /
``renderkit.manager.DataProcessor.export_gt`` behaviour: per-camera
intrinsics+extrinsics+image-paths, plus the small global metadata
header. Motion / Vicon / SMPL-X handling is intentionally omitted --
the example tasks do not need it, and adding it later does not
change the keys we already write.

GT export is the only mode of this script; ``--export_gt`` is accepted
for backward compatibility with saved task configs but has no effect.

Calibration is loaded via :func:`capture.load_calibration` (supports
``.xcp``, OpenCV JSON, and YAML).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Make ``capture`` importable when this script is invoked as
# ``python capture/run_ma_cap.py`` (cwd may be capture/ when the runner
# dispatches it).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from capture import Camera, load_calibration  # noqa: E402
from capture.discovery import find_image_cam_dirs, find_video_files  # noqa: E402
from capture.video_reader import VideoFrameReader  # noqa: E402

log = logging.getLogger(__name__)

# Same extension set as ``capture-hall-toolkit/loaders/images.py``.
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".pgm", ".tif", ".tiff", ".webp")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python capture/run_ma_cap.py",
        description=(
            "Build per-camera + global GT npz for one sequence. Accepts "
            "three input modes (mutually exclusive): --json (capture.json "
            "with ioi_root + calib + sequences), --videos_dir (directory of "
            "<cam_name>.mp4) + --calibration, or --images_root_dir (directory "
            "of <cam_name>/*.jpg) + --calibration."
        ),
    )
    # Input mode (exactly one of --json / --videos_dir / --images_root_dir).
    p.add_argument("--json", default=None,
                   help="Path to the capture.json describing this dataset.")
    p.add_argument("--videos_dir", default=None,
                   help="Directory of <cam_name>.mp4 files. Requires --calibration.")
    p.add_argument("--images_root_dir", default=None,
                   help="Directory of <cam_name>/*.jpg subdirectories. Requires --calibration.")
    p.add_argument("--calibration", default=None,
                   help="Calibration file (yaml/xcp/json). Required with "
                        "--videos_dir or --images_root_dir; overrides capture.json's "
                        "'calib' field when --json is also set.")
    p.add_argument("--seq_name", required=True,
                   help="Sequence name. With --json, must match capture.json's "
                        "'sequences' entry; with --videos_dir/--images_root_dir, "
                        "any user-chosen label.")
    p.add_argument("--cam_names", nargs="+", required=True,
                   help="Camera names to include. Must match calibration "
                        "entries; with --videos_dir, also must match MP4 stems.")
    p.add_argument("--out", required=True,
                   help="Output root; the script writes <out>/<seq_name>/gt/*.")
    p.add_argument("--fps", type=int, default=None,
                   help="FPS for global.npz. Defaults to capture.json's cam_fps "
                        "(--json mode), the video's fps (--videos_dir mode), "
                        "or 30 (--images_root_dir mode).")
    p.add_argument("--start", type=int, default=None,
                   help="First frame index (0-based, inclusive). Restricts the "
                        "sequence range every downstream step inherits. Default: 0.")
    p.add_argument("--end", type=int, default=None,
                   help="Last frame index (0-based, exclusive). Default: end of "
                        "sequence. Combined with --start, defines the canonical "
                        "frame range for this run; downstream steps read it from "
                        "the per-camera NPZ (frame_start, frame_end fields).")
    p.add_argument("--export_gt", action="store_true",
                   help="(Deprecated, no-op.) GT export is the only mode "
                        "of this script. Accepted for backward compatibility "
                        "with saved task configs; new presets omit it.")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v = INFO, -vv = DEBUG. Default WARNING.")
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


# ---------------------------------------------------------------------------
# capture.json shape
# ---------------------------------------------------------------------------

def _load_capture_json(path: str) -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    # capture_root is the current name; ioi_root is the legacy alias
    # (IOI was a brand name of the original capture rig). Either is fine.
    if "capture_root" not in cfg and "ioi_root" not in cfg:
        raise ValueError(
            f"{path}: missing required top-level key 'capture_root' "
            f"(or legacy 'ioi_root')"
        )
    for required in ("calib", "sequences"):
        if required not in cfg:
            raise ValueError(f"{path}: missing required top-level key {required!r}")
    # Anchor relative paths to the capture.json's directory so the JSONs
    # are portable (no per-machine absolute paths baked in).
    base = os.path.dirname(os.path.abspath(path))
    for key in ("capture_root", "ioi_root", "calib"):
        v = cfg.get(key)
        if isinstance(v, str) and v and not os.path.isabs(v):
            cfg[key] = os.path.normpath(os.path.join(base, v))
    return cfg


# ---------------------------------------------------------------------------
# Per-camera image-path discovery
# ---------------------------------------------------------------------------

def _gather_images(cam_dir: Path) -> List[str]:
    out: List[str] = []
    for ext in IMG_EXTENSIONS:
        out.extend(glob.glob(str(cam_dir / f"*{ext}")))
    out.sort()
    return out


def _discover_image_paths(ioi_seq_dir: Path, cam_names: List[str]) -> Dict[str, List[str]]:
    paths: Dict[str, List[str]] = {}
    for cam_name in cam_names:
        cam_dir = ioi_seq_dir / cam_name
        if not cam_dir.is_dir():
            raise FileNotFoundError(
                f"camera dir not found: {cam_dir} "
                f"(expected as a subdir of ioi_root/{ioi_seq_dir.name}/)"
            )
        imgs = _gather_images(cam_dir)
        if not imgs:
            raise FileNotFoundError(
                f"no images in {cam_dir} (extensions: {IMG_EXTENSIONS})"
            )
        paths[cam_name] = imgs
        log.info("camera %s: %d images", cam_name, len(imgs))
    return paths


# ---------------------------------------------------------------------------
# NPZ writers
# ---------------------------------------------------------------------------

def _write_global_npz(
    out_path: Path,
    *,
    seq_name: str,
    ioi_seq_dir: Path,
    frames_len: int,
    fps: int,
    frame_start: int = 0,
    frame_end: int = None,
) -> None:
    """Minimum-viable global.npz. Motion/Vicon fields intentionally omitted.

    ``frame_start`` / ``frame_end`` define the canonical range every
    downstream step inherits (videos workflow: frame indices in the
    backing MP4; images workflow: indices into the sorted image list).
    """
    if frame_end is None:
        frame_end = frame_start + frames_len
    payload = {
        # Provenance / inputs.
        "params_paths": np.array([]),           # no SMPL-X params loaded
        "c3d_path": None,                        # no Vicon C3D loaded
        "ioi_dir": str(ioi_seq_dir),
        # Sequence metadata.
        "seq_name": seq_name,
        "frames_len": int(frames_len),
        "fps": int(fps),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
    }
    np.savez(out_path, **payload)
    log.info("wrote %s (frames [%d:%d] = %d frames @ %d fps)",
             out_path, frame_start, frame_end, frames_len, fps)


def _write_cam_npz(
    out_path: Path,
    *,
    cam: Camera,
    image_paths: List[str],
    ioi_seq_dir: Path,
    frames_len: int,
    video_path: str = "",
    frame_start: int = 0,
    frame_end: int = None,
) -> None:
    """Per-camera NPZ matching the downstream-step contract.

    ``image_paths`` may be empty when the source is a video (no frames
    extracted to disk). ``video_path`` is the absolute path to the
    backing MP4 in that case; downstream code that wants frames reads
    via :class:`capture.VideoFrameReader`.

    ``frame_start`` / ``frame_end`` are the canonical frame range for
    this camera. Downstream :func:`frame_source_from_cam_data` reads
    them to construct ``VideoFrameReader(video_path, start, end)``.
    """
    if frame_end is None:
        frame_end = frame_start + frames_len
    abs_paths = [str(p) for p in image_paths[:frames_len]]
    rel_paths = [os.path.relpath(p, str(ioi_seq_dir)) for p in abs_paths] if abs_paths else []

    cam_int = cam.intrinsics.astype(np.float32)            # (3, 3)
    cam_ext = cam.T_cam_world.astype(np.float32)           # (4, 4) world->cam

    payload: dict = {
        "img_abs_path": np.array(abs_paths),
        "img_rel_path": np.array(rel_paths),
        "cam_name": np.array(cam.name),
        "cam_ext": cam_ext,
        "cam_int": cam_int,
        "cam_img_w": int(cam.width),
        "cam_img_h": int(cam.height),
        "cam_portrait": bool(cam.height > cam.width),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "video_path": np.array(video_path),  # empty string when image-sourced
        # Generic distortion contract -- carries every model (radtan /
        # opencv_brown / vicon_radial_2). Downstream readers prefer these.
        # The legacy ``vicon_radial_2`` key below is retained so older readers
        # (which only know that key) keep working.
        "distortion_model": np.array(cam.distortion_model),
        "distortion_coeffs": np.array(cam.distortion_coeffs, dtype=np.float64),
        # Legacy: only populated when the source format was Vicon XCP (5-param
        # radial); otherwise None. Kept for back-compat -- downstream consumers
        # that still read this key already handle the None case.
        "vicon_radial_2": (
            np.array(cam.distortion_coeffs, dtype=np.float64)
            if cam.distortion_model == "vicon_radial_2" else None
        ),
        # No motion data -> assume body is in every frame. Downstream that
        # doesn't index by this is unaffected; downstream that does sees
        # the contract preserved.
        "is_body_in_img": np.ones((frames_len,), dtype=bool),
    }
    np.savez(out_path, **payload)
    log.info("wrote %s (%d frames%s)", out_path, frames_len,
             f", video_path={video_path}" if video_path else "")


def _select_camera(calib_cameras: Dict[str, Camera], cam_name: str) -> Camera:
    """Resolve a calibration camera by the runner-supplied name.

    Tolerates the common ``IOI_NN`` <-> ``camNN`` discrepancy that older
    XCPs exhibit when ``use_deviceid`` was on.
    """
    if cam_name in calib_cameras:
        return calib_cameras[cam_name]
    # Best-effort fallback: numeric-suffix alignment between e.g.
    # ``cam01`` and ``IOI_01``.
    suffix = "".join(c for c in cam_name if c.isdigit())
    if suffix:
        for k, v in calib_cameras.items():
            ksuf = "".join(c for c in k if c.isdigit())
            if ksuf == suffix:
                log.warning(
                    "camera %s not found by name; matched %s by numeric suffix",
                    cam_name, k,
                )
                return v
    raise KeyError(
        f"camera {cam_name!r} not found in calibration "
        f"(available: {sorted(calib_cameras)})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _validate_input_mode(args) -> str:
    """Enforce the input-mode mutex and return the active mode name."""
    set_modes = [
        m for m, v in (
            ("--json", args.json),
            ("--videos_dir", args.videos_dir),
            ("--images_root_dir", args.images_root_dir),
        ) if v
    ]
    if not set_modes:
        sys.stderr.write(
            "error: one of --json / --videos_dir / --images_root_dir is required.\n"
        )
        sys.exit(2)
    if len(set_modes) > 1:
        sys.stderr.write(
            f"error: {' and '.join(set_modes)} are mutually exclusive.\n"
        )
        sys.exit(2)
    if (args.videos_dir or args.images_root_dir) and not args.calibration:
        sys.stderr.write(
            "error: --calibration is required when using "
            "--videos_dir or --images_root_dir.\n"
        )
        sys.exit(2)
    return set_modes[0].lstrip("-")


def _apply_range(start, end, full_len):
    """Clamp (start, end) to [0, full_len]; return (s, e, sliced_len)."""
    s = max(0, int(start)) if start is not None else 0
    e = min(int(full_len), int(end)) if end is not None else int(full_len)
    e = max(s, e)
    return s, e, e - s


def _ingest_json_mode(args):
    """Image-dirs driven by capture.json (legacy path)."""
    capture_cfg = _load_capture_json(args.json)
    # Prefer the current 'capture_root'; fall back to legacy 'ioi_root'.
    ioi_root = capture_cfg.get("capture_root") or capture_cfg["ioi_root"]
    calib_path = args.calibration or capture_cfg["calib"]
    fps = args.fps if args.fps is not None else int(capture_cfg.get("cam_fps", 30))

    ioi_seq_dir = Path(ioi_root) / args.seq_name
    if not ioi_seq_dir.is_dir():
        sys.stderr.write(
            f"error: sequence directory not found: {ioi_seq_dir}\n"
            f"  (ioi_root from capture.json: {ioi_root})\n"
        )
        sys.exit(2)

    image_paths = _discover_image_paths(ioi_seq_dir, args.cam_names)
    full_len = min(len(paths) for paths in image_paths.values())
    s, e, sliced = _apply_range(args.start, args.end, full_len)
    # Slice each camera's image list to the canonical range.
    image_paths = {c: paths[s:e] for c, paths in image_paths.items()}
    return {
        "calib_path": calib_path,
        "fps": fps,
        "ioi_seq_dir": ioi_seq_dir,
        "frames_len": sliced,
        "frame_start": s,
        "frame_end": e,
        "image_paths": image_paths,
        "video_paths": {},
    }


def _ingest_videos_mode(args):
    """Discover one MP4 per camera; frames_len = min over cameras."""
    video_files = find_video_files(args.videos_dir, cam_names=args.cam_names)
    video_paths: Dict[str, str] = {}
    n_frames_per_cam: Dict[str, int] = {}
    fps_per_cam: Dict[str, float] = {}
    for vp in video_files:
        stem = os.path.splitext(os.path.basename(vp))[0]
        if stem not in args.cam_names:
            continue
        # No start/end here — read full-video metadata; we apply the
        # range *after* computing the across-camera min so the range
        # is meaningful in global frame coordinates.
        reader = VideoFrameReader(vp)
        video_paths[stem] = os.path.abspath(vp)
        n_frames_per_cam[stem] = reader.n_frames
        fps_per_cam[stem] = reader.fps
    missing = [c for c in args.cam_names if c not in video_paths]
    if missing:
        sys.stderr.write(
            f"error: no .mp4 found under {args.videos_dir} for cameras: {missing}\n"
        )
        sys.exit(2)
    full_len = min(n_frames_per_cam.values())
    s, e, sliced = _apply_range(args.start, args.end, full_len)
    fps = args.fps if args.fps is not None else int(round(next(iter(fps_per_cam.values()))))
    return {
        "calib_path": args.calibration,
        "fps": fps,
        "ioi_seq_dir": Path(os.path.abspath(args.videos_dir)),
        "frames_len": sliced,
        "frame_start": s,
        "frame_end": e,
        "image_paths": {c: [] for c in args.cam_names},
        "video_paths": video_paths,
    }


def _ingest_images_root_mode(args):
    """Discover one image-dir per camera under args.images_root_dir."""
    cam_dirs = find_image_cam_dirs(args.images_root_dir, cam_names=args.cam_names)
    image_paths: Dict[str, List[str]] = {}
    for d in cam_dirs:
        stem = os.path.basename(d)
        if stem not in args.cam_names:
            continue
        image_paths[stem] = _gather_images(Path(d))
    missing = [c for c in args.cam_names if c not in image_paths or not image_paths[c]]
    if missing:
        sys.stderr.write(
            f"error: no images found under {args.images_root_dir} for cameras: {missing}\n"
        )
        sys.exit(2)
    full_len = min(len(p) for p in image_paths.values())
    s, e, sliced = _apply_range(args.start, args.end, full_len)
    image_paths = {c: paths[s:e] for c, paths in image_paths.items()}
    fps = args.fps if args.fps is not None else 30
    return {
        "calib_path": args.calibration,
        "fps": fps,
        "ioi_seq_dir": Path(os.path.abspath(args.images_root_dir)),
        "frames_len": sliced,
        "frame_start": s,
        "frame_end": e,
        "image_paths": image_paths,
        "video_paths": {},
    }


def synthesize_ma_cap_npzs(
    out_dir: str,
    seq_name: str,
    cam_names: List[str],
    *,
    calibration_path: str,
    videos_dir: str = None,
    images_root_dir: str = None,
    capture_json: str = None,
    fps: int = None,
    start: int = None,
    end: int = None,
) -> Path:
    """Write the ma_cap NPZ contract directly, bypassing the CLI.

    Used by standalone ma_3d / ma_vis runs to synthesize a minimal
    ``<out_dir>/<seq>/gt/`` tree from a calibration file plus a frame
    source (videos, image dirs, or capture.json). ``start`` / ``end``
    optionally restrict the canonical frame range — every downstream
    step inherits it via the per-camera NPZ. Returns the path to the
    populated ``gt/`` directory.
    """
    class _Args:
        pass
    a = _Args()
    a.json = capture_json
    a.videos_dir = videos_dir
    a.images_root_dir = images_root_dir
    a.calibration = calibration_path
    a.cam_names = list(cam_names)
    a.seq_name = seq_name
    a.fps = fps
    a.start = start
    a.end = end

    mode = _validate_input_mode(a)
    if mode == "json":
        ctx = _ingest_json_mode(a)
    elif mode == "videos_dir":
        ctx = _ingest_videos_mode(a)
    else:
        ctx = _ingest_images_root_mode(a)

    calib = load_calibration(ctx["calib_path"])
    seq_out_dir = Path(out_dir) / seq_name
    gt_dir = seq_out_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    _write_global_npz(
        gt_dir / "global.npz",
        seq_name=seq_name,
        ioi_seq_dir=ctx["ioi_seq_dir"],
        frames_len=ctx["frames_len"],
        fps=ctx["fps"],
        frame_start=ctx["frame_start"],
        frame_end=ctx["frame_end"],
    )
    for cam_name in cam_names:
        cam = _select_camera(dict(calib.cameras), cam_name)
        _write_cam_npz(
            gt_dir / f"{cam_name}.npz",
            cam=cam,
            image_paths=ctx["image_paths"].get(cam_name, []),
            ioi_seq_dir=ctx["ioi_seq_dir"],
            frames_len=ctx["frames_len"],
            video_path=ctx["video_paths"].get(cam_name, ""),
            frame_start=ctx["frame_start"],
            frame_end=ctx["frame_end"],
        )
    return gt_dir


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    _validate_input_mode(args)  # exits on failure
    t0 = time.perf_counter()
    gt_dir = synthesize_ma_cap_npzs(
        args.out,
        args.seq_name,
        args.cam_names,
        calibration_path=args.calibration,
        videos_dir=args.videos_dir,
        images_root_dir=args.images_root_dir,
        capture_json=args.json,
        fps=args.fps,
        start=args.start,
        end=args.end,
    )

    elapsed = time.perf_counter() - t0
    print(
        f"ma_cap done: {gt_dir.parent} "
        f"({len(args.cam_names)} cameras, {elapsed:.2f}s)"
    )


if __name__ == "__main__":
    main()
