"""Per-camera SMPL-X overlay video rendering.

Vendored from the upstream ``smplx_overlay.py``. The polished
version drops:

* the ``paths.string_path_to_windows`` cluster shim (we do not run on
  Windows clusters in this codebase),
* the silent ``mm`` -> ``m`` heuristic on the extrinsic translation
  (``if abs(t).max() > 200: t /= 1000``) -- if your calibration is in
  millimetres, fix it upstream,
* the unused ``render_front_view_colored`` code path,
* hard-coded ``3008x4112`` fallback resolution -- the camera npz already
  supplies ``cam_img_w``/``cam_img_h``.

``pyrender`` is imported lazily (inside :class:`OverlayRenderer`'s init)
so that simply importing this module does not require a working OpenGL
context. On Linux, the module sets ``PYOPENGL_PLATFORM=egl`` if not
already set; offscreen rendering still requires an EGL-capable GPU.

``ffmpeg`` is optional. If present in ``$PATH``, mp4v outputs are
re-encoded to H.264/yuv420p for browser compatibility; otherwise the
mp4v output is kept as-is.
"""
from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .cameras import Camera
from .motion import PersonMotion
from .projection import world_to_cam

log = logging.getLogger(__name__)

DEFAULT_OPACITY = 0.6
DEFAULT_FPS = 30


# The renderer needs an OpenGL backend; on Linux we default to EGL so it
# can run headless. On macOS/Windows pyrender picks the right thing itself.
if platform.system() != "Windows":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# Process-pool worker globals (filled by the initializer).
_WORKER_VERTICES: Optional[List[np.ndarray]] = None
_WORKER_BODY_IDS: Optional[List[int]] = None
_WORKER_FACES: Optional[np.ndarray] = None
_WORKER_MAX_FRAMES: Optional[int] = None


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraOverlayResult:
    cam_name: str
    video_path: Optional[Path]   # None if rendering was skipped or failed
    frames: int
    seconds: float


# ---------------------------------------------------------------------------
# pyrender wrapper
# ---------------------------------------------------------------------------

def _create_offscreen_renderer(pyrender, width: int, height: int):
    """OffscreenRenderer with EGL device fallback.

    EGL device 0 is not always usable: inside containers (apptainer binds the
    host ``/dev/dri``, so mesa enumerates the physical GPU first) the GPU's
    EGL vendor stack may be unreachable and ``eglInitialize`` fails, while a
    later enumerated device (mesa's software renderer) works fine. When the
    default choice fails and the user hasn't pinned ``EGL_DEVICE_ID``, probe
    the other device ids before giving up.
    """
    make = lambda: pyrender.OffscreenRenderer(  # noqa: E731
        viewport_width=width, viewport_height=height, point_size=1.0)
    try:
        return make()
    except Exception:
        if (os.environ.get("PYOPENGL_PLATFORM") != "egl"
                or "EGL_DEVICE_ID" in os.environ):
            raise
    last_exc: Optional[Exception] = None
    for dev in range(1, 8):
        os.environ["EGL_DEVICE_ID"] = str(dev)
        try:
            renderer = make()
            log.warning(
                "default EGL device failed to initialize; using "
                "EGL_DEVICE_ID=%d (likely the software renderer)", dev)
            return renderer
        except Exception as e:  # try the next device
            last_exc = e
    os.environ.pop("EGL_DEVICE_ID", None)
    raise last_exc


class OverlayRenderer:
    """Render multiple SMPL-X meshes onto a background image plane.

    A separate :class:`OverlayRenderer` is created per camera (and per
    output resolution) to avoid touching pyrender's stateful internals
    across cameras with different intrinsics.
    """

    def __init__(
        self,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        faces: np.ndarray,
        opacity: float = DEFAULT_OPACITY,
    ) -> None:
        try:
            import pyrender
        except ImportError as e:  # pragma: no cover  (env error)
            raise RuntimeError(
                "pyrender is required for overlay rendering. "
                "Install it (and an EGL-capable GPU stack) or run with "
                "--skip-overlay."
            ) from e

        try:
            self._renderer = _create_offscreen_renderer(pyrender, width, height)
        except Exception as e:
            backend = os.environ.get("PYOPENGL_PLATFORM", "<unset>")
            device = os.environ.get("EGL_DEVICE_ID", "<unset>")
            raise RuntimeError(
                "Failed to initialize pyrender offscreen renderer "
                f"(PYOPENGL_PLATFORM={backend}, EGL_DEVICE_ID={device}). "
                "Pass --skip-overlay to bypass overlay rendering."
            ) from e
        self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy
        self._faces = faces
        self._opacity = float(opacity)

    def render(
        self,
        verts_per_person_cam_frame: Sequence[np.ndarray],
        colors_rgb: Sequence[Tuple[float, float, float]],
        background_bgr: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render one frame.

        Args:
            verts_per_person_cam_frame: Each entry is ``(V, 3)`` mesh vertices
                already in the camera frame (``T_cam_world @ p_world``).
            colors_rgb: Per-mesh ``(r, g, b)`` floats in ``[0, 1]``.
            background_bgr: Optional ``(H, W, 3)`` uint8 image to alpha-blend
                under the rendered meshes. ``None`` returns the rendered
                meshes alone (RGB ordering, ready for cv2 ``VideoWriter``).
        """
        import pyrender
        import trimesh

        scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=np.zeros(3))
        scene.add(
            pyrender.camera.IntrinsicsCamera(
                fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy
            ),
            pose=np.eye(4),
        )
        scene.add(
            pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0),
            pose=trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0]),
        )
        scene.add(
            pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0),
            pose=trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0]),
        )
        # OpenCV camera frame has +Y down, +Z forward; pyrender's IntrinsicsCamera
        # expects the legacy OpenGL convention (+Y up). 180-deg flip about X.
        flip_x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])

        for v, color in zip(verts_per_person_cam_frame, colors_rgb):
            mesh = trimesh.Trimesh(v, self._faces, process=False)
            mesh.apply_transform(flip_x)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode="OPAQUE",
                baseColorFactor=(color[0], color[1], color[2], 1.0),
            )
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False))

        rgba, _ = self._renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        if background_bgr is None:
            return rgba[:, :, :3][:, :, ::-1].copy()  # RGB -> BGR

        bg = background_bgr.astype(np.float32)
        rgb = rgba[:, :, :3].astype(np.float32)
        bgr = rgb[:, :, ::-1]
        alpha = (rgba[:, :, 3:4].astype(np.float32) / 255.0) * self._opacity
        out = bgr * alpha + bg * (1.0 - alpha)
        return out.astype(np.uint8)

    def close(self) -> None:
        try:
            self._renderer.delete()
        except Exception:
            pass

    def __enter__(self) -> "OverlayRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Per-camera rendering
# ---------------------------------------------------------------------------

def _ensure_even(value: int) -> int:
    value = max(2, int(value))
    return value if value % 2 == 0 else max(2, value - 1)


def _compute_output_size(
    src_w: int, src_h: int, target_long_side: Optional[int]
) -> Tuple[int, int, float, float]:
    """Resize keeping aspect ratio so the long side equals ``target_long_side``.

    Returns ``(out_w, out_h, scale_x, scale_y)``. ``target_long_side <= 0``
    or ``None`` keeps the source size (1:1 scale).
    """
    if target_long_side is None or int(target_long_side) <= 0:
        return int(src_w), int(src_h), 1.0, 1.0
    target = max(2, int(target_long_side))
    scale = target / float(max(src_w, src_h))
    out_w = _ensure_even(int(round(src_w * scale)))
    out_h = _ensure_even(int(round(src_h * scale)))
    return out_w, out_h, out_w / float(src_w), out_h / float(src_h)


def _reencode_to_h264(path: Path) -> None:
    """Best-effort re-encode to H.264/yuv420p for browser playback.

    Silently no-ops if ``ffmpeg`` is not on the PATH or the encode fails.
    """
    if shutil.which("ffmpeg") is None or not path.exists():
        return
    tmp = path.with_suffix(".tmp.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(tmp),
            ],
            check=True,
        )
        os.replace(tmp, path)
    except subprocess.CalledProcessError:
        if tmp.exists():
            tmp.unlink()


def _resolve_image_path(cam: Camera, frame_idx: int, prefix: str) -> Optional[Path]:
    if cam.image_paths is None or frame_idx >= len(cam.image_paths):
        return None
    raw = cam.image_paths[frame_idx]
    if prefix:
        return (Path(prefix) / raw).resolve(strict=False)
    return Path(raw)


def _make_frame_source(cam: Camera, image_prefix: str = ""):
    """Build a :class:`capture.FrameSource` for ``cam``'s backing frames.

    Priority: ``cam.video_path`` (videos-mode → :class:`VideoSource`) >
    ``cam.image_paths`` (NPZ / image-dir mode → :class:`ImageFileSource`).
    Returns ``None`` if neither is available (the overlay renders on a
    black canvas in that case).

    ``image_prefix`` is prepended to each image path when set (used when
    the calibration NPZ holds paths from a different machine / mount).
    Absolute paths in ``image_paths`` ignore the prefix per ``pathlib``
    semantics.
    """
    # Lazy import + sys.path bump: this module may run in a worker pool
    # process under cwd=visualization/, so capture/ isn't auto-importable.
    import os as _os, sys as _sys
    _repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    from capture.frame_source import ImageFileSource, VideoSource  # noqa: E402
    from capture.video_reader import VideoFrameReader  # noqa: E402

    if cam.video_path:
        # ma_cap stamps frame_start/frame_end on the per-camera NPZ so the
        # overlay's local-index 0 maps to source-video frame `frame_start`,
        # keeping backgrounds aligned with ma_3d's meshes when only a
        # sub-range of the video was processed. Without these the reader
        # would default to source frame 0 → off-by-`start_frame` sync bug.
        try:
            reader = VideoFrameReader(
                cam.video_path,
                start=cam.frame_start,
                end=cam.frame_end,
            )
        except (OSError, RuntimeError) as e:
            log.warning("camera %s: cannot open video_path %s (%s)", cam.name, cam.video_path, e)
            return None
        return VideoSource(reader, camera_name=cam.name)
    if cam.image_paths:
        if image_prefix:
            paths = [str(Path(image_prefix) / p) for p in cam.image_paths]
        else:
            paths = list(cam.image_paths)
        return ImageFileSource(paths, camera_name=cam.name)
    return None


def _read_frame_bgr(source, idx: int):
    """Read frame ``idx`` from ``source`` as a BGR numpy array.

    ``source.read_rgb`` returns RGB; the overlay pipeline is cv2-native
    (BGR). Returns ``None`` on read failure (caller falls back to black).
    """
    try:
        rgb = source.read_rgb(idx)
    except (OSError, IndexError, RuntimeError) as e:
        log.warning("frame %d read failed: %s", idx, e)
        return None
    if rgb is None:
        return None
    return rgb[..., ::-1].copy()  # RGB → BGR (cv2 convention)


def _render_one_camera(
    cam: Camera,
    motions: Sequence[PersonMotion],
    faces: np.ndarray,
    out_dir: Path,
    *,
    fps: int,
    resolution: Optional[int],
    max_frames: Optional[int],
    image_prefix: str,
    colors_rgb: Sequence[Tuple[float, float, float]],
    opacity: float,
    undistort: bool = False,
) -> CameraOverlayResult:
    import cv2
    _undistort_fn = None
    if undistort:
        # Lazy import + sys.path bump: this module may run in a worker pool
        # process under cwd=visualization/, so capture/ isn't auto-importable.
        import os as _os, sys as _sys
        _repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from capture.undistort import undistort_rgb as _undistort_fn  # noqa: E402

    t0 = time.perf_counter()
    body_ids = [m.body_id for m in motions]
    if not body_ids:
        log.info("camera %s: no motions, skipping", cam.name)
        return CameraOverlayResult(cam.name, None, 0, 0.0)

    # Build a single frame source — handles all three input modes:
    # video (cam.video_path), image dir / NPZ (cam.image_paths), or
    # neither (None → render on black canvas).
    source = _make_frame_source(cam, image_prefix)
    source_kind = (
        "video" if (source is not None and cam.video_path)
        else "images" if source is not None
        else "none"
    )

    # The number of frames available is the min over (motion lengths,
    # source frame count if known, optional --max-frames cap).
    n_motion_frames = min(m.vertices.shape[0] for m in motions)
    n_source_frames = len(source) if source is not None else n_motion_frames
    frames_to_render = min(n_motion_frames, n_source_frames)
    if max_frames is not None and max_frames > 0:
        frames_to_render = min(frames_to_render, int(max_frames))
    if frames_to_render <= 0:
        log.info("camera %s: zero frames to render", cam.name)
        return CameraOverlayResult(cam.name, None, 0, time.perf_counter() - t0)

    # Probe the first frame to discover the source resolution; fall back
    # to (cam.width, cam.height) which the npz always provides.
    first_img = None
    if source is not None:
        first_img = _read_frame_bgr(source, 0)
        if first_img is not None and _undistort_fn is not None:
            first_img = _undistort_fn(first_img, cam)
    if first_img is not None:
        src_h, src_w = first_img.shape[:2]
    else:
        src_w, src_h = cam.width, cam.height
        if src_w <= 0 or src_h <= 0:
            log.warning(
                "camera %s: no source frame and missing cam_img_w/h; skipping",
                cam.name,
            )
            return CameraOverlayResult(cam.name, None, 0, time.perf_counter() - t0)
        log.info(
            "camera %s: no source frame (source_kind=%s); "
            "rendering on a black canvas %dx%d",
            cam.name, source_kind, src_w, src_h,
        )

    out_w, out_h, sx, sy = _compute_output_size(src_w, src_h, resolution)
    if (out_w, out_h) != (src_w, src_h):
        log.info(
            "camera %s: resizing overlay from %dx%d to %dx%d",
            cam.name, src_w, src_h, out_w, out_h,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cam.name}.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (out_w, out_h)
    )
    if not writer.isOpened():
        log.warning("camera %s: failed to open VideoWriter for %s", cam.name, out_path)
        return CameraOverlayResult(cam.name, None, 0, time.perf_counter() - t0)

    try:
        with OverlayRenderer(
            fx=float(cam.intrinsics[0, 0]) * sx,
            fy=float(cam.intrinsics[1, 1]) * sy,
            cx=float(cam.intrinsics[0, 2]) * sx,
            cy=float(cam.intrinsics[1, 2]) * sy,
            width=out_w, height=out_h,
            faces=faces, opacity=opacity,
        ) as renderer:
            for frame_idx in range(frames_to_render):
                bg = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                if source is not None:
                    loaded = _read_frame_bgr(source, frame_idx)
                    if loaded is not None:
                        if _undistort_fn is not None:
                            loaded = _undistort_fn(loaded, cam)
                        if loaded.shape[:2] != (out_h, out_w):
                            loaded = cv2.resize(loaded, (out_w, out_h))
                        bg = loaded

                verts_cam = [
                    world_to_cam(m.vertices[frame_idx], cam.extrinsics) for m in motions
                ]
                colors = [
                    colors_rgb[m.body_id % len(colors_rgb)] for m in motions
                ]
                rendered = renderer.render(verts_cam, colors, background_bgr=bg)
                writer.write(rendered)
    finally:
        writer.release()

    elapsed = time.perf_counter() - t0
    log.info(
        "camera %s: wrote %s (%d frames, %.2fs, %.2f fps)",
        cam.name, out_path, frames_to_render, elapsed,
        frames_to_render / elapsed if elapsed > 0 else 0.0,
    )
    _reencode_to_h264(out_path)
    return CameraOverlayResult(cam.name, out_path, frames_to_render, elapsed)


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------

def render_overlay_videos(
    cameras: Sequence[Camera],
    motions: Sequence[PersonMotion],
    faces: np.ndarray,
    out_dir,
    *,
    fps: int = DEFAULT_FPS,
    resolution: Optional[int] = 1280,
    max_frames: Optional[int] = None,
    num_workers: int = 1,
    image_prefix: str = "",
    colors_rgb: Optional[Sequence[Tuple[float, float, float]]] = None,
    opacity: float = DEFAULT_OPACITY,
    undistort: bool = False,
) -> List[CameraOverlayResult]:
    """Render an overlay mp4 per camera.

    Args:
        cameras: Cameras to render.
        motions: Per-person predicted vertex sequences.
        faces: ``(F, 3)`` SMPL-X face indices.
        out_dir: Directory to write ``<cam_name>.mp4`` files into.
        fps: Output video FPS.
        resolution: Long-side target in pixels (preserves aspect ratio).
            ``None`` or ``<=0`` keeps the source camera resolution.
        max_frames: Optional cap on frames per camera.
        num_workers: 1 = single process. >1 spawns a process pool with one
            ``OverlayRenderer`` per worker (each has its own pyrender context).
        image_prefix: Prefix to prepend to ``cam.image_paths`` entries (used
            when the calibration npz holds paths from a different machine
            and you mounted the dataset elsewhere).
        colors_rgb: Per-person ``(r, g, b)`` floats in ``[0, 1]``. Defaults
            to a 10-colour palette generated by :mod:`distinctipy` if the
            module is available, else evenly-spaced HSV.
        opacity: Overlay alpha in ``[0, 1]``.

    Returns:
        One :class:`CameraOverlayResult` per input camera, in the same order.
    """
    out_dir = Path(out_dir)
    if colors_rgb is None:
        colors_rgb = _default_palette(max(10, max((m.body_id for m in motions), default=0) + 1))
    fps = max(1, int(fps))
    num_workers = max(1, int(num_workers))

    if num_workers == 1 or len(cameras) == 1:
        return [
            _render_one_camera(
                cam, motions, faces, out_dir,
                fps=fps, resolution=resolution, max_frames=max_frames,
                image_prefix=image_prefix, colors_rgb=colors_rgb, opacity=opacity,
                undistort=undistort,
            )
            for cam in cameras
        ]

    return _render_in_parallel(
        cameras, motions, faces, out_dir,
        fps=fps, resolution=resolution, max_frames=max_frames,
        image_prefix=image_prefix, colors_rgb=colors_rgb, opacity=opacity,
        undistort=undistort,
        num_workers=num_workers,
    )


def _default_palette(n: int) -> List[Tuple[float, float, float]]:
    try:
        import distinctipy  # type: ignore
        return [tuple(c) for c in distinctipy.get_colors(n, pastel_factor=0.5, rng=0)]
    except ImportError:
        # Even-spaced HSV. Good enough as a fallback.
        import colorsys
        return [colorsys.hsv_to_rgb(i / max(n, 1), 0.55, 0.9) for i in range(n)]


# ---------------------------------------------------------------------------
# Process-pool path
# ---------------------------------------------------------------------------

def _init_worker(motions: Sequence[PersonMotion], faces: np.ndarray) -> None:
    global _WORKER_VERTICES, _WORKER_BODY_IDS, _WORKER_FACES, _WORKER_MAX_FRAMES
    _WORKER_VERTICES = [m.vertices for m in motions]
    _WORKER_BODY_IDS = [m.body_id for m in motions]
    _WORKER_FACES = faces
    _WORKER_MAX_FRAMES = (
        min(v.shape[0] for v in _WORKER_VERTICES) if _WORKER_VERTICES else 0
    )


def _worker_render_camera(payload):
    cam, out_dir, fps, resolution, max_frames, image_prefix, colors_rgb, opacity, undistort = payload
    motions = [
        PersonMotion(body_id=bid, vertices=v)
        for bid, v in zip(_WORKER_BODY_IDS or [], _WORKER_VERTICES or [])
    ]
    return _render_one_camera(
        cam, motions, _WORKER_FACES, Path(out_dir),
        fps=fps, resolution=resolution, max_frames=max_frames,
        image_prefix=image_prefix, colors_rgb=colors_rgb, opacity=opacity,
        undistort=undistort,
    )


def _render_in_parallel(
    cameras: Sequence[Camera],
    motions: Sequence[PersonMotion],
    faces: np.ndarray,
    out_dir: Path,
    *,
    fps: int,
    resolution: Optional[int],
    max_frames: Optional[int],
    image_prefix: str,
    colors_rgb: Sequence[Tuple[float, float, float]],
    opacity: float,
    num_workers: int,
    undistort: bool = False,
) -> List[CameraOverlayResult]:
    payloads = [
        (cam, str(out_dir), fps, resolution, max_frames, image_prefix, colors_rgb, opacity, undistort)
        for cam in cameras
    ]
    ordered: List[Optional[CameraOverlayResult]] = [None] * len(cameras)
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(list(motions), faces),
    ) as ex:
        future_to_idx = {ex.submit(_worker_render_camera, p): i for i, p in enumerate(payloads)}
        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                ordered[idx] = fut.result()
            except Exception as e:  # pragma: no cover  (only raised on real GPU)
                log.warning("camera %s failed: %s", cameras[idx].name, e)
                ordered[idx] = CameraOverlayResult(cameras[idx].name, None, 0, 0.0)
    return [r for r in ordered if r is not None]
