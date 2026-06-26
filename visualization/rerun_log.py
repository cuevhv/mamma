"""Write a scene ``.rrd`` file via the ``rerun-sdk``.

Encapsulates everything that ``MultiViewSystem`` did in the upstream
``mv-rerun`` code, minus the dataset-name dispatch and the dead
image-streaming branches. Public surface is :class:`RerunSceneLogger`.

Convention: cameras are logged under ``world/cameras/<cam_name>``,
ground plane under ``world/ground``, predicted meshes under
``world/meshes/<person_id>``, and 2D landmark projections under
``world/cameras/<cam_name>/image/<person_id>_keypoints``.

The Rerun timeline is named ``time`` and ticks in seconds (``frame_id /
fps``). The SDK's time API changed in 0.28.x; we use a single helper
that works with both old and new calls.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional, Sequence

import numpy as np

from .cameras import Camera
from .motion import LandmarkData, PersonMotion, uncertainty_from_log_variance

log = logging.getLogger(__name__)

_CAM_TAG = "world/cameras"
_GROUND_TAG = "world/ground"
_MESHES_TAG = "world/meshes"


def _set_time_seconds(seconds: float) -> None:
    """Set the ``time`` timeline in seconds, working with rerun-sdk >=0.28 and older."""
    import rerun as rr

    try:
        rr.set_time("time", timestamp=seconds)
    except (AttributeError, TypeError):
        rr.set_time_seconds("time", seconds)


def _ffmpeg_bin() -> str:
    """Path to an ffmpeg binary: system ``ffmpeg`` first, else the one bundled
    with ``imageio-ffmpeg`` (so the H.264 backdrop works without a system install)."""
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def ffmpeg_available() -> bool:
    """True when an ffmpeg binary is reachable (system or bundled). Used to
    decide whether the default H.264 backdrop can run, or must fall back to JPEG."""
    import os
    import shutil
    if shutil.which("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg
        return os.path.exists(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return False


def _probe_video_codec(path: str) -> str:
    """Source video codec name ('h264', 'hevc', 'av1', ...); '' on failure."""
    import shutil
    import subprocess
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of",
             "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _video_cache_dir(src_path: str) -> str:
    """Writable cache dir for re-encoded H.264. Prefers next-to-source (so the
    encode is reused across runs/out-tags), falls back to the system temp dir."""
    import os
    import tempfile
    cand = os.path.join(os.path.dirname(os.path.abspath(src_path)),
                        ".mamma_rrd_video_cache")
    try:
        os.makedirs(cand, exist_ok=True)
        probe = os.path.join(cand, ".w")
        open(probe, "w").close()
        os.remove(probe)
        return cand
    except OSError:
        d = os.path.join(tempfile.gettempdir(), "mamma_rrd_video_cache")
        os.makedirs(d, exist_ok=True)
        return d


def _ensure_h264(src_path, width, height, frame_start, frame_end, crf):
    """Path to an H.264 mp4 of ``src_path`` at ``width``x``height`` for frames
    ``[frame_start, frame_end)`` — re-encoding **only when necessary** and caching.

    Re-encode is skipped (the original is returned) when the source is already
    H.264 at the requested framing, since H.264 is the one codec the rerun web
    viewer decodes everywhere. Otherwise we decode -> re-encode H.264 (this is
    also what makes HEVC/AV1 sources viewable). ``-bf 0`` keeps access units in
    display order so each maps 1:1 to a timeline frame.
    """
    import os
    import hashlib
    import subprocess
    import cv2

    fs = int(frame_start or 0)
    fe = None if frame_end is None else int(frame_end)
    codec = _probe_video_codec(src_path)
    cap = cv2.VideoCapture(src_path)
    nw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    nh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ntot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    need_trim = fs > 0 or (fe is not None and fe < ntot)
    need_resize = (int(width), int(height)) != (nw, nh)
    if codec == "h264" and not need_trim and not need_resize:
        return src_path  # already viewer-ready at the requested framing

    cache = _video_cache_dir(src_path)
    try:
        mtime = int(os.path.getmtime(src_path))
    except OSError:
        mtime = 0
    # Bump _ENCODE_RECIPE whenever the ffmpeg recipe changes so stale cached
    # re-encodes are regenerated rather than silently reused.
    _ENCODE_RECIPE = "v3-bt709-main-gop30-lvl40"
    key = hashlib.md5(
        f"{os.path.abspath(src_path)}:{mtime}:{width}x{height}:{fs}:{fe}:"
        f"{crf}:{_ENCODE_RECIPE}".encode()
    ).hexdigest()[:12]
    out = os.path.join(cache, f"{key}.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out  # cache hit — no re-encode

    vf = []
    if need_trim:
        end_expr = f":end_frame={fe}" if fe is not None else ""
        vf.append(f"trim=start_frame={fs}{end_expr}")
        vf.append("setpts=PTS-STARTPTS")
    # Convert the source (whatever its matrix is) to BT.709 limited-range
    # yuv420p, then tag it fully below. The browser's WebCodecs decoder renders
    # BLACK when the stream carries an unusual matrix (e.g. bt470bg) or 'unknown'
    # transfer/primaries — native decoders tolerate it, WebCodecs does not. So we
    # normalise + fully specify the colour signalling for cross-browser playback.
    vf.append(f"scale={int(width)}:{int(height)}:out_color_matrix=bt709")
    vf.append("format=yuv420p")
    tmp = out + ".tmp.mp4"
    subprocess.run(
        [_ffmpeg_bin(), "-y", "-i", src_path, "-vf", ",".join(vf), "-an",
         # WebCodecs (the browser decoder) is far stricter than the native one:
         #  - 'main' profile + an explicit, sufficient level (4.0 covers <=1080p)
         #    so the stream isn't rejected for an under-reported level;
         #  - frequent keyframes (1s GOP, no scene-cut jitter) so the viewer can
         #    always start decoding from a nearby IDR — a long default GOP leaves
         #    only one keyframe, which can render black on seek/scrub in-browser;
         #  - bf=0 keeps access units in display order for 1:1 timeline mapping.
         "-c:v", "libx264", "-profile:v", "main", "-level:v", "4.0",
         "-preset", "medium", "-crf", str(int(crf)),
         "-bf", "0", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
         "-pix_fmt", "yuv420p",
         "-colorspace", "bt709", "-color_primaries", "bt709",
         "-color_trc", "bt709", "-color_range", "tv",
         "-movflags", "+faststart", tmp],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True,
    )
    os.replace(tmp, out)
    return out


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex smooth normals via area-weighted face-normal accumulation.

    Equivalent to trimesh's default vertex_normals but ~5-10x faster (no
    trimesh import, no mesh validation, just vectorized numpy). The face
    normal ``cross(v1 - v0, v2 - v0)`` is *not* normalized before scatter-
    summing into vertex slots, so each face's contribution is weighted by
    its area — the standard recipe for smooth shading.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)  # (F, 3) — magnitude = 2 * area
    vn = np.zeros_like(vertices, dtype=np.float32)
    np.add.at(vn, faces[:, 0], face_normals)
    np.add.at(vn, faces[:, 1], face_normals)
    np.add.at(vn, faces[:, 2], face_normals)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(norms, 1e-12)


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """Snap a near-rotation matrix to the closest proper orthonormal one.

    Floating-point cam_ext blocks loaded from npz can drift; this keeps the
    Rerun viewer happy without changing the underlying pose meaningfully.
    """
    U, _, Vt = np.linalg.svd(R)
    R_fixed = U @ Vt
    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1.0
        R_fixed = U @ Vt
    return R_fixed


def compute_floor_height(
    motions: Sequence[PersonMotion], *, up_vec, percentile: float = 5.0
) -> float:
    """Robust floor "height" along the (signed) up vector.

    Returns the ``percentile``-th value of the per-frame minima of the up-axis
    projection ``verts @ up_vec``. For ``up_vec = +Z`` this is exactly the old
    ``verts[:,:,2].min`` (back-compatible); for a signed axis like ``-Y`` it
    correctly tracks the floor on the negative side.
    """
    if not motions:
        return 0.0
    up = np.asarray(up_vec, dtype=np.float64)
    per_frame_mins = []
    for motion in motions:
        proj = np.asarray(motion.vertices, dtype=np.float64) @ up   # (F, V)
        per_frame_mins.append(proj.min(axis=1))
    return float(np.percentile(np.concatenate(per_frame_mins), percentile))


class RerunSceneLogger:
    """Stateful wrapper around the rerun-sdk for a single scene log.

    Usage:
        with RerunSceneLogger(rrd_path="scene.rrd", fps=30) as logger:
            logger.log_world_up("z")
            logger.log_cameras(cameras)
            logger.log_ground(floor_height=0.0, up_vec=[0, 0, 1])
            logger.log_meshes(motions, faces, colors)
            logger.log_landmark_projections(cameras, landmarks_by_cam, colors)
    """

    def __init__(
        self,
        rrd_path: Optional[str],
        *,
        fps: int = 30,
        app_id: str = "MAMMA visualization",
        display_scale: float = 1.0,
        image_long_edge: Optional[int] = None,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if display_scale <= 0:
            raise ValueError(f"display_scale must be positive, got {display_scale}")
        if image_long_edge is not None and image_long_edge <= 0:
            raise ValueError(
                f"image_long_edge must be positive when set, got {image_long_edge}"
            )

        import rerun as rr  # raises ImportError early if missing

        self._rr = rr
        self.rrd_path = rrd_path
        self.fps = int(fps)
        self.display_scale = float(display_scale)
        # When set, drives a per-camera scale that overrides display_scale
        # so the Pinhole, 2D landmarks, and any logged image all share one
        # downscaled pixel grid. See _effective_scale.
        self.image_long_edge: Optional[int] = (
            int(image_long_edge) if image_long_edge is not None else None
        )
        # spawn=True opens a viewer; only when no rrd file is requested.
        rr.init(app_id, spawn=(rrd_path is None))
        if rrd_path is not None:
            rr.save(rrd_path)
        # Ship a default blueprint with the recording so the viewer opens with a
        # fresh auto-layout for THIS sequence, instead of reusing a cached layout
        # from a previously-opened .rrd (Rerun persists blueprints per app_id, which
        # otherwise carries over stale per-camera windows and zoom). Best-effort:
        # older rerun-sdk without the blueprint API just keeps the plain recording.
        try:
            import rerun.blueprint as rrb
            rr.send_blueprint(
                rrb.Blueprint(auto_layout=True, auto_views=True),
                make_active=True, make_default=True,
            )
        except Exception:  # noqa: BLE001
            pass
        _set_time_seconds(0.0)

    # ---- context manager -------------------------------------------------

    def __enter__(self) -> "RerunSceneLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # rerun-sdk flushes on interpreter shutdown; nothing else to release.
        pass

    # ---- timing ----------------------------------------------------------

    def go_to_frame(self, frame_id: int) -> None:
        _set_time_seconds(frame_id / self.fps)

    # ---- cameras ---------------------------------------------------------

    def _effective_scale(self, cam: Camera) -> float:
        """Per-camera display scale.

        When ``image_long_edge`` is set, the scale is derived per-camera
        so the resulting Pinhole resolution matches the thumbnail we plan
        to log on that entity. Otherwise we fall back to the user's
        ``display_scale`` (current behaviour).
        """
        if self.image_long_edge is None:
            return self.display_scale
        long_edge = max(int(cam.width), int(cam.height))
        if long_edge <= 0:
            return self.display_scale
        return float(self.image_long_edge) / float(long_edge)

    def log_cameras(self, cameras: Iterable[Camera]) -> None:
        for cam in cameras:
            self._log_camera(cam)

    def _log_camera(self, cam: Camera) -> None:
        rr = self._rr
        scale = self._effective_scale(cam)
        display = cam.scaled(scale) if scale != 1.0 else cam

        rr.log(
            entity_path=f"{_CAM_TAG}/{cam.name}",
            entity=rr.Pinhole(
                resolution=(display.width, display.height),
                image_from_camera=display.intrinsics.flatten(),
            ),
        )
        # T_cam_world -> T_world_cam (camera pose in world). Snap rotation
        # to a clean orthonormal so Rerun's viewer doesn't reject it.
        R_cw = _orthonormalize(cam.extrinsics[:3, :3])
        t_cw = cam.extrinsics[:3, 3]
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        rr.log(
            entity_path=f"{_CAM_TAG}/{cam.name}",
            entity=rr.Transform3D(mat3x3=R_wc, translation=t_wc),
        )

    # ---- ground ----------------------------------------------------------

    def log_world_up(self, up_axis: str) -> None:
        """Log world ``ViewCoordinates`` so the viewer orients to the up-axis.

        Without this the web viewer guesses the up direction (scenes from non-Z-up
        rigs render rotated). ``up_axis`` is a signed axis string (e.g. ``-y``).
        Best-effort: older rerun-sdk / a missing import just keeps the plain recording.
        """
        rr = self._rr
        try:
            from capture.calibration import up_axis_to_viewcoords  # ma_vis adds repo root
            vc = getattr(rr.ViewCoordinates, up_axis_to_viewcoords(up_axis), None)
        except Exception:  # noqa: BLE001
            vc = None
        if vc is None:
            return
        try:
            rr.log("world", vc, static=True)
        except TypeError:
            try:
                rr.log("world", vc, timeless=True)
            except TypeError:
                rr.log("world", vc)

    def log_ground(
        self, *, floor_height: float = 0.0, size: float = 10.0, up_vec,
        center=(0.0, 0.0),
    ) -> None:
        rr = self._rr
        up = np.asarray(up_vec, dtype=np.float64)
        idx = int(np.argmax(np.abs(up)))
        sign = 1.0 if up[idx] >= 0 else -1.0
        a0, a1 = [a for a in (0, 1, 2) if a != idx]
        # ``center`` offsets the quad on the two non-up (plane) axes so it sits
        # under the subjects; ``size`` is the half-extent. Sizing/centering to the
        # people keeps the viewer's auto-fit framed on them, not a big origin floor.
        plane_coord = floor_height * sign
        corners = [(-size, size), (size, size), (-size, -size), (size, -size)]
        coords = np.zeros((4, 3), dtype=np.float64)
        for i, (c0, c1) in enumerate(corners):
            coords[i, a0] = center[0] + c0
            coords[i, a1] = center[1] + c1
            coords[i, idx] = plane_coord
        normal = up / (np.linalg.norm(up) + 1e-12)
        ground = rr.Mesh3D(
            vertex_positions=coords,
            triangle_indices=np.array([[0, 1, 2], [1, 3, 2]]),
            vertex_normals=np.tile(normal, (4, 1)),
            vertex_colors=np.tile(np.array([80, 80, 80], dtype=np.uint8), (4, 1)),
        )
        rr.log(entity_path=_GROUND_TAG, entity=ground)

    # ---- meshes ----------------------------------------------------------

    def log_meshes(
        self,
        motions: Sequence[PersonMotion],
        faces: np.ndarray,
        colors: Sequence[Sequence[float]],
        *,
        max_workers: int = 4,
    ) -> None:
        """Log per-frame meshes for every person in ``motions``.

        Each per-frame log writes the **full** Mesh3D archetype: vertex
        positions, vertex normals (for smooth shading), triangle
        indices, and the per-person albedo. Threads parallelize the
        per-frame normals computation across persons; the main thread
        serializes ``go_to_frame`` + ``rr.log`` so the rerun timeline
        state isn't raced across threads.

        Why full Mesh3D per frame, not a static-topology +
        temporal-positions split: an earlier optimization logged
        triangle_indices + albedo + frame-0 vertex_positions as static
        and per-frame logs as a partial update (positions + normals
        only). In practice rerun's viewer rendered the static frame-0
        positions even when the temporal log shipped fresh positions —
        the "normals change but mesh doesn't move" symptom. Writing the
        full archetype every frame removes the static/temporal blend
        for vertex_positions entirely and costs ~21 KB/frame extra in
        the .rrd (acceptable).
        """
        if not motions:
            return

        rr = self._rr
        faces_u32 = np.asarray(faces, dtype=np.uint32)
        # Normals computation reads vertex indices through `faces` once
        # per frame; cache the (V, 3) int form so we don't re-cast
        # inside the hot loop.
        faces_i = faces_u32.astype(np.int64, copy=False)

        # Per-person albedo tuple computed once.
        def _to_rgba(c):
            c = np.array(c, dtype=np.float64)
            if c.max() <= 1.0:
                c = (c * 255.0).clip(0, 255)
            else:
                c = c.clip(0, 255)
            return (int(c[0]), int(c[1]), int(c[2]), 255)

        albedos = [_to_rgba(colors[i % len(colors)]) for i in range(len(motions))]

        def _compute_person_frames(motion):
            items = []
            for frame_id in range(len(motion.vertices)):
                verts = motion.vertices[frame_id]
                normals = _vertex_normals(verts, faces_i)
                items.append((frame_id, verts, normals))
            return items

        if max_workers <= 1 or len(motions) == 1:
            results = [_compute_person_frames(m) for m in motions]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_compute_person_frames, motions))

        # Single-threaded log loop owns the timeline state.
        for idx, (motion, items) in enumerate(zip(motions, results)):
            entity = f"{_MESHES_TAG}/{motion.body_id:02d}"
            albedo = albedos[idx]
            for frame_id, verts, normals in items:
                self.go_to_frame(frame_id)
                rr.log(
                    entity_path=entity,
                    entity=rr.Mesh3D(
                        vertex_positions=verts,
                        vertex_normals=normals,
                        triangle_indices=faces_u32,
                        albedo_factor=albedo,
                    ),
                )

    # ---- 2D landmarks ----------------------------------------------------

    def log_landmark_projections(
        self,
        cameras: Iterable[Camera],
        landmarks_by_cam,
        colors: Sequence[Sequence[float]],
    ) -> None:
        """Project per-camera 2D landmark predictions into the rerun viewer.

        ``landmarks_by_cam`` is the dict returned by
        :func:`visualization.motion.load_landmarks`. Cameras without a
        matching entry are silently skipped.

        Coordinates are rescaled to the *display* (downscaled) image plane
        per ``self.display_scale``, matching the rig logged in
        :meth:`log_cameras`.
        """
        rr = self._rr
        for cam in cameras:
            data = landmarks_by_cam.get(cam.name)
            if data is None:
                continue
            self._log_one_camera_landmarks(cam, data, colors)

    def _log_one_camera_landmarks(
        self, cam: Camera, data: LandmarkData, colors: Sequence[Sequence[float]]
    ) -> None:
        rr = self._rr
        scale = self._effective_scale(cam)
        cx_orig = cam.width / 2.0
        cy_orig = cam.height / 2.0
        cx_disp = (cam.width * scale) / 2.0
        cy_disp = (cam.height * scale) / 2.0

        unc = (
            uncertainty_from_log_variance(data.log_variance)
            if data.log_variance is not None
            else None
        )

        n_frames, n_persons = data.landmarks.shape[:2]
        for frame_id in range(n_frames):
            for person_id in range(n_persons):
                points = data.landmarks[frame_id, person_id].astype(np.float64).copy()
                # Center, scale, recenter -- the upstream pattern. Equivalent to
                # scaling about the image centre rather than (0,0).
                points[:, 0] = (points[:, 0] - cx_orig) * scale + cx_disp
                points[:, 1] = (points[:, 1] - cy_orig) * scale + cy_disp

                visibility = data.visibility[frame_id, person_id]
                base = np.array(
                    colors[person_id % len(colors)], dtype=np.float64
                )
                if base.max() <= 1.0:
                    base = base * 255.0
                color_u8 = np.zeros((points.shape[0], 3), dtype=np.uint8)
                visible_mask = visibility > 0.5
                color_u8[visible_mask] = base.clip(0, 255).astype(np.uint8)
                if unc is not None:
                    # Slight tint by uncertainty channel (kept for parity).
                    u = unc[frame_id, person_id]
                    color_u8[:, 0] = (color_u8[:, 0] * (1.0 - u)).astype(np.uint8)

                self.go_to_frame(frame_id)
                rr.log(
                    entity_path=(
                        f"{_CAM_TAG}/{cam.name}/image/"
                        f"{person_id:02d}_keypoints"
                    ),
                    entity=rr.Points2D(positions=points, colors=color_u8),
                )

    # ---- per-camera image streams ---------------------------------------

    def log_camera_image_streams(
        self,
        cameras: Iterable[Camera],
        *,
        jpeg_quality: int = 75,
        num_workers: Optional[int] = None,
    ) -> None:
        """Log a JPEG image stream onto each camera's ``image`` entity.

        Decodes each camera's source video (or image directory) and logs
        a downscaled JPEG per frame to ``world/cameras/<cam>/image``. The
        target pixel grid matches the Pinhole resolution declared by
        :meth:`_log_camera` via :meth:`_effective_scale`, so 2D landmarks
        overlay correctly.

        Performance:

        * **Sequential video decode**: we hold one ``cv2.VideoCapture``
          open per camera and read frames in order. Bypasses
          :class:`VideoFrameReader`'s per-frame open/seek/close cycle
          (which is ~50-100x slower on 4K H.264/H.265).
        * **Threads across cameras**: each camera runs in its own worker
          since the videos are independent files and cv2 decode releases
          the GIL. Workers return JPEG buffers; the main thread serializes
          ``rr.log`` calls so timeline ordering stays clean regardless of
          rerun's thread semantics.

        Cameras without a frame source (e.g. distribution calibration-only
        NPZs) are skipped with a warning.
        """
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError(
                f"jpeg_quality must be in [1, 100], got {jpeg_quality}"
            )

        import cv2  # raises ImportError early if missing
        import os as _os
        import sys as _sys
        _repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from capture.frame_source import ImageFileSource  # noqa: E402

        cam_list = [c for c in cameras]
        usable = [c for c in cam_list if c.video_path or c.image_paths]
        for c in cam_list:
            if c not in usable:
                log.warning(
                    "cam %s: no video_path or image_paths; skipping image stream",
                    c.name,
                )
        if not usable:
            return

        if num_workers is None:
            workers = min(len(usable), 4)
        else:
            workers = max(1, int(num_workers))

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        def _encode_one(cam: Camera):
            scale = self._effective_scale(cam)
            W = max(1, int(round(cam.width * scale)))
            H = max(1, int(round(cam.height * scale)))
            out = []
            if cam.video_path:
                # Sequential decode: open once, seek to start, then read
                # forward. ~50-100x faster than VideoFrameReader for 4K
                # H.264/H.265 where each random-access seek decodes from
                # the nearest keyframe.
                cap = cv2.VideoCapture(cam.video_path)
                if not cap.isOpened():
                    log.warning("cam %s: cannot open %s", cam.name, cam.video_path)
                    return cam.name, out
                try:
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    start = cam.frame_start if cam.frame_start is not None else 0
                    end = cam.frame_end if cam.frame_end is not None else total
                    start = max(0, min(int(start), total))
                    end = max(start, min(int(end), total))
                    if start > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    for local_idx in range(end - start):
                        ok, bgr = cap.read()
                        if not ok:
                            break
                        small = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
                        ok, buf = cv2.imencode(".jpg", small, encode_params)
                        if ok:
                            out.append((local_idx, bytes(buf)))
                finally:
                    cap.release()
            else:
                # ma_cap pre-slices img_abs_path to the canonical range.
                source = ImageFileSource(list(cam.image_paths))
                for frame_id in range(len(source)):
                    rgb = source.read_rgb(frame_id)
                    if rgb is None:
                        continue
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    small = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", small, encode_params)
                    if ok:
                        out.append((frame_id, bytes(buf)))
            return cam.name, out

        rr = self._rr
        from concurrent.futures import ThreadPoolExecutor, as_completed

        log.info(
            "image streams: %d cameras, %d worker thread(s)",
            len(usable), workers,
        )
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_encode_one, cam) for cam in usable]
            # as_completed lets main start logging the first finished cam
            # while others are still decoding. Memory cap is one camera's
            # worth of JPEG buffers per outstanding future.
            for fut in as_completed(futures):
                cam_name, items = fut.result()
                entity = f"{_CAM_TAG}/{cam_name}/image"
                for frame_id, jpeg in items:
                    self.go_to_frame(frame_id)
                    rr.log(
                        entity_path=entity,
                        entity=rr.EncodedImage(
                            contents=jpeg, media_type="image/jpeg",
                        ),
                    )

    def log_camera_video_streams(
        self,
        cameras: Iterable[Camera],
        crf: int = 20,
        num_workers: Optional[int] = None,
    ) -> None:
        """Opt-in alternative to :meth:`log_camera_image_streams`: log each
        camera's source video as a re-encoded H.264 ``rr.AssetVideo`` (with a
        per-frame ``rr.VideoFrameReference``) instead of a per-frame JPEG stream.

        ~10-15x smaller ``.rrd`` and viewer-compatible for any source codec (we
        always re-encode to H.264, which also fixes HEVC/AV1-in-viewer). Video
        frames land on the same ``frame_id / fps`` timeline the overlays use, so
        2D landmarks stay aligned. The re-encode is skipped when the source is
        already H.264 at the requested framing, and cached otherwise. Cameras
        with only image directories (no ``video_path``) are skipped with a
        warning.
        """
        rr = self._rr
        cam_list = [c for c in cameras]
        vids = [c for c in cam_list if getattr(c, "video_path", None)]
        for c in cam_list:
            if not getattr(c, "video_path", None):
                log.warning(
                    "cam %s: --rerun-video needs a source video; skipping its "
                    "backdrop", c.name,
                )
        if not vids:
            return
        workers = (min(len(vids), 4) if num_workers is None
                   else max(1, int(num_workers)))

        def _prep(cam: Camera):
            scale = self._effective_scale(cam)
            # even dims: H.264 + yuv420p require width/height divisible by 2.
            w = max(2, (int(round(cam.width * scale)) // 2) * 2)
            h = max(2, (int(round(cam.height * scale)) // 2) * 2)
            fs = cam.frame_start if cam.frame_start is not None else 0
            return cam, _ensure_h264(cam.video_path, w, h, fs, cam.frame_end, crf)

        log.info("camera video streams: %d cameras, %d worker(s), crf %d",
                 len(vids), workers, crf)
        prepped = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in [ex.submit(_prep, c) for c in vids]:
                try:
                    prepped.append(fut.result())
                except Exception as e:  # noqa: BLE001 - re-encode is best-effort
                    log.warning("video re-encode failed (skipping backdrop): %s", e)

        # Rerun logging stays on one thread (single global recording).
        for cam, mp4 in prepped:
            entity = f"{_CAM_TAG}/{cam.name}/image"
            video = rr.AssetVideo(path=mp4)
            rr.log(entity, video, static=True)
            ts_ns = np.asarray(video.read_frame_timestamps_nanos())
            if ts_ns.size == 0:
                log.warning("cam %s: re-encoded video has no frames", cam.name)
                continue
            secs = np.arange(ts_ns.size, dtype=float) / float(self.fps)
            try:
                rr.send_columns(
                    entity,
                    indexes=[rr.TimeColumn("time", timestamp=secs)],
                    columns=rr.VideoFrameReference.columns_nanos(ts_ns),
                )
            except Exception as e:  # noqa: BLE001
                log.warning("cam %s: VideoFrameReference columnar log failed (%s); "
                            "video logged without per-frame refs", cam.name, e)
