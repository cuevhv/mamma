from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import cv2
import imageio_ffmpeg
import numpy as np

ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}
MASKED_OUTPUTS_VIDEO = "masked_outputs.mp4"

# See core/pipeline.py FFMPEG_EXE for rationale (statically-linked ffmpeg that
# sidesteps system-ffmpeg and Apptainer --nv library-injection issues).
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create MP4 videos as collages of multiple camera views for each sequence in a dataset."
        )
    )
    parser.add_argument("--dataset_dir", required=True, help="Root folder with sequence data")
    parser.add_argument("--fps", type=float, default=30, help="Output video FPS")
    parser.add_argument(
        "--tile_size",
        type=int,
        default=256,
        help="Size of each camera tile in the collage (pixels)",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=0,
        help="Number of columns in collage (0 = auto-calculate)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum number of frames to include in the video (0 = no limit)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of parallel processes (0 = one per CPU up to number of sequences)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=True, help="Regenerate videos even if they exist"
    )
    return parser.parse_args()


def find_sequences(dataset_dir: Path) -> List[Path]:
    return [p for p in sorted(dataset_dir.iterdir()) if p.is_dir()]


def _make_frame_collage(
    tiles: List[np.ndarray], columns: int, tile_size: int
) -> np.ndarray | None:
    """Create a collage from a list of image tiles."""
    if not tiles:
        return None
    cols = columns if columns > 0 else max(1, int(math.ceil(math.sqrt(len(tiles)))))
    rows = int(math.ceil(len(tiles) / cols))
    total_cells = rows * cols
    if total_cells > len(tiles):
        filler = np.full((tile_size, tile_size, 3), 0, dtype=np.uint8)
        tiles = tiles + [filler] * (total_cells - len(tiles))
    collage = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        collage[
            r * tile_size : (r + 1) * tile_size, c * tile_size : (c + 1) * tile_size
        ] = tile
    return collage


def _label_tile(tile: np.ndarray, text: str) -> np.ndarray:
    """Stamp a camera name in the tile's top-left corner (in place).

    White text over a black outline so it reads on both the letterbox padding
    and bright image content; size scales with the tile. Callers must not pass
    a shared buffer (label the copy, not the reusable black filler tile).
    """
    scale = max(0.35, tile.shape[0] / 640.0)
    thickness = max(1, int(round(1.5 * scale)))
    org = (int(16 * scale), int(40 * scale))
    for color, extra in (((0, 0, 0), 1), ((255, 255, 255), 0)):
        cv2.putText(tile, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness + extra, cv2.LINE_AA)
    return tile


def _overlay_frame_label(image_bgr: np.ndarray, frame_id: int) -> np.ndarray:
    """Add a header with frame ID to an image."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    header_h = max(48, image_bgr.shape[0] // 18)
    header = np.full((header_h, image_bgr.shape[1], 3), 10, dtype=np.uint8)
    label = f"Frame {frame_id:06d}"
    cv2.putText(
        header,
        label,
        (12, header_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([header, image_bgr], axis=0)


def find_cameras(seq_dir: Path) -> List[Path]:
    """Find all camera directories that contain masked_outputs subdirectory."""
    cameras = []
    for p in sorted(seq_dir.iterdir()):
        if p.is_dir() and (p / "masked_outputs").is_dir():
            cameras.append(p)
    return cameras


def list_frames(masked_outputs_dir: Path) -> List[Path]:
    """List all image frames in a masked_outputs directory."""
    if not masked_outputs_dir.exists() or not masked_outputs_dir.is_dir():
        return []
    frames = [
        p
        for p in sorted(masked_outputs_dir.iterdir())
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    ]
    return frames


def _probe_video_frame_count(video_path: Path) -> int:
    """Return readable frame count for a video (best-effort)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        cap.release()
        return frame_count

    # Some codecs/containers do not expose CAP_PROP_FRAME_COUNT reliably.
    frame_count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        frame_count += 1
    cap.release()
    return frame_count


def _read_first_source_frame(source: Dict[str, Any]) -> np.ndarray | None:
    source_type = source["type"]
    if source_type == "images":
        frame_paths: List[Path] = source["frames"]
        if not frame_paths:
            return None
        return cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)

    if source_type == "video":
        video_path: Path = source["video_path"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        return frame

    return None


def _read_source_frame(source: Dict[str, Any], frame_idx: int) -> np.ndarray | None:
    source_type = source["type"]

    if source_type == "images":
        frame_paths: List[Path] = source["frames"]
        if frame_idx >= len(frame_paths):
            return None
        return cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)

    if source_type == "video":
        if frame_idx >= int(source["length"]):
            return None
        cap: cv2.VideoCapture | None = source.get("cap")
        if cap is None:
            return None
        ok, frame = cap.read()
        if not ok:
            return None
        return frame

    return None


class _FfmpegH264Writer:
    """Drop-in replacement for cv2.VideoWriter that produces browser-friendly H.264 MP4.

    Pipes raw BGR frames to ffmpeg's stdin; ffmpeg encodes with libx264 + yuv420p
    and writes an MP4 with +faststart (moov atom at the head, so the video starts
    playing before the full file is fetched).
    """

    def __init__(self, width: int, height: int, output_path: Path, fps: float):
        # libx264 with yuv420p requires even width/height. pad with 1px if odd.
        padded_w = width + (width % 2)
        padded_h = height + (height % 2)

        cmd = [
            FFMPEG_EXE, "-y", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{fps}",
            "-i", "-",
            "-vf", f"pad={padded_w}:{padded_h}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            str(output_path),
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except FileNotFoundError as e:
            raise RuntimeError(f"ffmpeg binary not found at '{FFMPEG_EXE}'.") from e
        self._output_path = output_path
        self._opened = True

    def write(self, frame: np.ndarray) -> None:
        if not self._opened or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            # ffmpeg died mid-stream; .release() will surface the exit code.
            self._opened = False

    def release(self) -> None:
        if not self._opened:
            return
        self._opened = False
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        ret = self._proc.wait()
        if ret != 0:
            logging.warning("ffmpeg exited %d while writing %s", ret, self._output_path)


def prepare_writer(
    width: int, height: int, output_path: Path, fps: float
) -> _FfmpegH264Writer:
    """Create a browser-friendly H.264 video writer for the given dimensions."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions: {width}x{height}")
    if fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}")

    return _FfmpegH264Writer(width, height, output_path, fps)


def _resize_frame(frame: np.ndarray, tile_size: int) -> np.ndarray:
    """Resize frame to tile_size x tile_size with padding to preserve aspect ratio."""
    h, w = frame.shape[:2]

    # If already the correct size, return as is
    if h == tile_size and w == tile_size:
        return frame

    # Calculate scale to fit the frame within tile_size while preserving aspect ratio
    scale = min(tile_size / h, tile_size / w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Resize the frame
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black tile and place the resized frame in the center
    tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    top = (tile_size - new_h) // 2
    left = (tile_size - new_w) // 2
    tile[top : top + new_h, left : left + new_w] = resized

    return tile


def process_collage_body(
    seq_dir: Path,
    seq_name: str,
    cameras: List[Path],
    fps: float,
    tile_size: int,
    columns: int,
    max_frames: int,
    overwrite: bool,
) -> None:
    """Create a collage video for all cameras in a sequence."""
    sources: List[Dict[str, Any]] = []
    for cam in cameras:
        masked_dir = cam / "masked_outputs"
        frame_paths = list_frames(masked_dir)
        if frame_paths:
            sources.append(
                {
                    "camera": cam.name,
                    "type": "images",
                    "frames": frame_paths,
                    "length": len(frame_paths),
                }
            )
            continue

        video_path = masked_dir / MASKED_OUTPUTS_VIDEO
        if video_path.exists() and video_path.is_file():
            frame_count = _probe_video_frame_count(video_path)
            if frame_count > 0:
                sources.append(
                    {
                        "camera": cam.name,
                        "type": "video",
                        "video_path": video_path,
                        "length": frame_count,
                    }
                )
                continue
            logging.warning(
                "Camera %s has '%s' but no readable frames in %s",
                cam.name,
                MASKED_OUTPUTS_VIDEO,
                seq_name,
            )
        else:
            logging.warning(
                "Camera %s has neither frame images nor '%s' in %s",
                cam.name,
                MASKED_OUTPUTS_VIDEO,
                seq_name,
            )

        sources.append({"camera": cam.name, "type": "empty", "length": 0})

    num_frames = max(int(source["length"]) for source in sources) if sources else 0
    if num_frames == 0:
        logging.warning("No readable frames found for sequence %s", seq_name)
        return

    # Apply max_frames limit if specified
    if max_frames > 0:
        num_frames = min(num_frames, max_frames)

    output_dir = seq_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "collage.mp4"

    if output_path.exists() and not overwrite:
        logging.info("Skipping existing video %s", output_path)
        return

    # Read first frame from each camera/source to determine collage dimensions.
    black_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    first_tiles = []
    for source in sources:
        first_frame = _read_first_source_frame(source)
        if first_frame is None:
            first_tiles.append(black_tile)
        else:
            first_tiles.append(_resize_frame(first_frame, tile_size))

    if not first_tiles:
        logging.warning("Could not build initial collage tiles for %s", seq_name)
        return

    # Create a sample collage to determine output dimensions
    sample_collage = _make_frame_collage(first_tiles, columns, tile_size)
    if sample_collage is None:
        logging.warning("Could not create sample collage for %s", seq_name)
        return

    collage_h, collage_w = sample_collage.shape[:2]
    writer = prepare_writer(collage_w, collage_h, output_path, fps)

    try:
        for source in sources:
            if source["type"] != "video":
                continue
            cap = cv2.VideoCapture(str(source["video_path"]))
            if cap.isOpened():
                source["cap"] = cap
            else:
                logging.warning(
                    "Failed to open camera %s video '%s' for sequence %s",
                    source["camera"],
                    source["video_path"],
                    seq_name,
                )
                source["type"] = "empty"
                source["length"] = 0

        for frame_idx in range(num_frames):
            tiles = []
            for source in sources:
                frame = _read_source_frame(source, frame_idx)
                if frame is None:
                    # Black frame if this source has fewer frames or fails to
                    # decode — copied, since black_tile is shared and the label
                    # differs per camera.
                    tile = black_tile.copy()
                else:
                    tile = _resize_frame(frame, tile_size)
                tiles.append(_label_tile(tile, source["camera"]))

            collage = _make_frame_collage(tiles, columns, tile_size)
            if collage is not None:
                # Ensure the frame is C-contiguous and has correct dtype
                collage = np.ascontiguousarray(collage, dtype=np.uint8)
                writer.write(collage)
    finally:
        for source in sources:
            cap = source.get("cap")
            if cap is not None:
                cap.release()
        writer.release()

    logging.info(
        "Wrote collage video %s (%d frames, %d cameras) for sequence %s",
        output_path,
        num_frames,
        len(sources),
        seq_name,
    )


def process_sequence(
    seq_dir: Path, fps: float, tile_size: int, columns: int, max_frames: int, overwrite: bool
) -> None:
    seq_name = seq_dir.name
    cameras = find_cameras(seq_dir)
    if not cameras:
        logging.warning("No camera folders found in %s", seq_name)
        return

    process_collage_body(seq_dir, seq_name, cameras, fps, tile_size, columns, max_frames, overwrite)


def _worker(task: tuple[Path, float, int, int, int, bool]) -> None:
    seq_dir, fps, tile_size, columns, max_frames, overwrite = task
    try:
        process_sequence(seq_dir, fps, tile_size, columns, max_frames, overwrite)
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.exception("Failed processing %s: %s", seq_dir.name, exc)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset_dir = Path(args.dataset_dir).expanduser()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    sequences = find_sequences(dataset_dir)
    if not sequences:
        logging.error("No sequences found in %s", dataset_dir)
        return

    num_workers = args.num_workers or min(len(sequences), mp.cpu_count() or 1)
    tasks = [
        (seq, args.fps, args.tile_size, args.columns, args.max_frames, args.overwrite) for seq in sequences
    ]

    if num_workers <= 1:
        for task in tasks:
            _worker(task)
    else:
        with mp.Pool(processes=num_workers) as pool:
            pool.map(_worker, tasks)


if __name__ == "__main__":
    main()

