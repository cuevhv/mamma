from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Iterable, List

import cv2

ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create MP4 videos per camera and body for each sequence in a dataset."
        )
    )
    parser.add_argument("--dataset_dir", required=True, help="Root folder with sequence data")
    parser.add_argument("--fps", type=float, default=3, help="Output video FPS")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of parallel processes (0 = one per CPU up to number of sequences)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Regenerate videos even if they exist"
    )
    parser.add_argument(
        "--cleanup_frames",
        action="store_true",
        help="Delete source camera/body frame directories after each MP4 is written",
    )
    return parser.parse_args()


def find_sequences(dataset_dir: Path) -> List[Path]:
    return [p for p in sorted(dataset_dir.iterdir()) if p.is_dir()]


def find_cameras(seq_dir: Path) -> List[Path]:
    return [p for p in sorted(seq_dir.iterdir()) if p.is_dir()]


def find_bodies(cameras: Iterable[Path]) -> List[str]:
    body_sets = []
    for cam in cameras:
        bodies = {p.name for p in cam.iterdir() if p.is_dir()}
        if bodies:
            body_sets.append(bodies)
    if not body_sets:
        return []
    union = set.union(*body_sets)
    return sorted(union)


def list_frames(body_dir: Path) -> List[Path]:
    if not body_dir.exists() or not body_dir.is_dir():
        return []
    frames = [
        p
        for p in sorted(body_dir.iterdir())
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    ]
    return frames


def _reencode_to_h264(path: Path) -> None:
    """Re-encode an mp4v file in place to browser-friendly H.264/yuv420p.

    OpenCV's pip wheel can only encode mp4v (MPEG-4 Part 2), which browsers
    won't play inline, so we transcode the finished file with ffmpeg. Best-effort:
    no-ops if ffmpeg is unavailable or the encode fails.
    """
    import os
    import subprocess
    p = str(path)
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not os.path.exists(p):
        return
    tmp = p + ".tmp.mp4"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-i", p, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", tmp],
            check=True,
        )
        os.replace(tmp, p)
    except (subprocess.CalledProcessError, OSError):
        if os.path.exists(tmp):
            os.remove(tmp)


def prepare_writer(frames: List[Path], output_path: Path, fps: float) -> tuple[cv2.VideoWriter, tuple[int, int]]:
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Could not read image {frames[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    return writer, (width, height)


def process_body_camera(
    output_dir: Path,
    seq_name: str,
    body: str,
    camera: Path,
    fps: float,
    overwrite: bool,
    cleanup_frames: bool,
) -> None:
    body_dir = camera / body
    frames = list_frames(body_dir)
    if not frames:
        logging.warning("No frames for body %s in camera %s", body, camera.name)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{camera.name}.mp4"
    if output_path.exists() and not overwrite:
        logging.info("Skipping existing video %s", output_path)
        return

    writer, target_size = prepare_writer(frames, output_path, fps)
    target_w, target_h = target_size

    try:
        for frame_path in frames:
            img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not read image {frame_path}")
            if img.shape[1] != target_w or img.shape[0] != target_h:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()

    _reencode_to_h264(output_path)

    logging.info(
        "Wrote %s (%d frames) for sequence %s body %s camera %s",
        output_path,
        len(frames),
        seq_name,
        body,
        camera.name,
    )

    if cleanup_frames:
        try:
            shutil.rmtree(body_dir)
            logging.info("Removed frame directory %s after MP4 creation", body_dir)
        except OSError as exc:
            logging.warning("Failed to remove frame directory %s: %s", body_dir, exc)


def process_sequence(
    seq_dir: Path, fps: float, overwrite: bool, cleanup_frames: bool = False
) -> None:
    seq_name = seq_dir.name
    cameras = find_cameras(seq_dir)
    if not cameras:
        logging.warning("No camera folders found in %s", seq_name)
        return

    bodies = find_bodies(cameras)
    if not bodies:
        logging.warning("No common bodies across cameras in %s", seq_name)
        return

    videos_root = seq_dir / "videos"

    for body in bodies:
        body_out = videos_root / body
        for camera in cameras:
            process_body_camera(
                body_out, seq_name, body, camera, fps, overwrite, cleanup_frames
            )

    if cleanup_frames:
        for camera in cameras:
            try:
                if not any(camera.iterdir()):
                    camera.rmdir()
                    logging.info("Removed empty camera directory %s", camera)
            except OSError as exc:
                logging.warning("Failed to remove camera directory %s: %s", camera, exc)


def _worker(task: tuple[Path, float, bool, bool]) -> None:
    seq_dir, fps, overwrite, cleanup_frames = task
    try:
        process_sequence(seq_dir, fps, overwrite, cleanup_frames)
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
    tasks = [(seq, args.fps, args.overwrite, args.cleanup_frames) for seq in sequences]

    if num_workers <= 1:
        for task in tasks:
            _worker(task)
    else:
        with mp.Pool(processes=num_workers) as pool:
            pool.map(_worker, tasks)


if __name__ == "__main__":
    main()

