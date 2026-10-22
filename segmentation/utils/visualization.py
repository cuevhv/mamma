"""Visualization utilities for ma_masks pipeline outputs."""

import os
from pathlib import Path

import cv2
import numpy as np

OVERLAY_DIR = "masked_outputs"


def stale_overlay_cameras(seq_out_path):
    """Cameras whose ``rig_vote_report.json`` is newer than their overlays.

    The tracker draws ``masked_outputs`` while it writes the masks; the rig vote
    renames mask entries afterwards. A report newer than every overlay file
    marks overlays that still show the pre-vote identities, also when an
    earlier finalize stopped between the vote and the re-render.
    """
    stale = []
    for report in sorted(Path(seq_out_path).glob("*/rig_vote_report.json")):
        overlays = [p.stat().st_mtime_ns for p in (report.parent / OVERLAY_DIR).glob("*")
                    if p.suffix.lower() in (".jpg", ".mp4")]
        if overlays and report.stat().st_mtime_ns > max(overlays):
            stale.append(report.parent.name)
    return stale


def render_overlays(frames, mask_dir, out_dir, workers=4):
    """Write ``frame_FFFF.jpg`` overlays of the masks in ``mask_dir``.

    Same convention as the tracker's overlays: every mask adds half of the
    tab10 colour of its id - 1 to the frame, saved at quarter resolution.
    Frames are decoded in order and drawn by ``workers`` threads, with a few
    frames in flight. Returns the number of frames written.
    """
    from concurrent.futures import ThreadPoolExecutor
    from core.mask_archive import MaskArchiveReader
    from utils.drawing import show_mask_cv2

    os.makedirs(out_dir, exist_ok=True)
    with MaskArchiveReader(mask_dir) as reader, ThreadPoolExecutor(workers) as pool:
        by_frame = {}
        for name in reader.names():
            f, pid = (int(x) for x in name[len("mask_"):-len(".png")].split("_"))
            by_frame.setdefault(f, []).append((pid, name))
        half_colors = {pid: show_mask_cv2(np.ones((2, 2), bool), obj_id=pid - 1)[0, 0, :3]
                       .astype(np.float32) * 0.5
                       for entries in by_frame.values() for pid, _ in entries}

        def draw(f, rgb):
            img = rgb.astype(np.float32)
            for pid, name in by_frame[f]:
                img[reader.read(name) > 127] += half_colors[pid]
            h, w = img.shape[:2]
            # Channels are resized independently: resizing before the RGB to BGR
            # flip writes the same pixels as the tracker's flip-then-resize.
            small = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (max(1, w // 4), max(1, h // 4)))
            cv2.imwrite(os.path.join(out_dir, f"frame_{f:04d}.jpg"), small[:, :, ::-1])

        todo = [f for f in sorted(by_frame) if f < len(frames)]
        pending = []
        for f in todo:
            pending.append(pool.submit(draw, f, frames.read_rgb(f)))
            if len(pending) > 2 * workers:
                pending.pop(0).result()
        for p in pending:
            p.result()
    return len(todo)


def generate_cross_camera_summary(seq_out_path, log_fn=None):
    """Generate per-person cross-camera consistency images.

    For each person, stacks their person_XX_crop_summary.png from all cameras
    into one tall image. Makes it easy to visually verify that the same person
    has consistent ID across all cameras.

    Saves to: seq_out_path/cross_camera_summary/person_XX.png

    Args:
        seq_out_path: Sequence output directory containing camera subdirs.
        log_fn: Optional logging function (level, message). Falls back to print.
    """
    def _log(level, msg):
        if log_fn:
            log_fn(level, msg)
        else:
            print(f"[{level}] {msg}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        seq_dir = Path(seq_out_path)
        cam_dirs = sorted([
            d for d in seq_dir.iterdir()
            if d.is_dir() and (d / "masks").is_dir()
        ])

        if len(cam_dirs) < 2:
            return

        # Find all person IDs across cameras
        all_person_ids = set()
        for cam_dir in cam_dirs:
            for f in cam_dir.glob("person_*_crop_summary.png"):
                name = f.stem
                parts = name.split("_")
                if len(parts) >= 2:
                    try:
                        all_person_ids.add(int(parts[1]))
                    except ValueError:
                        pass

        if not all_person_ids:
            return

        out_dir = seq_dir / "cross_camera_summary"
        out_dir.mkdir(exist_ok=True)

        for pid in sorted(all_person_ids):
            cam_images = []
            cam_labels = []
            for cam_dir in cam_dirs:
                summary_path = cam_dir / f"person_{pid:02d}_crop_summary.png"
                if summary_path.exists():
                    cam_images.append(PILImage.open(summary_path))
                    cam_labels.append(cam_dir.name)

            if not cam_images:
                continue

            n_cams = len(cam_images)
            fig, axes = plt.subplots(n_cams, 1, figsize=(16, 4 * n_cams))
            if n_cams == 1:
                axes = [axes]

            for ax, img, label in zip(axes, cam_images, cam_labels):
                ax.imshow(img)
                ax.set_ylabel(label, fontsize=14, fontweight='bold', rotation=0,
                              labelpad=80, va='center')
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(f"Person {pid} — Cross-Camera Consistency",
                         fontsize=16, fontweight='bold', y=1.01)
            plt.tight_layout()
            plt.savefig(out_dir / f"person_{pid:02d}.png", dpi=100,
                        bbox_inches='tight', facecolor='white')
            plt.close()

        _log("INFO", f"Saved cross-camera summaries to '{out_dir}' ({len(all_person_ids)} people).")
    except Exception as exc:
        _log("WARN", f"Failed to generate cross-camera summary: {exc}")
