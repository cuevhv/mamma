"""
Debug visualizations for the multi-view re-ID step.

Two kinds of output are written to  <save_dir>/reid_debug/ :

  reid_stats.png
      Bar chart of corrections per camera and a correction-detail table.

  reid_cam<N>_<tag>.mp4   (one video per selected camera)
      Each frame is split in two halves separated by a yellow divider:
        Top    : BEFORE re-ID  – keypoints coloured by body_id using the
                 raw (possibly swapped) assignment.
        Bottom : AFTER re-ID   – same frame with the corrected assignment.
      Cameras where a swap was detected are tagged "CORRECTED"; a couple of
      unchanged cameras are included as a sanity-check reference.
"""

import os
import colorsys

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── per-person BGR colours (up to 8 people) ─────────────────────────────────
_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_IMG_SCALE = 4   # downsample factor when loading images for display


# ── image helpers ─────────────────────────────────────────────────────────────

def _build_subject_colors(body_ids):
    """
    Create one distinct BGR color per subject via evenly-spaced HSV hues.
    """
    n = max(len(body_ids), 1)
    out = {}
    for idx, body_id in enumerate(body_ids):
        r, g, b = colorsys.hsv_to_rgb(float(idx) / float(n), 0.9, 1.0)
        out[body_id] = (int(b * 255), int(g * 255), int(r * 255))
    return out


def _load_frame(cam_metadata_fn, imgs_pth, frame_t, scale=_IMG_SCALE, meta=None):
    """
    Load one frame from disk; return None on any failure.

    Pass a pre-loaded `meta` dict to avoid re-reading the NPZ on every call
    (important when iterating over many frames for a video).
    """
    try:
        if meta is None:
            meta = np.load(cam_metadata_fn, allow_pickle=True)
        if "img_abs_path" in meta:
            raw = str(meta["img_abs_path"][frame_t])
        elif "img_rel_path" in meta:
            raw = str(meta["img_rel_path"][frame_t])
        else:
            return None

        path = os.path.normpath(os.path.join(imgs_pth, raw)).replace("\\", "/")
        img  = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        return cv2.resize(img, (w // scale, h // scale))
    except Exception:
        return None


def _draw_keypoints(canvas, kpts_TL, vis_TL, frame_t, color, scale=_IMG_SCALE, radius=2):
    """
    Draw visible joints for one person at frame_t onto canvas (in-place).

    kpts_TL : [T, L, >=2] tensor or ndarray  (pixel coords in original resolution)
    vis_TL  : [T, L]      tensor or ndarray  or None
    """
    def _np(x):
        return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    uv = _np(kpts_TL[frame_t])   # [L, >=2]
    if vis_TL is not None:
        vis = _np(vis_TL[frame_t])  # [L]
    else:
        vis = np.ones(uv.shape[0], dtype=np.float32)

    for j in range(len(vis)):
        if vis[j] > 0.5 and uv[j, 0] > 1 and uv[j, 1] > 1:
            x = int(uv[j, 0] / scale)
            y = int(uv[j, 1] / scale)
            cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)


def _label(canvas, text, pos=(8, 22), font_scale=0.55, color=(255, 255, 255)):
    """White text with black outline for readability on any background."""
    cv2.putText(canvas, text, pos, _FONT, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, pos, _FONT, font_scale, color,     1, cv2.LINE_AA)


def _fit_frame(img, h, w):
    """Crop or pad img so it is exactly (h, w, 3)."""
    img = img[:h, :w]
    ph = h - img.shape[0]
    pw = w - img.shape[1]
    if ph > 0 or pw > 0:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), constant_values=40)
    return img


# ── main visualisation entry-points ──────────────────────────────────────────

def _reencode_to_h264(path: str) -> None:
    """Re-encode an mp4v file in place to browser-friendly H.264/yuv420p.

    OpenCV's pip wheel can only encode mp4v (MPEG-4 Part 2), which browsers
    won't play inline, so we transcode the finished file with ffmpeg. Best-effort:
    no-ops if ffmpeg is unavailable or the encode fails.
    """
    import shutil
    import subprocess
    path = str(path)
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not os.path.exists(path):
        return
    tmp = path + ".tmp.mp4"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-i", path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", tmp],
            check=True,
        )
        os.replace(tmp, path)
    except (subprocess.CalledProcessError, OSError):
        if os.path.exists(tmp):
            os.remove(tmp)


def visualize_reid(
    pts2d_before,       # {body_id: [cam_tensor[T,L,>=2], ...]}
    pts2d_vis_before,   # {body_id: [cam_tensor[T,L], ...]}  or  {bid: None}
    pts2d_after,        # same structure, after re-assignment
    pts2d_vis_after,
    corrections,        # list of {cam_id, group_idx, from_body_id, to_body_id}
    cam_metadata_fns,   # [cam0.npz, cam1.npz, ...]
    imgs_pth,
    save_dir,
    fps=25,
):
    """
    For each selected camera write one MP4 video where each video frame shows:
      - top half   : BEFORE re-ID keypoints (colour = body_id)
      - yellow bar : divider
      - bottom half: AFTER re-ID keypoints
    Corrected cameras are always included; up to 2 unchanged cameras are
    appended as a sanity-check reference.
    """
    out_dir = os.path.join(save_dir, "reid_debug")
    os.makedirs(out_dir, exist_ok=True)

    body_ids = sorted(pts2d_before.keys())
    subject_colors = _build_subject_colors(body_ids)
    n_cams   = len(cam_metadata_fns)
    T        = pts2d_before[body_ids[0]][0].shape[0]

    corrected_cams   = sorted({c["cam_id"] for c in corrections})
    uncorrected_cams = [v for v in range(n_cams) if v not in corrected_cams]
    cams_to_show     = corrected_cams + uncorrected_cams[:1]

    _DIVIDER_H = 4

    for cam_id in cams_to_show:
        if cam_id >= n_cams:
            continue
        is_corrected = cam_id in corrected_cams
        tag = "CORRECTED" if is_corrected else "no-change"

        cam_meta_fn = cam_metadata_fns[cam_id]
        cam_name = os.path.splitext(os.path.basename(cam_meta_fn))[0]

        # Load metadata once for the entire camera (avoids re-reading NPZ per frame)
        try:
            meta = np.load(cam_meta_fn, allow_pickle=True)
            img_w = int(meta["cam_img_w"]) // _IMG_SCALE
            img_h = int(meta["cam_img_h"]) // _IMG_SCALE
        except Exception:
            meta  = None
            img_w, img_h = 512, 512

        frame_w = img_w
        frame_h = img_h * 2 + _DIVIDER_H

        out_fn = os.path.join(out_dir, f"reid_{cam_name}_{tag}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_fn, fourcc, fps, (frame_w, frame_h))

        for t in range(T):
            bg = _load_frame(cam_meta_fn, imgs_pth, t, meta=meta)
            if bg is None:
                bg = np.full((img_h, img_w, 3), 40, dtype=np.uint8)

            before_img = _fit_frame(bg.copy(), img_h, frame_w)
            after_img  = _fit_frame(bg.copy(), img_h, frame_w)

            for bid in body_ids:
                color = subject_colors[bid]
                vis_b = pts2d_vis_before[bid]
                vis_a = pts2d_vis_after[bid]
                _draw_keypoints(before_img,
                                pts2d_before[bid][cam_id],
                                vis_b[cam_id] if vis_b is not None else None,
                                t, color)
                _draw_keypoints(after_img,
                                pts2d_after[bid][cam_id],
                                vis_a[cam_id] if vis_a is not None else None,
                                t, color)

            _label(before_img, f"BEFORE  {cam_name} f{t}")
            _label(after_img,  f"AFTER({tag})  {cam_name} f{t}")

            sep   = np.full((_DIVIDER_H, frame_w, 3), (0, 200, 200), dtype=np.uint8)
            frame = np.vstack([before_img, sep, after_img])
            writer.write(frame)

        writer.release()
        _reencode_to_h264(out_fn)
        print(f"[reid_viz] saved {out_fn}  ({T} frames @ {fps} fps)")


def plot_reid_stats(corrections, n_cams, n_bodies, save_dir):
    """
    Save reid_stats.png with:
      Left  : bar chart – number of corrections per camera.
      Right : table listing every individual swap (cam, group, from→to body_id).
    """
    out_dir = os.path.join(save_dir, "reid_debug")
    os.makedirs(out_dir, exist_ok=True)

    # per-camera correction counts
    corr_per_cam = np.zeros(n_cams, dtype=int)
    for c in corrections:
        if c["cam_id"] < n_cams:
            corr_per_cam[c["cam_id"]] += 1

    fig, (ax_bar, ax_tbl) = plt.subplots(1, 2, figsize=(max(14, n_cams // 2), 6))

    # ── bar chart ────────────────────────────────────────────────────────────
    cam_labels = [f"cam{i}" for i in range(n_cams)]
    bar_colors = ["#d94040" if v > 0 else "#4070c8" for v in corr_per_cam]
    ax_bar.bar(range(n_cams), corr_per_cam, color=bar_colors)
    ax_bar.set_xticks(range(n_cams))
    ax_bar.set_xticklabels(cam_labels, rotation=90, fontsize=7)
    ax_bar.set_xlabel("Camera")
    ax_bar.set_ylabel("Corrections (groups swapped)")
    ax_bar.set_title(
        f"Re-ID corrections per camera  "
        f"({sum(corr_per_cam > 0)}/{n_cams} cams affected, "
        f"{len(corrections)} total swaps)\n"
        f"Red = has corrections, blue = no change"
    )
    ax_bar.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── correction table ─────────────────────────────────────────────────────
    ax_tbl.axis("off")
    if corrections:
        rows = sorted(corrections, key=lambda c: (c["cam_id"], c["group_idx"]))
        cell_text  = [[f"cam{c['cam_id']}", str(c["group_idx"]),
                       str(c["from_body_id"]), f"→ {c['to_body_id']}"]
                      for c in rows]
        col_labels = ["Camera", "Group", "From body_id", "To body_id"]
        tbl = ax_tbl.table(cellText=cell_text, colLabels=col_labels,
                           cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.5)
        ax_tbl.set_title(f"All corrections  (total: {len(corrections)})", pad=10)
    else:
        ax_tbl.text(0.5, 0.5,
                    "No corrections needed\n(labels already consistent across cameras)",
                    ha="center", va="center", transform=ax_tbl.transAxes,
                    fontsize=12, color="#444444")
        ax_tbl.set_title("Correction details")

    plt.tight_layout()
    out_fn = os.path.join(out_dir, "reid_stats.png")
    plt.savefig(out_fn, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[reid_viz] saved {out_fn}")

