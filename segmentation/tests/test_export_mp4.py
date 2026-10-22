"""The overlay MP4 holds exactly one video frame per image, in order, at the requested fps."""
import types

import cv2
import numpy as np

from core.pipeline import SegmentMultipleFrames


def test_overlay_mp4_keeps_every_frame(tmp_path):
    n = 40
    for i in range(n):
        im = np.zeros((48, 64, 3), np.uint8)
        im[:, :, 1] = i * 6  # a distinct brightness per frame
        cv2.imwrite(str(tmp_path / f"frame_{i:04d}.jpg"), im)
    stub = types.SimpleNamespace(_log_warn=print, _log_info=print, _log=print)
    out = SegmentMultipleFrames._export_image_dir_to_mp4(
        stub, str(tmp_path), fps=30.0, video_filename="masked_outputs.mp4", remove_images=False)
    cap = cv2.VideoCapture(out)
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 30.0) < 1e-3
    green = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        green.append(float(frame[:, :, 1].mean()))
    assert len(green) == n
    assert np.allclose(green, np.arange(n) * 6, atol=4)
