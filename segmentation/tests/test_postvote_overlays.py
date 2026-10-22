"""Overlays follow the rig-vote relabel (process_sequence finalize).

The tracker's own overlay writer (``_build_masks_from_remap``) produces the
masks and overlays of a tiny two-camera sequence; a rig vote is simulated on
camera A by renaming its mask entries and writing its report. The redrawn
overlays of A must equal what the tracker writes for the post-vote ids, and
camera B must stay byte-identical.
"""
import json
import os
import time

import cv2
import numpy as np

from core import rig_vote
from core.frame_source import frame_source_from_cam_data
from core.pipeline import SegmentMultipleFrames, _overlay_half_color
from process_sequence import _rerender_relabeled_overlays
from utils.visualization import stale_overlay_cameras

H, W, T = 48, 64, 4
SWAP = {1: (1, 0), 2: (1, 0)}  # camera A, frames 1-2: ids 1 and 2 exchanged
EXPORTS = {"export_masked_outputs_mp4": True, "masked_outputs_fps": 30.0,
           "remove_masked_outputs_images": False}


def _segmenter(exports):
    seg = SegmentMultipleFrames.__new__(SegmentMultipleFrames)
    seg.assignment_config = {"exports": dict(exports), "masks": {}}
    seg.start_frame = seg.end_frame = None
    seg._log_info = seg._log_warn = lambda *a, **k: None
    return seg


def _cam(root, name, gray, n=T):
    """Flat frames, frame t at grey level ``gray + 10 t``."""
    d = root / "frames" / name
    d.mkdir(parents=True)
    paths = []
    for t in range(n):
        paths.append(str(d / f"{t:06d}.png"))
        cv2.imwrite(paths[-1], np.full((H, W, 3), gray + 10 * t, np.uint8))
    return {"cam_name": np.array(name), "img_abs_path": np.array(paths)}


def _segments(swapped=()):
    left = np.zeros((H, W), bool)
    left[:, :W // 2] = True
    return {t: ({0: ~left, 1: left} if t in swapped else {0: left, 1: ~left})
            for t in range(T)}


def _track(seg, cam_data, out_dir, swapped=()):
    seg._build_masks_from_remap(frame_source_from_cam_data(cam_data), _segments(swapped),
                                {0: 0, 1: 1}, str(out_dir))


def _files(d):
    return {p: open(os.path.join(d, p), "rb").read() for p in sorted(os.listdir(d))}


def _video(path):
    cap, out = cv2.VideoCapture(str(path)), []
    ok, frame = cap.read()
    while ok:
        out.append(frame)
        ok, frame = cap.read()
    cap.release()
    return out


def _vote_on_a(seq):
    """Tracker overlays predate the vote; the vote renames A's masks and reports."""
    old = time.time() - 100
    for p in seq.glob("*/masked_outputs/*"):
        os.utime(p, (old, old))
    # Rig vote on camera A: rename the mask entries and write its report.
    rig_vote.permute_mask_ids(str(seq / "A" / "masks"), SWAP)
    with open(seq / "A" / "rig_vote_report.json", "w") as fh:
        json.dump([{"start": 1, "end": 2, "new_ids": [2, 1]}], fh)


def _sequence(tmp_path):
    seq = tmp_path / "seq"
    seg = _segmenter(EXPORTS)
    cams = [_cam(tmp_path, "A", 100), _cam(tmp_path, "B", 150)]
    for cd in cams:
        _track(seg, cd, seq / str(cd["cam_name"]))
    _vote_on_a(seq)
    return seq, seg, cams


def test_relabeled_camera_overlays_follow_new_ids(tmp_path):
    seq, seg, cams = _sequence(tmp_path)
    a_out, b_out = seq / "A" / "masked_outputs", seq / "B" / "masked_outputs"
    a_before, b_before = _files(a_out), _files(b_out)
    assert stale_overlay_cameras(str(seq)) == ["A"]

    # Overlays switched off: nothing is redrawn.
    _rerender_relabeled_overlays(_segmenter(dict(EXPORTS, skip_masked_outputs=True)),
                                 str(seq), cams)
    assert _files(a_out) == a_before

    _rerender_relabeled_overlays(seg, str(seq), cams)
    a_after = _files(a_out)
    assert _files(b_out) == b_before
    assert set(a_after) == set(a_before)
    for f in (0, 3):
        assert a_after[f"frame_{f:04d}.jpg"] == a_before[f"frame_{f:04d}.jpg"]
    for f in (1, 2):
        assert a_after[f"frame_{f:04d}.jpg"] != a_before[f"frame_{f:04d}.jpg"]

    # Identical to the tracker's overlays for the post-vote ids.
    ref = tmp_path / "ref"
    _track(_segmenter(EXPORTS), cams[0], ref, swapped=(1, 2))
    ref_files = _files(ref / "masked_outputs")
    for name, data in a_after.items():
        if name.endswith(".jpg"):
            assert data == ref_files[name], name
    got, want = _video(a_out / "masked_outputs.mp4"), _video(ref / "masked_outputs" / "masked_outputs.mp4")
    assert len(got) == len(want) >= T
    assert all(np.array_equal(g, w) for g, w in zip(got, want))

    # Colours: on frame 1 the left half now carries id 2, the right half id 1.
    img = cv2.imread(str(a_out / "frame_0001.jpg"))[:, :, ::-1].astype(np.float32)
    h, w = img.shape[:2]
    left, right = img[h // 2, w // 8], img[h // 2, 7 * w // 8]
    assert np.abs(left - (110 + _overlay_half_color(1))).max() < 8
    assert np.abs(right - (110 + _overlay_half_color(0))).max() < 8

    # Idempotent: the redrawn overlays are newer than the report.
    assert stale_overlay_cameras(str(seq)) == []
    _rerender_relabeled_overlays(seg, str(seq), cams)
    assert _files(a_out) == a_after


def test_trimmed_run_uses_the_tracker_frame_range(tmp_path):
    seg = _segmenter(EXPORTS)
    seg.start_frame, seg.end_frame = 2, 2 + T
    cd = _cam(tmp_path, "A", 40, n=T + 3)
    trimmed = seg._apply_frame_range(cd, None)[0]
    seq = tmp_path / "seq"
    _track(seg, trimmed, seq / "A")
    _vote_on_a(seq)
    _rerender_relabeled_overlays(seg, str(seq), [cd])
    _track(seg, trimmed, tmp_path / "ref", swapped=(1, 2))
    got, want = _files(seq / "A" / "masked_outputs"), _files(tmp_path / "ref" / "masked_outputs")
    for name in (f"frame_{f:04d}.jpg" for f in range(T)):
        assert got[name] == want[name], name


def test_failed_redraw_keeps_tracker_overlays(tmp_path):
    seq, seg, cams = _sequence(tmp_path)
    a_out = seq / "A" / "masked_outputs"
    before = _files(a_out)
    for p in cams[0]["img_abs_path"]:
        os.remove(str(p))
    _rerender_relabeled_overlays(seg, str(seq), cams)
    assert _files(a_out) == before
    assert not (seq / "A" / "masked_outputs.rigvote_tmp").exists()


def test_no_report_is_a_noop(tmp_path):
    seq = tmp_path / "seq"
    seg = _segmenter(EXPORTS)
    cd = _cam(tmp_path, "A", 100)
    _track(seg, cd, seq / "A")
    before = _files(seq / "A" / "masked_outputs")
    assert stale_overlay_cameras(str(seq)) == []
    _rerender_relabeled_overlays(seg, str(seq), [cd])
    assert _files(seq / "A" / "masked_outputs") == before
