"""Track-then-match cross-camera identity: the seed prescreen, the whole-tracklet
remap to init identities, and the cross-camera strategy resolver.

All synthetic: painted square people in a small frame, a stub CLIP encoder that
maps each paint colour to a fixed feature vector, and calibrated cameras that
differ by a pure x translation (horizontal epipolar lines, so the epipolar
distance is the row offset). No SAM, no CLIP weights, no GPU.
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.pipeline import DEFAULT_ASSIGNMENT_CONFIG, SegmentMultipleFrames

W, H = 200, 100  # epipolar sigma 24 px, hard cutoff 48 px at this size
D = 8
A, B, OUT, WEAK_B = 1, 2, 3, 4  # paint colours (0 is background)


def _vec(*pairs):
    v = torch.zeros(D)
    for i, x in pairs:
        v[i] = x
    return v


FEATS = {
    A: _vec((1, 1.0)),
    B: _vec((2, 1.0)),
    OUT: _vec((3, 1.0)),
    WEAK_B: _vec((2, 0.3), (4, float(np.sqrt(1 - 0.09)))),  # cosine 0.3 with B
}


def _encode(crops):
    """Stub CLIP: feature of the dominant non-background colour in the crop."""
    out = []
    for c in crops:
        vals = c[..., 0].ravel()
        out.append(FEATS[int(np.bincount(vals[vals > 0]).argmax())].clone())
    return out


def _seg(masks=None, matching=None, sam_version="sam2", output_path="/nonexistent"):
    seg = SegmentMultipleFrames.__new__(SegmentMultipleFrames)
    seg.assignment_config = copy.deepcopy(DEFAULT_ASSIGNMENT_CONFIG)
    seg.assignment_config["masks"].update(masks or {})
    seg.assignment_config["matching"].update(matching or {})
    seg.sam_version = sam_version
    seg.output_path = output_path
    seg.subject_feature_bank = {}
    seg.subject_crop_bank = {}
    seg._prescreen_refs = []
    seg._encode_clip_batch = _encode
    seg._log_info = lambda *a, **k: None
    seg._log_warn = lambda *a, **k: None
    seg._log_error = lambda *a, **k: None
    return seg


def _cam(tx, name):
    K = np.array([[1000.0, 0, W / 2], [0, 1000.0, H / 2], [0, 0, 1]])
    E = np.eye(4)
    E[0, 3] = tx
    return {"cam_name": name, "cam_int": K, "cam_ext": E}


def _init_masks(subjects, frame=0):
    """{oid: (colour, centroid_y)} -> init-camera mask_data (gallery + slim centroids)."""
    return {oid: {"features": [FEATS[colour].clone()], "frame": [frame],
                  "centroid_xy": [(W / 2, float(y))]}
            for oid, (colour, y) in subjects.items()}


class _Frames:
    cam_name = "tgt"
    image_size = (W, H)

    def __init__(self, images):
        self._images = images

    def __len__(self):
        return len(self._images)

    def read_rgb(self, idx):
        return self._images[idx]


def _scene(spec):
    """{frame: {local_id: (x, y, side, colour)}} -> (frames, video_segments)."""
    images, segments = {}, {}
    for f, people in spec.items():
        img = np.zeros((H, W, 3), np.uint8)
        segments[f] = {}
        for sid, (x, y, side, colour) in people.items():
            m = np.zeros((H, W), bool)
            m[y - side // 2:y + side // 2, x - side // 2:x + side // 2] = True
            img[m] = colour
            segments[f][sid] = m
        images[f] = img
    return _Frames(images), segments


def _remap(seg, mask_data, spec, src=None, tgt=None):
    frames, segments = _scene(spec)
    return seg._remap_tracklets_to_init(
        "tgt", frames, mask_data, sorted(segments[0]), segments, 0,
        source_cam_data=src, target_cam_data=tgt)


# --------------------------------------------------------------------------- #
# Tracklet remap to init identities
# --------------------------------------------------------------------------- #

def test_remap_assigns_confident_matches_and_drops_extras():
    seg = _seg()
    md = _init_masks({0: (A, 50), 1: (B, 50)})
    people = {10: (40, 50, 20, B), 11: (100, 50, 20, A), 12: (160, 50, 20, OUT)}
    remap = _remap(seg, md, {f: people for f in range(3)})
    assert remap == {10: 1, 11: 0}  # outsider 12 has no init identity
    assert seg._last_remap_scores == pytest.approx({10: 1.0, 11: 1.0})


def test_remap_rejects_forced_pair_below_combined_min_score():
    # Hungarian pairs the outsider with the subject nobody else matched;
    # the pair scores below combined_min_score and is dropped.
    seg = _seg(matching={"combined_min_score": 0.2})
    md = _init_masks({0: (A, 50), 1: (B, 50)})
    people = {10: (40, 50, 20, A), 12: (160, 50, 20, OUT)}
    assert _remap(seg, md, {f: people for f in range(3)}) == {10: 0}


def test_remap_epipolar_resolves_identical_appearance():
    # Both subjects look the same; only the epipolar term can separate them.
    seg = _seg()
    md = _init_masks({0: (A, 20), 1: (A, 75)})
    people = {10: (60, 75, 20, A), 11: (140, 20, 20, A)}
    remap = _remap(seg, md, {f: people for f in range(3)},
                   src=_cam(0.0, "init"), tgt=_cam(1.0, "tgt"))
    assert remap == {10: 1, 11: 0}


def test_remap_updates_bank_only_for_confident_matches():
    seg = _seg(matching={"combined_min_score": 0.2, "online_bank_update_min_score": 0.35})
    md = _init_masks({0: (A, 50), 1: (B, 50)})
    people = {10: (40, 50, 20, A), 11: (160, 50, 20, WEAK_B)}
    remap = _remap(seg, md, {f: people for f in range(3)})
    # The weak match (0.3) is accepted as an identity but not trusted for the bank.
    assert remap == {10: 0, 11: 1}
    assert seg._last_remap_scores[11] == pytest.approx(0.3, abs=1e-4)
    assert list(seg.subject_feature_bank) == [0]
    assert torch.allclose(seg.subject_feature_bank[0][0], FEATS[A])


@pytest.mark.parametrize("min_ratio, tiny_weight", [
    (0.01, 0.1),                                   # tiny (0.005) is at or below the floor
    (0.001, 0.1 + 0.9 * (0.005 - 0.001) / 0.019),  # tiny is on the soft ramp
])
def test_remap_min_mask_area_ratio_is_a_soft_weight(min_ratio, tiny_weight):
    # Tracklet 10 is large (A) on frame 0 and tiny (B) on frames 1-2; tracklet 11
    # is tiny on every frame. Small masks are down-weighted, never filtered out.
    seg = _seg(matching={"min_mask_area_ratio": min_ratio})
    md = _init_masks({0: (A, 50), 1: (B, 50)})
    spec = {
        0: {10: (50, 50, 40, A), 11: (150, 50, 10, B)},
        1: {10: (50, 50, 10, B), 11: (150, 50, 10, B)},
        2: {10: (50, 50, 10, B), 11: (150, 50, 10, B)},
    }
    remap = _remap(seg, md, spec)
    assert remap == {10: 0, 11: 1}
    expected = FEATS[A] + 2 * tiny_weight * FEATS[B]
    expected = expected / expected.norm()
    assert torch.allclose(seg.subject_feature_bank[0][0], expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# Seed prescreen
# --------------------------------------------------------------------------- #

def _det(y, colour, x=100, half=8):
    crop = np.full((2 * half, 2 * half, 3), colour, np.uint8)
    return (crop, None, np.array([x - half, y - half, x + half, y + half], np.float32), 0.9)


def _prescreen(seg, dets, md, tgt=None):
    frames = _Frames({0: np.zeros((H, W, 3), np.uint8)})
    return seg._prescreen_seed_candidates(
        "tgt", frames, dets, 0, md, _cam(0.0, "init"),
        _cam(1.0, "tgt") if tgt is None else tgt)


def test_prescreen_keeps_top_candidates_for_single_subject(tmp_path):
    seg = _seg(masks={"seed_candidates_per_subject": 3}, output_path=str(tmp_path))
    md = _init_masks({0: (A, 40)})
    # Six look-alikes; the three nearest the subject's epipolar line survive.
    dets = [_det(y, A) for y in (95, 40, 100, 45, 90, 50)]
    out = _prescreen(seg, dets, md)
    assert len(out) == 3
    assert all(o is d for o, d in zip(out, [dets[1], dets[3], dets[5]]))


def test_prescreen_reserves_slots_per_subject(tmp_path):
    # Subject 1's only candidate (occluded: off its line, weak appearance) is
    # outscored by all four of subject 0's candidates, yet keeps a slot.
    seg = _seg(masks={"seed_candidates_per_subject": 2}, output_path=str(tmp_path))
    md = _init_masks({0: (A, 20), 1: (B, 60)})
    dets = [_det(y, A) for y in (20, 22, 24, 26)] + [_det(70, WEAK_B)]
    out = _prescreen(seg, dets, md)
    assert [id(o) for o in out] == [id(dets[i]) for i in (0, 1, 3, 4)]


@pytest.mark.parametrize("case", ["few_detections", "no_calibration", "mode_off", "no_subjects"])
def test_prescreen_noop_when_condition_does_not_hold(tmp_path, case):
    masks = {"seed_candidates_per_subject": 3}
    if case == "mode_off":
        masks["seed_prescreen"] = "off"
    seg = _seg(masks=masks, output_path=str(tmp_path))

    def _never(_crops):
        raise AssertionError("prescreen scored candidates on a no-op path")
    seg._encode_clip_batch = _never

    md = {} if case == "no_subjects" else _init_masks({0: (A, 40)})
    n = 3 if case == "few_detections" else 6
    dets = [_det(40 + 10 * i, A) for i in range(n)]
    tgt = {"cam_name": "tgt"} if case == "no_calibration" else None
    assert _prescreen(seg, dets, md, tgt=tgt) is dets


# --------------------------------------------------------------------------- #
# Cross-camera strategy resolution
# --------------------------------------------------------------------------- #

TTM, MTT = "track_then_match", "match_then_track"


@pytest.mark.parametrize("backend, setting, calib, expected", [
    ("sam3_prompt_light", "auto", False, TTM),
    ("sam3_prompt_light", "auto", True, TTM),
    ("sam2", "auto", True, TTM),
    ("sam3", "auto", True, TTM),
    ("sam2", None, True, TTM),          # key absent -> auto
    ("sam2", "auto", False, MTT),
    ("sam3", "auto", None, MTT),
    ("sam2", "match_then_track", True, MTT),
    ("sam3_prompt_light", "match_then_track", False, MTT),
    ("sam3", " Track_Then_Match ", False, TTM),
    ("sam2", "bogus", True, MTT),       # unknown values fall back to the legacy path
])
def test_cross_camera_strategy_resolution(backend, setting, calib, expected):
    seg = _seg(sam_version=backend)
    if setting is None:
        seg.assignment_config["masks"].pop("cross_camera_strategy")
    else:
        seg.assignment_config["masks"]["cross_camera_strategy"] = setting
    assert seg._cross_camera_strategy(calibration_available=calib) == expected


@pytest.mark.parametrize("calibrated, expected", [(True, TTM), (False, MTT)])
def test_process_new_video_dispatches_on_resolved_strategy(tmp_path, calibrated, expected):
    seg = _seg(sam_version="sam2")
    seg._process_new_video_track_then_match = lambda *a, **k: TTM
    seg._process_new_video_match_then_track = lambda *a, **k: MTT
    tgt = _cam(1.0, "tgt")
    if not calibrated:
        del tgt["cam_ext"]
    out = seg.process_new_video(SimpleNamespace(cam_name="tgt"), {}, str(tmp_path / "tgt"),
                                source_cam_data=_cam(0.0, "init"), target_cam_data=tgt)
    assert out == expected
