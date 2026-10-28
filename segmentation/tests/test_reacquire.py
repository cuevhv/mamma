"""Unit tests for re-acquisition of people the cleanup left maskless
(SegmentMultipleFrames._reacquire_lost_people). Synthetic square masks, a real
disk-backed MaskStore, a fake tracker and detector; no SAM, no GPU."""
import numpy as np

from core.mask_store import MaskStore
from core.pipeline import SegmentMultipleFrames

H, W, SQ = 100, 200, 40


def _square(x, size=SQ):
    m = np.zeros((H, W), dtype=bool)
    m[30:30 + size, x:x + size] = True
    return m


def _box(x, size=SQ):
    return [x, 30, x + size, 30 + size]


def _store(pos, n=80):
    """pos(f) -> {oid: x or None}."""
    vs = MaskStore()
    for f in range(n):
        vs.set_frame(f, {o: _square(x) for o, x in pos(f).items() if x is not None})
    return vs


class _FakeTracker:
    """Fresh-session tracker: each prompted box becomes a static square mask,
    unless ``hijack`` maps an obj id to another id's box (a session that
    collapses onto the other person)."""

    def __init__(self, hijack=None):
        self.resets = 0
        self.prompts = []
        self.hijack = hijack or {}
        self.tracked = []

    def reset_state(self, state):
        self.resets += 1
        self.prompts = []

    def add_new_points_or_box(self, state, frame_idx, obj_id, box):
        self.prompts.append((int(frame_idx), int(obj_id), [float(v) for v in box]))

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track, reverse=False):
        import torch
        boxes = {o: b for _f, o, b in self.prompts}
        oids = sorted(boxes)
        step = -1 if reverse else 1
        for k in range(int(max_frame_num_to_track) + 1):
            f = int(start_frame_idx) + step * k
            self.tracked.append(f)
            planes = []
            for o in oids:
                b = boxes[self.hijack.get(o, o)]
                planes.append(torch.from_numpy(_square(int(b[0])).astype(np.float32)) * 2 - 1)
            yield f, oids, torch.stack(planes)[:, None]


def _seg(tracker, dets):
    seg = SegmentMultipleFrames.__new__(SegmentMultipleFrames)
    seg.assignment_config = {}
    seg.logs = []
    seg._log_info = seg.logs.append
    seg._log_warn = seg.logs.append
    seg.predictor = tracker
    seg._state_frame_rgb = lambda state, fidx: np.zeros((H, W, 3), np.uint8)
    seg._collect_yolo_detections = lambda img, conf, include_clip_features=False: (
        None, [(None, None, np.asarray(b, float), 0.9) for b in dets])
    return seg


def _xs(vs, f, oid):
    m = vs.frame(f).get(oid)
    return None if m is None or not m.any() else int(np.nonzero(m)[1].min())


def test_lost_person_is_retracked_from_a_fresh_start():
    # id1 was zeroed on frames 20-59 (a cleaned-up collapse) while the dancer
    # stayed visible at x=150; id0 is tracked throughout at x=30.
    vs = _store(lambda f: {0: 30, 1: None if 20 <= f < 60 else 150})
    tr = _FakeTracker()
    seg = _seg(tr, [_box(30), _box(150)])
    out = seg._reacquire_lost_people({}, vs, cam_name="T")
    assert tr.resets == 1
    assert {o for _f, o, _b in tr.prompts} == {0, 1}      # everyone on the frame prompted
    for f in (20, 40, 59):
        assert _xs(out, f, 1) == 150
        assert _xs(out, f, 0) == 30
    out.close()


def test_start_frame_search_stops_at_the_first_clean_frame():
    # Gap 20-59, candidates every 15 frames: the first one already has the lost
    # dancer uncovered and well apart, so the detector runs exactly once.
    vs = _store(lambda f: {0: 30, 1: None if 20 <= f < 60 else 150})
    tr = _FakeTracker()
    seg = _seg(tr, [_box(30), _box(150)])
    calls = []
    detect = seg._collect_yolo_detections
    seg._collect_yolo_detections = lambda *a, **k: (calls.append(1), detect(*a, **k))[1]
    out = seg._reacquire_lost_people({}, vs, cam_name="T")
    assert len(calls) == 1
    assert tr.prompts[0][0] == 20                          # starts at the gap start
    assert _xs(out, 59, 1) == 150
    out.close()


def test_nothing_happens_without_an_uncovered_person():
    # The detector only sees the tracked person: id1 is genuinely not visible.
    vs = _store(lambda f: {0: 30, 1: None if 20 <= f < 60 else 150})
    tr = _FakeTracker()
    seg = _seg(tr, [_box(30)])
    out = seg._reacquire_lost_people({}, vs, cam_name="T")
    assert tr.resets == 0 and tr.prompts == []
    assert _xs(out, 40, 1) is None
    out.close()


def test_a_session_that_collapses_writes_nothing_and_stops():
    # The fresh session puts id1 on id0's person: every fill would duplicate
    # an existing mask, so nothing is written and the direction stalls out.
    vs = _store(lambda f: {0: 30, 1: None if 10 <= f < 75 else 150})
    tr = _FakeTracker(hijack={1: 0})
    seg = _seg(tr, [_box(30), _box(150)])
    seg.assignment_config = {"masks": {"reacquire_stall_frames": 5, "reacquire_max_rounds": 1}}
    out = seg._reacquire_lost_people({}, vs, cam_name="T")
    assert all(_xs(out, f, 1) is None for f in range(10, 75))
    assert len(tr.tracked) <= 2 * 6           # each direction gave up after the stall
    out.close()


class _SplitBodyTracker(_FakeTracker):
    """Both ids claim the person at x=30; their logits alternate per pixel, so
    the disjoint argmax splits that one body between them like a checkerboard."""

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track, reverse=False):
        import torch
        oids = sorted({o for _f, o, _b in self.prompts})
        body = torch.from_numpy(_square(30).astype(np.float32))
        check = torch.from_numpy((np.indices((H, W)).sum(0) % 2).astype(np.float32))
        step = -1 if reverse else 1
        for k in range(int(max_frame_num_to_track) + 1):
            f = int(start_frame_idx) + step * k
            self.tracked.append(f)
            planes = [body * (1 + (check if i % 2 else 1 - check)) * 2 - 1 for i in range(len(oids))]
            yield f, oids, torch.stack(planes)[:, None]


def test_two_ids_sharing_one_body_are_rejected():
    vs = _store(lambda f: {0: None, 1: None} if 20 <= f < 60 else {0: 30, 1: 150})
    tr = _SplitBodyTracker()
    seg = _seg(tr, [_box(150), _box(30)])
    seg.assignment_config = {"masks": {"reacquire_max_rounds": 1}}
    out = seg._reacquire_lost_people({}, vs, cam_name="T")
    assert all(_xs(out, f, o) is None for f in range(20, 60) for o in (0, 1))
    out.close()


def test_both_missing_ids_are_assigned_by_nearest_known_position():
    # Duplicate suppression zeroed BOTH ids on 20-59; each comes back on the
    # person nearest to where it was last seen.
    vs = _store(lambda f: {0: None, 1: None} if 20 <= f < 60 else {0: 30, 1: 150})
    tr = _FakeTracker()
    seg = _seg(tr, [_box(150), _box(30)])
    out = seg._reacquire_lost_people({}, vs, cam_name="T")
    for f in (20, 40, 59):
        assert _xs(out, f, 0) == 30
        assert _xs(out, f, 1) == 150
    out.close()


def test_one_visible_person_for_two_missing_ids_who_crossed_is_left_alone():
    # Both ids are gone on 20-59 and only one person is visible (x=30). Before
    # the gap id0 stood there, after it id1 does: the two dancers crossed, so
    # the visible person could be either and nothing is assigned.
    def pos(f):
        if 20 <= f < 60:
            return {0: None, 1: None}
        return {0: 30, 1: 150} if f < 20 else {0: 150, 1: 30}
    vs = _store(pos)
    tr = _FakeTracker()
    seg = _seg(tr, [_box(30)])
    out = seg._reacquire_lost_people({}, vs, cam_name="T")
    assert tr.prompts == []
    assert all(_xs(out, f, o) is None for f in range(20, 60) for o in (0, 1))
    out.close()


def test_one_visible_person_matching_the_same_id_on_both_sides_is_refilled():
    vs = _store(lambda f: {0: None, 1: None} if 20 <= f < 60 else {0: 30, 1: 150})
    tr = _FakeTracker()
    seg = _seg(tr, [_box(30)])
    out = seg._reacquire_lost_people({}, vs, cam_name="T")
    assert {o for _f, o, _b in tr.prompts} == {0}
    assert _xs(out, 40, 0) == 30 and _xs(out, 40, 1) is None
    out.close()


def test_frame_indices_that_do_not_start_at_zero():
    # Trimmed runs index the store from the first kept frame; the gap, the
    # prompt frame and the writes must all use those indices.
    vs = MaskStore()
    for f in range(100, 180):
        vs.set_frame(f, {0: _square(30)} if 120 <= f < 160 else {0: _square(30), 1: _square(150)})
    tr = _FakeTracker()
    seg = _seg(tr, [_box(30), _box(150)])
    out = seg._reacquire_lost_people({}, vs, cam_name="T")
    assert tr.prompts and tr.prompts[0][0] == 120
    assert all(_xs(out, f, 1) == 150 for f in (120, 140, 159))
    assert min(tr.tracked) >= 100
    out.close()


def test_implausibly_small_detection_is_not_taken_for_the_lost_person():
    # Only a distant, much smaller person is uncovered (e.g. a seated bystander).
    vs = _store(lambda f: {0: 30, 1: None if 20 <= f < 60 else 150})
    tr = _FakeTracker()
    seg = _seg(tr, [_box(30), _box(150, size=8)])
    out = seg._reacquire_lost_people({}, vs, cam_name="T")
    assert tr.prompts == []
    assert _xs(out, 40, 1) is None
    out.close()


def test_clean_store_is_untouched():
    vs = _store(lambda f: {0: 30, 1: 150})
    tr = _FakeTracker()
    seg = _seg(tr, [_box(30), _box(150)])
    out = seg._reacquire_lost_people({}, vs, cam_name="T")
    assert tr.resets == 0 and _xs(out, 40, 1) == 150
    out.close()
