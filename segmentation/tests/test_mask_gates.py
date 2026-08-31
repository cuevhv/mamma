"""Unit tests for the identity-cleanup mechanisms around SAM propagation:
centroid-outlier detection, area-spike detection, seed centrality weighting,
the near-identical run scanner, re-latch truncation (absence / jump-resolved /
duplicate-suppressed paths), and their bit-identical-off guarantees.

All synthetic: square masks in a small frame, a real disk-backed MaskStore,
no SAM, no GPU.
"""
import numpy as np
import pytest

from core.mask_store import MaskStore
from core.pipeline import (
    SegmentMultipleFrames,
    _argmax_disjoint_masks,
    _detect_centroid_outlier_runs,
    _resolve_auto_flag,
    _seed_centrality_weight,
    _transient_area_spikes,
    _weighted_unit_mean,
)

H, W = 100, 200
SQ = 40  # area 1600 px: comfortably above the truncation "solid" floor (10 * pre_area_px)


def _seg():
    seg = SegmentMultipleFrames.__new__(SegmentMultipleFrames)
    seg.assignment_config = {}
    seg._log_info = lambda *a, **k: None
    seg._log_warn = lambda *a, **k: None
    seg._log_error = lambda *a, **k: None
    return seg


def _square(x, y=30):
    m = np.zeros((H, W), dtype=bool)
    m[y:y + SQ, max(0, x):max(0, x) + SQ] = True
    return m


def _store(pos):
    """pos: {frame: {oid: x-position or None}} -> MaskStore."""
    vs = MaskStore()
    for f, oids in sorted(pos.items()):
        masks = {}
        for oid, x in oids.items():
            if x is not None:
                masks[oid] = _square(x)
        vs.set_frame(f, masks)
    return vs


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def test_centrality_weight_shape_long_clip():
    # >= 3000 frames: full strength
    assert _seed_centrality_weight(2500, 5001, bias=0.5) == 1.0       # center
    assert _seed_centrality_weight(0, 5001, bias=0.5) == pytest.approx(0.5)
    assert _seed_centrality_weight(5000, 5001, bias=0.5) == pytest.approx(0.5)
    assert _seed_centrality_weight(0, 5001, bias=0.0) == 1.0          # disabled
    assert _seed_centrality_weight(0, 1, bias=0.5) == 1.0             # degenerate


def test_centrality_weight_inert_on_short_clips():
    # <= 1000 frames: fully inert (GT study: unscaled bias cost 17 recall
    # points on a 740-frame crowded clip by overriding a better-separated
    # endpoint seed); 2000 frames: half strength.
    assert _seed_centrality_weight(0, 740, bias=0.5) == 1.0
    assert _seed_centrality_weight(0, 1000, bias=0.5) == 1.0
    # n=2000: strength = 0.5 * (2000-1000)/2000 = 0.25 -> endpoint 0.75
    assert _seed_centrality_weight(0, 2000, bias=0.5) == pytest.approx(0.75)


def test_centroid_outlier_detects_transient_teleport():
    # steady drift, 2-frame teleport far away, then back
    c = {f: (100 + f, 50.0) for f in range(40)}
    c[20] = (600.0, 50.0)
    c[21] = (600.0, 50.0)
    runs = _detect_centroid_outlier_runs(c, abs_floor=50.0)
    assert runs == [(20, 21)]


def test_centroid_outlier_spares_fast_motion():
    # genuinely fast motion lies on its own bridge: no flags
    c = {f: (10.0 + 30 * f, 50.0) for f in range(40)}
    assert _detect_centroid_outlier_runs(c, abs_floor=50.0) == []


def test_centroid_outlier_ignores_sustained_move():
    # a sustained relocation (> max_gap frames) is not transient
    c = {f: ((100.0, 50.0) if f < 15 or f >= 30 else (600.0, 50.0)) for f in range(45)}
    assert _detect_centroid_outlier_runs(c, abs_floor=50.0, max_gap=4) == []


def test_centroid_outlier_ignores_distant_brackets():
    # an excursion right after a long absence: the pre-absence bracket cannot
    # certify a return, so a genuine reappearance must not be zeroed
    c = {f: (0.0, 50.0) for f in range(21)}
    c[120] = (800.0, 50.0)
    c[121] = (800.0, 50.0)
    c.update({f: (0.0, 50.0) for f in range(122, 143)})
    assert _detect_centroid_outlier_runs(c, abs_floor=100.0) == []
    # loosening the bracket bound restores the drop
    assert _detect_centroid_outlier_runs(
        c, abs_floor=100.0, bracket_max_gap=200) == [(120, 121)]


def test_centroid_outlier_no_false_flags_on_smooth_motion_with_dropout():
    # fast smooth (no-teleport) trajectories with random dropout and one long
    # occlusion must never be flagged; index-space bridge interpolation and
    # unbounded brackets both false-fired on exactly this
    rng = np.random.default_rng(0)
    n_flagged = 0
    for _ in range(100):
        v = np.zeros(2)
        p = rng.uniform(200, 1800, size=2)
        traj = {}
        drop = rng.random(300) < 0.25
        occ_start = int(rng.integers(50, 220))
        drop[occ_start:occ_start + 40] = True
        for f in range(300):
            v = 0.9 * v + rng.normal(0.0, 15.0, size=2)
            p = p + v
            if not drop[f]:
                traj[f] = (float(p[0]), float(p[1]))
        if len(traj) >= 5:
            n_flagged += len(_detect_centroid_outlier_runs(traj, abs_floor=300.0))
    assert n_flagged == 0


def test_area_spikes():
    areas = {f: 100 for f in range(10)}
    areas[5] = 500                                   # transient spike
    assert _transient_area_spikes(areas, spike_ratio=3.0) == [5]
    growing = {f: 100 * (f + 1) for f in range(10)}  # sustained growth
    assert _transient_area_spikes(growing, spike_ratio=3.0) == []


def test_area_spikes_ignore_sustained_step():
    # a permanent step-up (person approaching the camera) is not a spike:
    # the frame must exceed spike_ratio times BOTH neighbours
    step = {0: 100, 1: 100, 2: 100, 3: 400, 4: 400, 5: 400}
    assert _transient_area_spikes(step, spike_ratio=3.0) == []


def test_area_spikes_ignore_distant_neighbors():
    # exit-large / re-enter-small across a long absence is not a spike
    areas = {f: 4000 for f in range(100, 141)}
    areas[900] = 900
    assert _transient_area_spikes(areas, spike_ratio=3.0) == []


def test_duplicate_suppression_compares_common_boundaries():
    # B jumps 100 px onto the run and has no post-run centroid; A moves 10 px
    # at the run start and a lot AFTER the run (unrelated motion). Only the
    # start boundary is measured for both, so B is the culprit; averaging A's
    # end-boundary motion in used to force an abstain.
    seg = _seg()
    series = {
        "A": [(0.0, 0.0)] * 4 + [(10.0, 0.0)] * 3 + [(210.0, 0.0)] * 3,
        "B": [(0.0, 0.0)] * 4 + [(100.0, 0.0)] * 3 + [None] * 3,
    }
    culprits, keeper, _ = seg._pick_duplicate_suppression(
        series, "A", "B", run_s=4, run_e=6, pre_frames=2, min_jump_px=60.0)
    assert culprits == ["B"]
    assert keeper == "A"


def test_weighted_unit_mean_ignores_raw_norms():
    # a low-weight sample with a large raw norm must not dominate the mean
    # direction: influence comes from the weight alone
    import torch
    a = torch.tensor([[1.0, 0.0]])          # unit norm, weight 0.9
    b = torch.tensor([[0.0, 10.0]])         # 10x norm, weight 0.1
    avg = _weighted_unit_mean(torch.cat([a, b]), torch.tensor([0.9, 0.1]))
    assert float(avg[0]) > float(avg[1])    # direction follows the 0.9 weight
    assert float(avg.norm()) == pytest.approx(1.0)


def _cam(tx):
    """Synthetic calibrated camera translated tx meters along x."""
    K = np.array([[1000.0, 0, 500], [0, 1000.0, 400], [0, 0, 1]])
    E = np.eye(4)
    E[0, 3] = tx
    return {"cam_name": f"cam_tx{tx}", "cam_int": K, "cam_ext": E}


def test_epipolar_reference_set_uses_completed_cameras():
    seg = _seg()
    init_cam, target = _cam(0.0), _cam(2.0)
    done_cam = _cam(1.0)
    no_calib = {"cam_name": "nocalib"}
    seg._prescreen_refs = [
        {"cam_name": "cam_tx1.0", "cam_data": done_cam, "mask_dir": "/nonexistent",
         "obj_ids": [0, 1], "mask_data": {0: {"frame": [5], "centroid_xy": [(600.0, 400.0)]}}},
        {"cam_name": "nocalib", "cam_data": no_calib, "mask_dir": "/nonexistent",
         "obj_ids": [0], "mask_data": None},
        # the target itself must never be its own reference
        {"cam_name": "cam_tx2.0", "cam_data": target, "mask_dir": "/nonexistent",
         "obj_ids": [0], "mask_data": None},
    ]
    init_md = {0: {"frame": [5], "centroid_xy": [(500.0, 400.0)]}}
    refs = seg._epipolar_reference_set("cam_tx2.0", init_cam, target, init_md)
    assert [r[0] for r in refs] == ["init", "cam_tx1.0"]
    # both ref-point getters work and F-matrices score a consistent pair
    for name, F, point_fn in refs:
        p = point_fn(0, 5)
        assert p is not None
        score, dist = seg._epipolar_score(F, x1=np.array([p[0], p[1], 1.0]),
                                          x2=np.array([500.0, 400.0, 1.0]),
                                          sigma_px=100.0, max_dist_px=500.0)
        # pure-x translation: epipolar lines are horizontal, same y -> dist 0
        assert score == pytest.approx(1.0)
    # missing subject resolves to None, not a crash
    assert refs[1][2](7, 5) is None


def test_default_config_matches_shipped_yaml():
    # The shipped yamls are the config truth; the in-code defaults exist for
    # library use and partial configs and must never drift from them.
    import os
    import yaml
    from core.pipeline import DEFAULT_ASSIGNMENT_CONFIG
    yml = yaml.safe_load(open(os.path.join(
        os.path.dirname(__file__), os.pardir, "configs", "sam2.yaml")))
    for section in ("matching", "masks"):
        code = DEFAULT_ASSIGNMENT_CONFIG.get(section, {})
        shipped = yml.get(section, {})
        for key in set(code) & set(shipped):
            assert code[key] == shipped[key], (
                f"{section}.{key}: code default {code[key]!r} != "
                f"sam2.yaml {shipped[key]!r}")


def test_argmax_disjoint_masks():
    import torch
    lg = torch.full((2, 1, 20, 20), -5.0)
    lg[0, 0, 2:12, 2:12] = 3.0      # obj A claims a square strongly
    lg[1, 0, 8:18, 8:18] = 1.0      # obj B overlaps A's corner, weaker
    out = _argmax_disjoint_masks([7, 9], lg)
    a, b = out[7], out[9]
    assert not (a & b).any()                     # pixel-disjoint by construction
    assert a[5, 5] and a[10, 10] and not b[10, 10]  # overlap goes to the higher logit
    assert b[15, 15]                             # B keeps its uncontested region
    # union preserved: every positive pixel belongs to exactly one object
    assert int(a.sum() + b.sum()) == int(((lg[0, 0] > 0) | (lg[1, 0] > 0)).sum())


def test_resolve_auto_flag():
    # booleans and boolean-ish strings pass through
    assert _resolve_auto_flag(True, None, 1500) is True
    assert _resolve_auto_flag(False, 9999, 1500) is False
    assert _resolve_auto_flag("true", None, 1500) is True
    assert _resolve_auto_flag("off", 9999, 1500) is False
    # auto gates on clip length; unknown length resolves to off
    assert _resolve_auto_flag("auto", 1499, 1500) is False
    assert _resolve_auto_flag("auto", 1500, 1500) is True
    assert _resolve_auto_flag("auto", None, 1500) is False


def test_mask_store_thresholds_strictly_positive():
    # storage contract: foreground iff value > 0 (not merely nonzero)
    from core.mask_store import _as_stored, _from_stored
    bits, shape = _as_stored(np.array([[-1.0, 0.0, 3.0]]))
    assert _from_stored(bits, shape).tolist() == [[False, False, True]]


# --------------------------------------------------------------------------- #
# Run scanner
# --------------------------------------------------------------------------- #

def test_scan_finds_bridged_run_positions():
    pos = {}
    for f in range(40):
        pos[f] = {0: 30, 1: (150 if f < 20 else 30)}   # identical from 20
    vs = _store(pos)
    scan = _seg()._scan_near_identical_runs(vs, [0, 1])
    vs.close()
    assert scan is not None
    assert scan["runs"] == [(0, 1, 20, 39)]


def test_scan_returns_none_when_never_overlapping():
    vs = _store({f: {0: 20, 1: 150} for f in range(30)})
    assert _seg()._scan_near_identical_runs(vs, [0, 1]) is None
    vs.close()


# --------------------------------------------------------------------------- #
# Truncation paths
# --------------------------------------------------------------------------- #

def _zeroed(vs, oid, frames):
    return [f for f in frames if not vs.frame(f).get(oid, np.zeros(1)).any()]


def test_truncation_absence_path_works_without_gate():
    # id1 present, vanishes >= pre_frames, re-acquires onto id0: default path.
    pos = {}
    for f in range(40):
        x1 = 150 if f < 10 else (None if f < 20 else 30)
        pos[f] = {0: 30, 1: x1}
    vs = _store(pos)
    out = _seg()._truncate_relatched_tracklets("T", vs, [0, 1], resolve_by_jump=False)
    assert len(_zeroed(out, 1, range(20, 40))) == 20   # run zeroed
    assert not _zeroed(out, 1, range(0, 10))            # pre-run kept
    assert not _zeroed(out, 0, range(40))               # keeper intact
    out.close()


def test_truncation_ambiguous_abstains_when_gate_off():
    # both present before an instant takeover: with the gate off nothing changes
    pos = {f: {0: 30, 1: (150 if f < 20 else 30)} for f in range(40)}
    vs = _store(pos)
    out = _seg()._truncate_relatched_tracklets("T", vs, [0, 1], resolve_by_jump=False)
    assert not _zeroed(out, 0, range(40)) and not _zeroed(out, 1, range(20, 40))
    out.close()


def test_truncation_jump_resolved_zeroes_intruder():
    pos = {f: {0: 30, 1: (150 if f < 20 else 30)} for f in range(40)}
    vs = _store(pos)
    out = _seg()._truncate_relatched_tracklets(
        "T", vs, [0, 1], resolve_by_jump=True, image_size=(W, H))
    assert len(_zeroed(out, 1, range(20, 40))) == 20
    assert not _zeroed(out, 0, range(40))
    out.close()


def test_duplicate_suppression_picks_weaker_boundary_continuity():
    # id1 converges GRADUALLY (4 px steps: below the strict jump floor) but
    # snaps 40 px away at the run end; id0 never moves. Strict resolver
    # abstains; suppression must zero id1 only.
    pos = {}
    for f in range(46):
        if f < 16:
            x1 = 90
        elif f < 20:
            x1 = 46 - 4 * (f - 15)          # 42, 38, 34, 30... gradual approach
        elif f <= 35:
            x1 = 30                          # aligned with id0: the run
        else:
            x1 = 70                          # 40 px snap-out
        pos[f] = {0: 30, 1: x1}
    vs = _store(pos)
    out = _seg()._truncate_relatched_tracklets(
        "T", vs, [0, 1], resolve_by_jump=True, image_size=(W, H))
    assert len(_zeroed(out, 1, range(20, 36))) == 16    # culprit suppressed
    assert not _zeroed(out, 0, range(46))                # keeper intact
    out.close()


class _FakePredictor:
    """Records prompts; re-propagation emits the caller-supplied masks.

    Mirrors the two SAM2 semantics the recovery code depends on: the prompt
    frame itself is re-emitted first (real processing_order starts AT
    start_frame_idx), and the recovery code must have cleared the lost
    object's tracked flags / non-cond memory (asserted by the tests via the
    sam2-shaped state dict)."""

    def __init__(self, emit):
        self.prompts = []
        self._emit = emit  # {frame: {oid: bool mask}}

    def _obj_id_to_idx(self, state, obj_id):
        return int(obj_id)

    def add_new_points_or_box(self, state, frame_idx, obj_id, box):
        self.prompts.append((int(frame_idx), int(obj_id)))

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track,
                           reverse=False):
        import torch
        step = -1 if reverse else 1
        for k in range(0, int(max_frame_num_to_track) + 1):  # incl. prompt frame
            f = int(start_frame_idx) + step * k
            masks = self._emit.get(f)
            if masks is None:
                continue
            oids = sorted(masks)
            logits = torch.stack(
                [torch.from_numpy(masks[o].astype(np.float32)) * 2 - 1 for o in oids])
            yield f, oids, logits


def _sam2_state(n_frames, oids=(0, 1)):
    """A sam2-shaped inference_state carrying the bits recovery must reset."""
    return {
        "frames_tracked_per_obj": {o: {f: {} for f in range(n_frames)} for o in oids},
        "output_dict_per_obj": {
            o: {"non_cond_frame_outputs": {f: "old" for f in range(n_frames)},
                "cond_frame_outputs": {}}
            for o in oids},
    }


def test_recover_collapsed_runs_reseeds_and_repropagates():
    # id1 jumps onto id0 at frame 20 and stays identical through 49 (a 30-frame
    # collapse). The (stubbed) detector always sees the second person at x=150,
    # uncovered by the merged masks, so recovery must: prompt tracklet 1 there,
    # re-propagate the sub-range, and leave no sustained run behind.
    pos = {}
    for f in range(60):
        pos[f] = {0: 30, 1: (150 if f < 20 or f >= 50 else 30)}
    vs = _store(pos)

    emit = {f: {0: _square(30), 1: _square(150)} for f in range(20, 50)}
    seg = _seg()
    seg.assignment_config = {"masks": {"recover_collapsed_tracklets": True}}
    seg.predictor = _FakePredictor(emit)
    seg._state_frame_rgb = lambda state, fidx: np.zeros((H, W, 3), np.uint8)
    seg._collect_yolo_detections = lambda img, conf, include_clip_features=False: (
        None, [(None, None, [30, 30, 30 + SQ, 30 + SQ], 0.9),
               (None, None, [150, 30, 150 + SQ, 30 + SQ], 0.9)])

    state = _sam2_state(60)
    out = seg._recover_collapsed_runs(state, vs, cam_name="T", image_size=(W, H))

    assert seg.predictor.prompts and seg.predictor.prompts[0][1] == 1  # intruder re-seeded
    # SAM2 semantics: the lost object's tracked flags + non-cond memory inside
    # the run MUST be cleared, or the re-propagation recomputes the prompt
    # frame without the click and destroys it (review finding 6).
    prompt_f = seg.predictor.prompts[0][0]
    assert prompt_f not in state["frames_tracked_per_obj"][1]
    assert prompt_f not in state["output_dict_per_obj"][1]["non_cond_frame_outputs"]
    assert 35 not in state["output_dict_per_obj"][1]["non_cond_frame_outputs"]
    assert 35 in state["output_dict_per_obj"][0]["non_cond_frame_outputs"]  # keeper untouched
    # the re-propagated sub-range now has id1 back on its own person
    for f in (25, 35, 49):
        m1 = out.frame(f)[1]
        xs = np.nonzero(m1)[1]
        assert xs.min() >= 140, f"frame {f} not recovered"
    # and no sustained near-identical run remains
    scan = seg._scan_near_identical_runs(out, [0, 1])
    runs = [r for r in (scan["runs"] if scan else []) if r[3] - r[2] + 1 >= 30]
    assert runs == []
    out.close()


def test_recover_abstains_on_gradual_convergence_three_people():
    # THREE tracked people; two of them stack with NO takeover-scale jump
    # (conga-line style). A wrong re-seed could leak the third person's
    # identity into the pair, so recovery must abstain.
    pos = {}
    for f in range(60):
        pos[f] = {0: 30, 1: (33 if f < 20 or f >= 50 else 30), 2: 110}
    vs = _store(pos)
    seg = _seg()
    seg.assignment_config = {"masks": {"recover_collapsed_tracklets": True}}
    seg.predictor = _FakePredictor({})
    seg._state_frame_rgb = lambda state, fidx: np.zeros((H, W, 3), np.uint8)
    seg._collect_yolo_detections = lambda img, conf, include_clip_features=False: (
        None, [(None, None, [30, 30, 30 + SQ, 30 + SQ], 0.9),
               (None, None, [110, 30, 110 + SQ, 30 + SQ], 0.9),
               (None, None, [150, 30, 150 + SQ, 30 + SQ], 0.9)])
    state = _sam2_state(60, oids=(0, 1, 2))
    out = seg._recover_collapsed_runs(state, vs, cam_name="T", image_size=(W, H))
    assert seg.predictor.prompts == []          # abstained: no re-seed
    xs = np.nonzero(out.frame(35)[1])[1]
    assert xs.min() < 60                        # masks untouched
    out.close()


def test_recover_reseeds_two_person_drift_without_jump():
    # Exactly TWO tracked people, one verifiably maskless while the pair
    # stacks, and no jump signature (far-from-seed gradual drift). With two
    # people a re-seed cannot leak onto a bystander, so recovery must fire.
    pos = {}
    for f in range(60):
        pos[f] = {0: 30, 1: (33 if f < 20 or f >= 50 else 30)}
    vs = _store(pos)
    emit = {f: {0: _square(30), 1: _square(150)} for f in range(20, 50)}
    seg = _seg()
    seg.assignment_config = {"masks": {"recover_collapsed_tracklets": True}}
    seg.predictor = _FakePredictor(emit)
    seg._state_frame_rgb = lambda state, fidx: np.zeros((H, W, 3), np.uint8)
    seg._collect_yolo_detections = lambda img, conf, include_clip_features=False: (
        None, [(None, None, [30, 30, 30 + SQ, 30 + SQ], 0.9),
               (None, None, [150, 30, 150 + SQ, 30 + SQ], 0.9)])
    state = _sam2_state(60)
    out = seg._recover_collapsed_runs(state, vs, cam_name="T", image_size=(W, H))
    assert seg.predictor.prompts, "two-person drift must be re-seeded"
    xs = np.nonzero(out.frame(35)[1])[1]
    assert xs.min() >= 140                      # lost dancer re-acquired
    out.close()


def test_recover_ignores_tracked_bystander():
    # A third, correctly-tracked person (id 2 at x=90) must never be mistaken
    # for the lost dancer: their box is covered by their OWN mask. With no
    # genuinely uncovered detection, recovery must place no prompt at all
    # (the pre-fix code unioned only the collapsed pair's masks and would
    # have re-seeded onto the bystander).
    pos = {f: {0: 30, 1: (150 if f < 20 else 30), 2: 90} for f in range(60)}
    vs = _store(pos)
    seg = _seg()
    seg.assignment_config = {"masks": {"recover_collapsed_tracklets": True}}
    seg.predictor = _FakePredictor({})
    seg._state_frame_rgb = lambda state, fidx: np.zeros((H, W, 3), np.uint8)
    seg._collect_yolo_detections = lambda img, conf, include_clip_features=False: (
        None, [(None, None, [30, 30, 30 + SQ, 30 + SQ], 0.9),
               (None, None, [90, 30, 90 + SQ, 30 + SQ], 0.9)])
    out = seg._recover_collapsed_runs(_sam2_state(60, (0, 1, 2)), vs,
                                      cam_name="T", image_size=(W, H))
    assert seg.predictor.prompts == []
    out.close()


def test_recover_skips_when_person_never_detected():
    # detector only ever sees ONE person: nothing to recover; masks unchanged
    # and no prompt is placed (the run falls through to suppression later).
    pos = {f: {0: 30, 1: (150 if f < 20 else 30)} for f in range(60)}
    vs = _store(pos)
    seg = _seg()
    seg.assignment_config = {"masks": {"recover_collapsed_tracklets": True}}
    seg.predictor = _FakePredictor({})
    seg._state_frame_rgb = lambda state, fidx: np.zeros((H, W, 3), np.uint8)
    seg._collect_yolo_detections = lambda img, conf, include_clip_features=False: (
        None, [(None, None, [30, 30, 30 + SQ, 30 + SQ], 0.9)])
    out = seg._recover_collapsed_runs({}, vs, cam_name="T", image_size=(W, H))
    assert seg.predictor.prompts == []
    assert out.frame(40)[1].any()  # untouched
    out.close()


def test_clip_start_run_suppressed_with_gate_on():
    # A run that starts at frame 0 has no pre-run evidence; with the gate on
    # it must still be suppressed via END-boundary continuity (task 152
    # Stage A validation: frames 0-464 shipped as duplicates), and left
    # untouched with the gate off.
    pos = {}
    for f in range(40):
        x1 = 30 if f <= 25 else 110      # identical from the start, 80 px snap-out
        pos[f] = {0: 30, 1: x1}
    vs = _store(pos)
    out = _seg()._truncate_relatched_tracklets(
        "T", vs, [0, 1], resolve_by_jump=True, image_size=(W, H))
    run = range(0, 26)
    assert len(_zeroed(out, 1, run)) == 26      # intruder (end-jumper) zeroed
    assert not _zeroed(out, 0, range(40))
    out.close()

    vs2 = _store(pos)
    out2 = _seg()._truncate_relatched_tracklets(
        "T", vs2, [0, 1], resolve_by_jump=False)
    assert not _zeroed(out2, 1, run)            # gate off: unchanged
    out2.close()


def test_duplicate_suppression_zeroes_both_when_symmetric():
    # both converge gradually and separate gradually: no identity information;
    # neither may be exported as a duplicate.
    pos = {}
    for f in range(50):
        if f < 10:
            x0, x1 = 40, 20
        elif f < 20:                          # 1 px/frame convergence
            x0, x1 = 40 - (f - 10), 20 + (f - 10)
        elif f <= 34:
            x0 = x1 = 30                      # the run
        else:                                 # gradual separation
            x0, x1 = 30 + min(f - 34, 10), 30 - min(f - 34, 10)
        pos[f] = {0: x0, 1: x1}
    vs = _store(pos)
    out = _seg()._truncate_relatched_tracklets(
        "T", vs, [0, 1], resolve_by_jump=True, image_size=(W, H))
    run = [f for f in range(50) if 20 <= f <= 34]
    assert len(_zeroed(out, 0, run)) == len(run)
    assert len(_zeroed(out, 1, run)) == len(run)
    assert not _zeroed(out, 0, range(0, 10))             # outside untouched
    assert not _zeroed(out, 1, range(44, 50))
    out.close()


# ---- recovery verify-or-zero -------------------------------------------


class _MemSegments:
    """Minimal in-memory stand-in for MaskStore's frame API."""

    def __init__(self):
        self.d = {}

    def frame(self, f):
        return {k: v.copy() for k, v in self.d.get(f, {}).items()}

    def set_frame(self, f, masks):
        self.d[f] = {k: v.copy() for k, v in masks.items()}


def _blob(x, y, r=6, h=80, w=120):
    m = np.zeros((h, w), bool)
    m[max(0, y - r):y + r, max(0, x - r):x + r] = True
    return m


def test_collapse_flags_scrap_and_split_not_contact():
    from core.pipeline import _collapse_flags
    seg = _MemSegments()
    frames = list(range(12))
    box = (0, 0, 40, 40)                        # person box, area 1600
    for f in frames:
        seg.set_frame(f, {1: _blob(30, 40, 10), 2: _blob(90, 40, 10)})
    seg.set_frame(3, {1: _blob(30, 40, 10), 2: _blob(32, 41, 3)})   # scrap on 1
    seg.set_frame(4, {1: _blob(28, 40, 7), 2: _blob(38, 40, 7)})    # split body
    seg.set_frame(5, {1: _blob(30, 40, 10), 2: _blob(58, 40, 4)})   # occluded neighbor
    flags, scanned = _collapse_flags(seg, frames, [(0, box)], 0, 11, margin=0)
    assert scanned == 12
    assert {(frames[i], o) for i, o in flags} == {(3, 2), (4, 1)} or \
           {(frames[i], o) for i, o in flags} == {(3, 2), (4, 2)}


def test_zero_flagged_masks_blanks_only_flagged():
    from core.pipeline import _zero_flagged_masks
    seg = _MemSegments()
    frames = list(range(6))
    for f in frames:
        seg.set_frame(f, {1: _blob(30, 40, 10), 2: _blob(32, 41, 3)})
    n = _zero_flagged_masks(seg, frames, [(2, 2), (3, 2)])
    assert n == 2
    assert not seg.frame(2)[2].any() and not seg.frame(3)[2].any()
    assert seg.frame(1)[2].any() and seg.frame(4)[2].any()
    assert all(seg.frame(f)[1].any() for f in frames)


def test_unify_failed_run_labels_stops_alternation():
    from core.pipeline import _unify_failed_run_labels
    seg = _MemSegments()
    frames = list(range(21))
    for f in list(range(0, 5)) + list(range(16, 21)):     # healthy boundaries
        seg.set_frame(f, {1: _blob(30, 40, 6), 2: _blob(90, 40, 6)})
    for f in range(5, 16):                                # survivor alternates
        seg.set_frame(f, {(1 if f % 2 == 0 else 2): _blob(30, 40, 6)})
    relabeled, blanked = _unify_failed_run_labels(
        seg, frames, run_s=5, run_e=15, margin=0, jump_px=25.0)
    assert relabeled > 0 and blanked == 0
    for f in range(5, 16):
        masks = seg.frame(f)
        assert masks.get(1) is not None and masks[1].any()
        assert 2 not in masks or not masks[2].any()


def test_unify_failed_run_labels_blanks_unanchored_chain():
    from core.pipeline import _unify_failed_run_labels
    seg = _MemSegments()
    frames = list(range(21))
    for f in list(range(0, 5)) + list(range(16, 21)):
        seg.set_frame(f, {1: _blob(30, 40, 6), 2: _blob(90, 40, 6)})
    for f in range(5, 16):                     # far from both boundary anchors
        seg.set_frame(f, {1: _blob(60, 70, 6)})
    relabeled, blanked = _unify_failed_run_labels(
        seg, frames, run_s=5, run_e=15, margin=0, jump_px=15.0)
    assert blanked == 11 and relabeled == 0
    for f in range(5, 16):
        masks = seg.frame(f)
        assert all(not m.any() for m in masks.values())


def test_drop_single_frame_islands():
    from core.pipeline import _drop_single_frame_islands

    class _Seg(_MemSegments):
        def frames(self):
            return sorted(self.d)

    seg = _Seg()
    for f in range(10):
        seg.set_frame(f, {1: _blob(30, 40, 6)})
    seg.set_frame(4, {1: _blob(30, 40, 6), 2: _blob(90, 40, 6)})   # island
    seg.set_frame(0, {1: _blob(30, 40, 6), 2: _blob(90, 40, 6)})   # clip start
    n = _drop_single_frame_islands(seg)
    assert n == 1
    assert not seg.frame(4)[2].any()
    assert seg.frame(0)[2].any()                # clip end untouched
    assert all(seg.frame(f)[1].any() for f in range(10))
