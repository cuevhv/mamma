"""Cross-camera consensus: synthetic rigs with known swaps. Numpy-only."""
import os

import numpy as np
import pytest

from core import consensus, geometry

K = np.array([[1000.0, 0, 1000], [0, 1000.0, 750], [0, 0, 1]])


def _cam(tx, ty=0.0):
    E = np.eye(4)
    E[0, 3], E[1, 3] = tx, ty
    return {"cam_int": K, "cam_ext": E}


def _lookat_cam(eye, target, k=None):
    fwd = np.asarray(target, float) - np.asarray(eye, float)
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, -1.0, 0.0])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = -R @ np.asarray(eye, float)
    return {"cam_int": K if k is None else k, "cam_ext": E}


def _project(cam, X):
    E = np.asarray(cam["cam_ext"])
    Kc = np.asarray(cam["cam_int"])
    x = E[:3, :3] @ X + E[:3, 3]
    u = Kc @ (x / x[2])
    return float(u[0]), float(u[1])


def _world_tracks(n=40):
    """Two people walking parallel lines in world space, 4m deep."""
    t = {}
    for f in range(n):
        t[f] = {0: np.array([-0.5 + 0.01 * f, 0.0, 4.0]),
                1: np.array([+0.5 - 0.01 * f, 0.2, 4.0])}
    return t


def _cam_tracks(cam, world, swap_after=None):
    out = {0: {}, 1: {}}
    for f, people in world.items():
        for pid, X in people.items():
            oid = pid
            if swap_after is not None and f > swap_after:
                oid = 1 - pid
            out[oid][f] = _project(cam, X)
    return out


def _rig():
    cams = {"A": _cam(0.0), "B": _cam(1.5), "C": _cam(-1.5, 0.5)}
    world = _world_tracks()
    return cams, world


def _fmats(cams, a, b):
    return (geometry.compute_fundamental_matrix(cams[a], cams[b]),
            geometry.compute_fundamental_matrix(cams[b], cams[a]))


def test_pair_vote_detects_relative_swap():
    cams, world = _rig()
    ta = _cam_tracks(cams["A"], world, swap_after=20)
    tb = _cam_tracks(cams["B"], world)
    f_ab, f_ba = _fmats(cams, "A", "B")
    assert consensus.pair_vote(ta, tb, f_ab, f_ba, 10, [0, 1],
                               min_margin_px=10.0) == (0, 1)
    assert consensus.pair_vote(ta, tb, f_ab, f_ba, 30, [0, 1],
                               min_margin_px=10.0) == (1, 0)


def test_pair_vote_survives_centroid_bias():
    # Shift every centroid of one camera 30px sideways (silhouette-asymmetry
    # bias): the relative permutation must still be read correctly because a
    # wrong-person hypothesis costs person-separation distances, far above
    # the bias. Regression test for the measured failure of line-to-bbox
    # voting (person-sized boxes saturate every hypothesis to zero).
    cams, _ = _rig()
    world = {f: {0: np.array([-1.0 + 0.004 * f, 0.0, 4.0]),
                 1: np.array([+1.0 - 0.004 * f, 0.6, 4.5])}
             for f in range(40)}
    ta = _cam_tracks(cams["A"], world, swap_after=20)
    biased = {oid: {f: (x + 30.0, y) for f, (x, y) in trk.items()}
              for oid, trk in ta.items()}
    tb = _cam_tracks(cams["B"], world)
    f_ab, f_ba = _fmats(cams, "A", "B")
    assert consensus.pair_vote(biased, tb, f_ab, f_ba, 30, [0, 1],
                               min_margin_px=40.0) == (1, 0)


def test_pair_vote_abstains_when_people_colocated():
    cams, _ = _rig()
    world = {f: {0: np.array([0.0, 0.0, 4.0]), 1: np.array([0.02, 0.0, 4.0])}
             for f in range(40)}
    ta = _cam_tracks(cams["A"], world, swap_after=20)
    tb = _cam_tracks(cams["B"], world)
    f_ab, f_ba = _fmats(cams, "A", "B")
    assert consensus.pair_vote(ta, tb, f_ab, f_ba, 30, [0, 1],
                               min_margin_px=40.0) is None


def test_pair_regimes_drops_transients_and_merges():
    votes = ([(f, (0, 1)) for f in range(0, 20)]
             + [(20, (1, 0))]                        # single-sample flip
             + [(f, (0, 1)) for f in range(21, 40)]
             + [(f, (1, 0)) for f in range(40, 60)])
    regs = consensus.pair_regimes(votes, min_seg=5)
    assert [r[2] for r in regs] == [(0, 1), (1, 0)]
    assert regs[0][0] == 0 and regs[0][1] == 39
    assert regs[1][0] == 40 and regs[1][1] == 59


def test_detect_events_clusters_and_records_relations():
    ident, swap = (0, 1), (1, 0)
    series = {
        frozenset(("A", "B")): [(0, 20, ident), (24, 60, swap)],
        frozenset(("A", "C")): [(0, 22, ident), (26, 60, swap)],
        frozenset(("B", "C")): [(0, 60, ident)],
    }
    events = consensus.detect_events(series)
    assert len(events) == 1
    ev = events[0]
    assert ev.lo == 20 and ev.hi == 26
    assert ev.relations[frozenset(("A", "B"))] == (ident, swap)
    assert ev.relations[frozenset(("B", "C"))] == (ident, ident)


def test_solve_event_yields_gauge_pair():
    ident, swap = (0, 1), (1, 0)
    ev = consensus._Event(20, 26, {
        frozenset(("A", "B")): (ident, swap),
        frozenset(("A", "C")): (ident, swap),
        frozenset(("B", "C")): (ident, ident),
    })
    sols = consensus.solve_event(ev, ["A", "B", "C"], 2)
    as_sets = {tuple(sorted(c for c, s in sol.items() if s != ident))
               for sol in sols}
    assert as_sets == {("A",), ("B", "C")}   # complement pair, nothing else


def test_resolve_mode():
    assert consensus.resolve_mode("auto", 2) is True
    assert consensus.resolve_mode("auto", 1) is False
    assert consensus.resolve_mode(False, 5) is False
    assert consensus.resolve_mode("true", 0) is True


def test_relabel_masks_two_phase_swap(tmp_path):
    import cv2
    d = tmp_path / "masks"
    d.mkdir()
    a = np.zeros((10, 10), np.uint8); a[2:5, 2:5] = 255
    b = np.zeros((10, 10), np.uint8); b[6:9, 6:9] = 255
    for f in range(3):
        cv2.imwrite(str(d / f"mask_{f:04d}_01.png"), a)
        cv2.imwrite(str(d / f"mask_{f:04d}_02.png"), b)
    seg = consensus.Segment(start_frame=1, end_frame=2, permutation=(1, 0),
                            margin_px=99.0)
    n = consensus.relabel_masks(str(d), [0, 1], seg)
    assert n == 4
    m1 = cv2.imread(str(d / "mask_0001_01.png"), 0)
    assert (m1[6:9, 6:9] > 0).all()          # swapped content
    m0 = cv2.imread(str(d / "mask_0000_01.png"), 0)
    assert (m0[2:5, 2:5] > 0).all()          # untouched outside the segment
    assert not list(d.glob("*.consensus_tmp"))


def test_line_to_bbox_distance():
    line = np.array([0.0, 1.0, -100.0])          # horizontal line y=100
    assert geometry.line_to_bbox_distance(line, (0, 90, 50, 110)) == 0.0
    assert geometry.line_to_bbox_distance(line, (0, 120, 50, 150)) == pytest.approx(20.0)
    assert geometry.line_to_bbox_distance(line, (0, 40, 50, 60)) == pytest.approx(40.0)


def test_track_at_interpolation():
    track = {10: (100.0, 200.0), 40: (160.0, 260.0)}
    assert consensus.track_at(track, 10) == (100.0, 200.0)
    assert consensus.track_at(track, 25) == pytest.approx((130.0, 230.0))
    assert consensus.track_at(track, 5) is None            # no left bracket
    assert consensus.track_at(track, 25, max_gap=10) is None  # gap too wide


# --------------------------------------------------------------------------
# end-to-end: PNG rigs through run_cross_camera_consensus
# --------------------------------------------------------------------------

def _write_rig(tmp_path, cams, world, swaps):
    """Mask PNG tree for each camera; ``swaps`` maps cam -> swap-after frame."""
    import cv2
    for name, cam in cams.items():
        d = tmp_path / name / "masks"
        d.mkdir(parents=True)
        for f, people in world.items():
            for pid, X in people.items():
                oid = pid
                if swaps.get(name) is not None and f > swaps[name]:
                    oid = 1 - pid
                u, v = _project(cam, X)
                m = np.zeros((1500, 2000), np.uint8)
                cv2.circle(m, (int(round(u)), int(round(v))), 12, 255, -1)
                cv2.imwrite(str(d / f"mask_{f:04d}_{oid + 1:02d}.png"), m)


def _read_labels(tmp_path, cam, world):
    """Recovered person id per (frame, label) via nearest projection."""
    import cv2
    out = {}
    for f, people in world.items():
        for lbl in (0, 1):
            p = tmp_path / cam / "masks" / f"mask_{f:04d}_{lbl + 1:02d}.png"
            m = cv2.imread(str(p), 0)
            if m is None or m.max() == 0:
                continue
            ys, xs = np.nonzero(m)
            c = np.array([xs.mean(), ys.mean()])
            cams_proj = {pid: np.array(_project(CAMS_E2E[cam], X))
                         for pid, X in people.items()}
            out[(f, lbl)] = min(cams_proj, key=lambda k: np.linalg.norm(cams_proj[k] - c))
    return out


CAMS_E2E = {}


def _e2e_cfg():
    return {"consensus_min_margin_px": 8.0,
            "consensus_min_segment_frames": 6,
            "consensus_stride": 3}


def test_end_to_end_repairs_minority_swap(tmp_path):
    """One camera's files swapped mid-sequence while everyone stays
    separated: the minority hypothesis costs one camera's separation, the
    complement costs two, so the swapped camera is repaired, frame-exact."""
    global CAMS_E2E
    CAMS_E2E = cams = {"A": _cam(0.0), "B": _cam(1.5), "C": _cam(-1.5, 0.5)}
    world = {f: {0: np.array([-0.9 + 0.004 * f, 0.0, 4.0]),
                 1: np.array([+0.9 - 0.004 * f, 0.3, 4.3])}
             for f in range(60)}
    _write_rig(tmp_path, cams, world, {"A": 30})
    reports = consensus.run_cross_camera_consensus(
        str(tmp_path), cams | {}, cfg=_e2e_cfg(), log=lambda m: None)
    rep = {r.cam_name: r for r in reports}
    assert len(rep["A"].relabeled) == 1
    seg = rep["A"].relabeled[0]
    assert abs(seg.start_frame - 31) <= 1 and seg.end_frame >= 59
    assert not rep["B"].relabeled and not rep["C"].relabeled
    for cam in cams:
        labels = _read_labels(tmp_path, cam, world)
        wrong = [k for k, pid in labels.items() if pid != k[1]]
        assert wrong == []
    # idempotence: a second run must change nothing
    reports2 = consensus.run_cross_camera_consensus(
        str(tmp_path), cams | {}, cfg=_e2e_cfg(), log=lambda m: None)
    assert not any(r.relabeled for r in reports2)


def test_end_to_end_constant_inverted_camera(tmp_path):
    """A camera whose labels are inverted from the very first frame (failed
    seeding) shows constant pair relations and produces no events; only the
    static gauge layer can catch it. The anchor camera defines truth."""
    global CAMS_E2E
    CAMS_E2E = cams = {"A": _cam(0.0), "B": _cam(1.5), "C": _cam(-1.5, 0.5)}
    world = {f: {0: np.array([-0.9 + 0.004 * f, 0.0, 4.0]),
                 1: np.array([+0.9 - 0.004 * f, 0.3, 4.3])}
             for f in range(60)}
    _write_rig(tmp_path, cams, world, {"A": -1})          # swapped throughout
    cfg = _e2e_cfg() | {"consensus_anchor_cam": "B", "consensus_anchor_frame": 0}
    reports = consensus.run_cross_camera_consensus(
        str(tmp_path), cams | {}, cfg=cfg, log=lambda m: None)
    rep = {r.cam_name: r for r in reports}
    assert len(rep["A"].relabeled) == 1
    seg = rep["A"].relabeled[0]
    assert seg.start_frame == 0 and seg.end_frame >= 59
    assert not rep["B"].relabeled and not rep["C"].relabeled
    for cam in cams:
        labels = _read_labels(tmp_path, cam, world)
        assert [k for k, pid in labels.items() if pid != k[1]] == []


def test_end_to_end_2v2_contact_disambiguation(tmp_path):
    """Two of four cameras track through a 2D crossing and swap there; the
    other two view from the side where the people never come close. The
    pairwise votes alone cannot tell the two hypotheses apart (global-flip
    gauge); continuity must pick the pair that crossed, never the pair that
    stayed separated. Regression test for majority-free gauge anchoring."""
    global CAMS_E2E
    CAMS_E2E = cams = {
        "A": _cam(-0.2, ty=-0.3), "B": _cam(0.2, ty=0.35),
        "C": _lookat_cam([4.5, -1.2, 4.0], [0.0, 0.0, 4.1]),
        "D": _lookat_cam([-4.5, -1.0, 4.0], [0.0, 0.0, 4.1]),
    }
    # front cams see the pair cross near f30; side cams keep them separated
    # by the 0.8m depth gap; heights break the planar epipolar degeneracy
    world = {f: {0: np.array([-0.8 + 0.025 * f, 0.0, 3.7]),
                 1: np.array([+0.8 - 0.025 * f, 0.15, 4.5])}
             for f in range(60)}
    _write_rig(tmp_path, cams, world, {"A": 30, "B": 31})
    reports = consensus.run_cross_camera_consensus(
        str(tmp_path), cams | {}, cfg=_e2e_cfg(), log=lambda m: None)
    rep = {r.cam_name: r for r in reports}
    assert rep["A"].relabeled and rep["B"].relabeled
    assert not rep["C"].relabeled and not rep["D"].relabeled
    for cam in cams:
        labels = _read_labels(tmp_path, cam, world)
        wrong = [k for k, pid in labels.items() if pid != k[1]]
        assert len(wrong) <= 4        # crossing frames themselves are ambiguous


def test_solve_event_two_person_tolerates_noisy_pair():
    ident, swap = (0, 1), (1, 0)
    ev = consensus._Event(20, 26, {
        frozenset(("A", "B")): (ident, swap),
        frozenset(("A", "C")): (ident, swap),
        frozenset(("A", "D")): (ident, ident),   # noisy: contradicts the rest
        frozenset(("B", "C")): (ident, ident),
        frozenset(("B", "D")): (ident, ident),
        frozenset(("C", "D")): (ident, ident),
    })
    sols = consensus.solve_event(ev, ["A", "B", "C", "D"], 2)
    as_sets = {tuple(sorted(c for c, s in sol.items() if s != ident))
               for sol in sols}
    assert as_sets == {("A",), ("B", "C", "D")}


def test_continuity_cost_absence_is_no_evidence():
    tracks = {0: {10: (100.0, 100.0), 11: (110.0, 100.0)},
              1: {10: (140.0, 100.0)}}          # id 1 unobserved at f11
    cost, frame = consensus.continuity_cost(tracks, (1, 0), [0, 1], 10, 10)
    assert cost == pytest.approx(40.0) and frame == 10
    # No frame in the bracket observes the full exchange: no evidence,
    # never a free swap.
    cost, frame = consensus.continuity_cost({0: {11: (0.0, 0.0)}, 1: {}},
                                            (1, 0), [0, 1], 10, 12)
    assert cost is None and frame == 11


def test_camera_reliability_flags_flipping_camera():
    ident, swap = (0, 1), (1, 0)
    series = {frozenset(("A", "B")): [(0, 200, ident)],
              frozenset(("A", "D")): [(0, 90, ident), (100, 200, swap)],
              frozenset(("B", "D")): [(0, 90, ident), (100, 200, swap)]}
    rel = consensus.camera_reliability(series, ["A", "B", "D"])
    assert rel["A"] == 1.0 and rel["B"] == 1.0
    assert rel["D"] < 0.6


def test_reliability_prior_prevents_majority_flip():
    """The 159fix failure shape: one unreliable, mask-absent camera versus
    three healthy cameras that all watched the same physical contact. The
    solver must flip the suspect, never the majority."""
    ident, swap = (0, 1), (1, 0)
    ev = consensus._Event(100, 110, {})
    comp = ["A", "B", "C", "D"]
    solutions = [{"A": ident, "B": ident, "C": ident, "D": swap},
                 {"A": swap, "B": swap, "C": swap, "D": ident}]
    # A/B/C observe both people close together (contact); D observes nothing.
    close = {0: {f: (500.0, 500.0) for f in range(100, 111)},
             1: {f: (530.0, 500.0) for f in range(100, 111)}}
    dense = {"A": close, "B": close, "C": close, "D": {}}
    reliability = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.55}
    winner, scored, reason = consensus.score_event_hypotheses(
        solutions, comp, dense, [0, 1], ev, reliability, 40.0)
    assert winner is not None, reason
    flipped = [c for c in comp if winner[1][c] != ident]
    assert flipped == ["D"]
    # Same geometry but no reliability gap: priors alone must not decide.
    winner, _scored, reason = consensus.score_event_hypotheses(
        solutions, comp, dense, [0, 1], ev,
        {c: 1.0 for c in comp}, 40.0)
    assert winner is None
    assert "no continuity evidence" in reason


def test_genuine_multi_camera_swap_still_repaired():
    """Two cameras that both watched the contact and both flipped must stay
    repairable: the flip penalty cancels between equal-cardinality,
    equal-reliability hypotheses and continuity decides."""
    ident, swap = (0, 1), (1, 0)
    ev = consensus._Event(100, 110, {})
    comp = ["A", "B", "C", "D"]
    solutions = [{"A": swap, "B": swap, "C": ident, "D": ident},
                 {"A": ident, "B": ident, "C": swap, "D": swap}]
    close = {0: {105: (500.0, 500.0)}, 1: {105: (520.0, 500.0)}}
    far = {0: {105: (300.0, 500.0)}, 1: {105: (900.0, 500.0)}}
    dense = {"A": close, "B": close, "C": far, "D": far}
    winner, _scored, reason = consensus.score_event_hypotheses(
        solutions, comp, dense, [0, 1], ev,
        {c: 1.0 for c in comp}, 40.0)
    assert winner is not None, reason
    flipped = sorted(c for c in comp if winner[1][c] != ident)
    assert flipped == ["A", "B"]


def test_smeared_event_abstains(tmp_path, monkeypatch):
    """A vote void wider than MAX_EVENT_FRAMES between regimes must abstain
    rather than attribute a swap across the smear."""
    import cv2
    cams, world = _rig()
    n = 420
    long_world = {}
    for f in range(n):
        long_world[f] = {0: np.array([-0.5 + 0.001 * f, 0.0, 4.0]),
                         1: np.array([+0.5 - 0.001 * f, 0.2, 4.0])}
    for name, cam in cams.items():
        d = tmp_path / name / "masks"
        d.mkdir(parents=True)
        for f in range(n):
            if 40 <= f < 380:      # long absence smears the transition
                continue
            for pid, X in long_world[f].items():
                oid = pid if f < 40 or name != "A" else 1 - pid
                u, v = _project(cam, X)
                m = np.zeros((1500, 2000), np.uint8)
                cv2.circle(m, (int(u), int(v)), 12, 255, -1)
                cv2.imwrite(str(d / f"mask_{f:04d}_{oid + 1:02d}.png"), m)
    logs = []
    reports = consensus.run_cross_camera_consensus(
        str(tmp_path), cams,
        cfg={"consensus_min_segment_frames": 20, "consensus_stride": 5},
        log=logs.append)
    assert any("bracket wider" in line for line in logs)
    assert not any(r.relabeled for r in reports)


def test_pair_regimes_rejects_sparse_votes():
    """A few near-threshold votes stretched over a long span must not form a
    regime when the sampling grid shows most samples were indecisive."""
    ident, swap = (0, 1), (1, 0)
    grid = list(range(0, 300, 5))
    dense = [(f, ident) for f in range(0, 100, 5)]
    sparse = [(120, swap), (165, swap), (210, swap), (255, swap)]
    regs = consensus.pair_regimes(dense + sparse, 45, grid=grid)
    assert [r[2] for r in regs] == [ident]
    # The same votes without grid information keep the old span behavior.
    regs = consensus.pair_regimes(dense + sparse, 45)
    assert [r[2] for r in regs] == [ident, swap]
