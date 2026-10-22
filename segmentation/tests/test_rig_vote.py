"""Rig-vote identity relabel: synthetic 12-camera rigs (two and three people) with known swaps."""
import json
import os

import cv2
import numpy as np
import pytest

from core import rig_vote
from core.mask_archive import MaskArchiveReader, MaskArchiveWriter

W, H, F = 320, 240, 300.0
N_CAMS, T, DS = 12, 80, 4
SWAP_CAM, SWAP_LO, SWAP_HI = 3, 20, 59
CFG = {"rig_vote_downsample": DS, "rig_vote_workers": 2}


# --------------------------------------------------------------------------
# synthetic rig
# --------------------------------------------------------------------------

def _lookat(eye, target):
    """World-to-camera (R, t) with z forward, x right, y down; world z-up."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    return R, -R @ eye


def _make_cameras():
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    cams = []
    for i in range(N_CAMS):
        a = 2 * np.pi * i / N_CAMS
        eye = [4.0 * np.cos(a), 4.0 * np.sin(a), 1.5]
        R, t = _lookat(eye, [0.0, 0.0, 0.9])
        cams.append(rig_vote.RigCamera(f"cam{i:02d}", K, R, t, W, H))
    return cams


def _cylinder_points(radius=0.2, height=1.7, step=0.03):
    ax = np.arange(-radius, radius + 1e-9, step)
    xy = np.array([(x, y) for x in ax for y in ax if x * x + y * y <= radius ** 2])
    z = np.arange(0.0, height + 1e-9, step)
    return np.concatenate([np.c_[xy, np.full(len(xy), zz)] for zz in z])


def _centres(t):
    ang = 2 * np.pi * t / 400.0
    c = 0.7 * np.array([np.cos(ang), np.sin(ang)])
    return c, -c


def _render(cam, pts, t):
    """Both people's masks (uint8 0/255) at one frame, with occlusion."""
    depth = []
    for c in _centres(t):
        X = pts + np.array([c[0], c[1], 0.0])
        uv, z = cam.project(X)
        d = np.full((H, W), np.inf)
        ok = np.isfinite(uv).all(1) & (z > 0.1)
        u = np.round(uv[ok, 0]).astype(int)
        v = np.round(uv[ok, 1]).astype(int)
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        np.minimum.at(d, (v[inside], u[inside]), z[ok][inside])
        depth.append(d)
    kernel = np.ones((5, 5), np.uint8)
    out = []
    for k in range(2):
        m = (np.isfinite(depth[k]) & (depth[k] < depth[1 - k])).astype(np.uint8) * 255
        out.append(cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel))
    return out


@pytest.fixture(scope="module")
def rig():
    cams = _make_cameras()
    for c in cams:
        uv, z = c.project(np.array([[0.0, 0.0, 0.9]]))
        assert 0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H and z[0] > 0
    pts = _cylinder_points()
    masks = [[_render(c, pts, t) for t in range(T)] for c in cams]  # [cam][t][pid-1]
    return cams, masks


def _swapped(cam_frames, lo=SWAP_LO, hi=SWAP_HI):
    """One camera's per-frame mask pairs with ids exchanged on lo..hi."""
    return [pair[::-1] if lo <= t <= hi else pair for t, pair in enumerate(cam_frames)]


def _to_camera_masks(frames):
    """Reduced-resolution masks exactly as ``load_camera_masks`` builds them."""
    gh, gw = round(H / DS), round(W / DS)
    planes = []
    for pair in frames:
        row = []
        for img in pair:
            small = cv2.resize((img > 127).astype(np.uint8) * 255, (gw, gh),
                               interpolation=cv2.INTER_AREA)
            row.append(np.flatnonzero(small >= 64).astype(np.uint32))
        planes.append(row)
    return rig_vote.CameraMasks.from_planes((gh, gw), 2, planes)


def _cam_masks(per_cam):
    return [_to_camera_masks(fm) for fm in per_cam]


@pytest.fixture(scope="module")
def clean_result(rig):
    cams, masks = rig
    return rig_vote.compute_rig_vote(cams, _cam_masks(masks), T, CFG, log=lambda s: None)


def _write_rig(root, cams, per_cam, packed=True):
    for c, fm in zip(cams, per_cam):
        with MaskArchiveWriter(str(root / c.name / "masks"), packed=packed) as w:
            for t, pair in enumerate(fm):
                for k, img in enumerate(pair):
                    w.write(f"mask_{t:04d}_{k + 1:02d}.png", img)


def _cam_data(cams):
    out = {}
    for c in cams:
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = c.R, c.t
        out[c.name] = {"cam_int": c.K, "cam_ext": E, "cam_img_w": W, "cam_img_h": H}
    return out


# --------------------------------------------------------------------------
# voting
# --------------------------------------------------------------------------

def test_fixture_people_stay_apart():
    for t in range(T):
        a, b = _centres(t)
        assert np.linalg.norm(a - b) >= 0.8


def test_clean_rig_has_no_swaps(clean_result):
    assert not clean_result.swaps.any()
    assert not (clean_result.raw_states != 0).any()
    assert not clean_result.skipped


def test_injected_swap_is_localised(rig):
    cams, masks = rig
    per_cam = [_swapped(fm) if i == SWAP_CAM else fm for i, fm in enumerate(masks)]
    res = rig_vote.compute_rig_vote(cams, _cam_masks(per_cam), T, CFG, log=lambda s: None)
    row = res.swaps[SWAP_CAM]
    inner = np.arange(SWAP_LO + 2, SWAP_HI - 1)
    outer = np.r_[0:SWAP_LO - 2, SWAP_HI + 3:T]
    assert row[inner].all()
    assert not row[outer].any()
    others = np.delete(res.swaps, SWAP_CAM, axis=0)
    assert not others.any()
    assert res.evidence0[SWAP_CAM, 30] < 0
    assert res.evidence0[0, 30] > 0


# --------------------------------------------------------------------------
# viterbi_states (two subjects: keep / swap)
# --------------------------------------------------------------------------

def _viterbi(E, cost):
    """Two-subject decoding of a (C, T) evidence array; True = swapped."""
    C, T = E.shape
    M = np.zeros((T, C, 2, 2))
    M[:, :, 0, 0] = E.T
    M[:, :, 0, 1] = -E.T
    return rig_vote.viterbi_states(M, rig_vote.permutations(2), cost) != 0


def test_viterbi_negative_block_becomes_swap():
    E = np.full((1, 100), 2.0)
    E[0, 30:60] = -2.0
    out = _viterbi(E, 6.0)
    assert out.shape == (1, 100)
    assert out[0, 30:60].all()
    assert not out[0, :30].any() and not out[0, 60:].any()


def test_viterbi_single_frame_blip_does_not_flip():
    E = np.full((2, 50), 2.0)
    E[0, 25] = -2.0
    assert not _viterbi(E, 6.0).any()


def test_viterbi_zero_evidence_keeps_labels():
    assert not _viterbi(np.zeros((3, 40)), 6.0).any()


def test_viterbi_empty_sequence():
    assert _viterbi(np.zeros((2, 0)), 6.0).shape == (2, 0)


# --------------------------------------------------------------------------
# merge_guard
# --------------------------------------------------------------------------

def _guard_inputs():
    """T=120, one contact window at 55-64, two stretches around it."""
    n, c = 120, 12
    vox = 0.05 ** 3
    cores = np.zeros((n, 2, 4))
    cores[:, 0, :3] = [-1.0, 0.0, 0.9]
    cores[:, 1, :3] = [1.0, 0.0, 0.9]
    cores[:, :, 3] = 1000
    cores[55:65, 0, :3] = [-0.05, 0.0, 0.9]
    cores[55:65, 1, :3] = [0.05, 0.0, 0.9]
    E0 = np.full((c, n), 1.0)
    swaps = np.zeros((c, n), bool)
    # first stretch (0-54): 1 of 12 cameras disagrees
    E0[0, :55] = -1.0
    swaps[0, 10:40] = True
    # second stretch (65-119): 6 of 12 disagree
    E0[:6, 65:] = -1.0
    swaps[:6, 70:100] = True
    # a swap that runs through the contact window
    swaps[1, 50:70] = True
    return cores, E0, swaps, vox


def test_merge_guard_applies_and_abstains():
    cores, E0, swaps, vox = _guard_inputs()
    out, merged, stretches = rig_vote.merge_guard(cores, E0, swaps.astype(np.int64), vox, 0.3, 1 / 3)
    assert merged[55:65].all()
    assert not merged[:50].any() and not merged[70:].any()
    assert not out[:, merged].any()
    assert out[0, 10:40].all()
    assert not out[:6, 70:100].any()
    assert len(stretches) == 2
    first, second = stretches
    assert first["applied"] and first["start"] == 0
    assert not second["applied"]
    assert second["median_minority"] > 1 / 3
    assert second["end"] == 119


def test_merge_guard_bridges_window_swapped_on_both_sides():
    cores, E0, swaps, vox = _guard_inputs()
    E0[:6, 65:] = 1.0           # second stretch uncontested now
    swaps[:6, 70:100] = False
    out, merged, _ = rig_vote.merge_guard(cores, E0, swaps.astype(np.int64), vox, 0.3, 1 / 3)
    assert out[1, 50:70].all()           # swapped before and after: kept through the contact
    assert not out[0, 55:65].any()       # swapped on one side only: window left raw


def _fragment_evidence(rig, masks_of_cam):
    cams, masks = rig
    per_cam = [masks_of_cam if i == SWAP_CAM else fm for i, fm in enumerate(masks)]
    cm = _cam_masks(per_cam)
    grids = [m.grid for m in cm]
    box = rig_vote.capture_volume(cams, cm, T, 2)
    index = rig_vote.VoxelIndex(cams, grids, rig_vote.grid_points(box[0], box[1], 0.05), 0.05)
    perms = rig_vote.permutations(2)

    def evidence(t):
        M, _cores = rig_vote.frame_evidence(index, grids, rig_vote._frame_planes(cm, t, None, None), 2)
        return rig_vote.identity_margin(M[None], perms)[:, 0]
    return evidence


def test_fragment_does_not_cancel_body_evidence(rig):
    """A stray fragment of person 1 labelled as person 2 (person 2 itself
    unmasked) used to cancel the whole body's agreement."""
    cams, masks = rig
    t = 10
    body = masks[SWAP_CAM][t][0]
    ys = np.nonzero(body)[0]
    frag = body.copy()
    frag[:ys.min()] = 0
    frag[ys.min() + 8:] = 0                  # top 8 rows of the body: a few % of its area
    frames = list(masks[SWAP_CAM])
    frames[t] = [body, frag]
    e = _fragment_evidence(rig, frames)(t)
    assert e[SWAP_CAM] > rig_vote.DECISIVE_E


def test_merge_guard_flags_vanished_core_as_contact():
    cores, E0, swaps, vox = _guard_inputs()
    cores[:, 1, 3] = 1000
    cores[30:50, 1, 3] = 5   # below the minimum core size
    _, merged, _ = rig_vote.merge_guard(cores, E0, swaps.astype(np.int64), vox, 0.3, 1 / 3)
    assert merged[32:48].all()


# --------------------------------------------------------------------------
# swap_mask_ids
# --------------------------------------------------------------------------

def _dummy(seed):
    rng = np.random.default_rng(seed)
    return ((rng.random((24, 32)) > 0.5) * 255).astype(np.uint8)


def _write_dummy(d, packed):
    imgs = {"mask_0001_01.png": _dummy(1), "mask_0001_02.png": _dummy(2),
            "mask_0002_01.png": _dummy(3), "mask_0003_01.png": _dummy(4),
            "mask_0003_02.png": _dummy(5)}
    with MaskArchiveWriter(str(d), packed=packed) as w:
        for n, img in imgs.items():
            w.write(n, img)
    return imgs


def _read_all(d):
    with MaskArchiveReader(str(d)) as r:
        return {n: r.read(n) for n in r.names()}


@pytest.mark.parametrize("packed", [True, False])
def test_permute_mask_ids_both_layouts(tmp_path, packed):
    d = tmp_path / "masks"
    orig = _write_dummy(d, packed)
    n = rig_vote.permute_mask_ids(str(d), {1: (1, 0), 2: (1, 0)})
    assert n == 3
    got = _read_all(d)
    assert set(got) == (set(orig) - {"mask_0002_01.png"}) | {"mask_0002_02.png"}
    assert np.array_equal(got["mask_0001_01.png"], orig["mask_0001_02.png"])
    assert np.array_equal(got["mask_0001_02.png"], orig["mask_0001_01.png"])
    assert np.array_equal(got["mask_0002_02.png"], orig["mask_0002_01.png"])
    assert "mask_0002_01.png" not in got
    for name in ("mask_0003_01.png", "mask_0003_02.png"):
        assert np.array_equal(got[name], orig[name])
    assert rig_vote.permute_mask_ids(str(d), {1: (1, 0), 2: (1, 0)}) == 3
    back = _read_all(d)
    assert set(back) == set(orig)
    for name, img in orig.items():
        assert np.array_equal(back[name], img)
    assert not [f for f in os.listdir(d) if "tmp" in f]


@pytest.mark.parametrize("packed", [True, False])
def test_permute_mask_ids_noop(tmp_path, packed):
    d = tmp_path / "masks"
    _write_dummy(d, packed)
    assert rig_vote.permute_mask_ids(str(d), {}) == 0
    assert rig_vote.permute_mask_ids(str(d), {99: (1, 0)}) == 0


# --------------------------------------------------------------------------
# run_rig_vote
# --------------------------------------------------------------------------

def test_run_rig_vote_repairs_and_is_idempotent(tmp_path, rig):
    cams, masks = rig
    per_cam = [_swapped(fm) if i == SWAP_CAM else fm for i, fm in enumerate(masks)]
    _write_rig(tmp_path, cams, per_cam)
    cd = _cam_data(cams)
    logs = []
    res = rig_vote.run_rig_vote(str(tmp_path), cd, CFG, log=logs.append, apply=True)
    assert not res.skipped
    assert res.swaps[SWAP_CAM].any()

    cam3 = cams[SWAP_CAM].name
    with MaskArchiveReader(str(tmp_path / cam3 / "masks")) as r:
        for t in range(T):
            for k in range(2):
                got = r.read(f"mask_{t:04d}_{k + 1:02d}.png")
                assert np.array_equal(got, masks[SWAP_CAM][t][k]), (t, k)
    report = json.loads((tmp_path / cam3 / "rig_vote_report.json").read_text())
    assert report and all(s["new_ids"] == [2, 1] for s in report)
    summary = json.loads((tmp_path / "rig_vote_summary.json").read_text())
    assert cam3 in summary["relabeled"]
    assert summary["cameras"] == [c.name for c in cams]
    other = cams[0].name
    assert not (tmp_path / other / "rig_vote_report.json").exists()

    res2 = rig_vote.run_rig_vote(str(tmp_path), cd, CFG, log=logs.append, apply=True)
    assert not res2.swaps.any()


def test_run_rig_vote_apply_false_touches_nothing(tmp_path, rig):
    cams, masks = rig
    per_cam = [_swapped(fm) if i == SWAP_CAM else fm for i, fm in enumerate(masks)]
    _write_rig(tmp_path, cams, per_cam)
    zpath = tmp_path / cams[SWAP_CAM].name / "masks" / "masks.zip"
    before = zpath.read_bytes()
    res = rig_vote.run_rig_vote(str(tmp_path), _cam_data(cams), CFG,
                                log=lambda s: None, apply=False)
    assert res.swaps[SWAP_CAM].any()
    assert zpath.read_bytes() == before
    assert not (tmp_path / "rig_vote_summary.json").exists()


# --------------------------------------------------------------------------
# skips and modes
# --------------------------------------------------------------------------

def test_too_many_subjects_skips_and_touches_nothing(tmp_path, rig):
    cams, masks = rig
    sub = cams[:6]
    _write_rig(tmp_path, sub, masks[:6])
    d = tmp_path / sub[0].name / "masks"
    with MaskArchiveReader(str(d)) as r:
        items = {n: r.read(n) for n in r.names()}
    items["mask_0000_07.png"] = items["mask_0000_01.png"]   # more than MAX_SUBJECTS ids
    with MaskArchiveWriter(str(d)) as w:
        for n, img in items.items():
            w.write(n, img)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    res = rig_vote.run_rig_vote(str(tmp_path), _cam_data(sub), CFG, log=lambda s: None)
    assert res.skipped and "[1, 2, 7]" in res.skipped
    after = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_too_few_cameras_skips(tmp_path, rig):
    cams, masks = rig
    _write_rig(tmp_path, cams[:3], masks[:3])
    res = rig_vote.run_rig_vote(str(tmp_path), _cam_data(cams[:3]), CFG, log=lambda s: None)
    assert res.skipped


@pytest.mark.parametrize("value,n,min_n,expected", [
    ("auto", 11, 12, False),
    ("auto", 12, 12, True),
    (True, 4, 12, True),
    (True, 1, 12, False),
    ("true", 4, 12, True),
    ("false", 32, 12, False),
    (False, 32, 12, False),
])
def test_resolve_mode(value, n, min_n, expected):
    assert rig_vote.resolve_mode(value, n, min_n) is expected


# --------------------------------------------------------------------------
# lens models
# --------------------------------------------------------------------------

def _lens_points():
    rng = np.random.default_rng(1)
    return rng.uniform([-0.6, -0.4, 2.5], [0.6, 0.4, 4.0], (200, 3))


def test_opencv_lens_matches_project_points():
    K = np.array([[300.0, 0, 160], [0, 310.0, 120], [0, 0, 1]])
    dist = (-0.21, 0.05, 0.001, -0.002, 0.01)
    cam = rig_vote.camera_from_cam_data("c", {
        "cam_int": K, "cam_ext": np.eye(4), "cam_img_w": 320, "cam_img_h": 240,
        "distortion_model": "opencv_brown", "distortion_coeffs": np.array(dist)})
    X = _lens_points()
    uv, _ = cam.project(X)
    ref, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K, np.array(dist))
    ok = np.isfinite(uv).all(1)
    assert ok.mean() > 0.9
    assert np.abs(uv[ok] - ref.reshape(-1, 2)[ok]).max() < 1e-6


def test_undistorted_frames_use_pinhole():
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    cd = {"cam_int": K, "cam_ext": np.eye(4), "cam_img_w": 320, "cam_img_h": 240,
          "distortion_model": "opencv_brown", "distortion_coeffs": np.array([-0.3, 0.1, 0, 0, 0]),
          "_undistort": True}
    cam = rig_vote.camera_from_cam_data("c", cd)
    X = _lens_points()
    uv, z = cam.project(X)
    pin = (X / X[:, 2:3]) @ K.T
    ok = np.isfinite(uv).all(1)
    assert cam.model is None
    assert np.abs(uv[ok] - pin[ok, :2]).max() < 1e-9


# --------------------------------------------------------------------------
# three people
# --------------------------------------------------------------------------

def _multi_masks(per_cam, n):
    gh, gw = round(H / DS), round(W / DS)
    out = []
    for frames in per_cam:
        planes = []
        for masks_t in frames:
            row = []
            for img in masks_t:
                small = cv2.resize((img > 127).astype(np.uint8) * 255, (gw, gh),
                                   interpolation=cv2.INTER_AREA)
                row.append(np.flatnonzero(small >= 64).astype(np.uint32))
            planes.append(row)
        out.append(rig_vote.CameraMasks.from_planes((gh, gw), n, planes))
    return out


def _centres3(t):
    out = []
    for k in range(3):
        ang = 2 * np.pi * (t / 400.0 + k / 3.0)
        out.append(0.8 * np.array([np.cos(ang), np.sin(ang)]))
    return out


def _render3(cam, pts, t):
    depth = []
    for c in _centres3(t):
        X = pts + np.array([c[0], c[1], 0.0])
        uv, z = cam.project(X)
        d = np.full((H, W), np.inf)
        ok = np.isfinite(uv).all(1) & (z > 0.1)
        u = np.round(uv[ok, 0]).astype(int)
        v = np.round(uv[ok, 1]).astype(int)
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        np.minimum.at(d, (v[inside], u[inside]), z[ok][inside])
        depth.append(d)
    depth = np.stack(depth)
    front = depth.argmin(0)
    kernel = np.ones((5, 5), np.uint8)
    return [cv2.morphologyEx((np.isfinite(depth[k]) & (front == k)).astype(np.uint8) * 255,
                             cv2.MORPH_CLOSE, kernel) for k in range(3)]


@pytest.fixture(scope="module")
def rig3():
    cams = _make_cameras()
    pts = _cylinder_points()
    return cams, [[_render3(c, pts, t) for t in range(T)] for c in cams]


def test_multi_three_people(rig3):
    """A 3-cycle in one camera mid-clip and a camera with two ids exchanged
    from frame 0 are both repaired; everything else is left alone."""
    cams, masks = rig3
    cyc, inv = 3, 7
    per_cam = []
    for i, fm in enumerate(masks):
        if i == cyc:
            fm = [[m[2], m[0], m[1]] if SWAP_LO <= t <= SWAP_HI else m for t, m in enumerate(fm)]
        if i == inv:
            fm = [[m[1], m[0], m[2]] for m in fm]
        per_cam.append(fm)
    clean = rig_vote.compute_rig_vote(cams, _multi_masks(masks, 3), T, CFG,
                                            log=lambda s: None)
    assert not clean.swaps.any()
    res = rig_vote.compute_rig_vote(cams, _multi_masks(per_cam, 3), T, CFG,
                                          log=lambda s: None)
    perms = res.perms
    # applying the decided permutation must restore every camera's true ids
    for i, fm in enumerate(per_cam):
        for t in range(T):
            p = perms[res.states[i, t]]
            restored = [None] * 3
            for k in range(3):
                restored[p[k]] = fm[t][k]
            truth = masks[i][t]
            if res.merged[t]:
                continue
            if i == cyc and t in (SWAP_LO - 1, SWAP_LO, SWAP_HI, SWAP_HI + 1):
                continue   # boundary frames: switch placement may be off by one
            assert all(np.array_equal(restored[k], truth[k]) for k in range(3)), (i, t)


def test_multi_skips_outside_subject_range(tmp_path, rig):
    cams, masks = rig
    for c in cams:
        with MaskArchiveWriter(str(tmp_path / c.name / "masks")) as w:
            w.write("mask_0000_01.png", masks[0][0][0])
    res = rig_vote.run_rig_vote(str(tmp_path), _cam_data(cams), CFG, log=lambda s: None)
    assert "subjects" in res.skipped


def test_permute_mask_ids_cycle(tmp_path):
    imgs = {f"mask_0001_{k:02d}.png": _dummy(k) for k in (1, 2, 3)}
    with MaskArchiveWriter(str(tmp_path), packed=True) as w:
        for n, img in imgs.items():
            w.write(n, img)
    assert rig_vote.permute_mask_ids(str(tmp_path), {1: (1, 2, 0)}) == 3
    got = _read_all(tmp_path)
    assert np.array_equal(got["mask_0001_02.png"], imgs["mask_0001_01.png"])
    assert np.array_equal(got["mask_0001_03.png"], imgs["mask_0001_02.png"])
    assert np.array_equal(got["mask_0001_01.png"], imgs["mask_0001_03.png"])


# --------------------------------------------------------------------------
# optimised vote against a straightforward reference (random small scenes)
# --------------------------------------------------------------------------

def _reference_evidence(index, grids, planes, n):
    """frame_evidence written out directly: every label's votes on every
    voxel, every camera's leave-one-out cores label by label."""
    C = len(planes)
    M = np.zeros((C, n, n), np.float64)
    cores = np.full((n, 4), np.nan)
    has = np.array([any(len(p) for p in pl) for pl in planes])
    count = (lambda ps: np.bincount(np.concatenate(ps), minlength=index.n)
             if ps else np.zeros(index.n, np.int64))
    votes = np.stack([count([index.voxels_of(i, pl[k]) for i, pl in enumerate(planes) if has[i]])
                      for k in range(n)])
    fg = count([index.voxels_of(i, np.unique(np.concatenate(pl)))
                for i, pl in enumerate(planes) if has[i]])

    def core_of(occ, v, k):
        others = np.delete(v, k, axis=0).max(0)
        return occ & (v[k] >= rig_vote.CORE_VOTES) & (v[k] >= rig_vote.CORE_RATIO * others)

    cand = np.flatnonzero(fg >= rig_vote.MIN_VOTES)
    if cand.size == 0:
        return M, cores
    gathered = index.idx[:, cand]
    ne = ((gathered < index.cells[:, None]) & has[:, None]).sum(0)
    fgc, vc = fg[cand], votes[:, cand]
    occ = fgc >= np.maximum(rig_vote.MIN_VOTES, rig_vote.OCC_FRAC * ne)
    for k in range(n):
        core = core_of(occ, vc, k)
        if core.sum() * index.voxel_m3 >= rig_vote.CORE_MIN_M3:
            cores[k, :3] = index.V[cand[core]].mean(0)
            cores[k, 3] = core.sum()
    kernel = np.ones((3, 3), np.uint8)
    for i, pl in enumerate(planes):
        if not has[i]:
            continue
        gh, gw = grids[i]
        L = np.zeros(gh * gw + 1, np.uint8)
        for k, cells_k in enumerate(pl):
            L[cells_k] |= np.uint8(1 << k)
        L[-1] = 255
        g = L[gathered[i]]
        seen = g != 255
        g = np.where(seen, g, 0)
        others = vc - np.stack([(g >> k) & 1 for k in range(n)]).astype(np.int64)
        occ_i = (fgc - (g > 0)) >= np.maximum(rig_vote.MIN_VOTES, rig_vote.OCC_FRAC * (ne - seen))
        regions = []
        for k in range(n):
            img = np.zeros(gh * gw + 1, np.uint8)
            img[gathered[i][core_of(occ_i, others, k)]] = 1
            regions.append(cv2.dilate(img[:-1].reshape(gh, gw), kernel).ravel().astype(bool))
        cover = np.sum(regions, axis=0)
        sizes = [len(p) for p in pl]
        big = max(max(sizes), 1)
        for k, own in enumerate(pl):
            if not len(own):
                continue
            share = np.array([(regions[j] & (cover == 1))[own].mean() for j in range(n)])
            order = np.argsort(-share, kind="stable")
            for j in range(n):
                alt = share[order[1]] if j == order[0] else share[order[0]]
                M[i, k, j] = len(own) / big * (share[j] - alt)
        if min(sizes) < rig_vote.FRAGMENT_SHARE * big:
            M[i] *= rig_vote.INCOMPLETE_WEIGHT
    return M, cores


def _random_scene(seed, n, frames=6):
    """Per frame, per camera, n masks at the reduced grid: projected people
    plus label permutations, dropped masks, fragments, overlaps and noise,
    and a camera without masks."""
    rng = np.random.default_rng(seed)
    cams = _make_cameras()
    gh, gw = round(H / DS), round(W / DS)
    pts = _cylinder_points(step=0.06)
    out = []
    for _ in range(frames):
        centres = rng.uniform(-1.2, 1.2, (n, 2))
        planes = []
        for i, cam in enumerate(cams):
            pl = []
            for c in centres:
                uv, z = cam.project(pts + np.array([c[0], c[1], 0.0]))
                ok = np.isfinite(uv).all(1) & (z > 0.1)
                col = np.floor(uv[ok, 0] * gw / W).astype(int)
                row = np.floor(uv[ok, 1] * gh / H).astype(int)
                inside = (col >= 0) & (col < gw) & (row >= 0) & (row < gh)
                pl.append(np.unique(row[inside] * gw + col[inside]).astype(np.uint32))
            pl = [pl[k] for k in rng.permutation(n)]
            r = rng.random(4)
            if r[0] < 0.15:
                pl[rng.integers(n)] = np.zeros(0, np.uint32)
            if r[1] < 0.15 and len(pl[0]):
                pl[1] = pl[0][: max(1, len(pl[0]) // 20)]
            if r[2] < 0.15:
                pl[0] = np.union1d(pl[0], pl[1]).astype(np.uint32)
            if r[3] < 0.3:
                pl[-1] = np.union1d(pl[-1], rng.integers(0, gh * gw, 30)).astype(np.uint32)
            planes.append(pl if i != 5 else [np.zeros(0, np.uint32)] * n)
        out.append(planes)
    grids = [(gh, gw)] * len(cams)
    index = rig_vote.VoxelIndex(cams, grids, rig_vote.grid_points(np.array([-2.0, -2.0, 0.0]),
                                                                  np.array([2.0, 2.0, 2.0]), 0.08), 0.08)
    return index, grids, out


@pytest.mark.parametrize("seed,n", [(0, 2), (1, 3), (2, 4), (3, 6)])
def test_frame_evidence_matches_reference(seed, n):
    index, grids, frames = _random_scene(seed, n)
    counter = rig_vote.ForegroundCounter(index)
    checked = 0
    for planes in frames:
        Mr, cr = _reference_evidence(index, grids, planes, n)
        fg = counter.update(planes)
        assert np.array_equal(fg, rig_vote._foreground(index, planes, rig_vote._has_masks(planes)))
        for got in (rig_vote.frame_evidence(index, grids, planes, n),
                    rig_vote.frame_evidence(index, grids, planes, n, fg)):
            assert got[0].tobytes() == Mr.tobytes()
            assert got[1].tobytes() == cr.tobytes()
        checked += bool(np.isfinite(cr[:, 3]).any())
    assert checked, "scenes must produce vote cores"


def test_label_votes_and_top_two_match_reference():
    rng = np.random.default_rng(7)
    g = rng.integers(0, 64, (300, 500)).astype(np.uint8)      # more cameras than one byte lane holds
    ref = np.stack([((g >> k) & 1).sum(0) for k in range(6)])
    assert np.array_equal(rig_vote._label_votes(g, 6), ref)
    votes = rng.integers(0, 4, (5, 2000)).astype(np.int16)     # small counts: many ties
    m1, a1, m2, t2 = rig_vote._top_two(votes)
    for v in range(votes.shape[1]):
        col = votes[:, v]
        k = int(np.argmax(col))
        rest = np.delete(col, k)
        assert (m1[v], a1[v], m2[v]) == (col.max(), k, rest.max())
        assert t2[v] == sum(1 << j for j in range(5) if j != k and col[j] == rest.max())
