"""
Unit tests for the opt-in occlusion-aware gating.

Run from the optimization/ directory:  python -m pytest tests/test_occlusion_gating.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch

# Make `utils` and `losses` importable when invoked from anywhere.
_OPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)

from utils.occlusion_gating import compute_occlusion_gates  # noqa: E402
from losses.losses import proj_pts_loss  # noqa: E402


def _vis(*rows):
    """Build a list (per camera) of [T, N] visibility tensors from [N]-rows.

    Each positional arg is one camera's per-landmark visibility (single frame).
    """
    return [torch.tensor([r], dtype=torch.float32) for r in rows]


# --------------------------------------------------------------------------- #
# compute_occlusion_gates
# --------------------------------------------------------------------------- #

def test_monotonic_in_visibility():
    # One camera, visibilities increasing across landmarks -> gates increasing.
    vis = _vis([0.02, 0.1, 0.3, 0.6, 0.95])
    cfg = dict(enabled=True, mode="soft", tau=0.30, beta=0.05, temporal=False)
    gates = compute_occlusion_gates(vis, cfg)
    g = gates[0][0].numpy()
    assert np.all(np.diff(g) > 0), f"gate must be monotonic in visibility, got {g}"


def test_soft_gate_suppresses_below_tau():
    vis = _vis([0.05, 0.95])
    cfg = dict(enabled=True, mode="soft", tau=0.30, beta=0.05, temporal=False)
    gates = compute_occlusion_gates(vis, cfg)
    g = gates[0][0].numpy()
    assert g[0] < 1e-2, f"occluded landmark must be driven ~0, got {g[0]}"
    assert g[1] > 0.9, f"visible landmark keeps ~full weight, got {g[1]}"


def test_trim_hard_zeros_below_tau():
    # Two cameras so keep_top_m=1 protects the better one; the worse one is the
    # one we expect to be hard-zeroed below tau.
    vis = _vis([0.6, 0.05], [0.1, 0.9])  # cam0 landmarks, cam1 landmarks
    cfg = dict(enabled=True, mode="trim", tau=0.30, keep_top_m=1, temporal=False)
    gates = compute_occlusion_gates(vis, cfg)
    g0 = gates[0][0].numpy()
    g1 = gates[1][0].numpy()
    # landmark 0: cam0=0.6 (kept, >=tau), cam1=0.1 (below tau, not top) -> 0
    assert g0[0] == pytest.approx(0.6)
    assert g1[0] == pytest.approx(0.0)
    # landmark 1: cam1=0.9 kept, cam0=0.05 below tau and not top -> 0
    assert g1[1] == pytest.approx(0.9)
    assert g0[1] == pytest.approx(0.0)


def test_keep_top_m_never_fully_zeros_a_landmark():
    # All cameras see a landmark poorly (all below tau). keep_top_m=1 must keep
    # the single best view non-zero so the landmark is never fully unconstrained.
    vis = _vis([0.04, 0.0], [0.02, 0.0], [0.01, 0.0])  # 3 cams, landmark0 low-but-present
    cfg = dict(enabled=True, mode="hybrid", tau=0.30, beta=0.05, keep_top_m=1, temporal=False)
    gates = compute_occlusion_gates(vis, cfg)
    col0 = np.array([gates[c][0, 0].item() for c in range(3)])
    # The best view is restored to its full visibility; the rest are negligible.
    assert col0[0] == pytest.approx(0.04), f"best view kept at its visibility, got {col0}"
    assert col0.max() == pytest.approx(0.04)
    assert np.all(col0[1:] < 1e-3), f"non-top views must be negligible, got {col0}"


def test_temporal_uses_median_not_single_frame():
    # cam sees a landmark only in 1 of 5 frames -> median is low -> gated.
    T = 5
    v = torch.zeros(T, 1)
    v[2, 0] = 0.9  # a single high frame
    cfg = dict(enabled=True, mode="soft", tau=0.30, beta=0.05, temporal=True)
    gates = compute_occlusion_gates([v], cfg)
    g = gates[0][:, 0].numpy()
    assert np.all(g < 1e-2), f"a transiently-visible landmark stays gated, got {g}"


def test_soft_mode_has_no_safety_net():
    # soft mode: even the best (still-low) view is suppressed, no keep_top_m.
    vis = _vis([0.05], [0.04])
    cfg = dict(enabled=True, mode="soft", tau=0.30, beta=0.05, keep_top_m=1, temporal=False)
    gates = compute_occlusion_gates(vis, cfg)
    assert gates[0][0, 0].item() < 1e-2
    assert gates[1][0, 0].item() < 1e-2


# --------------------------------------------------------------------------- #
# proj_pts_loss integration (OFF == original behaviour)
# --------------------------------------------------------------------------- #

def _toy_scene(N=6, T=2, C=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts3d = torch.randn(T, N, 3, generator=g)
    pts3d[..., 2] += 5.0  # in front of cameras
    intr, extr, pts2d, ppw = [], [], [], []
    for _ in range(C):
        K = torch.eye(3); K[0, 0] = K[1, 1] = 500.0; K[0, 2] = K[1, 2] = 128.0
        Rt = torch.eye(4)
        intr.append(K.unsqueeze(0).repeat(T, 1, 1))
        extr.append(Rt.unsqueeze(0).repeat(T, 1, 1))
        # observed 2D = projection + noise, with a sigma channel
        Xc = pts3d
        proj = torch.matmul(Xc, K.T)
        proj = proj / proj[..., 2:3]
        obs = proj[..., :2] + torch.randn(T, N, 2, generator=g)
        sigma = torch.full((T, N, 1), 0.1)
        pts2d.append(torch.cat([obs, sigma], dim=-1))
        ppw.append(torch.rand(T, N, generator=g))
    return pts2d, pts3d, intr, extr, ppw


def test_off_equals_original_floor():
    pts2d, pts3d, intr, extr, ppw = _toy_scene()
    base = proj_pts_loss(pts2d, pts3d, intr, extr, loss="geman_mcclure",
                         per_point_weight=ppw, vis_clip_value=0.8, weight=20.0)
    # gating None and gating disabled must both equal the original.
    none = proj_pts_loss(pts2d, pts3d, intr, extr, loss="geman_mcclure",
                         per_point_weight=ppw, vis_clip_value=0.8,
                         occlusion_gating=None, weight=20.0)
    off = proj_pts_loss(pts2d, pts3d, intr, extr, loss="geman_mcclure",
                        per_point_weight=ppw, vis_clip_value=0.8,
                        occlusion_gating={"enabled": False}, weight=20.0)
    assert torch.allclose(base, none)
    assert torch.allclose(base, off)


def test_gating_reduces_low_visibility_contribution():
    # Make one camera fully occluded (vis ~0). With the floor it still gets
    # weight >= vis_clip_value; with gating it should be driven toward zero,
    # changing the total loss.
    pts2d, pts3d, intr, extr, ppw = _toy_scene()
    ppw = [w.clone() for w in ppw]
    ppw[0][:] = 0.02  # camera 0 occluded everywhere
    floor = proj_pts_loss(pts2d, pts3d, intr, extr, loss="geman_mcclure",
                          per_point_weight=ppw, vis_clip_value=0.8, weight=20.0)
    gated = proj_pts_loss(pts2d, pts3d, intr, extr, loss="geman_mcclure",
                          per_point_weight=ppw, vis_clip_value=0.8,
                          occlusion_gating={"enabled": True, "mode": "hybrid",
                                              "tau": 0.30, "beta": 0.05,
                                              "keep_top_m": 1, "temporal": True},
                          weight=20.0)
    assert not torch.allclose(floor, gated), "gating must change the loss for occluded views"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
