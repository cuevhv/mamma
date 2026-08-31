import torch


def triangulate_batch(pts2d_list, Ks_list, extrs_list,
                      vis_list=None, img_size_px=512,
                      use_uncertainties=True, channel_is_logvar=False,
                      vis_thresh=0.0, clamp_sigma=(1.0, 24.0),
                      weights_list=None):
    """
    pts2d_list: list of [T, N, 2 or 3] (pixel coords)
    Ks_list: list of [T,3,3]
    extrs_list: list of [T,4,4]
    vis_list: list of [T,N] or None
    weights_list: optional list of [T,N] per-point multipliers
        folded into each observation's weight — a near-zero weight makes
        that view's point count ~nothing in the SVD without changing the
        validity mask (so the >=2-views rule still sees it as an observation).
    Returns:
        pts3d_world [T,N,3], valid_mask [T,N]
    """
    print("Triangulating batch with {} views ...".format(len(pts2d_list)))
    device = Ks_list[0].device
    dtype  = Ks_list[0].dtype
    n_cams = len(pts2d_list)
    T, N = pts2d_list[0].shape[:2]

    Ps = []
    for c in range(n_cams):
        K, E = Ks_list[c], extrs_list[c]
        Ps.append(K @ E[:, :3, :])                          # [T,3,4]
    Ps = torch.stack(Ps, dim=1)                             # [T,n_cams,3,4]

    pts2d = torch.stack(pts2d_list, dim=1)                  # [T,n_cams,N,2 or 3]
    vis   = torch.stack(vis_list, dim=1) if vis_list is not None else \
            torch.ones(T, n_cams, N, device=device, dtype=dtype)

    valid = (vis > vis_thresh) & (pts2d[..., :2].abs().sum(-1) > 0)
    valid_f = valid.float()

    if use_uncertainties and pts2d.shape[-1] == 3:
        third = pts2d[..., 2]
        sigma = torch.exp(0.5 * third) * (img_size_px / 2.0) if channel_is_logvar else \
                third * (img_size_px / 2.0)
        sigma = sigma.clamp(*clamp_sigma)
        w = 1.0 / (sigma * sigma)
    else:
        w = torch.ones_like(valid_f)
    if weights_list is not None:
        w = w * torch.stack(weights_list, dim=1).to(device=w.device, dtype=w.dtype)
    w = w * valid_f
    w_sqrt = torch.sqrt(w + 1e-12)

    u, v = pts2d[..., 0], pts2d[..., 1]
    P0, P1, P2 = Ps[..., 0, :], Ps[..., 1, :], Ps[..., 2, :]

    r1 = (u.unsqueeze(-1) * P2.unsqueeze(2) - P0.unsqueeze(2)) * w_sqrt.unsqueeze(-1)
    r2 = (v.unsqueeze(-1) * P2.unsqueeze(2) - P1.unsqueeze(2)) * w_sqrt.unsqueeze(-1)
    A = torch.cat([r1, r2], dim=1).permute(0, 2, 1, 3).contiguous()  # [T,N,2n,4]

    # Flatten batch for a single SVD call
    TN, M, D = T*N, A.shape[2], 4
    A_flat = A.view(TN, M, D)
    # SVD(A): A = U S Vh, right singular vector = last row of Vh
    # do svd in cpu because it's faster
    A_flat = A_flat.cpu()
    U, S, Vh = torch.linalg.svd(A_flat, full_matrices=False)
    Xh = Vh[:, -1, :]                                       # [TN, 4]
    Xh = Xh / (Xh[:, 3:4] + 1e-8)
    Xw = Xh[:, :3].view(T, N, 3)
    Xw = Xw.to(device=device, dtype=dtype)

    cam_count = valid.sum(dim=1)
    ok = cam_count >= 2
    Xw = Xw.masked_fill(~ok.unsqueeze(-1), 0.0)
    print("Triangulation done.")
    return Xw, ok


def triangulate_batch_irls(pts2d_list, Ks_list, extrs_list, vis_list=None,
                           iters=2, r0_px=20.0, img_whs=None, **kw):
    """Robust (IRLS) triangulation: solve once with equal weights, then re-solve
    with per-observation Cauchy weights ``1/(1+(r/r0)^2)`` where ``r`` is each
    observation's reprojection residual against the current consensus. Weights
    derive from CROSS-VIEW AGREEMENT only: a view that barely sees a person
    disagrees with the consensus by hundreds of px and self-suppresses.

    ``r0_px`` is defined at the image diagonal of the rig the default was
    validated on (2056x1504). When ``img_whs`` — a list of per-camera
    (width, height) — is given, r0 is rescaled per camera by its actual
    diagonal, making the weighting resolution-independent. Without it,
    ``r0_px`` is used as-is.
    """
    ref_diag = (2056.0 ** 2 + 1504.0 ** 2) ** 0.5
    r0s = [r0_px] * len(pts2d_list) if img_whs is None else \
          [r0_px * float(w * w + h * h) ** 0.5 / ref_diag for w, h in img_whs]
    Xw, ok = triangulate_batch(pts2d_list, Ks_list, extrs_list, vis_list=vis_list, **kw)
    for _ in range(int(iters)):
        Xh = torch.cat([Xw, torch.ones_like(Xw[..., :1])], dim=-1)      # [T,N,4]
        weights = []
        for c in range(len(pts2d_list)):
            P = Ks_list[c] @ extrs_list[c][:, :3, :]                    # [T,3,4]
            proj = torch.einsum("tij,tnj->tni", P, Xh)
            xy = proj[..., :2] / (proj[..., 2:3].abs() + 1e-8)
            r = (xy - pts2d_list[c][..., :2]).norm(dim=-1)              # [T,N] px
            observed = pts2d_list[c][..., :2].abs().sum(-1) > 0
            weights.append((1.0 / (1.0 + (r / r0s[c]) ** 2)) * observed)
        Xw, ok = triangulate_batch(pts2d_list, Ks_list, extrs_list, vis_list=vis_list,
                                   weights_list=weights, **kw)
    return Xw, ok


# ─────────────────────────────────────────────────────────────────────────────
# Temporal outlier gate on a triangulated 3D trajectory (opt-in, --tri-outlier-gate)
#
# Bad masks / re-ID occasionally make a majority of cameras agree on a wrong 3D
# point for a few isolated frames. The optimizer then locks onto it and the body
# "teleports" metres away and back (a one/two-frame flicker). These pure helpers
# detect such transient excursions on the triangulated trajectory and report the
# frame runs to bridge, so the caller can repair the seed and drop the offending
# 2D before optimizing. No optimizer state here, so they unit-test in isolation.


def _masked_median_center(X, valid_mask):
    """Per-frame robust body centre: median over the frame's valid landmarks.

    X:          [T, N, 3] triangulated world points.
    valid_mask: [T, N] bool.
    Returns (centers [T,3], usable [T] bool); usable[t] is False when frame t has
    no valid landmark (centre left at 0 there and ignored by callers).
    """
    T = X.shape[0]
    centers = X.new_zeros((T, 3))
    usable = torch.zeros(T, dtype=torch.bool, device=X.device)
    for t in range(T):
        m = valid_mask[t]
        if bool(m.any()):
            centers[t] = X[t][m].median(dim=0).values
            usable[t] = True
    return centers, usable


def _rolling_median_center(centers, usable, half_window):
    """Temporal median of `centers` over a centred window, ignoring unusable frames.
    The window median tracks genuine locomotion, so sustained motion is never
    treated as an outlier (only transient deviations from the local trend are)."""
    T = centers.shape[0]
    out = centers.clone()
    for t in range(T):
        lo, hi = max(0, t - half_window), min(T, t + half_window + 1)
        sel = usable[lo:hi]
        if bool(sel.any()):
            out[t] = centers[lo:hi][sel].median(dim=0).values
    return out


def detect_translation_outlier_runs(X, valid_mask, k=8.0, abs_floor=0.30,
                                    half_window=7, max_gap=4):
    """Detect transient global-translation outliers on a triangulated trajectory.

    A frame is flagged when its robust centre deviates from the local temporal
    median by more than ``thr = max(abs_floor, k * MAD(step))``, where MAD(step)
    is the median absolute frame-to-frame step of that body (a robust per-body
    motion scale). Only *transient* flagged runs (length <= max_gap) bounded on
    both sides by usable, non-flagged frames are returned as interpolatable gaps.

    Returns (runs, flagged, info):
      runs:    list of (a, b) inclusive index ranges safe to bridge.
      flagged: [T] bool of every frame over threshold.
      info:    {'thr', 'mad', 'n_flagged'} diagnostics.
    """
    centers, usable = _masked_median_center(X, valid_mask)
    med = _rolling_median_center(centers, usable, half_window)
    dev = torch.linalg.norm(centers - med, dim=1)                      # [T]

    step = torch.linalg.norm(centers[1:] - centers[:-1], dim=1)        # [T-1]
    pair_ok = usable[1:] & usable[:-1]
    if bool(pair_ok.any()):
        s = step[pair_ok]
        mad = float((s - s.median()).abs().median())
    else:
        mad = 0.0
    thr = max(float(abs_floor), float(k) * mad)

    flagged = (dev > thr) & usable
    runs = []
    T = X.shape[0]
    t = 0
    while t < T:
        if bool(flagged[t]):
            a = t
            while t < T and bool(flagged[t]):
                t += 1
            b = t - 1
            interpolatable = (
                (b - a + 1) <= max_gap
                and a - 1 >= 0 and b + 1 < T
                and bool(usable[a - 1]) and bool(usable[b + 1])
                and not bool(flagged[a - 1]) and not bool(flagged[b + 1])
            )
            # Confirmation: keep the run only if its frames genuinely leave the
            # straight bridge between the bounding frames. Sustained (near-linear)
            # motion lies on its own bridge, so this rejects the boundary false
            # positives the one-sided window median can produce during fast motion;
            # a real teleport is far off the bridge and survives.
            if interpolatable:
                c0, c1 = centers[a - 1], centers[b + 1]
                span = (b + 1) - (a - 1)
                excursion = 0.0
                for tt in range(a, b + 1):
                    w = (tt - (a - 1)) / span
                    pred = (1.0 - w) * c0 + w * c1
                    excursion = max(excursion, float(torch.linalg.norm(centers[tt] - pred)))
                if excursion > thr:
                    runs.append((a, b))
        else:
            t += 1
    return runs, flagged, {"thr": thr, "mad": mad, "n_flagged": int(flagged.sum())}


def interpolate_points_over_runs(X, runs):
    """Linear per-landmark interpolation of X:[T,N,3] across each (a,b) run, using
    the bounding frames X[a-1] and X[b+1]. Mutates X in place and returns it."""
    for a, b in runs:
        L = b - a + 1
        x0, x1 = X[a - 1], X[b + 1]                                    # [N,3] each
        for i in range(1, L + 1):
            w = i / (L + 1)
            X[a + i - 1] = (1.0 - w) * x0 + w * x1
    return X
