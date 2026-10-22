import numpy as np
import torch
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment
# ------------------------------------------------------------
# Geometry helpers (epipolar, fundamental matrix)
# ------------------------------------------------------------

def _skew(v):
    """v: [...,3] -> [...,3,3] skew-symmetric matrix."""
    O = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack([
        torch.stack([ O, -vz,  vy], dim=-1),
        torch.stack([ vz,  O, -vx], dim=-1),
        torch.stack([-vy,  vx,  O], dim=-1),
    ], dim=-2)


def _fundamental_from_world2cam_batch(Ka, Ra, ta, Kb, Rb, tb):
    """
    Compute fundamental matrix F_ba from world->cam extrinsics and intrinsics.
    Supports broadcasting over leading batch dims (e.g., [T,3,3] / [T,3]).
    F_ba such that x_b^T F_ba x_a = 0 (homogeneous pixel coords).
    """
    # relative motion: a -> b in camera coordinates (broadcast over leading dims)
    R_ba = Rb @ Ra.transpose(-1, -2)  # [...,3,3]
    t_ba = tb - (R_ba @ ta.unsqueeze(-1)).squeeze(-1)  # [...,3]

    # essential and fundamental matrices
    E = _skew(t_ba) @ R_ba  # [...,3,3]
    Kb_inv_T = torch.linalg.inv(Kb).transpose(-1, -2)  # [...,3,3]
    Ka_inv   = torch.linalg.inv(Ka)                    # [...,3,3]
    F = Kb_inv_T @ E @ Ka_inv                          # [...,3,3]
    return F

def _fundamental_from_world2cam(Ka, Ra, ta, Kb, Rb, tb):
    """
    Compute fundamental matrix F_ba from world->cam extrinsics and intrinsics.
    F_ba such that x_b^T F_ba x_a = 0 (both in homogeneous pixel coords).
    """
    # relative motion: a -> b in camera coordinates
    R_ba = Rb @ Ra.transpose(-1, -2)         # [3,3]
    t_ba = tb - Rb @ Ra.transpose(-1, -2) @ ta  # [3]

    E = _skew(t_ba) @ R_ba                   # essential matrix
    F = torch.inverse(Kb).transpose(-1, -2) @ E @ torch.inverse(Ka)
    return F


def _stack_h(x):
    """x: [...,2] -> homogeneous [...,3]."""
    return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)


def _point_line_distance(x, l):
    """
    x: [...,3] point in homogeneous coords
    l: [...,3] line in homogeneous coords
    returns distance in pixels.
    """
    num = torch.abs((x * l).sum(dim=-1))      # |x^T l|
    den = torch.sqrt(l[..., 0]**2 + l[..., 1]**2 + 1e-8)
    return num / den


# ------------------------------------------------------------
# Geometric affinity (MVPose Eq. 3 + sigmoid mapping)
# ------------------------------------------------------------

def geometric_affinity_pose_pair(
        uv_a, vis_a,
        uv_b, vis_b,
        F_ba,
        geom_scale=50.0,
        dg_thresh=80.0,
        min_joint=5
    ):
    """
    Compute geometric affinity A_g between two 2D poses (dense landmarks or joints),
    as in Sec. 3.2 / Eq. (3) of the paper, but generalized to your dense landmarks.

    uv_a, uv_b: [L,2] in pixels
    vis_a, vis_b: [L] in [0,1]
    F_ba: [3,3] s.t. x_b^T F_ba x_a = 0

    Returns scalar affinity in [0,1].
    """
    device = uv_a.device
    L = uv_a.shape[0]

    # visibility mask: only landmarks seen reasonably well in both views
    mask = (vis_a > 0.5) & (vis_b > 0.5)
    if mask.sum() < min_joint:
        return torch.tensor(0.0, device=device)

    ua = uv_a[mask]
    ub = uv_b[mask]
    ua_h = _stack_h(ua)      # [M,3]
    ub_h = _stack_h(ub)      # [M,3]

    # epipolar line in b for each point in a: l_b = F_ba * x_a
    l_b = (F_ba @ ua_h.transpose(0, 1)).transpose(0, 1)   # [M,3]
    d_a2b = _point_line_distance(ub_h, l_b)

    # epipolar line in a for each point in b: l_a = F_ba^T * x_b
    F_ab = F_ba.transpose(0, 1)
    l_a = (F_ab @ ub_h.transpose(0, 1)).transpose(0, 1)   # [M,3]
    d_b2a = _point_line_distance(ua_h, l_a)

    # Dg: mean symmetric distance (Eq. (3) in the paper)
    Dg = 0.5 * (d_a2b.mean() + d_b2a.mean())

    # If geometry is too bad, cut it off (Eq. (2): if Dg > th, set affinity=0)
    if Dg > dg_thresh:
        return torch.tensor(0.0, device=device)

    # Map distance -> affinity in (0,1). Paper uses sigmoid; we use exp(-Dg/scale),
    # which is equivalent up to monotonic transform.
    A_g = torch.exp(-Dg / geom_scale)
    return A_g


# ------------------------------------------------------------
# Multi-view matching: pairwise Hungarian + cycle consistency
# ------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0]*n

    def find(self, x):
        px = self.parent[x]
        if px != x:
            self.parent[x] = self.find(px)
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _hungarian_on_affinity(A, bad_value=1e6):
    """
    A: [Ma,Mb] affinity in [0,1]
    Return row_inds, col_inds (np arrays) as in linear_sum_assignment,
    minimizing cost = 1 - A (with very bad entries for 0-affinity).
    """
    if isinstance(A, torch.Tensor):
        A_np = A.detach().cpu().numpy()
    else:
        A_np = A

    # convert to cost
    cost = 1.0 - A_np
    # if affinity is exactly 0 => discourage match by large cost
    cost[A_np <= 0.0] = bad_value
    row, col = linear_sum_assignment(cost)
    return row, col


# ------------------------------------------------------------
# Main: MVPose-style association + triangulation on your data
# ------------------------------------------------------------

def mvpose_style_associate_and_triangulate(
        poses, pts2d, pts2d_vis_weight,
        K, extrinsics,
        triangulate_batch_fn,
        skip_first_t_frames=5,
        only_ten_first=False,
        img_size_px=512,
        channel_is_logvar=False,
        geom_scale=50.0,
        dg_thresh=80.0,
        min_visible_L=10,
        min_geom_joints=5,
        affinity_match_thresh=0.4,
        min_views_for_group=2
    ):
    """
    MVPose-style multi-view, multi-person association + 3D triangulation for your data.

    Inputs from `self` (as in your triangulate_3d_from_2d_points):
      - pts2d[body_id][cam]:        [T, L, D] (D=2 or 3; our code uses the first 2 dims)
      - pts2d_vis_weight[body_id][cam]: [T, L]
      - K[cam]:                    [T, 3,3]
      - extrinsics[cam]:           [T, 4,4] (world->cam)

    `triangulate_batch_fn` should have signature like your:
      triang_funcs.triangulate_batch(pts2d_list, Ks, EX,
                                     vis_list=None,
                                     use_uncertainties=False,
                                     img_size_px=512,
                                     channel_is_logvar=False)

    Returns:
      groups_per_t: dict[t] -> list of groups; each group is:
          {
            "views":    {view_id: det_local_idx},
            "body_ids": {view_id: original_body_id}
          }
      triX_per_t:   dict[t] -> list of [L,3] tensors (3D points per group)
      valid_per_t:  dict[t] -> list of [L] bool tensors (valid mask per group)
    """
    L = pts2d[0][0].shape[1]  # number of landmarks
    l_hand_idx = [3, 5, 7, 8, 10, 12, 18, 20, 27, 37]
    r_hand_idx = [259, 261, 263, 264, 266, 268, 274, 276, 283, 293]
    valid_idx = list([i for i in range(L) if i not in l_hand_idx and i not in r_hand_idx])
    device = K[0].device
    dtype  = K[0].dtype

    body_ids = list(poses.keys())   # consistent with your usage
    Ncams    = len(K)
    T_total  = K[0].shape[0]

    # frame range as in your code
    if only_ten_first:
        start_frame = 5 + skip_first_t_frames
        end_frame   = min(590 + skip_first_t_frames, T_total)
    else:
        start_frame = skip_first_t_frames
        end_frame   = T_total

    groups_per_t = {}
    triX_per_t   = {}
    valid_per_t  = {}

    for t in range(start_frame, end_frame):
        # ----------------------------------------------------
        # 1) Build detections per view for this frame (Sec 3.1)
        #    Here, each (body_id, view) with enough visible landmarks
        #    is treated as a 2D "pose" / detection.
        # ----------------------------------------------------
        detections_per_view = [[] for _ in range(Ncams)]

        for v in range(Ncams):
            for bid in body_ids:
                uv_full  = pts2d[bid][v][:, valid_idx]            # [T,L,D]
                vis_full = pts2d_vis_weight[bid][v][:, valid_idx] # [T,L]

                uv  = uv_full[t]                         # [L,D]
                vis = vis_full[t]                        # [L]

                # require enough visible landmarks
                if (vis > 0.5).sum().item() < min_visible_L:
                    continue

                # ignore purely empty predictions
                if uv[..., :2].abs().sum() == 0:
                    continue

                detections_per_view[v].append({
                    "body_id": bid,
                    "uv": uv.to(device=device, dtype=dtype),
                    "vis": vis.to(device=device, dtype=torch.float32),
                })

        total_dets = sum(len(dv) for dv in detections_per_view)
        if total_dets == 0:
            groups_per_t[t] = []
            triX_per_t[t]   = []
            valid_per_t[t]  = []
            continue

        # ----------------------------------------------------
        # 2) Assign a global node index to each detection
        #    (for cycle-consistent matching using UnionFind)
        # ----------------------------------------------------
        view_det_to_node = {}
        node_to_view_det = []
        node_counter = 0

        for v in range(Ncams):
            view_det_to_node[v] = {}
            for i in range(len(detections_per_view[v])):
                view_det_to_node[v][i] = node_counter
                node_to_view_det.append((v, i))
                node_counter += 1

        uf = UnionFind(node_counter)

        # ----------------------------------------------------
        # 3) For each pair of views (i,j):
        #    - compute geometric affinity matrix A_ij (Sec 3.2)
        #    - run Hungarian on cost = 1 - A_ij
        #    - keep good matches, add edges for union-find
        # ----------------------------------------------------
        for a in range(Ncams):
            for b in range(a+1, Ncams):
                dets_a = detections_per_view[a]
                dets_b = detections_per_view[b]
                Ma, Mb = len(dets_a), len(dets_b)
                if Ma == 0 or Mb == 0:
                    continue

                Ka = K[a][t]              # [3,3]
                Ea = extrinsics[a][t]     # [4,4]
                Ra, ta = Ea[:3, :3], Ea[:3, 3]

                Kb = K[b][t]
                Eb = extrinsics[b][t]
                Rb, tb = Eb[:3, :3], Eb[:3, 3]

                F_ba = _fundamental_from_world2cam(Ka, Ra, ta, Kb, Rb, tb)

                A_ab = torch.zeros((Ma, Mb), device=device, dtype=dtype)

                # compute geometric affinity for every pair of detections
                for i in range(Ma):
                    uv_i  = dets_a[i]["uv"][..., :2]   # [L,2]
                    vis_i = torch.ones_like(dets_a[i]["vis"])           # [L]

                    for j in range(Mb):
                        uv_j  = dets_b[j]["uv"][..., :2]
                        vis_j = torch.ones_like(dets_b[j]["vis"])

                        A_g = geometric_affinity_pose_pair(
                            uv_i, vis_i,
                            uv_j, vis_j,
                            F_ba,
                            geom_scale=geom_scale,
                            dg_thresh=dg_thresh,
                            min_joint=min_geom_joints
                        )

                        # In the original paper, they combine appearance and geometry
                        # as A_ij = sqrt(Aa_ij * Ag_ij) if Dg <= th, else 0 (Eq. 2).
                        # Here we don't have appearance, so treat Aa_ij = 1:
                        A_ab[i, j] = A_g

                if A_ab.max() <= 0:
                    continue

                # Hungarian on cost = 1 - affinity (partial permutation)
                row_ind, col_ind = _hungarian_on_affinity(A_ab)

                # keep only high-affinity matches, and add edges for union-find
                for i, j in zip(row_ind, col_ind):
                    if A_ab[i, j] >= affinity_match_thresh:
                        ni = view_det_to_node[a][i]
                        nj = view_det_to_node[b][j]
                        uf.union(ni, nj)

        # ----------------------------------------------------
        # 4) Extract cycle-consistent clusters (connected components)
        #    Each cluster = 2D poses of same person across views (Fig. 2c in paper)
        # ----------------------------------------------------
        clusters = {}
        for node in range(node_counter):
            root = uf.find(node)
            clusters.setdefault(root, []).append(node)

        groups = []
        triX_list = []
        valid_list = []

        for root, nodes in clusters.items():
            group_views_to_det = {}
            group_views_to_bid = {}

            for node in nodes:
                v, di = node_to_view_det[node]
                if v not in group_views_to_det:
                    group_views_to_det[v] = di
                    group_views_to_bid[v] = detections_per_view[v][di]["body_id"]
                # if multiple detections from same view in one cluster, we keep first

            if len(group_views_to_det) < min_views_for_group:
                continue  # not enough views for robust triangulation

            # ------------------------------------------------
            # 5) Triangulate this cluster (Sec 3.3: reconstruct
            #    3D pose from multi-view 2D pose; we call your
            #    triangulate_batch_fn instead of 3DPS).
            # ------------------------------------------------
            # Use any view in group to define L, D
            any_v  = next(iter(group_views_to_det.keys()))
            any_di = group_views_to_det[any_v]
            uv_any = detections_per_view[any_v][any_di]["uv"]  # [L,D]
            L = uv_any.shape[0]
            D = uv_any.shape[1]

            pts2d_list = []
            vis_list   = []
            Ks_batch   = []
            EX_batch   = []

            for v in range(Ncams):
                if v in group_views_to_det:
                    di  = group_views_to_det[v]
                    det = detections_per_view[v][di]
                    uv  = det["uv"]                   # [L,D]
                    vis = det["vis"]                  # [L]
                    pts2d_list.append(uv.view(1, L, D))
                    vis_list.append(vis.view(1, L))
                else:
                    pts2d_list.append(torch.zeros((1, L, D), device=device, dtype=dtype))
                    vis_list.append(torch.zeros((1, L), device=device, dtype=torch.float32))

                Kt  = K[v][t]
                Ext = extrinsics[v][t]
                Ks_batch.append(Kt.view(1, 3, 3))
                EX_batch.append(Ext.view(1, 4, 4))

            # Now call your triangulator
            Xw, valid = triangulate_batch_fn(
                pts2d_list, Ks_batch, EX_batch,
                vis_list=vis_list,
                use_uncertainties=(D == 3),
                img_size_px=img_size_px,
                channel_is_logvar=channel_is_logvar
            )   # expect [1,L,3], [1,L]

            triX_list.append(Xw[0])     # [L,3]
            valid_list.append(valid[0]) # [L]

            groups.append({
                "views":    group_views_to_det,  # view -> local det idx
                "body_ids": group_views_to_bid,  # view -> original body_id
            })

    return groups_per_t, triX_per_t, valid_per_t

def geometric_affinity_trajectory_pair(
        uv_a_T, vis_a_T,   # [T', L, 2], [T', L]
        uv_b_T, vis_b_T,   # [T', L, 2], [T', L]
        K_a_T, E_a_T,      # [T',3,3], [T',4,4]
        K_b_T, E_b_T,      # [T',3,3], [T',4,4]
        geom_scale=50.0,
        dg_thresh=80.0,
        min_joint=5,
        use_uncertainties=False,
        img_size_px=512
    ):
    """
    Compute geometric affinity Ag between two **trajectories** of 2D poses
    across time, by averaging the per-frame symmetric epipolar distance Dg(t).

    uv_a_T, uv_b_T: [T', L, 2] in pixels
    vis_a_T, vis_b_T: [T', L] in [0,1]
    K_*, E_*: per-frame camera intrinsics/extrinsics.

    Returns scalar affinity in [0,1].
    """
    device = uv_a_T.device
    T_slice, L, D = uv_a_T.shape

    # split XY and optional sigma
    ua_xy_T = uv_a_T[..., :2]   # [T,L,2]
    ub_xy_T = uv_b_T[..., :2]   # [T,L,2]

    # # Vectorized over time (batch the per-frame computation)
    # device = uv_a_T.device
    # T_slice = uv_a_T.shape[0]

    sigma_a_T = None
    sigma_b_T = None
    # use_uncertainties = False

    if use_uncertainties and D == 3:
        # Third channel encodes uncertainty; convert to sigma in pixels.
        third_a = uv_a_T[..., 2]   # [T,L]
        third_b = uv_b_T[..., 2]   # [T,L]

        # assume third channel is normalized sigma in [-1,1]
        # map to px: [−1,1] -> [0, img_size_px]
        sigma_a_T = ((third_a) * 0.5) * img_size_px
        sigma_b_T = ((third_b) * 0.5) * img_size_px
        # clamp to reasonable range to avoid insane weights
        sigma_a_T = sigma_a_T.clamp(min=1.0, )
        sigma_b_T = sigma_b_T.clamp(min=1.0, )


    # homogeneous coords: [T, L, 3]
    ua_h = _stack_h(ua_xy_T)    # [T, L, 3]
    ub_h = _stack_h(ub_xy_T)    # [T, L, 3]

    U, S, V = torch.linalg.svd(E_a_T[:, :3, :3], full_matrices=False)
    E_a_T[:, :3, :3] = U @ V
    U, S, V = torch.linalg.svd(E_b_T[:, :3, :3], full_matrices=False)
    E_b_T[:, :3, :3] = U @ V

    # per-frame camera extrinsics
    Ra_T = E_a_T[:, :3, :3]    # [T,3,3]
    ta_T = E_a_T[:, :3, 3]     # [T,3]
    Rb_T = E_b_T[:, :3, :3]
    tb_T = E_b_T[:, :3, 3]

    # compute batch fundamental matrices: F_ba [T,3,3]
    F_ba_T = _fundamental_from_world2cam_batch(K_a_T, Ra_T, ta_T, K_b_T, Rb_T, tb_T)

    # epipolar lines: l_b = F_ba * x_a, l_a = F_ba^T * x_b
    # ua_h.transpose(1,2): [T,3,L] -> matmul -> [T,3,L] -> transpose -> [T,L,3]
    l_b_T = (F_ba_T @ ua_h.transpose(1, 2)).transpose(1, 2)   # [T,L,3]
    F_ab_T = F_ba_T.transpose(-1, -2)
    l_a_T = (F_ab_T @ ub_h.transpose(1, 2)).transpose(1, 2)    # [T,L,3]

    # distances per-frame per-joint: [T,L]
    d_a2b_T = _point_line_distance(ub_h, l_b_T)
    d_b2a_T = _point_line_distance(ua_h, l_a_T)

    # visibility mask per-frame per-joint
    mask_T = (vis_a_T > 0.5) & (vis_b_T > 0.5)   # [T,L]
    valid_counts_per_frame = mask_T.sum(dim=1)   # [T]

    # frames with enough joints
    valid_frames = valid_counts_per_frame >= min_joint
    valid_count = int(valid_frames.sum().item())
    if valid_count == 0:
        return torch.tensor(0.0, device=device)

    # uncertainty-aware weights
    if use_uncertainties and sigma_a_T is not None and sigma_b_T is not None:
        # variance sum
        var_sum_T = sigma_a_T.pow(2) + sigma_b_T.pow(2)   # [T,L]
        # weight ∝ 1 / variance, zero where invisible
        w_T = mask_T.float() / var_sum_T.clamp(min=1.0)   # [T,L]
    else:
        # no uncertainty → uniform weights on visible joints
        w_T = mask_T.float()                              # [T,L]
    # masked mean per-frame (avoid div-by-zero by clamping denominator)

   # 7) Per-frame weighted symmetric distance
    #    First, symmetric distance per joint:
    d_sym_T = 0.5 * (d_a2b_T + d_b2a_T)                  # [T,L]

    # per-frame normalization over joints
    w_sum_per_frame = w_T.sum(dim=1)                     # [T]
    # frames that have at least some weight (implicit min_joint already above)
    frame_has_weight = w_sum_per_frame > 0
    valid_frames = valid_frames & frame_has_weight
    if not valid_frames.any():
        return torch.tensor(0.0, device=device)

    w_sum_per_frame = w_sum_per_frame.clamp(min=1e-6)
    Dg_per_frame = (d_sym_T * w_T).sum(dim=1) / w_sum_per_frame  # [T]

    # 8) Global Dg: weighted average over frames
    #    (frames with more confident joints contribute more)
    frame_weights = w_sum_per_frame[valid_frames]        # [T_valid]
    Dg_valid = Dg_per_frame[valid_frames]                # [T_valid]

    # normalized frame weights
    fw_norm = frame_weights / frame_weights.sum().clamp(min=1e-6)
    Dg = (Dg_valid * fw_norm).sum()

    # 9) Threshold & map to affinity
    if Dg > dg_thresh:
        return torch.tensor(0.0, device=device)

    A_g = torch.exp(-Dg / geom_scale)
    return A_g


def mvpose_style_associate_and_triangulate_temporal(
        poses, pts2d, pts2d_vis_weight,
        K, extrinsics,
        triangulate_batch_fn,
        skip_first_t_frames=5,
        only_ten_first=False,
        img_size_px=512,
        channel_is_logvar=False,
        geom_scale=50.0,
        dg_thresh=80.0,
        min_visible_L=10,       # threshold over all T×L vis
        min_geom_joints=5,
        affinity_match_thresh=0.4,
        min_views_for_group=2
    ):
    """
    Temporal version:

    Each (body_id, cam) is a **trajectory** with shape [T', L, D].
    We:
      - use the whole trajectory to compute geometric affinity between views
      - run Hungarian + union-find to get multi-view groups (persons)
      - triangulate 3D trajectories [T', L, 3] per group.

    Inputs:
      poses: dict[body_id] -> (unused here, just for ids)
      pts2d[body_id][cam]:        [T, L, D] (D=2 or 3)
      pts2d_vis_weight[body_id][cam]: [T, L]
      K[cam]:                    [T, 3,3]
      extrinsics[cam]:           [T, 4,4]

    Returns:
      groups:     list of groups; each group is:
          {
            "views":    {view_id: det_local_idx},
            "body_ids": {view_id: original_body_id}
          }
      triX_list:  list of [T', L, 3] tensors (3D trajectories per group)
      valid_list: list of [T', L] bool tensors (valid mask per group)
    """
    L_total = pts2d[0][0].shape[1]  # total number of landmarks

    # remove hands if you want, like before
    l_hand_idx = [3, 5, 7, 8, 10, 12, 18, 20, 27, 37]
    r_hand_idx = [259, 261, 263, 264, 266, 268, 274, 276, 283, 293]
    valid_idx = [i for i in range(L_total)
                 if i not in l_hand_idx and i not in r_hand_idx]

    device = K[0].device
    dtype  = K[0].dtype

    body_ids = list(poses.keys())
    Ncams    = len(K)
    T_total  = K[0].shape[0]

    # time window
    if only_ten_first:
        start_frame = 5 + skip_first_t_frames
        end_frame   = min(590 + skip_first_t_frames, T_total)
    else:
        start_frame = skip_first_t_frames
        end_frame   = T_total

    T_slice = end_frame - start_frame

    # --------------------------------------------------------
    # 1) Build detections per view: now **one detection = trajectory**
    # --------------------------------------------------------
    detections_per_view = [[] for _ in range(Ncams)]

    for v in range(Ncams):
        for bid in body_ids:
            # [T, L, D] -> [T_slice, L_valid, D]
            uv_full  = pts2d[bid][v][start_frame:end_frame, :] #[:, valid_idx]
            vis_full = pts2d_vis_weight[bid][v][start_frame:end_frame, :] #[:, valid_idx]

            # require enough visible points over the entire sequence
            if (vis_full > 0.5).sum().item() < min_visible_L:
                continue

            # ignore trajectories that are completely zero everywhere
            if uv_full[..., :2].abs().sum() == 0:
                continue

            detections_per_view[v].append({
                "body_id": bid,
                "uv_T": uv_full.to(device=device, dtype=dtype),                # [T',L,D]
                "vis_T": vis_full.to(device=device, dtype=torch.float32),      # [T',L]
            })

    total_dets = sum(len(dv) for dv in detections_per_view)
    if total_dets == 0:
        return [], [], []

    # --------------------------------------------------------
    # 2) Assign global node indices for union-find
    # --------------------------------------------------------
    view_det_to_node = {}
    node_to_view_det = []
    node_counter = 0

    for v in range(Ncams):
        view_det_to_node[v] = {}
        for i in range(len(detections_per_view[v])):
            view_det_to_node[v][i] = node_counter
            node_to_view_det.append((v, i))
            node_counter += 1

    uf = UnionFind(node_counter)

    # --------------------------------------------------------
    # 3) Pairwise view–view trajectory affinities + Hungarian
    # --------------------------------------------------------
    for a in range(Ncams):
        for b in range(a+1, Ncams):
            dets_a = detections_per_view[a]
            dets_b = detections_per_view[b]
            Ma, Mb = len(dets_a), len(dets_b)
            if Ma == 0 or Mb == 0:
                continue

            # camera params for the whole window [T_slice,...]
            K_a_T = K[a][start_frame:end_frame]           # [T',3,3]
            E_a_T = extrinsics[a][start_frame:end_frame]  # [T',4,4]

            K_b_T = K[b][start_frame:end_frame]
            E_b_T = extrinsics[b][start_frame:end_frame]

            A_ab = torch.zeros((Ma, Mb), device=device, dtype=dtype)

            for i in range(Ma):
                uv_a_T  = dets_a[i]["uv_T"][..., valid_idx, :]      # [T',L,2]
                vis_a_T = torch.ones_like(dets_a[i]["vis_T"][:, valid_idx])              # [T',L]
                vis_a_T = dets_a[i]["vis_T"][:, valid_idx]

                for j in range(Mb):
                    uv_b_T  = dets_b[j]["uv_T"][..., valid_idx, :]
                    vis_b_T = torch.ones_like(dets_b[j]["vis_T"][:, valid_idx])              # [T',L]
                    vis_b_T = dets_b[j]["vis_T"][:, valid_idx]

                    # print(f"Computing geometric affinity between trajs {i},{j} in views [{a}, {b}] ...")
                    A_g = geometric_affinity_trajectory_pair(
                        uv_a_T, vis_a_T,
                        uv_b_T, vis_b_T,
                        K_a_T, E_a_T,
                        K_b_T, E_b_T,
                        geom_scale=geom_scale,
                        dg_thresh=dg_thresh,
                        min_joint=min_geom_joints, # 100
                        use_uncertainties=False, #uv_b_T.shape[-1] == 3,
                        img_size_px=img_size_px,
                    )

                    A_ab[i, j] = A_g
            print(f"Affinity matrix between view {a} and {b}:")
            print(A_ab)

            if A_ab.max() <= 0:
                continue

            row_ind, col_ind = _hungarian_on_affinity(A_ab)

            for i, j in zip(row_ind, col_ind):
                if A_ab[i, j] >= 0.2: #affinity_match_thresh: # we have to check if we use 0.2 or affinity_match_thresh
                    ni = view_det_to_node[a][i]
                    nj = view_det_to_node[b][j]
                    uf.union(ni, nj)
                else:
                    print(f"Skipping match between trajs {i},{j} in views [{a}, {b}] due to low affinity {A_ab[i,j]:.3f}")

    # --------------------------------------------------------
    # 4) Extract multi-view, cycle-consistent clusters (persons)
    # --------------------------------------------------------
    clusters = {}
    for node in range(node_counter):
        root = uf.find(node)
        clusters.setdefault(root, []).append(node)

    groups = []
    triX_list = []
    valid_list = []

    # --------------------------------------------------------
    # 5) Triangulate full trajectories for each group
    # --------------------------------------------------------
    for root, nodes in clusters.items():
        group_views_to_det = {}
        group_views_to_bid = {}

        for node in nodes:
            v, di = node_to_view_det[node]
            if v not in group_views_to_det:
                group_views_to_det[v] = di
                group_views_to_bid[v] = detections_per_view[v][di]["body_id"]
            # if multiple detections from same view, we keep the first

        if len(group_views_to_det) < min_views_for_group:
            continue

        # shape from any member
        any_v  = next(iter(group_views_to_det.keys()))
        any_di = group_views_to_det[any_v]
        uv_any_T = detections_per_view[any_v][any_di]["uv_T"]  # [T',L,D]
        T_slice, L_valid, D = uv_any_T.shape

        pts2d_list = []
        vis_list   = []
        Ks_batch   = []
        EX_batch   = []

        for v in range(Ncams):
            if v in group_views_to_det:
                di  = group_views_to_det[v]
                det = detections_per_view[v][di]
                uv_T  = det["uv_T"]                   # [T',L,D]
                vis_T = det["vis_T"]                  # [T',L]
                pts2d_list.append(uv_T)               # [T',L,D]
                vis_list.append(vis_T)                # [T',L]
            else:
                pts2d_list.append(torch.zeros((T_slice, L_valid, D),
                                              device=device, dtype=dtype))
                vis_list.append(torch.zeros((T_slice, L_valid),
                                            device=device, dtype=torch.float32))

            K_T  = K[v][start_frame:end_frame]        # [T',3,3]
            EX_T = extrinsics[v][start_frame:end_frame]  # [T',4,4]
            Ks_batch.append(K_T)
            EX_batch.append(EX_T)

        Xw, valid = triangulate_batch_fn(
            pts2d_list, Ks_batch, EX_batch,
            vis_list=vis_list,
            use_uncertainties=(D == 3),
            img_size_px=img_size_px,
            channel_is_logvar=channel_is_logvar
        )  # expected [T',L,3], [T',L]

        triX_list.append(Xw)      # full trajectory
        valid_list.append(valid)

        groups.append({
            "views":    group_views_to_det,  # view -> local det index
            "body_ids": group_views_to_bid,  # view -> original body_id
        })

    print('finish temporal association: found', len(groups), 'groups.')
    print("views")
    for i, group in enumerate(groups):
        print(i, group['views'].values())
    print('body_ids')
    for i, group in enumerate(groups):
        print(i, group['body_ids'].values())

    groups, triX_list, valid_list, _ = propagate_ids_via_reprojection(
        groups, triX_list, valid_list,
        pts2d, pts2d_vis_weight, K, extrinsics,
        skip_first_t_frames=skip_first_t_frames,
        only_ten_first=only_ten_first,
        img_size_px=img_size_px,
        max_mean_reproj_error=60.0,
        min_time_overlap=0.2,
        triangulate_batch_fn=triangulate_batch_fn,
    )

    return groups, triX_list, valid_list


def animate_pointcloud_bodies_body_first(
    pts_seq,
    color_seq=None,
    out_html="sequence.html",
    colorscale="Viridis",
):
    """
    Animate multiple bodies' point clouds over time in 3D using Plotly.

    Parameters
    ----------
    pts_seq : list of tensors/arrays
        pts_seq[b] -> (T, N, 3)
    color_seq : list of tensors/arrays or None
        color_seq[b] -> (T, N) with values in [0, 1]
    out_html : str
        Output HTML filename.
    colorscale : str
        Plotly colorscale name.
    """

    B = len(pts_seq)
    if B == 0:
        raise ValueError("pts_seq is empty")

    T = pts_seq[0].shape[0]
    for b in range(1, B):
        if pts_seq[b].shape[0] != T:
            raise ValueError(f"Body {b} has different T={pts_seq[b].shape[0]} != {T}")

    if color_seq is not None and len(color_seq) != B:
        raise ValueError("color_seq must be a list with one entry per body")

    def to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        return np.asarray(x)

    def maybe_subsample(pts_np, colors_np=None):
        N = pts_np.shape[0]
        if N == 512:
            idx = np.concatenate(
                [np.arange(0, 150, dtype=np.int64),
                 np.arange(255, 255 + 150, dtype=np.int64)]
            )
            pts_np = pts_np[idx, :]
            if colors_np is not None:
                colors_np = colors_np[idx]
        return pts_np, colors_np

    # -------------------- Compute global min/max --------------------
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])
    for b in range(B):
        pts_flat = to_numpy(pts_seq[b]).reshape(-1, 3)
        mins = np.minimum(mins, pts_flat.min(axis=0))
        maxs = np.maximum(maxs, pts_flat.max(axis=0))
    padding = 0.02 * (maxs - mins + 1e-9)
    mins -= padding
    maxs += padding

    body_fixed_colors = [f"hsl({int(360 * i / B)}, 80%, 50%)" for i in range(B)]
    default_size = 4

    # -------------------- Initial frame --------------------
    init_data = []
    for b in range(B):
        pts = to_numpy(pts_seq[b][0])
        if color_seq is not None and color_seq[b] is not None:
            c = to_numpy(color_seq[b][0])
        else:
            c = None
        pts, c = maybe_subsample(pts, c)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        if c is not None:
            marker_dict = dict(
                size=default_size,
                opacity=0.8,
                color=c,
                cmin=0.0, cmax=1.0,
                colorscale=colorscale,
                showscale=(b == 0),
            )
        else:
            marker_dict = dict(
                size=default_size,
                opacity=0.8,
                color=body_fixed_colors[b],
            )
        init_data.append(
            go.Scatter3d(x=x, y=y, z=z, mode="markers", name=f"Body {b}", marker=marker_dict)
        )

    fig = go.Figure(data=init_data)

    # -------------------- Frames --------------------
    frames = []
    for t in range(T):
        frame_data = []
        for b in range(B):
            pts = to_numpy(pts_seq[b][t])
            if color_seq is not None and color_seq[b] is not None:
                c = to_numpy(color_seq[b][t])
            else:
                c = None
            pts, c = maybe_subsample(pts, c)
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            if c is not None:
                marker_dict = dict(
                    size=default_size,
                    opacity=0.8,
                    color=c,
                    cmin=0.0, cmax=1.0,
                    colorscale=colorscale,
                    showscale=False,
                )
            else:
                marker_dict = dict(
                    size=default_size,
                    opacity=0.8,
                    color=body_fixed_colors[b],
                )
            frame_data.append(
                go.Scatter3d(x=x, y=y, z=z, mode="markers",
                             name=f"Body {b}", marker=marker_dict)
            )
        frames.append(go.Frame(data=frame_data, name=str(t)))
    fig.frames = frames

    # -------------------- Slider for animation --------------------
    sliders = [{
        "steps": [
            {"method": "animate",
             "label": str(t),
             "args": [[str(t)], {"mode": "immediate",
                                 "frame": {"duration": 0, "redraw": True},
                                 "transition": {"duration": 0}}]}
            for t in range(T)
        ],
        "transition": {"duration": 0},
        "x": 0,
        "y": 0,
        "currentvalue": {"font": {"size": 16}, "prefix": "Frame: ", "visible": True},
        "len": 1.0,
    }]

    # -------------------- Interactive point size slider --------------------
    sizes = list(range(1, 11))  # point sizes from 1 to 10
    size_buttons = [
        {
            "label": str(s),
            "method": "restyle",
            "args": [{"marker.size": [s for _ in range(B)]}],
        }
        for s in sizes
    ]

    # -------------------- Layout --------------------
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[mins[0], maxs[0]], title="x"),
            yaxis=dict(range=[mins[1], maxs[1]], title="y"),
            zaxis=dict(range=[mins[2], maxs[2]], title="z"),
            aspectmode="cube",
        ),
        sliders=sliders,
        updatemenus=[
            # Animation buttons
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 1.05,
                "direction": "left",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 40, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 0}}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                    },
                ],
            },
            # Point size control
            {
                "buttons": size_buttons,
                "direction": "right",
                "x": 0.3,
                "y": 1.15,
                "showactive": True,
                "type": "buttons",
            },
        ],
        margin=dict(l=0, r=0, t=40, b=0),
        width=900,
        height=800,
        title="3D Point Cloud Animation with Adjustable Point Size",
    )

    fig.write_html(out_html)
    print(f"Saved interactive sequence to {out_html}")


def _project_world_to_image(Xw_TL3, K_T, E_T):
    """
    Xw_TL3: [T, L, 3] world coords
    K_T:    [T, 3, 3]
    E_T:    [T, 4, 4] (world -> cam)

    Returns:
        uv_TL2: [T, L, 2] in pixels
        z_TL:  [T, L] depth in camera
    """
    T_slice, L, _ = Xw_TL3.shape
    R_T = E_T[:, :3, :3]                    # [T,3,3]
    t_T = E_T[:, :3, 3].unsqueeze(1)        # [T,1,3]

    # world -> camera
    X_cam = Xw_TL3 @ R_T.transpose(1, 2) + t_T   # [T,L,3]

    # project
    uvw = X_cam @ K_T.transpose(1, 2)      # [T,L,3]
    z = uvw[..., 2].clamp(min=1e-6)        # [T,L]
    uv = uvw[..., :2] / z.unsqueeze(-1)    # [T,L,2]
    return uv, z


def propagate_ids_via_reprojection(
    groups,
    triX_list,
    valid_list,
    pts2d,                # dict[body_id][view] -> [T, L, D]
    pts2d_vis_weight,     # dict[body_id][view] -> [T, L]
    K,                    # list[view] -> [T, 3, 3]
    extrinsics,           # list[view] -> [T, 4, 4]
    poses=None,           # dict[body_id] (just to get body_ids if you want)
    skip_first_t_frames=5,
    only_ten_first=False,
    img_size_px=512,
    max_mean_reproj_error=25.0,  # px
    min_time_overlap=0.2,        # fraction of frames that must have valid points
    triangulate_batch_fn=None,
):
    """
    For each group/person, reproject its 3D trajectory to *all* views and try to
    assign an existing 2D track (body_id) in views that had nothing assigned.

    Inputs:
      groups:     list of dicts, as from mvpose_style_associate_and_triangulate_temporal:
                  {
                    "views":    {view_id: local_det_idx},   # we ignore local_det_idx here
                    "body_ids": {view_id: original_body_id}
                  }
      triX_list:  list[k] -> [T_slice, L_valid, 3]  (3D world coords per group)
      valid_list: list[k] -> [T_slice, L_valid] bool mask
      pts2d[body_id][view]: [T, L, D]  (D>=2)
      pts2d_vis_weight[body_id][view]: [T, L]
      K[view]: [T,3,3]
      extrinsics[view]: [T,4,4]
      skip_first_t_frames, only_ten_first: must match values used for triangulation

    Returns:
      updated_groups   (same list, but with more "body_ids" entries),
      assignments_info (optional debug dict)
    """
    print("Propagating IDs via reprojection...")
    device = K[0].device
    dtype  = K[0].dtype

    # figure out frame window used in triangulation
    T_total = K[0].shape[0]
    if only_ten_first:
        start_frame = 5 + skip_first_t_frames
        end_frame   = min(590 + skip_first_t_frames, T_total)
    else:
        start_frame = skip_first_t_frames
        end_frame   = T_total
    T_slice = end_frame - start_frame

    # determine which landmarks were used (we reused this pattern before)
    # if you already have valid_idx from triangulation, pass it in instead
    any_body = next(iter(pts2d.keys()))
    any_view = 0
    L_total = pts2d[any_body][any_view].shape[1]

    l_hand_idx = [3, 5, 7, 8, 10, 12, 18, 20, 27, 37]
    r_hand_idx = [259, 261, 263, 264, 266, 268, 274, 276, 283, 293]
    valid_idx = [i for i in range(L_total)
                 if i not in l_hand_idx and i not in r_hand_idx]
    valid_idx = list(range(L_total))  # if you want all joints
    L_valid = len(valid_idx)

    body_ids_all = list(pts2d.keys())
    Ncams        = len(K)

    # avoid assigning the same 2D track (body_id) to multiple groups in same view
    used_body_ids_per_view = {v: set() for v in range(Ncams)}
    for g, group in enumerate(groups):
        for v, bid in group["body_ids"].items():
            used_body_ids_per_view[v].add(bid)

    assignments_info = {
        "per_group": []  # for debugging / inspection
    }

    for g_idx, group in enumerate(groups):
        Xw = triX_list[g_idx]         # [T_slice, L_valid, 3]
        valid_mask = valid_list[g_idx]  # [T_slice, L_valid] bool

        # sanity
        assert Xw.shape[0] == T_slice, "T_slice mismatch with triX_list"
        assert Xw.shape[1] == L_valid, "L_valid mismatch with triX_list"

        group_body_ids = group["body_ids"]
        known_views = set(group_body_ids.keys())

        group_assign_debug = {"new_assignments": [], "skipped": []}

        # for each view, try to assign if missing
        for v in range(Ncams):
            if v in known_views:
                continue  # already has an id from the multi-view matching

            # camera params for this view and time window
            K_T  = K[v][start_frame:end_frame]          # [T_slice,3,3]
            E_T  = extrinsics[v][start_frame:end_frame] # [T_slice,4,4]

            # project the 3D trajectory to this view
            uv_proj_TL2, z_TL = _project_world_to_image(Xw, K_T, E_T)  # [T,L,2], [T,L]
            # ignore behind-camera points
            proj_valid = (z_TL > 0.1) & valid_mask     # [T,L]

            best_bid = None
            best_err = None

            # search over body_ids in this view that are not yet taken
            for bid in body_ids_all:
                if bid in used_body_ids_per_view[v]:
                    continue  # already used by another group in this view

                # get the 2D track for this body/view, restricted to the same time window
                pts2d_TLD = pts2d[bid][v][start_frame:end_frame]      # [T_slice,L,D]
                vis_TL    = pts2d_vis_weight[bid][v][start_frame:end_frame]  # [T_slice,L]

                if pts2d_TLD.shape[0] != T_slice:
                    continue  # mismatch, skip

                uv_TL2 = pts2d_TLD[:, valid_idx, :2]    # [T_slice,L_valid,2]
                vis_TL = vis_TL[:, valid_idx]           # [T_slice,L_valid]

                # joint visibility AND 3D validity
                joint_mask = (vis_TL > 0.5) & proj_valid  # [T,L_valid]

                # require enough joints and enough frames
                joints_per_frame = joint_mask.sum(dim=1)       # [T]
                frames_with_enough_joints = joints_per_frame >= 5
                num_good_frames = frames_with_enough_joints.sum().item()

                if num_good_frames == 0:
                    print("View", v, "has no valid projections for group", g_idx, "; inserting zeros.")
                    continue

                if num_good_frames < min_time_overlap * T_slice:
                    print("View", v, "has insufficient temporal overlap for group", g_idx, "; inserting zeros.")
                    continue  # not enough temporal overlap

                # compute reprojection error only where valid
                diff = uv_proj_TL2[:, valid_idx, :] - uv_TL2  # [T,L_valid,2]
                err = torch.norm(diff, dim=-1)               # [T,L_valid]
                err = err * joint_mask.float()

                # mean error over valid joints/frames
                denom = joint_mask.float().sum(dim=1).clamp(min=1e-6)  # [T]
                mean_err_per_frame = (err.sum(dim=1) / denom)          # [T]
                mean_err = mean_err_per_frame[frames_with_enough_joints].mean()

                mean_err_val = float(mean_err.item())

                if best_err is None or mean_err_val < best_err:
                    best_err = mean_err_val
                    best_bid = bid

            # if we found a good candidate
            if best_bid is not None and best_err is not None and best_err <= max_mean_reproj_error:
                group_body_ids[v] = best_bid
                used_body_ids_per_view[v].add(best_bid)
                group_assign_debug["new_assignments"].append(
                    {"view": v, "body_id": best_bid, "mean_error": best_err}
                )
            else:
                group_assign_debug["skipped"].append(
                    {"view": v, "reason": "no candidate" if best_bid is None else f"best_err={best_err:.2f}"}
                )

        assignments_info["per_group"].append(group_assign_debug)

    print("Reprojection-based ID propagation done.")
    # ------------------------------------------------------------------
    # STEP 2: Re-triangulate using updated group["body_ids"]
    # ------------------------------------------------------------------
    triX_list_new  = []
    valid_list_new = []

    for g_idx, group in enumerate(groups):
        group_body_ids = group["body_ids"]

        # pick any view in the group to define D and L_valid
        any_v = next(iter(group_body_ids.keys()))
        any_bid = group_body_ids[any_v]
        any_pts2d_TLD = pts2d[any_bid][any_v][start_frame:end_frame]  # [T_slice,L_total,D]
        T_slice_check, L_total_check, D = any_pts2d_TLD.shape
        assert T_slice_check == T_slice

        pts2d_list = []
        vis_list   = []
        Ks_batch   = []
        EX_batch   = []

        for v in range(Ncams):
            if v in group_body_ids:
                bid = group_body_ids[v]
                pts2d_TLD = pts2d[bid][v][start_frame:end_frame]  # [T_slice,L_total,D]
                vis_TL    = pts2d_vis_weight[bid][v][start_frame:end_frame]  # [T_slice,L_total]

                pts2d_valid = pts2d_TLD[:, valid_idx, :]  # [T_slice,L_valid,D]
                vis_valid   = vis_TL[:, valid_idx]        # [T_slice,L_valid]

                pts2d_list.append(pts2d_valid.to(device=device, dtype=dtype))
                vis_list.append(vis_valid.to(device=device))
            else:
                # no track in this view for this group
                print("View", v, "has no assigned body_id for group", g_idx, "; inserting zeros.")
                pts2d_list.append(torch.zeros((T_slice, L_valid, D),
                                              device=device, dtype=dtype))
                vis_list.append(torch.zeros((T_slice, L_valid),
                                            device=device))

            K_T  = K[v][start_frame:end_frame]          # [T_slice,3,3]
            EX_T = extrinsics[v][start_frame:end_frame] # [T_slice,4,4]
            Ks_batch.append(K_T.to(device=device, dtype=dtype))
            EX_batch.append(EX_T.to(device=device, dtype=dtype))

        # call your triangulation
        print(f"Re-triangulating group {g_idx} with views {list(group_body_ids.keys())} ...")
        Xw_new, valid_new = triangulate_batch_fn(
            pts2d_list, Ks_batch, EX_batch,
            vis_list=vis_list,
            use_uncertainties=(True and D == 3),
            img_size_px=img_size_px,
            channel_is_logvar=False,
        )   # [T_slice,L_valid,3], [T_slice,L_valid]

        triX_list_new.append(Xw_new)
        valid_list_new.append(valid_new)


    return groups, triX_list_new, valid_list_new, assignments_info
