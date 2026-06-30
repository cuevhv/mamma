"""
Occlusion-aware gating for the multi-view reprojection loss.

Background
----------
Useful when many occluded views collectively out-vote the few views that
actually see a body part (e.g. legs behind a table).

When explicitly enabled, this module replaces the default per-landmark
visibility weighting (which clamps occluded landmarks up to a minimum weight)
with a **occlusion gate**: a per-(camera, frame, landmark) multiplier in
``[0, 1]`` that is driven toward zero as the predicted visibility drops, so
occluded detections stop contributing and the views that do see the part
dominate.

Grounding
---------------------
* Detect when many occluded views agree on a wrong location.
* Down-weight low-confidence keypoints before fusing across views.
* The ``keep_top_m`` safety is a multi-view trimmed estimator: never zero a
  landmark in *all* views, so a part seen by >=1 camera is never left fully
  unconstrained (it falls back to the surviving view + pose prior + kinematics).

The function is intentionally **visibility-only**: the existing per-keypoint
aleatoric uncertainty (sigma -> 1/sigma^2 precision) keeps acting *among the
surviving views* inside ``proj_pts_loss``; gating only decides *which* views
survive.  This keeps the change to a single, well-localised multiplier.
"""

from typing import List

import torch

_VALID_MODES = ("soft", "hybrid", "trim")


def compute_occlusion_gates(
    vis_list: List[torch.Tensor],
    cfg: dict,
) -> List[torch.Tensor]:
    """
    Compute per-camera gate weights from per-camera visibilities.

    Each gate is a weight in ``[0, 1]`` multiplying a landmark's reprojection
    contribution. For ``soft``/``hybrid`` mode::

        gate = v * sigmoid((v - tau) / beta)

    where ``v`` is the landmark's visibility in that view — its temporal median
    over frames when ``temporal`` is set, else the per-frame value. Views with
    ``v`` well above ``tau`` keep ~their visibility, views well below collapse
    toward zero, and ``keep_top_m`` then restores the best views so a landmark is
    never zeroed in every view. (The ``enabled`` switch lives one level up, in the
    config block read by ``proj_pts_loss`` — this function is only called when on.)

    Parameters
    ----------
    vis_list : list (length C) of ``[T, N]`` float tensors
        Predicted visibility in ``[0, 1]`` per camera, frame and landmark.
    cfg : dict
        Keys (all optional, with defaults):
          - ``mode``: gate shape (default "hybrid"):
              "soft"   = sigmoid gate only, no safety net (even the best view is
                         suppressed below ``tau``);
              "hybrid" = sigmoid gate + ``keep_top_m`` safety (smooth, never zeros
                         a landmark in all views);
              "trim"   = hard cutoff (weight 0 below ``tau``) + ``keep_top_m`` safety.
          - ``tau``: visibility threshold / sigmoid centre. Landmarks below it are
            treated as occluded and driven toward zero weight (default 0.30).
          - ``beta``: transition softness around ``tau`` (soft/hybrid only) —
            smaller = sharper near-step, larger = gentler ramp (default 0.05).
          - ``keep_top_m``: keep the m most-visible views per landmark ungated, so a
            landmark seen by >=1 camera is never fully zeroed (hybrid/trim; soft
            ignores it) (default 1).
          - ``temporal``: decide on the temporal *median* visibility instead of
            per-frame, so a consistently-low landmark is gated but a transient
            single-frame dip is not (default True).

    Returns
    -------
    gates_list : list (length C) of ``[T, N]`` tensors in ``[0, 1]``.
        The gate applies to every landmark; the per-keypoint uncertainty term in
        ``proj_pts_loss`` keeps weighting the surviving views.
    """
    if not vis_list:
        return vis_list

    mode = str(cfg.get("mode", "hybrid")).lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"occlusion_gating.mode must be one of {_VALID_MODES}, got {mode!r}")

    tau = float(cfg.get("tau", 0.30))
    beta = max(float(cfg.get("beta", 0.05)), 1e-6)
    keep_top_m = int(cfg.get("keep_top_m", 1))
    temporal = bool(cfg.get("temporal", True))

    vis = torch.stack([v.float() for v in vis_list], dim=0)  # [C, T, N]
    vis = torch.nan_to_num(vis, nan=0.0).clamp_(0.0, 1.0)
    C = vis.shape[0]

    # Decision signal: temporally-aggregated (robust median over frames) so a
    # landmark that is *consistently* low in a view is gated, while transient
    # single-frame dips are not.
    if temporal:
        vis_used = torch.median(vis, dim=1, keepdim=True).values.expand_as(vis)
    else:
        vis_used = vis

    if mode == "trim":
        gate = vis_used * (vis_used >= tau).to(vis_used.dtype)
    else:  # soft / hybrid: smooth redescending gate
        gate = vis_used * torch.sigmoid((vis_used - tau) / beta)

    # keep_top_m safety (hybrid, trim): for every (frame, landmark) restore the
    # m most-visible cameras to their ungated weight so the landmark is never
    # fully zeroed across all views.  Soft mode deliberately has no safety net.
    if mode in ("hybrid", "trim") and keep_top_m > 0 and C > 0:
        m = min(keep_top_m, C)
        top_idx = torch.topk(vis_used, k=m, dim=0).indices          # [m, T, N]
        keep = torch.zeros_like(vis_used, dtype=torch.bool)
        keep.scatter_(0, top_idx, True)
        gate = torch.where(keep, torch.maximum(gate, vis_used), gate)

    return [gate[c] for c in range(C)]
