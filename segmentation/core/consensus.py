"""Cross-camera identity consensus (post-process, optional).

Repairs MID-TRACKLET identity swaps that single-camera tracking cannot see:
two similar-looking people cross paths, the tracker exchanges them, and every
downstream mask carries the wrong id from that frame on. No local signal
exists at the swap (no duplicate, no coverage gap), but the OTHER cameras
observed the crossing from angles where the people never overlapped.

The vote is built around a hard geometric constraint: epipolar consistency is
invariant to relabeling ALL cameras with the same permutation, so cross-view
geometry alone can never fix the global gauge, and any scheme that relabels
cameras toward an instantaneous majority can rotate whole windows into a
consistent but wrong state. What geometry does observe exactly is the
RELATIVE permutation between each camera pair over time. A swap inside one
camera flips that camera's relation to every other camera simultaneously, so
pairwise vote series localize swap events and attribute them to cameras, up
to one residual global flip per event. That last ambiguity is broken by
temporal continuity: a camera can only swap at a moment when its own tracks
are in contact or occluded, so the hypothesis whose cameras have the smallest
label-exchange displacement at the event wins.

Design constraints, deliberately:
  * Standalone: reads mask PNGs (the artifact it repairs) and the calibration
    already present in ``cam_data``; renames mask PNGs; never touches the
    tracker, its state, or the pipeline internals.
  * Numpy-only and centroid-based: a few thousand 3-vector operations per
    sequence; no GPU, no video decoding.
  * Single application: events are detected, attributed, composed per camera,
    and applied once; reruns on repaired output are no-ops.
  * Abstains under ambiguity: undersized regimes, indecisive pair votes, and
    continuity-cost ties are reported, never applied.

Enable via ``masks.cross_camera_consensus`` (``false`` default / ``auto`` =
on when >=2 calibrated cameras / ``true``). Runs once per sequence after all
cameras complete.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core import geometry

# A gauge hypothesis must beat the alternative by this cost ratio to be
# applied; ties mean geometry+continuity cannot say which side swapped.
DECISIVE_RATIO = 0.6


# --------------------------------------------------------------------------
# data loading (I/O layer)
# --------------------------------------------------------------------------

def densify_from_pngs(cam_out_dir: str, stride: int = 15, frames=None):
    """Centroid + bbox tracks read directly from the mask PNGs.

    Masks are the ground truth here: caches elsewhere in the pipeline go
    stale the moment files are renamed, so the vote only ever reads PNGs.
    ``frames`` restricts the scan to an explicit frame list (dense event
    refinement); otherwise every ``stride``-th frame is read.
    """
    import cv2
    mask_dir = os.path.join(cam_out_dir, "masks")
    cents: Dict[int, Dict[int, Tuple[float, float]]] = {}
    boxes: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    if frames is None:
        frames = sorted({int(os.path.basename(p).split("_")[1])
                         for p in glob.glob(os.path.join(mask_dir, "mask_*.png"))})
        frames = frames[::max(1, stride)]
    for f in frames:
        for p in glob.glob(os.path.join(mask_dir, f"mask_{f:04d}_*.png")):
            oid = int(os.path.basename(p).split("_")[2].split(".")[0]) - 1
            m = cv2.imread(p, 0)
            if m is None or m.max() == 0:
                continue
            ys, xs = np.nonzero(m > 127)
            if len(xs) < 25:
                continue
            cents.setdefault(oid, {})[f] = (float(xs.mean()), float(ys.mean()))
            boxes.setdefault(oid, {})[f] = (float(xs.min()), float(ys.min()),
                                            float(xs.max()), float(ys.max()))
    return cents, boxes


# --------------------------------------------------------------------------
# voting core (pure; unit-tested on synthetic rigs)
# --------------------------------------------------------------------------

def track_at(track: dict, frame: int, max_gap: int = 60):
    """Value at ``frame`` from a sparsely sampled track: exact hit, else
    linear interpolation between the bracketing samples (both within
    ``max_gap`` frames), else None. Cameras cache centroids/boxes at
    DIFFERENT sampled frames, so cross-camera lookups need this.
    """
    v = track.get(frame)
    if v is not None:
        return v
    lo = max((f for f in track if f < frame), default=None)
    hi = min((f for f in track if f > frame), default=None)
    if lo is None or hi is None or frame - lo > max_gap or hi - frame > max_gap:
        return None
    t = (frame - lo) / float(hi - lo)
    a, b = track[lo], track[hi]
    return tuple((1.0 - t) * x + t * y for x, y in zip(a, b))


def _compose(p: tuple, q: tuple) -> tuple:
    """(p o q)[k] = p[q[k]]: apply q first, then p. Index tuples."""
    return tuple(p[q[k]] for k in range(len(p)))


def _inverse(p: tuple) -> tuple:
    inv = [0] * len(p)
    for k, v in enumerate(p):
        inv[v] = k
    return tuple(inv)


def pair_vote(tracks_a, tracks_b, f_ab, f_ba, frame, ids,
              min_margin_px=40.0) -> Optional[tuple]:
    """Winning relative permutation between two cameras at one frame.

    cost(pi) = mean over ids i of the symmetric centroid point-to-epipolar-
    line distance between camera-a's tracklet i and camera-b's tracklet
    pi(i). Centroids carry a silhouette-asymmetry bias of some tens of px
    (measured; bbox distances saturate to zero on person-sized boxes), but a
    wrong-person hypothesis costs person-separation distances, so the margin
    gate separates them cleanly. Returns the index permutation, or None when
    evidence is missing or the margin is below ``min_margin_px``.
    """
    pa = {i: track_at(tracks_a.get(i, {}), frame) for i in ids}
    pb = {i: track_at(tracks_b.get(i, {}), frame) for i in ids}
    obs_a = [i for i in ids if pa[i] is not None]
    obs_b = [i for i in ids if pb[i] is not None]
    if len(obs_a) < 2 or len(obs_b) < 2:
        return None
    costs = {}
    for pi in itertools.permutations(range(len(ids))):
        dists = []
        for k, i in enumerate(ids):
            j = ids[pi[k]]
            if pa[i] is None or pb[j] is None:
                continue
            xa = np.array([*pa[i][:2], 1.0])
            xb = np.array([*pb[j][:2], 1.0])
            d1 = geometry.epipolar_distance(f_ab, xa, xb)
            d2 = geometry.epipolar_distance(f_ba, xb, xa)
            ds = [d for d in (d1, d2) if d is not None]
            if ds:
                dists.append(float(np.mean(ds)))
        if len(dists) < 2:
            return None
        costs[pi] = float(np.mean(dists))
    ranked = sorted(costs.items(), key=lambda kv: kv[1])
    if len(ranked) < 2 or ranked[1][1] - ranked[0][1] < min_margin_px:
        return None
    return ranked[0][0]


def pair_regimes(votes: List[Tuple[int, tuple]], min_seg: int):
    """Maximal sustained constant runs of a pairwise vote series.

    ``votes`` are (frame, permutation) for decisive samples only. Runs
    shorter than ``min_seg`` frames are dropped (transient flips at
    crossings); consecutive runs with equal permutations merge across the
    dropped noise. Returns [(start_frame, end_frame, permutation)].
    """
    runs = []
    for f, pi in votes:
        if runs and runs[-1][2] == pi:
            runs[-1][1] = f
        else:
            runs.append([f, f, pi])
    runs = [r for r in runs if r[1] - r[0] >= min_seg]
    merged = []
    for r in runs:
        if merged and merged[-1][2] == r[2]:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(r[0], r[1], r[2]) for r in merged]


@dataclass
class Segment:
    start_frame: int
    end_frame: int              # inclusive
    permutation: tuple          # global id for each of ids (identity order)
    margin_px: float


@dataclass
class ConsensusReport:
    cam_name: str
    sampled: int = 0
    relabeled: List[Segment] = field(default_factory=list)
    abstained: int = 0


@dataclass
class _Event:
    lo: int                     # last decisive frame of the old regimes
    hi: int                     # first decisive frame of the new regimes
    relations: Dict[frozenset, Tuple[tuple, tuple]]  # pair -> (old, new)


def _regime_at(regs, frame: int, after: bool):
    """Permutation of the sustained regime at/before ``frame`` (``after`` =
    at/after), or None outside all regimes."""
    if after:
        cand = [r for r in regs if r[1] >= frame]
        return cand[0][2] if cand else None
    cand = [r for r in regs if r[0] <= frame]
    return cand[-1][2] if cand else None


def detect_events(series: Dict[frozenset, List[Tuple[int, int, tuple]]]):
    """Regime transitions per pair, clustered into events across pairs.

    A transition of pair (a, b) between consecutive sustained regimes spans
    the frame interval between them. Transitions whose intervals overlap
    describe the same physical crossing and merge into one event. Each event
    records, for EVERY pair with regimes on both sides, the pair's relation
    before and after: unflipped pairs constrain the solve just as much as
    flipped ones.
    """
    raw = []
    for pair, regs in series.items():
        for r0, r1 in zip(regs, regs[1:]):
            if r0[2] != r1[2]:
                raw.append((r0[1], r1[0], pair))
    raw.sort()
    events: List[_Event] = []
    for lo, hi, _pair in raw:
        ev = next((e for e in events if lo <= e.hi and hi >= e.lo), None)
        if ev is None:
            events.append(_Event(lo, hi, {}))
        else:
            ev.lo, ev.hi = min(ev.lo, lo), max(ev.hi, hi)
    for ev in events:
        for pair, regs in series.items():
            old = _regime_at(regs, ev.lo, after=False)
            new = _regime_at(regs, ev.hi, after=True)
            if old is not None and new is not None:
                ev.relations[pair] = (old, new)
    return sorted(events, key=lambda e: e.lo)


def solve_event(event: _Event, cam_names: List[str], n_ids: int):
    """Per-camera change assignments explaining an event's pairwise
    relations, which transform as V' = s_b o V o s_a^-1.

    Two-person events are solved exhaustively: every assignment over the
    swap group is scored by how many pair relations it violates and the
    minimum-violation family is returned. Regime noise (a spurious short
    regime surviving the sustain gate) then costs one constraint instead of
    making the whole event unsolvable; the continuity tie-break and the
    ambiguity gate still arbitrate between surviving candidates. Larger id
    counts use exact propagation from a root camera, which returns the
    global-gauge family or nothing on conflict.
    """
    if n_ids == 2 and len(cam_names) <= 10:
        ident, swap = (0, 1), (1, 0)
        scored = []
        for bits in itertools.product((ident, swap), repeat=len(cam_names)):
            sol = dict(zip(cam_names, bits))
            viol = 0
            for pair, (old, new) in event.relations.items():
                a, b = sorted(pair)
                if a not in sol or b not in sol:
                    continue
                pred = _compose(_compose(sol[b], old), _inverse(sol[a]))
                if pred != new:
                    viol += 1
            scored.append((viol, sol))
        best = min(v for v, _s in scored)
        return [s for v, s in scored if v == best]
    solutions = []
    for h in itertools.permutations(range(n_ids)):
        s = {cam_names[0]: h}
        queue = [cam_names[0]]
        ok = True
        while queue and ok:
            a = queue.pop()
            for b in cam_names:
                pair = frozenset((a, b))
                if b == a or pair not in event.relations:
                    continue
                old, new = event.relations[pair]
                # stored maps run low->high camera name; orient a->b
                if a > b:
                    old, new = _inverse(old), _inverse(new)
                s_b = _compose(_compose(new, s[a]), _inverse(old))
                if b in s:
                    if s[b] != s_b:
                        ok = False
                        break
                else:
                    s[b] = s_b
                    queue.append(b)
        if ok and len(s) == len(cam_names):
            solutions.append(s)
    return solutions


def continuity_cost(dense_tracks, perm: tuple, ids, lo: int, hi: int):
    """Cheapest frame in [lo, hi] to apply ``perm`` to a camera's labels,
    as (cost_px, frame). The cost at a frame is the mean displacement of
    the exchanged centroids; a frame where an exchanged id is unobserved
    costs zero (swap hidden by occlusion). Identity is free.
    """
    moved = [k for k in range(len(ids)) if perm[k] != k]
    if not moved:
        return 0.0, lo
    best = (float("inf"), (lo + hi) // 2)
    for f in range(lo, hi + 1):
        ds = []
        occluded = False
        for k in moved:
            a = dense_tracks.get(ids[k], {}).get(f)
            b = dense_tracks.get(ids[perm[k]], {}).get(f)
            if a is None or b is None:
                occluded = True
                break
            ds.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
        cost = 0.0 if occluded else float(np.mean(ds))
        if cost < best[0]:
            best = (cost, f)
        if cost == 0.0:
            break
    return best


# --------------------------------------------------------------------------
# relabeling (I/O layer)
# --------------------------------------------------------------------------

def relabel_masks(mask_dir: str, ids: List[int], segment: Segment) -> int:
    """Apply a permutation to the mask PNG filenames inside a segment.

    Two-phase rename (to temp names first) so a swap can never collide.
    Returns the number of files renamed.
    """
    pi = {i: g for i, g in zip(ids, segment.permutation)}
    moves = []
    for f in range(segment.start_frame, segment.end_frame + 1):
        for i, g in pi.items():
            if i == g:
                continue
            src = os.path.join(mask_dir, f"mask_{f:04d}_{i + 1:02d}.png")
            if os.path.isfile(src):
                moves.append((src, os.path.join(mask_dir, f"mask_{f:04d}_{g + 1:02d}.png")))
    for src, _ in moves:
        os.replace(src, src + ".consensus_tmp")
    for src, dst in moves:
        os.replace(src + ".consensus_tmp", dst)
    return len(moves)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def resolve_mode(value, n_calibrated_cams: int) -> bool:
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "auto":
            return n_calibrated_cams >= 2
        return v in {"1", "true", "yes", "on"}
    return bool(value)


def run_cross_camera_consensus(seq_out_dir: str, cam_data_by_name: Dict[str, dict],
                               cfg: Optional[dict] = None, log=print) -> List[ConsensusReport]:
    """Detect per-camera swap events from pairwise vote series, attribute
    them with the continuity tie-break, and relabel each camera once with
    the composed correction. Writes ``consensus_report.json`` per corrected
    camera. Safe no-op on single-camera or uncalibrated runs."""
    cfg = cfg or {}
    min_margin = float(cfg.get("consensus_min_margin_px", 40.0))
    min_seg = int(cfg.get("consensus_min_segment_frames", 45))
    stride = int(cfg.get("consensus_stride", 15))
    cams = {c: d for c, d in cam_data_by_name.items()
            if os.path.isdir(os.path.join(seq_out_dir, c, "masks"))}
    tracks = {}
    for c in cams:
        cents, _boxes = densify_from_pngs(os.path.join(seq_out_dir, c), stride)
        if cents:
            tracks[c] = cents
    if len(tracks) < 2:
        return []
    names = sorted(tracks)
    ids = sorted(set().union(*[set(t) for t in tracks.values()]))
    ident = tuple(range(len(ids)))
    fmat = {}
    for a in names:
        for b in names:
            if a != b:
                fmat[(a, b)] = geometry.compute_fundamental_matrix(cams[a], cams[b])
    grid = sorted(set().union(*[set(f) for t in tracks.values() for f in t.values()]))
    reports = {c: ConsensusReport(cam_name=c, sampled=len(grid)) for c in names}

    series = {}
    for a, b in itertools.combinations(names, 2):
        if fmat[(a, b)] is None or fmat[(b, a)] is None:
            continue
        votes = []
        for f in grid:
            pi = pair_vote(tracks[a], tracks[b], fmat[(a, b)], fmat[(b, a)],
                           f, ids, min_margin_px=min_margin)
            if pi is not None:
                votes.append((f, pi))
        series[frozenset((a, b))] = pair_regimes(votes, min_seg)

    net = {c: ident for c in names}
    state: Dict[str, List[Tuple[int, tuple]]] = {c: [] for c in names}
    for ev in detect_events(series):
        dense = {c: densify_from_pngs(os.path.join(seq_out_dir, c), 1,
                                      frames=list(range(ev.lo, ev.hi + 1)))[0]
                 for c in names}
        # cameras whose pairs were indecisive across the event carry no
        # constraints; solve each connected evidence component on its own
        # and leave unconstrained cameras untouched
        for comp in _components(ev.relations):
            scored = []
            for sol in solve_event(ev, comp, len(ids)):
                per_cam = {c: continuity_cost(dense[c], sol[c], ids, ev.lo, ev.hi)
                           for c in comp}
                scored.append((sum(v[0] for v in per_cam.values()), sol, per_cam))
            scored.sort(key=lambda t: t[0])
            ambiguous = (not scored
                         or (len(scored) > 1
                             and (scored[0][0] > DECISIVE_RATIO * scored[1][0]
                                  or scored[1][0] < 1e-6)))
            if ambiguous:
                detail = (f"costs {scored[0][0]:.0f} vs {scored[1][0]:.0f}px"
                          if len(scored) > 1 else "constraints unsolvable")
                log(f"[consensus] event at frames {ev.lo}-{ev.hi}: gauge ambiguous "
                    f"({detail}), abstaining.")
                for c in comp:
                    reports[c].abstained += 1
                continue
            _, sol, per_cam = scored[0]
            ref = next((c for c in comp if sol[c] == ident), None)
            for c in comp:
                if sol[c] == ident:
                    continue
                swap_frame = _refine_swap_frame(
                    dense, tracks, fmat, c, ref, sol[c], ids, ev, min_margin,
                    fallback=per_cam[c][1])
                net[c] = _compose(sol[c], net[c])
                state[c].append((swap_frame, net[c]))

    last = _last_mask_frame(seq_out_dir, names)
    gauge = _static_gauges(names, ids, grid, series, state, cfg, log)
    for c in names:
        if gauge[c] is None:
            continue
        mask_dir = os.path.join(seq_out_dir, c, "masks")
        bounds = [(0, ident)] + sorted(state[c]) + [(last + 1, ident)]
        for (f0, perm), (f1, _next) in zip(bounds, bounds[1:]):
            if f1 - 1 < f0:
                continue
            fix = _compose(gauge[c], _inverse(perm))
            if fix == ident:
                continue
            seg = Segment(f0, f1 - 1, tuple(ids[fix[k]] for k in range(len(ids))),
                          margin_px=min_margin)
            n = relabel_masks(mask_dir, ids, seg)
            reports[c].relabeled.append(seg)
            log(f"[consensus] {c}: relabeled frames {seg.start_frame}-{seg.end_frame} "
                f"as {seg.permutation} ({n} files)")
        if reports[c].relabeled:
            with open(os.path.join(seq_out_dir, c, "consensus_report.json"), "w") as fh:
                json.dump([{"start": s.start_frame, "end": s.end_frame,
                            "permutation": list(s.permutation),
                            "margin_px": s.margin_px} for s in reports[c].relabeled],
                          fh, indent=2)
    return [reports[c] for c in names]


def _fix_at(state_c, gauge_c, ident, f):
    """Repair permutation a camera receives at frame ``f``."""
    perm = ident
    for f0, p in sorted(state_c):
        if f0 <= f:
            perm = p
    return _compose(gauge_c, _inverse(perm))


def _static_gauges(names, ids, grid, series, state, cfg, log):
    """Constant per-camera correction on top of the event chain.

    Events pin every label CHANGE, which leaves each camera one constant
    permutation short of absolute identity: a camera seeded with the wrong
    assignment shows constant pair relations and no events at all. The
    anchor camera at the anchor frame defines truth (the init camera at its
    init frame when the pipeline provides them); every other camera gets
    the constant that makes its post-repair relation to the anchored set
    identity, propagated over decisive pairs. Cameras whose samples do not
    agree on a single constant are left unrepaired and logged.
    """
    ident = tuple(range(len(ids)))
    anchor = (cfg or {}).get("consensus_anchor_cam")
    if anchor not in names:
        anchor = names[0]
    a_frame = int((cfg or {}).get("consensus_anchor_frame",
                                  grid[0] if grid else 0))
    # truth at the anchor: no repair at the anchor frame
    ev_perm = ident
    for f0, p in sorted(state[anchor]):
        if f0 <= a_frame:
            ev_perm = p
    gauge = {anchor: ev_perm}   # cancels the chain at the anchor frame
    abstain = set()
    pending = [c for c in names if c != anchor]
    while pending:
        progressed = False
        for c in list(pending):
            votes: Dict[tuple, int] = {}
            for ref in gauge:
                pair = frozenset((c, ref))
                regs = series.get(pair)
                if not regs:
                    continue
                for f in grid:
                    if not any(r[0] <= f <= r[1] for r in regs):
                        continue
                    rel = _regime_at(regs, f, after=False)
                    v = rel if min(c, ref) == ref else _inverse(rel)  # ref->c
                    f_ref = _fix_at(state[ref], gauge[ref], ident, f)
                    fix_c_chain = _fix_at(state[c], ident, ident, f)
                    # want: r_c o fix_c = f_ref o v_ref_to_c^-1
                    r = _compose(_compose(f_ref, _inverse(v)), _inverse(fix_c_chain))
                    votes[r] = votes.get(r, 0) + 1
            if not votes:
                continue
            best = max(votes.items(), key=lambda kv: kv[1])
            frac = best[1] / sum(votes.values())
            if frac < 0.9:
                log(f"[consensus] {c}: static gauge inconsistent "
                    f"({100 * frac:.0f}% agreement), leaving camera unrepaired.")
                abstain.add(c)
            else:
                gauge[c] = best[0]
                if best[0] != ident:
                    log(f"[consensus] {c}: constant relabel {best[0]} "
                        f"(camera-wide gauge, e.g. inverted seeding).")
            pending.remove(c)
            progressed = True
        if not progressed:
            break
    out = {}
    for c in names:
        out[c] = None if (c in abstain or c in pending) else gauge.get(c, ident)
    return out


def _components(relations) -> List[List[str]]:
    """Connected components of the pairwise-constraint graph, sorted."""
    adj: Dict[str, set] = {}
    for pair in relations:
        a, b = sorted(pair)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, comps = set(), []
    for start in sorted(adj):
        if start in seen:
            continue
        comp, queue = [], [start]
        seen.add(start)
        while queue:
            c = queue.pop()
            comp.append(c)
            for d in adj[c]:
                if d not in seen:
                    seen.add(d)
                    queue.append(d)
        comps.append(sorted(comp))
    return comps


def _last_mask_frame(seq_out_dir: str, names) -> int:
    last = 0
    for c in names:
        for p in glob.glob(os.path.join(seq_out_dir, c, "masks", "mask_*.png")):
            last = max(last, int(os.path.basename(p).split("_")[1]))
    return last


def _refine_swap_frame(dense, tracks, fmat, cam, ref, s_c, ids, event,
                       min_margin, fallback: int) -> int:
    """Frame-exact swap onset inside the event interval.

    Primary evidence: the dense per-frame vote of ``cam`` against an
    unswapped reference camera flips exactly where the labels flip: the
    onset is the first dense frame voting the post-event relation. Falls
    back to the camera's own closest-approach frame when no reference
    exists or the dense votes never become decisive.
    """
    if ref is None:
        return fallback
    pair = frozenset((cam, ref))
    if pair not in event.relations or fmat.get((ref, cam)) is None:
        return fallback
    old, new = event.relations[pair]
    if cam < ref:
        old, new = _inverse(old), _inverse(new)
    ref_tracks = {**tracks[ref], **{i: {**tracks[ref].get(i, {}), **dense[ref].get(i, {})}
                                    for i in dense[ref]}}
    for f in range(event.lo, event.hi + 1):
        pi = pair_vote(ref_tracks, dense[cam], fmat[(ref, cam)], fmat[(cam, ref)],
                       f, ids, min_margin_px=min_margin)
        if pi == new and pi != old:
            return f
    return fallback
