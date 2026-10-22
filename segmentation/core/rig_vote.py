"""Rig-vote identity relabel (post-process): the post-tracking identity repair of ma_masks.

Repairs per-camera identity errors on calibrated rigs by letting the whole
rig vote in 3D. The capture volume is divided into voxels; every camera's
masks label the voxels that project into them, and a voxel takes the label
most cameras agree on. Each camera is scored against the vote of the OTHER
cameras (leave-one-out), and a per-camera Viterbi over label permutations
(with a switching penalty) turns the per-frame scores into relabel segments;
the vote is recomputed with the corrected labels until it stops changing. A
whole-camera mislabel is a constant permutation, a mid-sequence swap a change
of permutation, a short takeover a brief one.

Per camera and frame the evidence is an N x N matrix: entry (k, j) is how
much better mask k fits rig label j than the best other label (its share on
label j's exclusive region minus the largest share on any other label's),
weighted by the mask's size relative to the camera's largest mask, so a
stray fragment cannot outvote a body. A permutation's score is the sum of its
entries; a camera-frame holding a missing or fragment mask counts
INCOMPLETE_WEIGHT.

A rig majority is itself only a gauge: if most cameras exchange two people at
a body contact, the vote follows them. The merge guard therefore splits the
sequence at contact windows (two vote cores merge or one vanishes) and drops
relabels on stretches the cameras are split on (median share of disagreeing
cameras above ``rig_vote_max_minority``) and inside a contact window, unless
the camera holds the same permutation on both sides of it.

Design constraints, deliberately:
  * Standalone: reads the mask files (the artifact it repairs) and the
    calibration already present in ``cam_data``; renames mask entries; never
    touches the tracker, its caches, or other outputs.
  * 2 to MAX_SUBJECTS subjects (the Viterbi state space is N!) and at least
    MIN_VOTES + 1 cameras with masks; other runs are skipped.
  * CPU-only: masks are voted at reduced resolution through a precomputed
    voxel-to-pixel lookup, so the per-frame work scales with the foreground.
  * Single application: segments are composed per camera and applied once;
    a rerun on repaired output is a no-op.

Enable via ``masks.rig_vote`` (``false`` / ``auto`` = on with at least
``rig_vote_min_cameras`` calibrated cameras / ``true``). Runs once per
sequence after all cameras complete (in the finalize cell on a camera
fan-out). Writes ``rig_vote_report.json`` per relabeled camera and
``rig_vote_summary.json`` per sequence.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_opening

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from capture.frame_source import _camera_from_cam_data  # noqa: E402
from capture.undistort import _radial_correction  # noqa: E402

DEFAULTS = {
    "rig_vote_min_cameras": 12,      # auto mode: calibrated cameras required
    "rig_vote_downsample": 8,        # mask grid = image size / downsample
    "rig_vote_voxel_m": 0.05,        # voxel edge
    "rig_vote_switch_cost": 6.0,     # Viterbi penalty for a keep/swap change
    "rig_vote_max_passes": 4,        # vote passes (stops early on convergence)
    "rig_vote_max_minority": 1 / 3,  # merge guard: max median disagreeing share
    "rig_vote_merge_sep_m": 0.30,    # 3D core distance treated as contact
    "rig_vote_workers": 8,           # decode threads
}

MASK_COVER = 64            # grid cell is foreground when >= 25% covered
NEAR_M = 0.2               # voxels closer than this to a camera are ignored
MIN_VOTES = 4              # occupancy: foreground in >= this many cameras ...
OCC_FRAC = 0.5             # ... and in >= this share of the cameras seeing it
CORE_VOTES = 3             # a core voxel needs >= 3 cameras for its label ...
CORE_RATIO = 2             # ... and twice the votes of the other label
DECISIVE_E = 0.3           # |evidence| above which a camera counts as (dis)agreeing
FRAGMENT_SHARE = 0.1       # a mask below this share of the camera's larger mask is a fragment ...
INCOMPLETE_WEIGHT = 0.5    # ... and a camera-frame with one mask or a fragment counts this much
CORE_MIN_M3 = 0.0025       # smallest vote core with a usable centroid
GUARD_CORE_M3 = 0.0375     # smaller cores read as a vanished person (contact)
MERGE_CLOSE, MERGE_OPEN = 9, 5   # contact-mask morphology (frames)
COARSE_M = 0.25            # capture-volume search grid
COARSE_PAD_M = 3.0         # search box = camera-centre box grown by this
VOLUME_MARGIN_M = 0.5      # fine grid = occupied coarse box grown by this
VOLUME_SAMPLES = 400       # frames sampled to find the capture volume
MAX_SUBJECTS = 6

_MASK_NAME = re.compile(r"mask_(\d+)_(\d+)\.png$")


# --------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------

@dataclass
class RigCamera:
    """Calibrated camera; projects world points to raw mask pixels."""
    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    model: Optional[str] = None
    coeffs: Tuple[float, ...] = ()

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """World points (N, 3) -> pixels (N, 2) and depth (N,). Points whose
        pinhole projection falls far outside the image get NaN pixels, so a
        distortion polynomial can never fold them back inside."""
        Xc = X @ self.R.T + self.t
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = Xc[:, 0] / z
            yn = Xc[:, 1] / z
        K = self.K
        u0 = K[0, 0] * xn + K[0, 1] * yn + K[0, 2]
        v0 = K[1, 1] * yn + K[1, 2]
        far = ((u0 < -0.25 * self.width) | (u0 > 1.25 * self.width)
               | (v0 < -0.25 * self.height) | (v0 > 1.25 * self.height))
        if self.model in ("radtan", "opencv_brown"):
            c = list(self.coeffs) + [0.0] * 5
            k1, k2, p1, p2, k3 = c[:5]
            r2 = xn * xn + yn * yn
            rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
            yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
            u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
            v = K[1, 1] * yd + K[1, 2]
        elif self.model == "vicon_radial_2":
            u, v = _radial_correction(u0, v0, *[float(x) for x in self.coeffs[:5]])
        else:
            u, v = u0, v0
        uv = np.stack([u, v], 1)
        uv[far] = np.nan
        return uv, z


def camera_from_cam_data(name: str, cd: dict,
                         image_hw: Optional[Tuple[int, int]] = None) -> RigCamera:
    """Build a :class:`RigCamera` from a pipeline ``cam_data`` dict.

    Masks are drawn on the frames the tracker saw: raw frames unless the run
    undistorted them (``_undistort``), in which case the pinhole model applies.
    """
    K = np.asarray(cd["cam_int"], dtype=np.float64).reshape(3, 3)
    E = np.asarray(cd["cam_ext"], dtype=np.float64)
    R, t = E[:3, :3], E[:3, 3]
    w = cd.get("cam_img_w")
    h = cd.get("cam_img_h")
    if w is None or h is None:
        if image_hw is None:
            raise ValueError(f"{name}: no image size in cam_data")
        h, w = image_hw
    model, coeffs = None, ()
    if not cd.get("_undistort"):
        dist = _camera_from_cam_data(cd)
        if dist is not None:
            model, coeffs = dist.distortion_model, tuple(dist.distortion_coeffs)
    return RigCamera(name, K, R, t, int(np.asarray(w)), int(np.asarray(h)), model, coeffs)


# --------------------------------------------------------------------------
# masks (I/O layer)
# --------------------------------------------------------------------------

def list_mask_entries(mask_dir: str) -> List[Tuple[int, int, str]]:
    """``(frame, pid, name)`` for every mask entry (packed or loose)."""
    from core.mask_archive import MaskArchiveReader
    with MaskArchiveReader(mask_dir) as reader:
        names = reader.names()
    out = []
    for n in names:
        m = _MASK_NAME.match(n)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), n))
    return out


@dataclass
class CameraMasks:
    """Reduced-resolution masks of one camera, ids 1..n_ids.

    ``pix[off[n*t+k]:off[n*t+k+1]]`` are the flat grid cells of id ``k+1``
    at frame ``t``.
    """
    grid: Tuple[int, int]
    n_ids: int
    pix: np.ndarray
    off: np.ndarray

    @property
    def n_frames(self) -> int:
        return (len(self.off) - 1) // self.n_ids

    def plane(self, t: int, k: int) -> np.ndarray:
        if t >= self.n_frames:
            return self.pix[:0]
        i = self.n_ids * t + k
        return self.pix[self.off[i]:self.off[i + 1]]

    @classmethod
    def from_planes(cls, grid, n_ids: int, planes: Sequence[Sequence[np.ndarray]]) -> "CameraMasks":
        flat = [p for frame in planes for p in frame]
        off = np.zeros(len(flat) + 1, np.int64)
        off[1:] = np.cumsum([len(p) for p in flat])
        pix = (np.concatenate(flat).astype(np.uint32) if flat and off[-1]
               else np.zeros(0, np.uint32))
        return cls(tuple(grid), int(n_ids), pix, off)


def load_camera_masks(mask_dir: str, entries, n_frames: int, n_ids: int, downsample: int,
                      pool: ThreadPoolExecutor) -> Tuple[Optional[CameraMasks], Optional[Tuple[int, int]]]:
    """Decode ids 1..n_ids into reduced grids; returns the masks and the full
    image size (rows, cols). A packed archive is fetched with one sequential
    read (per-entry reads on a network share are latency-bound)."""
    from core.mask_archive import ARCHIVE_NAME, MaskArchiveReader
    wanted = [(f, p, n) for f, p, n in entries if 1 <= p <= n_ids and f < n_frames]
    shape = {}
    archive = os.path.join(mask_dir, ARCHIVE_NAME)
    blobs, reader = None, None
    if os.path.isfile(archive):
        import io
        import zipfile
        with open(archive, "rb") as fh, zipfile.ZipFile(io.BytesIO(fh.read())) as z:
            blobs = {n: z.read(n) for _f, _p, n in wanted}
    else:
        reader = MaskArchiveReader(mask_dir)

    def decode(item):
        f, p, n = item
        img = (cv2.imdecode(np.frombuffer(blobs[n], np.uint8), cv2.IMREAD_GRAYSCALE)
               if blobs is not None else reader.read(n))
        if img is None:
            return f, p, None
        H, W = img.shape[:2]
        if shape.setdefault("hw", (H, W)) != (H, W):
            raise ValueError(f"{mask_dir}/{n}: mask is {H}x{W}, other masks of this camera are "
                             f"{shape['hw'][0]}x{shape['hw'][1]}")
        gh, gw = max(1, round(H / downsample)), max(1, round(W / downsample))
        small = cv2.resize((img > 127).astype(np.uint8) * 255, (gw, gh),
                           interpolation=cv2.INTER_AREA)
        return f, p, np.flatnonzero(small >= MASK_COVER).astype(np.uint32)

    try:
        decoded = list(pool.map(decode, wanted))
    finally:
        if reader is not None:
            reader.close()
    full = shape.get("hw")
    if full is None:
        return None, None
    grid = (max(1, round(full[0] / downsample)), max(1, round(full[1] / downsample)))
    empty = np.zeros(0, np.uint32)
    planes = [[empty] * n_ids for _ in range(n_frames)]
    for f, p, cells in decoded:
        if cells is not None:
            planes[f][p - 1] = cells
    return CameraMasks.from_planes(grid, n_ids, planes), full


def permute_mask_ids(mask_dir: str, frame_perms: Dict[int, Tuple[int, ...]]) -> int:
    """Rename entries by a per-frame permutation (0-based: camera id k becomes
    id ``perm[k]``); returns the number of entries renamed.

    Packed layout: streamed into a temp archive, then atomically swapped in.
    Loose layout: two-phase rename so renames can never collide.
    """
    def target(f, p):
        perm = frame_perms.get(f)
        if perm is None or not 1 <= p <= len(perm) or perm[p - 1] == p - 1:
            return None
        return f"mask_{f:04d}_{perm[p - 1] + 1:02d}.png"

    archive = os.path.join(mask_dir, "masks.zip")
    renamed = 0
    if os.path.isfile(archive):
        import zipfile
        tmp = f"{archive}.tmp.{os.getpid()}"
        with zipfile.ZipFile(archive, "r") as zin, \
                zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zout:
            for info in zin.infolist():
                name = info.filename
                m = _MASK_NAME.match(name)
                new = target(int(m.group(1)), int(m.group(2))) if m else None
                if new is not None:
                    name = new
                    renamed += 1
                zout.writestr(name, zin.read(info.filename))
        os.replace(tmp, archive)
        return renamed
    moves = []
    for f, p, n in list_mask_entries(mask_dir):
        new = target(f, p)
        if new is not None:
            moves.append((os.path.join(mask_dir, n), os.path.join(mask_dir, new)))
    for src, _ in moves:
        os.replace(src, src + ".rigvote_tmp")
    for src, dst in moves:
        os.replace(src + ".rigvote_tmp", dst)
    return len(moves)


# --------------------------------------------------------------------------
# voxel lookup
# --------------------------------------------------------------------------

class VoxelIndex:
    """Voxel centres and their grid cell in every camera (static rig).

    ``idx[i, v]`` is the flat cell of voxel ``v`` in camera ``i`` (sentinel
    ``cells[i]`` when out of view). The inverse lookup (cell -> voxels) is a
    CSR per camera, so a mask's voxels are gathered from its cells directly.
    """

    def __init__(self, cams: Sequence[RigCamera], grids: Sequence[Tuple[int, int]],
                 V: np.ndarray, voxel_m: float, min_views: int = MIN_VOTES):
        C = len(cams)
        self.voxel_m3 = float(voxel_m) ** 3
        idx = np.empty((C, len(V)), np.int64)
        for i, (cam, (gh, gw)) in enumerate(zip(cams, grids)):
            uv, z = cam.project(V)
            with np.errstate(invalid="ignore"):
                col = np.floor(uv[:, 0] * gw / cam.width)
                row = np.floor(uv[:, 1] * gh / cam.height)
                ok = (z > NEAR_M) & (col >= 0) & (col < gw) & (row >= 0) & (row < gh)
            idx[i] = np.where(ok, np.nan_to_num(row) * gw + np.nan_to_num(col), gh * gw)
        keep = ((idx < np.array([gh * gw for gh, gw in grids])[:, None]).sum(0)
                >= min_views)
        self.V = V[keep]
        self.idx = idx[:, keep].astype(np.int32)
        self.cells = np.array([gh * gw for gh, gw in grids], np.int64)
        self.n_views = (self.idx < self.cells[:, None]).sum(0)
        self.csr = []
        for i in range(C):
            pix = self.idx[i]
            vis = np.flatnonzero(pix < self.cells[i])
            order = vis[np.argsort(pix[vis], kind="stable")].astype(np.int32)
            cnt = np.bincount(pix[vis], minlength=int(self.cells[i]))
            self.csr.append((order, np.cumsum(cnt) - cnt, cnt))

    @property
    def n(self) -> int:
        return len(self.V)

    def voxels_of(self, i: int, cells: np.ndarray) -> np.ndarray:
        """Voxels projecting into ``cells`` of camera ``i``."""
        order, start, cnt = self.csr[i]
        lens = cnt[cells]
        total = int(lens.sum())
        if total == 0:
            return order[:0]
        ends = np.cumsum(lens)
        rep = np.repeat(start[cells] - ends + lens, lens) + np.arange(total)
        return order[rep]


def grid_points(lo: np.ndarray, hi: np.ndarray, step: float) -> np.ndarray:
    axes = [np.arange(a + step / 2, b, step) for a, b in zip(lo, hi)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


# --------------------------------------------------------------------------
# voting core (pure; unit-tested on synthetic rigs)
# --------------------------------------------------------------------------

def permutations(n: int) -> np.ndarray:
    """All permutations of n ids as rows, identity first."""
    return np.array(list(itertools.permutations(range(n))), np.int64)


def composition_table(perms: np.ndarray) -> np.ndarray:
    """``table[d, p]`` = index of the permutation q with q[k] = d[p[k]]
    (apply p, then d)."""
    index = {tuple(p): i for i, p in enumerate(perms.tolist())}
    S = len(perms)
    table = np.empty((S, S), np.int64)
    for d in range(S):
        for p in range(S):
            table[d, p] = index[tuple(perms[d][perms[p]].tolist())]
    return table


def _frame_planes(masks: Sequence[CameraMasks], t: int, state, perms):
    """Per camera, the masks of frame ``t`` in label order under ``state``."""
    planes = []
    for i, m in enumerate(masks):
        own = [m.plane(t, k) for k in range(m.n_ids)]
        if state is not None and state[i, t]:
            p = perms[state[i, t]]
            relabeled = [None] * m.n_ids
            for k in range(m.n_ids):
                relabeled[p[k]] = own[k]
            own = relabeled
        planes.append(own)
    return planes


def _has_masks(planes) -> np.ndarray:
    return np.array([any(len(p) for p in pl) for pl in planes])


def _foreground(index: VoxelIndex, planes, has) -> np.ndarray:
    """Per voxel, the number of cameras whose masks cover it."""
    fg = [index.voxels_of(i, np.unique(np.concatenate(pl))) for i, pl in enumerate(planes) if has[i]]
    return (np.bincount(np.concatenate(fg), minlength=index.n) if fg
            else np.zeros(index.n, np.int64))


class ForegroundCounter:
    """:func:`_foreground` carried from frame to frame: only the cells whose
    coverage changed are looked up (exact integer updates; a voxel lies in one
    cell per camera, so no update repeats a voxel)."""

    def __init__(self, index: VoxelIndex):
        self.index = index
        self.base = np.r_[0, np.cumsum(index.cells)]
        self.cover = np.zeros(self.base[-1], bool)
        self.fg = np.zeros(index.n, np.int32)

    def update(self, planes) -> np.ndarray:
        cover = np.zeros_like(self.cover)
        for i, pl in enumerate(planes):
            for cells in pl:
                cover[self.base[i] + cells] = True
        changed = np.flatnonzero(cover != self.cover)
        cut = np.searchsorted(changed, self.base)
        for i in range(len(planes)):
            d = changed[cut[i]:cut[i + 1]]
            if d.size:
                on = cover[d]
                self.fg[self.index.voxels_of(i, d[on] - self.base[i])] += 1
                self.fg[self.index.voxels_of(i, d[~on] - self.base[i])] -= 1
        self.cover = cover
        return self.fg


# byte k of _BYTE_LANES[b] is bit k of b: summing over cameras counts every label at once
_BYTE_LANES = np.array([sum(((b >> k) & 1) << (8 * k) for k in range(8)) for b in range(256)],
                       np.uint64)


def _label_votes(g: np.ndarray, n: int) -> np.ndarray:
    """(n, K) int16 votes per label from per-camera label bits ``g`` (C, K) uint8."""
    vc = np.zeros((n, g.shape[1]), np.int16)
    for a in range(0, len(g), 255):          # a byte lane holds up to 255 cameras
        lanes = _BYTE_LANES[g[a:a + 255]].sum(0)
        for k in range(n):
            vc[k] += ((lanes >> np.uint64(8 * k)) & np.uint64(255)).astype(np.int16)
    return vc


def _top_two(votes: np.ndarray):
    """Per voxel of ``votes`` (n, K) (non-negative): the largest count ``m1``,
    its first label ``a1``, the largest count of any other label ``m2`` and
    the bit set ``t2`` of the other labels holding ``m2``."""
    m1 = votes[0].copy()
    m2 = np.zeros_like(m1)
    a1 = np.zeros(m1.shape, np.int8)
    for k in range(1, len(votes)):
        x = votes[k]
        np.maximum(m2, np.minimum(m1, x), out=m2)
        a1[x > m1] = k
        np.maximum(m1, x, out=m1)
    t2 = np.zeros(m1.shape, np.uint8)
    for k in range(len(votes)):
        t2 |= ((votes[k] == m2) & (a1 != k)).astype(np.uint8) << k
    return m1, a1, m2, t2


def frame_evidence(index: VoxelIndex, grids, planes, n: int, fg: Optional[np.ndarray] = None):
    """Leave-one-out evidence of every camera at one frame.

    Returns ``(M, cores)``: ``M[i]`` is camera i's (n, n) float64 matrix,
    entry (k, j) = how much better its mask k fits rig label j than the best
    other label (size-weighted, INCOMPLETE_WEIGHT on incomplete camera-frames;
    zeros = no evidence), and the rig's vote cores as ``(n, 4)`` rows of
    centroid xyz and voxel count (NaN when too small), from all cameras.
    ``fg``: the frame's per-voxel foreground counts when already known.

    A voxel is a core of label k when it is occupied, k has at least
    CORE_VOTES votes and CORE_RATIO times the votes of every other label.
    Such a k is the unique top label, and leaving one camera out lowers each
    count by at most one, so the leave-one-out core of any camera can only be
    the rig's top label; it is decided from the top two counts alone.
    """
    C = len(planes)
    M = np.zeros((C, n, n), np.float64)
    cores = np.full((n, 4), np.nan)
    has = _has_masks(planes)
    if fg is None:
        fg = _foreground(index, planes, has)
    cand0 = np.flatnonzero(fg >= MIN_VOTES)
    if cand0.size == 0:
        return M, cores
    fgc = fg[cand0]
    ne = index.n_views[cand0]
    for i in np.flatnonzero(~has):
        ne = ne - (index.idx[i, cand0] < index.cells[i])
    # leave-one-out occupancy can only hold where it holds with one camera fewer,
    # and full occupancy implies it: everything below lives on these voxels
    loo = fgc >= np.maximum(MIN_VOTES, OCC_FRAC * (ne - 1))
    sel = np.flatnonzero(loo)
    if sel.size == 0:
        return M, cores
    vox, fgc, ne = cand0[sel], fgc[sel], ne[sel]
    gathered = index.idx[:, vox]
    # all cameras' label images (bit k = label k) in one table; out-of-view cells read 255
    base = np.r_[0, np.cumsum(index.cells + 1)]
    L = np.zeros(base[-1], np.uint8)
    L[base[1:] - 1] = 255
    for i, pl in enumerate(planes):
        for k, cells_k in enumerate(pl):
            L[base[i] + cells_k] |= np.uint8(1 << k)
    g = L[gathered + base[:-1, None]]
    seen = g != 255
    g[~seen] = 0
    m1, a1, m2, t2 = _top_two(_label_votes(g, n))
    occ = fgc >= np.maximum(MIN_VOTES, OCC_FRAC * ne)
    core = occ & (m1 >= CORE_VOTES) & (m1 >= CORE_RATIO * m2)
    for k in range(n):
        core_k = core & (a1 == k)
        if core_k.sum() * index.voxel_m3 >= CORE_MIN_M3:
            cores[k, :3] = index.V[vox[core_k]].mean(0)
            cores[k, 3] = core_k.sum()

    # leave camera i out: its label bits b lower each voted count by one. Occupancy
    # then holds where i sees no mask (``loo``), else where it holds with one vote fewer.
    occ_b = (fgc - 1) >= np.maximum(MIN_VOTES, OCC_FRAC * (ne - 1))
    # the top count loses at most one vote, the runner-up at most one: no other voxel can qualify
    p = np.flatnonzero((m1 >= CORE_VOTES) & (m1 >= CORE_RATIO * (m2 - 1)))
    b, m1, a1, m2, t2, occ_b = g[:, p], m1[p], a1[p], m2[p], t2[p], occ_b[p]
    o1 = m1 - ((b >> a1) & 1)
    o2 = m2 - ((b & t2) == t2)
    keep = (has[:, None] & seen[:, p] & ((b == 0) | occ_b)
            & (o1 >= CORE_VOTES) & (o1 >= CORE_RATIO * o2))
    # per camera, one image per label of its leave-one-out cores, flat in one buffer
    rbase = np.r_[0, np.cumsum(n * index.cells)]
    img = np.zeros(rbase[-1], np.uint8)
    img[((rbase[:-1, None] + index.cells[:, None] * a1) + gathered[:, p])[keep]] = 1
    kernel = np.ones((3, 3), np.uint8)
    for i in np.flatnonzero(has):
        pl = planes[i]
        gh, gw = grids[i]
        sizes = np.array([len(c) for c in pl])
        big = max(sizes.max(), 1)
        nz = np.flatnonzero(sizes)
        own = np.concatenate([pl[k] for k in nz])
        planes_i = img[rbase[i]:rbase[i + 1]].reshape(n, gh, gw)
        # regions are only read on the camera's own mask cells
        region = np.stack([cv2.dilate(planes_i[k], kernel).ravel()[own] for k in range(n)]) != 0
        excl = region & (region.sum(0) == 1)
        starts = np.r_[0, np.cumsum(sizes[nz])[:-1]]
        share = np.add.reduceat(excl, starts, axis=1, dtype=np.int64).T / sizes[nz][:, None]
        order = np.argsort(-share, axis=1, kind="stable")
        r = np.arange(len(nz))
        alt = np.repeat(share[r, order[:, 0]][:, None], n, axis=1)
        alt[r, order[:, 0]] = share[r, order[:, 1]]
        M[i, nz] = (sizes[nz] / big)[:, None] * (share - alt)
        if sizes.min() < FRAGMENT_SHARE * big:
            M[i] *= INCOMPLETE_WEIGHT
    return M, cores


def permutation_scores(M: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """(..., n, n) evidence -> (..., S) float32 score of every permutation."""
    n = perms.shape[1]
    picked = M[..., np.arange(n), perms]          # (..., S, n)
    total = picked[..., 0]
    for k in range(1, n):
        total = total + picked[..., k]
    return total.astype(np.float32)


def viterbi_states(M: np.ndarray, perms: np.ndarray, switch_cost: float) -> np.ndarray:
    """Per-camera permutation decoding of evidence ``M`` (T, C, n, n);
    returns (C, T) permutation indices (0 = keep the labels)."""
    T, C = M.shape[:2]
    S = len(perms)
    out = np.zeros((C, T), np.int64)
    if T == 0:
        return out
    rows = np.arange(C)
    states = np.arange(S)[None, :]
    back = np.zeros((T, C, S), np.int16 if S < 32768 else np.int32)
    acc = -permutation_scores(M[0], perms).astype(np.float64)
    for t in range(1, T):
        order = np.argsort(acc, 1, kind="stable")
        b1 = order[:, :1]
        b2 = order[:, 1:2] if S > 1 else b1
        m1 = np.take_along_axis(acc, b1, 1)
        m2 = np.take_along_axis(acc, b2, 1)
        is_best = states == b1
        cross = np.where(is_best, m2, m1) + switch_cost
        from_other = cross < acc
        back[t] = np.where(from_other, np.where(is_best, b2, b1), states)
        acc = np.minimum(acc, cross) - permutation_scores(M[t], perms)
    s = np.argmin(acc, 1)
    for t in range(T - 1, -1, -1):
        out[:, t] = s
        if t > 0:
            s = back[t, rows, s].astype(np.int64)
    return out


def identity_margin(M: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """(T, C) half the score gap between keeping the labels and the best
    relabel (two subjects: exactly core.rig_vote's evidence)."""
    sc = permutation_scores(M, perms).astype(np.float64)
    best_other = sc[..., 1:].max(-1) if sc.shape[-1] > 1 else sc[..., 0]
    return ((sc[..., 0] - best_other) / 2).T.astype(np.float32)


def merge_guard(cores: np.ndarray, E0: np.ndarray, states: np.ndarray, voxel_m3: float,
                merge_sep_m: float, max_minority: float):
    """Keep relabel ``states`` (C, T) only on stretches between contacts the
    rig is not split on, and inside a contact window only where a camera holds
    the same permutation on both sides. ``cores`` (T, n, 4): vote-core
    centroids and voxel counts under the final labels; ``E0`` (C, T):
    identity margins on the original labels. Returns
    ``(guarded_states, merged (T,), stretches)``."""
    T, n = cores.shape[:2]
    size = cores[..., 3] * voxel_m3
    valid = np.isfinite(size).all(1) & (np.nan_to_num(size).min(1) >= GUARD_CORE_M3)
    iu = np.triu_indices(n, 1)
    pts = cores[..., :3]
    pair = np.linalg.norm(pts[:, iu[0]] - pts[:, iu[1]], axis=-1)
    sep = np.nan_to_num(pair, nan=0.0).min(1) if len(iu[0]) else np.full(T, np.inf)
    merged = ~valid | (sep < merge_sep_m)
    merged = binary_opening(binary_closing(merged, np.ones(MERGE_CLOSE)), np.ones(MERGE_OPEN))
    pos = (E0 > DECISIVE_E).sum(0)
    neg = (E0 < -DECISIVE_E).sum(0)
    minority = neg / np.maximum(pos + neg, 1)
    d = np.diff(np.r_[0, merged.astype(int), 0])
    ends = np.flatnonzero(d == -1)
    bounds = sorted(set([0] + [int(x) for x in ends] + [T]))
    out = states.copy()
    stretches = []
    for a, b in zip(bounds, bounds[1:]):
        m = float(np.median(minority[a:b]))
        keep = m <= max_minority
        if not keep:
            out[:, a:b] = 0
        stretches.append({"start": a, "end": b - 1, "median_minority": round(m, 4),
                          "applied": bool(keep),
                          "relabeled_camera_frames": int((states[:, a:b] != 0).sum())})
    out[:, merged] = 0
    for a, b in _segments(merged):
        if a > 0 and b + 1 < T:
            same = (out[:, a - 1] == out[:, b + 1]) & (out[:, a - 1] != 0)
            out[same, a:b + 1] = out[same, a - 1][:, None]
    return out, merged, stretches


def capture_volume(cams: Sequence[RigCamera], masks: Sequence[CameraMasks],
                   n_frames: int, n: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Box around everything the rig sees as occupied on sampled frames
    (cells seen by at least half the cameras; see core.rig_vote)."""
    centres = np.array([c.centre for c in cams])
    lo, hi = centres.min(0) - COARSE_PAD_M, centres.max(0) + COARSE_PAD_M
    coarse = VoxelIndex(cams, [m.grid for m in masks], grid_points(lo, hi, COARSE_M), COARSE_M,
                        min_views=max(MIN_VOTES, int(np.ceil(OCC_FRAC * len(cams)))))
    if coarse.n == 0:
        return None
    hit = np.zeros(coarse.n, bool)
    counter = ForegroundCounter(coarse)
    for t in np.unique(np.linspace(0, n_frames - 1, min(n_frames, VOLUME_SAMPLES)).astype(int)):
        planes = _frame_planes(masks, t, None, None)
        has = _has_masks(planes)
        fg = counter.update(planes)
        cand = np.flatnonzero(fg >= MIN_VOTES)
        ne = ((coarse.idx[:, cand] < coarse.cells[:, None]) & has[:, None]).sum(0)
        hit[cand[fg[cand] >= OCC_FRAC * ne]] = True
    if not hit.any():
        return None
    occ = coarse.V[hit]
    return (occ.min(0) - COARSE_M / 2 - VOLUME_MARGIN_M,
            occ.max(0) + COARSE_M / 2 + VOLUME_MARGIN_M)


@dataclass
class RigVoteResult:
    cams: List[str] = field(default_factory=list)
    n_ids: int = 0
    perms: Optional[np.ndarray] = None        # (S, n) permutation table
    states: Optional[np.ndarray] = None       # (C, T) applied permutation indices
    raw_states: Optional[np.ndarray] = None   # (C, T) before the merge guard
    evidence0: Optional[np.ndarray] = None    # (C, T) identity margin on original labels
    cores: Optional[np.ndarray] = None        # (T, n, 4) final vote cores
    merged: Optional[np.ndarray] = None       # (T,) contact frames
    stretches: List[dict] = field(default_factory=list)
    passes: List[dict] = field(default_factory=list)
    n_voxels: int = 0
    bounds: Optional[List[List[float]]] = None
    skipped: str = ""

    @property
    def swaps(self) -> Optional[np.ndarray]:
        """(C, T) True where a camera-frame is relabelled."""
        return None if self.states is None else self.states != 0


def compute_rig_vote(cams: Sequence[RigCamera], masks: Sequence[CameraMasks],
                     n_frames: int, cfg: Optional[dict] = None,
                     log: Callable[[str], None] = print) -> RigVoteResult:
    """Pure decision step: per-camera relabel segments for an in-memory rig."""
    p = dict(DEFAULTS, **(cfg or {}))
    n = masks[0].n_ids
    perms = permutations(n)
    comp = composition_table(perms)
    res = RigVoteResult(cams=[c.name for c in cams], n_ids=n, perms=perms)
    grids = [m.grid for m in masks]
    box = capture_volume(cams, masks, n_frames, n)
    if box is None:
        res.skipped = "no occupied capture volume"
        return res
    step = float(p["rig_vote_voxel_m"])
    index = VoxelIndex(cams, grids, grid_points(box[0], box[1], step), step)
    res.n_voxels, res.bounds = index.n, [box[0].round(2).tolist(), box[1].round(2).tolist()]
    C, T = len(cams), n_frames
    state = np.zeros((C, T), np.int64)
    M = np.zeros((T, C, n, n), np.float64)
    cores = np.full((T, n, 4), np.nan)
    todo = np.arange(T)
    counter = ForegroundCounter(index)
    for k in range(int(p["rig_vote_max_passes"])):
        t0 = time.time()
        # a frame's vote depends only on that frame's labels: later passes
        # recompute just the frames where some camera changed
        for t in todo:
            planes = _frame_planes(masks, t, state, perms)
            M[t], cores[t] = frame_evidence(index, grids, planes, n, counter.update(planes))
        if k == 0:
            res.evidence0 = identity_margin(M, perms)
        delta = viterbi_states(M, perms, float(p["rig_vote_switch_cost"]))
        state = comp[delta, state]
        changed = delta != 0
        todo = np.flatnonzero(changed.any(0))
        res.cores = cores
        res.passes.append({"pass": k, "seconds": round(time.time() - t0, 1),
                           "changed": float(changed.mean()), "relabeled": float((state != 0).mean())})
        log(f"[rig_vote] pass {k}: changed {changed.mean():.4f} of camera-frames, "
            f"relabeled {(state != 0).mean():.4f} ({time.time() - t0:.0f}s)")
        if not changed.any():
            break
    res.raw_states = state
    res.states, res.merged, res.stretches = merge_guard(
        res.cores, res.evidence0, state, index.voxel_m3,
        float(p["rig_vote_merge_sep_m"]), float(p["rig_vote_max_minority"]))
    for s in res.stretches:
        if s["relabeled_camera_frames"] and not s["applied"]:
            log(f"[rig_vote] frames {s['start']}-{s['end']}: cameras split "
                f"(median minority {s['median_minority']:.2f}), abstaining.")
    return res


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def resolve_mode(value, n_calibrated_cams: int, min_cameras: int) -> bool:
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "auto":
            return n_calibrated_cams >= int(min_cameras)
        return v in {"1", "true", "yes", "on"} and n_calibrated_cams >= 2
    return bool(value) and n_calibrated_cams >= 2


def _segments(row: np.ndarray) -> List[Tuple[int, int]]:
    d = np.diff(np.r_[0, row.astype(int), 0])
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def _state_segments(row: np.ndarray) -> List[Tuple[int, int, int]]:
    """Maximal runs of one non-zero permutation index: (start, end, index)."""
    out, t, T = [], 0, len(row)
    while t < T:
        if row[t]:
            s = t
            while t + 1 < T and row[t + 1] == row[s]:
                t += 1
            out.append((s, t, int(row[s])))
        t += 1
    return out


def run_rig_vote(seq_out_dir: str, cam_data_by_name: Dict[str, dict],
                 cfg: Optional[dict] = None, log: Callable[[str], None] = print,
                 apply: bool = True) -> RigVoteResult:
    """Load the rig's masks, decide relabel segments, and (``apply``) rename
    the mask entries and write the reports. Safe no-op outside 2 to
    MAX_SUBJECTS subjects or with too few calibrated cameras."""
    p = dict(DEFAULTS, **(cfg or {}))
    names = sorted(c for c in cam_data_by_name
                   if os.path.isdir(os.path.join(seq_out_dir, c, "masks")))
    entries = {c: list_mask_entries(os.path.join(seq_out_dir, c, "masks")) for c in names}
    entries = {c: e for c, e in entries.items() if e}
    pids = sorted({pid for e in entries.values() for _f, pid, _n in e})
    res = RigVoteResult(cams=sorted(entries))
    n_ids = max(pids) if pids else 0
    if len(pids) < 2 or n_ids > MAX_SUBJECTS:
        res.skipped = f"mask ids {pids}: rig vote needs 2 to {MAX_SUBJECTS} subjects"
    elif len(entries) < MIN_VOTES + 1:
        res.skipped = f"{len(entries)} camera(s) with masks"
    if res.skipped:
        log(f"[rig_vote] skipped: {res.skipped}.")
        return res
    t0 = time.time()
    n_frames = 1 + max(f for e in entries.values() for f, _p, _n in e)
    cams, masks = [], []
    workers = max(1, min(int(p["rig_vote_workers"]), len(os.sched_getaffinity(0))))
    with ThreadPoolExecutor(workers) as pool:
        for c in res.cams:
            m, full = load_camera_masks(os.path.join(seq_out_dir, c, "masks"), entries[c],
                                        n_frames, n_ids, int(p["rig_vote_downsample"]), pool)
            if m is None:
                continue
            cams.append(camera_from_cam_data(c, cam_data_by_name[c], full))
            masks.append(m)
    log(f"[rig_vote] {len(cams)} camera(s), {n_frames} frame(s), {n_ids} subjects loaded "
        f"in {time.time() - t0:.0f}s.")
    res = compute_rig_vote(cams, masks, n_frames, p, log)
    if res.skipped or not apply:
        if res.skipped:
            log(f"[rig_vote] skipped: {res.skipped}.")
        return res

    relabeled = {}
    for i, c in enumerate(res.cams):
        segs = _state_segments(res.states[i])
        if not segs:
            continue
        frame_perms = {f: tuple(int(x) for x in res.perms[s]) for a, b, s in segs
                       for f in range(a, b + 1)}
        n = permute_mask_ids(os.path.join(seq_out_dir, c, "masks"), frame_perms)
        relabeled[c] = [(a, b, [int(x) + 1 for x in res.perms[s]]) for a, b, s in segs]
        with open(os.path.join(seq_out_dir, c, "rig_vote_report.json"), "w") as fh:
            json.dump([{"start": a, "end": b, "new_ids": perm} for a, b, perm in relabeled[c]],
                      fh, indent=2)
        log(f"[rig_vote] {c}: relabeled {len(frame_perms)} frame(s) in "
            f"{len(segs)} segment(s) ({n} files).")
    summary = {
        "params": {k: p[k] for k in DEFAULTS},
        "cameras": res.cams, "frames": n_frames, "subjects": n_ids, "voxels": res.n_voxels,
        "bounds": res.bounds, "passes": res.passes,
        "contact_windows": _segments(res.merged), "stretches": res.stretches,
        "relabeled": {c: [[a, b, perm] for a, b, perm in v] for c, v in relabeled.items()},
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(seq_out_dir, "rig_vote_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return res

