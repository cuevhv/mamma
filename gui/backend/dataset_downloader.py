"""Dataset download orchestrator — companion to data_readiness.py.

Wraps the ``data/download_mamma_*.sh`` scripts behind a single GUI:

  /api/datasets/catalog       — family + option matrix (no creds)
  /api/datasets/plan          — file plan from a user selection (no creds)
  /api/datasets/start         — kick off a download job (creds in body, one
                                shot, never persisted; same contract as
                                data_readiness.py's MPI flow)
  /api/datasets/job/<id>      — progress for one job
  /api/datasets/job/<id>/cancel  — request cancellation

The sequence lists shipped by the bash scripts are the canonical
inventory; we parse them at runtime from ``data/download_mamma_*.sh`` so
adding a new sequence in the shell script automatically surfaces in the
widget. We do NOT shell out — the actual HTTPS POSTs are driven by
``urllib.request`` here so credentials never appear in a process
argument list (no leak via ``/proc/<pid>/cmdline``).
"""
from __future__ import annotations

import dataclasses
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Optional

from flask import Flask, jsonify, request

# Reuse the same security-audited helpers and constants the readiness
# widget uses. _is_html_error and _safe_error in particular are the
# crucial bits — they ensure HTML "auth failed" pages are caught early
# and credentials never appear in any error string returned to the user.
from data_readiness import (
    _ALLOWED_HOSTS, _MPI_DOWNLOAD_URL, _DOWNLOAD_LIMIT_MSG,
    _check_url, _is_html_error, _safe_error,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"

# Same wire shape the bash scripts use:
#   POST https://download.is.tue.mpg.de/download.php?domain=mamma&resume=1&sfile=datasets/<rel>
#   body: username=<>&password=<>
_REMOTE_ROOT = "datasets"


# ─── bash-array parser ────────────────────────────────────────────────────

_ARRAY_BEGIN_RE = re.compile(r"^(?P<name>[A-Z_][A-Z0-9_]*)=\(\s*$", re.MULTILINE)
_ARRAY_END_RE   = re.compile(r"^\)\s*$", re.MULTILINE)


def _parse_bash_arrays(script_path: Path) -> dict:
    """Extract every ``VAR=( item1 item2 ... )`` array from a bash script.

    Tolerant of inline ``#`` comments and either quoted or unquoted items.
    Returns ``{name: tuple_of_strings}``. Items that look like
    ``"<int>:<path>"`` (used by the iphone script to embed person counts)
    are returned verbatim — the caller splits them."""
    if not script_path.is_file():
        return {}
    text = script_path.read_text()
    out: dict = {}
    for m in _ARRAY_BEGIN_RE.finditer(text):
        em = _ARRAY_END_RE.search(text, m.end())
        if not em:
            continue
        block = text[m.end():em.start()]
        items = []
        for line in block.splitlines():
            # Strip inline comment. All items in our scripts are simple
            # path-like strings without `#`, so this is safe.
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            if not line:
                continue
            if (line.startswith('"') and line.endswith('"')) or \
               (line.startswith("'") and line.endswith("'")):
                line = line[1:-1]
            items.append(line)
        out[m.group('name')] = tuple(items)
    return out


_arrays_cache: dict[str, dict] = {}


def _arrays(script_name: str) -> dict:
    if script_name not in _arrays_cache:
        _arrays_cache[script_name] = _parse_bash_arrays(_DATA_DIR / script_name)
    return _arrays_cache[script_name]


def _resolve_list(sel: dict, key: str, default: list) -> list:
    """Return ``sel[key]`` when present (even if empty), else ``default``.

    Use this instead of ``sel.get(key) or default``: that pattern treats
    an explicitly-passed empty list as "I didn't specify" and silently
    falls back to defaults, which means a user un-checking every option
    accidentally triggers a download-everything plan."""
    if key not in sel:
        return list(default)
    val = sel[key]
    if not isinstance(val, list):
        return list(default)
    return list(val)


# ─── Camera presets ───────────────────────────────────────────────────────

CAM_IOI32 = tuple(f"IOI_{i:02d}" for i in range(1, 33))
CAM_IOI16 = tuple(f"IOI_{i:02d}" for i in range(1, 17))
CAM_IPHONE = ("A001", "B001", "C001", "D001")


# ─── Family catalog ───────────────────────────────────────────────────────

# A "family" is one of the five .sh scripts. Each lists the available
# content groups (dance styles, locations, etc.), asset types, and the
# video variants (which are mutually exclusive). The expand() callable
# turns a user selection into a flat list of (sfile, local_relpath)
# tuples, where sfile is the wire path under "datasets/" and
# local_relpath is the on-disk path under data/.

@dataclasses.dataclass(frozen=True)
class _Family:
    id: str
    label: str
    description: str
    script: str
    content_groups: tuple   # tuple of (id, label, count)
    asset_types: tuple      # tuple of (id, label, description)
    video_variants: tuple   # tuple of (id, label, description). Includes
                            # "none" as the first entry. Empty if family
                            # has no video options (syn_wd).
    camera_kind: str        # 'ioi32', 'ioi16-or-32', 'iphone4', 'none'
    expand: Callable        # selection dict -> list[(sfile, local_relpath)]
    notes: str = ""


# ── Markerless dance ────────────────────────────────────────────────

_DANCE_GROUPS = (
    ("westcoastswing", "West Coast Swing", "WESTCOASTSWING_SEQUENCES"),
    ("bachata",        "Bachata",          "BACHATA_SEQUENCES"),
    ("breakdance",     "Breakdance",       "BREAKDANCE_SEQUENCES"),
    ("ballroom",       "Ballroom",         "BALLROOM_SEQUENCES"),
)

def _dance_person_count(ds_seq: str) -> int:
    """Mirror of dance_person_count() in download_mamma_dance.sh."""
    seq_name = ds_seq.rsplit("/", 1)[-1]
    if "Breakdance" not in ds_seq:
        return 2
    if re.search(r"_\d{5}(_\d+){1,3}$", seq_name):
        return 1
    if re.search(r"_\d{5}_\d{5}(_\d+){1,3}$", seq_name):
        return 2
    return 2


def _expand_dance(sel: dict) -> list[tuple[str, str]]:
    arr = _arrays("download_mamma_dance.sh")
    groups = _resolve_list(sel, "groups", [g[0] for g in _DANCE_GROUPS])
    asset_types = set(_resolve_list(sel, "asset_types", ["meta", "pred"]))
    video_variant = sel.get("video_variant") or "videos_crf24"
    cameras = _resolve_list(sel, "cameras", list(CAM_IOI32))
    seq_filter = set(sel.get("sequences") or [])

    group_to_arr = {g[0]: g[2] for g in _DANCE_GROUPS}
    seqs: list[str] = []
    for g in groups:
        seqs.extend(arr.get(group_to_arr.get(g, ""), ()))
    if seq_filter:
        seqs = [s for s in seqs if s in seq_filter]

    files: list[tuple[str, str]] = []
    for ds in seqs:
        n_people = _dance_person_count(ds)
        if "meta" in asset_types:
            files.append((f"{ds}/meta/global.npz",) * 2)
            for cam in cameras:
                f = f"{ds}/meta/{cam}.npz"; files.append((f, f))
        if "pred" in asset_types:
            for i in range(n_people):
                f = f"{ds}/pred/params_{i:02d}.npz"; files.append((f, f))
        if "preview" in asset_types:
            f = f"{ds}/preview/overlay_grid.mp4"; files.append((f, f))
        if video_variant and video_variant != "none":
            for cam in cameras:
                f = f"{ds}/{video_variant}/{cam}.mp4"; files.append((f, f))
    return files


# ── Markerless multi-people ────────────────────────────────────────

_MULTI_GROUPS = (
    ("3p", "3 people", "SEQUENCES_3P", 3),
    ("4p", "4 people", "SEQUENCES_4P", 4),
    ("5p", "5 people", "SEQUENCES_5P", 5),
    ("6p", "6 people", "SEQUENCES_6P", 6),
)


def _expand_multi(sel: dict) -> list[tuple[str, str]]:
    arr = _arrays("download_mamma_multi_people.sh")
    groups = _resolve_list(sel, "groups", [g[0] for g in _MULTI_GROUPS])
    asset_types = set(_resolve_list(sel, "asset_types", ["meta", "pred"]))
    video_variant = sel.get("video_variant") or "videos_crf24"
    cameras = _resolve_list(sel, "cameras", list(CAM_IOI32))
    seq_filter = set(sel.get("sequences") or [])

    group_to_arr = {g[0]: (g[2], g[3]) for g in _MULTI_GROUPS}
    seqs: list[tuple[str, int]] = []
    for g in groups:
        if g not in group_to_arr:
            continue
        arr_name, n_people = group_to_arr[g]
        for s in arr.get(arr_name, ()):
            seqs.append((s, n_people))
    if seq_filter:
        seqs = [(s, n) for (s, n) in seqs if s in seq_filter]

    files: list[tuple[str, str]] = []
    for ds, n_people in seqs:
        if "meta" in asset_types:
            files.append((f"{ds}/meta/global.npz",) * 2)
            for cam in cameras:
                f = f"{ds}/meta/{cam}.npz"; files.append((f, f))
        if "pred" in asset_types:
            for i in range(n_people):
                f = f"{ds}/pred/params_{i:02d}.npz"; files.append((f, f))
        if "preview" in asset_types:
            f = f"{ds}/preview/overlay_grid.mp4"; files.append((f, f))
        if video_variant and video_variant != "none":
            for cam in cameras:
                f = f"{ds}/{video_variant}/{cam}.mp4"; files.append((f, f))
    return files


# ── Markerless iPhone ─────────────────────────────────────────────

_IPHONE_GROUPS = (
    ("indoors",  "Indoor (16)",  "INDOORS_SEQUENCES"),
    ("outdoors", "Outdoor (26)", "OUTDOORS_SEQUENCES"),
)


def _expand_iphone(sel: dict) -> list[tuple[str, str]]:
    arr = _arrays("download_mamma_iphone.sh")
    groups = _resolve_list(sel, "groups", [g[0] for g in _IPHONE_GROUPS])
    asset_types = set(_resolve_list(sel, "asset_types", ["meta", "pred"]))
    video_variant = sel.get("video_variant") or "videos_light"
    cameras = _resolve_list(sel, "cameras", list(CAM_IPHONE))
    seq_filter = set(sel.get("sequences") or [])

    group_to_arr = {g[0]: g[2] for g in _IPHONE_GROUPS}
    entries: list[tuple[str, int]] = []   # (sequence_path, person_count)
    for g in groups:
        for entry in arr.get(group_to_arr.get(g, ""), ()):
            # Each entry is "<n_people>:<path>"
            if ":" not in entry:
                continue
            n_str, path = entry.split(":", 1)
            try:
                entries.append((path, int(n_str)))
            except ValueError:
                continue
    if seq_filter:
        entries = [(p, n) for (p, n) in entries if p in seq_filter]

    files: list[tuple[str, str]] = []
    for ds, n_people in entries:
        if "meta" in asset_types:
            files.append((f"{ds}/meta/global.npz",) * 2)
            for cam in cameras:
                f = f"{ds}/meta/{cam}.npz"; files.append((f, f))
        if "pred" in asset_types:
            for i in range(n_people):
                f = f"{ds}/pred/params_{i:02d}.npz"; files.append((f, f))
        if "preview" in asset_types:
            f = f"{ds}/preview/overlay_grid.mp4"; files.append((f, f))
        if video_variant and video_variant != "none":
            for cam in cameras:
                f = f"{ds}/{video_variant}/{cam}.mp4"; files.append((f, f))
    return files


# ── MammaEval ─────────────────────────────────────────────────────

# Eval is a single SEQUENCES array spanning three sub-families with
# different camera counts: singles=16 cams, extra=16 cams, dance=32 cams.
# We surface the sub-families as "groups" for the user even though the
# shell script treats them as one list.

_EVAL_GROUPS = (
    ("eval_singles", "Singles (22 seq · 16 cams)", "mamma_eval_singles"),
    ("eval_extra",   "Extra (12 seq · 16 cams)",   "mamma_eval_extra"),
    ("eval_dance",   "Dance (18 seq · 32 cams)",   "mamma_eval_dance"),
)


def _expand_eval(sel: dict) -> list[tuple[str, str]]:
    arr = _arrays("download_mamma_eval.sh")
    groups = set(_resolve_list(sel, "groups", [g[0] for g in _EVAL_GROUPS]))
    asset_types = set(_resolve_list(sel, "asset_types", ["gt"]))
    video_variant = sel.get("video_variant") or "videos_crf24"
    # For eval, the user CAN pass an explicit camera subset; an empty
    # camera list means "use defaults" rather than "no cameras".
    cli_cams = _resolve_list(sel, "cameras", [])
    seq_filter = set(sel.get("sequences") or [])

    prefix_to_group = {p: gid for (gid, _, p) in _EVAL_GROUPS}
    seqs_with_cams: list[tuple[str, list[str]]] = []
    for ds_seq in arr.get("SEQUENCES", ()):
        prefix = ds_seq.split("/", 1)[0]
        gid = prefix_to_group.get(prefix)
        if gid not in groups:
            continue
        if seq_filter and ds_seq not in seq_filter:
            continue
        # singles + extra ship with 16 cams, dance with 32; user cli
        # override (when present) clips to that selection.
        if prefix in ("mamma_eval_singles", "mamma_eval_extra"):
            default_cams = list(CAM_IOI16)
        else:
            default_cams = list(CAM_IOI32)
        if cli_cams:
            cams = [c for c in cli_cams if c in default_cams]
        else:
            cams = default_cams
        seqs_with_cams.append((ds_seq, cams))

    files: list[tuple[str, str]] = []
    for ds_seq, cams in seqs_with_cams:
        if "gt" in asset_types:
            files.append((f"{ds_seq}/gt/global.npz",) * 2)
            for cam in cams:
                f = f"{ds_seq}/gt/{cam}.npz"; files.append((f, f))
        if "masks" in asset_types:
            for cam in cams:
                f = f"{ds_seq}/masks/{cam}_masks.tar"; files.append((f, f))
        if "markers" in asset_types and ds_seq.startswith("mamma_eval_extra/"):
            for fname in ("vicon_m37.npy", "baseline_m37.npy", "labels_m37.npy"):
                f = f"{ds_seq}/markers/{fname}"; files.append((f, f))
        if "preview" in asset_types:
            for fname in ("overlay_grid.mp4", "masks_grid.mp4"):
                f = f"{ds_seq}/preview/{fname}"; files.append((f, f))
        if video_variant and video_variant != "none":
            for cam in cams:
                f = f"{ds_seq}/{video_variant}/{cam}.mp4"; files.append((f, f))
    return files


# ── MammaSyn (training webdataset) ────────────────────────────────

_SYN_GROUPS = (
    ("interactions", "Interactions (Harmony4D, Hi4D, Inter-X, …)", "INTERACTIONS_DATASETS"),
    ("singles",      "Singles (BEDLAM, MoYo)",                     "SINGLES_DATASETS"),
    ("hands",        "Hands (InterHand, SignAvatars)",             "HANDS_DATASETS"),
)


def _expand_syn(sel: dict) -> list[tuple[str, str]]:
    """For syn_wd we only know the top-level dataset names up-front; the
    .tar shards inside each dataset are listed in a remote
    `tar_train_list.txt`. We return one entry per *manifest* file here;
    the worker expands them at download time after fetching the manifest.

    The local path mirrors the bash script's behaviour: ``OUTPUT_DIR``
    in the script is ``data/training_webdataset``, so files actually
    land at ``data/training_webdataset/training_webdataset/<ds>/...``.
    Quirky, but matches the shipped script so the widget and the script
    are interchangeable."""
    arr = _arrays("download_mamma_syn_wd.sh")
    groups = _resolve_list(sel, "groups", [g[0] for g in _SYN_GROUPS])
    seq_filter = set(sel.get("sequences") or [])  # dataset-name filter
    group_to_arr = {g[0]: g[2] for g in _SYN_GROUPS}

    datasets: list[str] = []
    for g in groups:
        for ds in arr.get(group_to_arr.get(g, ""), ()):
            datasets.append(ds)
    if seq_filter:
        datasets = [d for d in datasets if d in seq_filter]

    files: list[tuple[str, str]] = []
    for ds in datasets:
        # sfile path that the .sh script uses for the manifests.
        for manifest in ("tar_train_list.txt", "train_data.txt", "get_dataset_list.sh"):
            s = f"training_webdataset/{ds}/{manifest}"
            files.append((s, f"training_webdataset/{s}"))
    return files


# ── Catalog assembly ─────────────────────────────────────────────

def _gc(group_id: str, label: str, count: int) -> dict:
    return {"id": group_id, "label": label, "count": count}


def _at(at_id: str, label: str, desc: str) -> dict:
    return {"id": at_id, "label": label, "description": desc}


def _vv(vv_id: str, label: str, desc: str) -> dict:
    return {"id": vv_id, "label": label, "description": desc}


FAMILIES: dict[str, _Family] = {
    "dance": _Family(
        id="dance",
        label="Markerless Dance",
        description="Multi-view dance with 32 IOI cameras. 4 styles, 123 sequences.",
        script="download_mamma_dance.sh",
        content_groups=_DANCE_GROUPS,
        asset_types=(
            ("meta",    "Meta",         "Camera params + per-frame metadata"),
            ("pred",    "Predictions",  "MAMMA SMPL-X params, per person"),
            ("preview", "Preview",      "Overlay-grid preview video"),
        ),
        video_variants=(
            ("none",         "No videos",      "Skip videos (metadata only)"),
            ("videos_crf24", "Light · CRF24",  "H.264, smallest"),
            ("videos_crf16", "Lossy · CRF16",  "H.264, medium"),
            ("videos",       "Original · CRF5","H.264, near-lossless"),
        ),
        camera_kind="ioi32",
        expand=_expand_dance,
    ),
    "multi": _Family(
        id="multi",
        label="Markerless Multi-People",
        description="3–6 person interactions, 32 IOI cameras. 34 sequences.",
        script="download_mamma_multi_people.sh",
        content_groups=_MULTI_GROUPS,
        asset_types=(
            ("meta",    "Meta",         "Camera params + per-frame metadata"),
            ("pred",    "Predictions",  "MAMMA SMPL-X params, per person"),
            ("preview", "Preview",      "Overlay-grid preview video"),
        ),
        video_variants=(
            ("none",         "No videos",      "Skip videos (metadata only)"),
            ("videos_crf24", "Light · CRF24",  "H.264, smallest"),
            ("videos_crf16", "Lossy · CRF16",  "H.264, medium"),
            ("videos",       "Original · CRF5","H.264, near-lossless"),
        ),
        camera_kind="ioi32",
        expand=_expand_multi,
    ),
    "iphone": _Family(
        id="iphone",
        label="Markerless iPhone",
        description="4-iPhone capture, indoor + outdoor. 42 sequences.",
        script="download_mamma_iphone.sh",
        content_groups=_IPHONE_GROUPS,
        asset_types=(
            ("meta",    "Meta",         "Camera params + per-frame metadata"),
            ("pred",    "Predictions",  "MAMMA SMPL-X params, per person"),
            ("preview", "Preview",      "Overlay-grid preview video"),
        ),
        video_variants=(
            ("none",          "No videos",     "Skip videos (metadata only)"),
            ("videos_light",  "Light · CRF24", "H.265, smaller"),
            ("videos",        "Original · CRF16", "H.265, full quality"),
        ),
        camera_kind="iphone4",
        expand=_expand_iphone,
    ),
    "eval": _Family(
        id="eval",
        label="MammaEval",
        description="Evaluation set with GT MoSh++ SMPL-X. 52 sequences across 3 sub-families.",
        script="download_mamma_eval.sh",
        content_groups=_EVAL_GROUPS,
        asset_types=(
            ("gt",      "GT SMPL-X",   "MoSh++ ground truth"),
            ("masks",   "GT masks",    "Per-camera mask tarballs"),
            ("markers", "Markers",     "Marker traces (Extra only)"),
            ("preview", "Preview",     "Overlay + mask grid videos"),
        ),
        video_variants=(
            ("none",         "No videos",       "Skip videos"),
            ("videos_crf24", "Light · CRF24",   "H.264, smallest"),
            ("videos_crf16", "Lossy · CRF16",   "H.264, medium"),
            ("videos",       "Original · CRF5", "H.264, near-lossless"),
        ),
        camera_kind="ioi16-or-32",
        expand=_expand_eval,
        notes="Singles/Extra ship with 16 cameras (IOI_01–16); Dance uses 32. Camera picks above 16 are auto-clipped for Singles/Extra.",
    ),
    "syn": _Family(
        id="syn",
        label="Synthetic training (WebDataset)",
        description="Synthetic SMPL-X renders for landmark training. ~29 datasets.",
        script="download_mamma_syn_wd.sh",
        content_groups=_SYN_GROUPS,
        asset_types=(),                # only one asset kind (the WD shards)
        video_variants=(),             # no video variants
        camera_kind="none",
        expand=_expand_syn,
        notes="Each dataset's .tar shards are listed in a remote manifest fetched at download time.",
    ),
}


def _bash_array_count(family: _Family, group_id: str) -> int:
    arr = _arrays(family.script)
    # Eval is special: all sequences live in one SEQUENCES array, grouped
    # by the leading path segment ("mamma_eval_singles/...", etc.).
    if family.id == "eval":
        for gid, _label, prefix in family.content_groups:
            if gid == group_id:
                return sum(
                    1 for s in arr.get("SEQUENCES", ())
                    if s.startswith(prefix + "/")
                )
        return 0
    for gid, _label, arr_name, *_rest in family.content_groups:
        if gid == group_id:
            return len(arr.get(arr_name, ()))
    return 0


def _family_record(family: _Family) -> dict:
    return {
        "id": family.id,
        "label": family.label,
        "description": family.description,
        "script": family.script,
        "content_groups": [
            _gc(g[0], g[1], _bash_array_count(family, g[0]))
            for g in family.content_groups
        ],
        "asset_types": [_at(at[0], at[1], at[2]) for at in family.asset_types],
        "video_variants": [_vv(v[0], v[1], v[2]) for v in family.video_variants],
        "camera_kind": family.camera_kind,
        "default_cameras":
            list(CAM_IOI32) if family.camera_kind in ("ioi32", "ioi16-or-32")
            else list(CAM_IPHONE) if family.camera_kind == "iphone4"
            else [],
        "max_ioi_cameras": 32 if family.camera_kind in ("ioi32", "ioi16-or-32") else 0,
        "notes": family.notes,
    }


def _list_family_sequences(family: _Family) -> list[dict]:
    """All sequences across all content groups for the given family.
    Used by the frontend to render the optional per-sequence subset
    picker."""
    arr = _arrays(family.script)
    if family.id == "eval":
        prefix_to_group = {p: gid for (gid, _, p) in _EVAL_GROUPS}
        return [
            {"path": ds, "group": prefix_to_group.get(ds.split("/", 1)[0])}
            for ds in arr.get("SEQUENCES", ())
        ]
    if family.id == "iphone":
        # entries are "n:path" — split them
        out = []
        for gid, _label, arr_name in family.content_groups:
            for entry in arr.get(arr_name, ()):
                if ":" not in entry: continue
                _n, path = entry.split(":", 1)
                out.append({"path": path, "group": gid})
        return out
    # dance / multi / syn
    out = []
    for gid, _label, arr_name, *_rest in family.content_groups:
        for s in arr.get(arr_name, ()):
            out.append({"path": s, "group": gid})
    return out


# ─── Job table ────────────────────────────────────────────────────────────

# RLock (not Lock): _run_job acquires the lock to inspect cancel state,
# then calls _update_job (which also acquires) — a non-reentrant lock
# would deadlock here.
_jobs_lock = threading.RLock()
_jobs: dict = {}


def _new_job(family_id: str, files_total: int) -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "family": family_id,
            "state": "downloading",
            "files_total": files_total,
            "files_done": 0,
            "files_failed": 0,
            "current_label": None,
            "current_bytes": 0,
            "current_total": 0,
            "bytes_done_total": 0,
            "error": None,
            "started_at": time.time(),
            "_cancel": False,
        }
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        rec = _jobs.get(job_id)
        if rec is not None:
            rec.update(kwargs)


def _job_view(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


# ─── Worker ───────────────────────────────────────────────────────────────

def _download_one_mpi(sfile: str, dest_path: Path, form_template: bytes,
                       job_id: str, label: str) -> str:
    """Download a single file via the MPI auth flow.
    Returns "ok", "skip", or "fail:<reason>"."""
    # Resume-like behaviour: if the destination already exists with non-
    # zero size and isn't an HTML error page, skip. Mirrors the .sh
    # mamma_is_valid_download.
    if dest_path.is_file() and dest_path.stat().st_size > 0:
        try:
            with open(dest_path, "rb") as f:
                head = f.read(256)
            if not _is_html_error(head):
                return "skip"
        except OSError:
            pass

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(dest_path.suffix + ".part")
    url = f"{_MPI_DOWNLOAD_URL}?domain=mamma&resume=1&sfile={urllib.parse.quote(f'{_REMOTE_ROOT}/{sfile}', safe='/')}"
    try:
        _check_url(url)
        req = urllib.request.Request(
            url, data=form_template, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            total_hdr = resp.headers.get("Content-Length")
            total = int(total_hdr) if total_hdr and total_hdr.isdigit() else 0
            _update_job(job_id, current_label=label, current_bytes=0,
                        current_total=total)
            downloaded = 0
            first = True
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    if first:
                        if _is_html_error(chunk):
                            raise RuntimeError(_DOWNLOAD_LIMIT_MSG)
                        first = False
                    f.write(chunk)
                    downloaded += len(chunk)
                    _update_job(job_id, current_bytes=downloaded)
        os.replace(tmp, dest_path)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        return f"fail:{_safe_error(exc)}"


def _run_job(job_id: str, family: _Family, plan: list[tuple[str, str]],
             form_template: bytes) -> None:
    """Iterate the plan, downloading each file. Honour cancellation."""
    bytes_total = 0
    for i, (sfile, local_relpath) in enumerate(plan):
        with _jobs_lock:
            rec = _jobs.get(job_id)
            if rec is None or rec.get("_cancel"):
                _update_job(job_id, state="cancelled")
                logger.info("Job %s cancelled at %d/%d", job_id, i, len(plan))
                return
        dest = (_DATA_DIR / local_relpath).resolve()
        # Defence in depth: refuse paths that escape the data directory.
        if not str(dest).startswith(str(_DATA_DIR.resolve()) + os.sep) and \
           str(dest) != str(_DATA_DIR.resolve()):
            _update_job(job_id, error=f"Refusing path outside data/: {local_relpath}")
            _update_job(job_id, state="error")
            return
        label = f"[{i + 1}/{len(plan)}] {sfile}"
        result = _download_one_mpi(sfile, dest, form_template, job_id, label)
        if result == "ok":
            try:
                bytes_total += dest.stat().st_size
            except OSError:
                pass
            with _jobs_lock:
                rec = _jobs.get(job_id)
                if rec is not None:
                    rec["files_done"] = rec.get("files_done", 0) + 1
                    rec["bytes_done_total"] = bytes_total
        elif result == "skip":
            with _jobs_lock:
                rec = _jobs.get(job_id)
                if rec is not None:
                    rec["files_done"] = rec.get("files_done", 0) + 1
        else:
            with _jobs_lock:
                rec = _jobs.get(job_id)
                if rec is not None:
                    rec["files_failed"] = rec.get("files_failed", 0) + 1
                    # Surface the first failure's reason as the job error
                    # so the UI can show what's wrong; we keep going so
                    # one transient blip doesn't kill a 1000-file run.
                    if rec.get("error") is None:
                        rec["error"] = result.split(":", 1)[1]
    # Final state
    with _jobs_lock:
        rec = _jobs.get(job_id)
        if rec is None:
            return
        if rec.get("_cancel"):
            rec["state"] = "cancelled"
        elif rec.get("files_failed", 0) > 0 and rec.get("files_done", 0) == 0:
            rec["state"] = "error"
        else:
            rec["state"] = "done"
        rec["current_label"] = None
        rec["current_bytes"] = 0
        rec["current_total"] = 0
    logger.info("Job %s finished: state=%s", job_id, _jobs[job_id]["state"])


# ─── Flask wiring ─────────────────────────────────────────────────────────

def register_routes(app: Flask) -> None:

    @app.get("/api/datasets/catalog")
    def _datasets_catalog():
        out = {"families": [_family_record(f) for f in FAMILIES.values()]}
        return jsonify(out)

    @app.get("/api/datasets/<family_id>/sequences")
    def _datasets_sequences(family_id):
        family = FAMILIES.get(family_id)
        if family is None:
            return jsonify({"error": "unknown family"}), 404
        return jsonify({"sequences": _list_family_sequences(family)})

    @app.post("/api/datasets/plan")
    def _datasets_plan():
        body = request.get_json(silent=True) or {}
        family_id = body.get("family") or ""
        family = FAMILIES.get(family_id)
        if family is None:
            return jsonify({"error": "unknown family"}), 404
        try:
            files = family.expand(body)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"plan failed: {type(exc).__name__}"}), 400
        return jsonify({
            "family": family_id,
            "files_total": len(files),
            "preview": [{"sfile": s, "local": p} for (s, p) in files[:6]],
        })

    @app.post("/api/datasets/start")
    def _datasets_start():
        body = request.get_json(silent=True) or {}
        family_id = body.get("family") or ""
        family = FAMILIES.get(family_id)
        if family is None:
            return jsonify({"error": "unknown family"}), 404
        username = str(body.get("username") or "")
        password = str(body.get("password") or "")
        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400
        try:
            files = family.expand(body)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"plan failed: {type(exc).__name__}"}), 400
        if not files:
            return jsonify({"error": "selection produced an empty plan"}), 400

        # Same credential-lifetime contract as data_readiness.py: build
        # the form body once, drop the local creds, hand the bytes to
        # the worker.
        form_body = urllib.parse.urlencode({
            "username": username,
            "password": password,
        }).encode("utf-8")
        del username, password
        logger.info(
            "Dataset download starting: family=%s files=%d",
            family_id, len(files),
        )

        job_id = _new_job(family_id, len(files))
        threading.Thread(
            target=_run_job, args=(job_id, family, files, form_body),
            daemon=True,
        ).start()
        return jsonify({"job_id": job_id, "files_total": len(files)})

    @app.get("/api/datasets/job/<job_id>")
    def _datasets_job(job_id):
        with _jobs_lock:
            rec = _jobs.get(job_id)
        if rec is None:
            return jsonify({"error": "unknown job"}), 404
        return jsonify(_job_view(rec))

    @app.post("/api/datasets/job/<job_id>/cancel")
    def _datasets_cancel(job_id):
        with _jobs_lock:
            rec = _jobs.get(job_id)
            if rec is None:
                return jsonify({"error": "unknown job"}), 404
            rec["_cancel"] = True
        return jsonify({"ok": True})
