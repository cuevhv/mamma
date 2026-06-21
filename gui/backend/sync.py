"""Filesystem ↔ DB reconciliation helpers used by the Database tab.

The runner writes per-step outputs at:
    <output_dir>/<step>/<output_id>/<dataset_name>/<seq_name>/

That layout makes it cheap to enumerate all (step, output_id, dataset,
seq) tuples that exist on disk by walking three levels under each step
directory. A run is keyed by `output_id` — the same identifier the
tasks table stores in `tasks.output_id`. Joining on output_id gives us
"what's on disk but not in the DB" and the inverse.

Kept as a plain module (no Flask imports) so it's easy to unit-test or
exercise from a notebook.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _safe_listdir(path: str) -> list[str]:
    try:
        return sorted(os.listdir(path))
    except (OSError, PermissionError):
        return []


def _dir_size_bytes(path: str, *, max_entries: int = 5000) -> int:
    """Sum file sizes recursively. Capped to avoid blowing the audit on a
    pathological tree — we just want a "rough scale" number, not a precise
    quota report."""
    total = 0
    seen = 0
    for root, _, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
            seen += 1
            if seen >= max_entries:
                return total
    return total


def scan_output_dirs(output_root: str, all_steps: Iterable[str]) -> dict[str, dict]:
    """Walk `<output_root>/<step>/<output_id>/<dataset>/<seq>/` and return:

        {
            output_id: {
                "outputId": str,
                "outputDir": str,        # what would go in tasks.output_path
                "dataset": str,          # set if all (step, run) entries agree
                "steps": [str, ...],     # which steps produced files
                "sequences": [str, ...], # union of seqs seen across steps
                "sizeBytes": int,        # rough total
            },
            ...
        }

    `output_root` is the user's `MAMMA_OUTPUT_DIR` (or the default `var/output/`).
    """
    if not output_root or not os.path.isdir(output_root):
        return {}

    runs: dict[str, dict] = {}
    for step in all_steps:
        step_dir = os.path.join(output_root, step)
        if not os.path.isdir(step_dir):
            continue
        for output_id in _safe_listdir(step_dir):
            run_dir = os.path.join(step_dir, output_id)
            if not os.path.isdir(run_dir):
                continue
            entry = runs.setdefault(output_id, {
                "outputId": output_id,
                "outputDir": output_root,
                "dataset": None,
                "steps": [],
                "sequences": set(),
                "sizeBytes": 0,
            })
            if step not in entry["steps"]:
                entry["steps"].append(step)
            entry["sizeBytes"] += _dir_size_bytes(run_dir)

            datasets = _safe_listdir(run_dir)
            for dataset in datasets:
                ds_dir = os.path.join(run_dir, dataset)
                if not os.path.isdir(ds_dir):
                    continue
                # Earlier steps in the loop set `dataset`; if a later step's
                # tree disagrees, fall back to "(mixed)" so the user knows
                # there's something weird going on.
                if entry["dataset"] is None:
                    entry["dataset"] = dataset
                elif entry["dataset"] != dataset and entry["dataset"] != "(mixed)":
                    entry["dataset"] = "(mixed)"

                for seq in _safe_listdir(ds_dir):
                    if os.path.isdir(os.path.join(ds_dir, seq)):
                        entry["sequences"].add(seq)

    # Convert sets to sorted lists so the JSON response is stable.
    for entry in runs.values():
        entry["sequences"] = sorted(entry["sequences"])
        entry["steps"] = sorted(entry["steps"])
    return runs


def infer_step_status(output_root: str, output_id: str, dataset: str | None,
                      step: str, seq: str) -> str:
    """Per-(step, seq) status inference: "Completed" if the expected
    output dir contains at least one file (recursively); "Failed" if the
    dir is missing or empty.

    This is a deliberately rough heuristic — many ML scripts write a
    sentinel "DONE" file but not all do, and parsing every step's log
    format is out of scope. The user can edit / re-run as needed."""
    if not dataset:
        return "Failed"
    seq_dir = os.path.join(output_root, step, output_id, dataset, seq)
    if not os.path.isdir(seq_dir):
        return "Failed"
    for _, _, files in os.walk(seq_dir):
        if files:
            return "Completed"
    return "Failed"


def infer_step_statuses(output_root: str, output_id: str, dataset: str | None,
                        steps: Iterable[str], sequences: Iterable[str]) -> dict[tuple[str, str], str]:
    """Bulk wrapper for `infer_step_status` — returns a {(step, seq): status} map."""
    out: dict[tuple[str, str], str] = {}
    for step in steps:
        for seq in sequences:
            out[(step, seq)] = infer_step_status(output_root, output_id, dataset, step, seq)
    return out


def task_outputs_exist(output_root: str, output_id: str,
                       steps: Iterable[str]) -> bool:
    """True if any `<output_root>/<step>/<output_id>/` directory exists.
    Used to flag DB rows whose outputs are gone — the orphan check."""
    if not output_root or not output_id:
        return False
    for step in steps:
        if os.path.isdir(os.path.join(output_root, step, output_id)):
            return True
    return False


def resolve_output_root(task_content: dict | None, default_output_root: str) -> str:
    """Best-effort resolution of which directory holds a task's outputs.
    Prefer the task.json's `global.out_dir`; fall back to the project
    default. Mirrors the runner's logic so the audit lines up with reality."""
    if isinstance(task_content, dict):
        out_dir = (task_content.get("global") or {}).get("out_dir")
        if isinstance(out_dir, str) and out_dir:
            return os.path.expanduser(out_dir)
    return default_output_root


# Step output dirs walked as a *fallback* when a capture has no input
# images on disk (e.g. imported via the Database tab when raw inputs are
# gone). Tried in this priority — least-derived first, since a raw input
# is nicer than a fitted-mesh render for "which capture is this?".
# Run-output step preference for the capture-card thumbnail. Only the
# final-rendered preview (``ma_vis``) makes sense as a card thumbnail:
# ma_cap writes NPZs (no image), ma_masks writes person-segmentation
# heatmaps (visually noisy and not what a user pictures when they
# scan the captures list), ma_2d writes landmark heatmaps, ma_3d
# writes meshes the user usually doesn't want to identify by. Keep
# ma_vis only.
_THUMBNAIL_STEP_PRIORITY = ("ma_vis",)
_THUMBNAIL_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def list_input_thumbnail_candidates(
    capture_content: dict | None,
    *,
    max_walk_entries: int = 50,
) -> list[dict]:
    """Return one representative input-frame candidate per camera for the
    first sequence. Powers the "edit thumbnail" picker — the user sees one
    frame per camera angle and clicks the one they want.

    Each entry: { cam: str, path: str }. Returns [] when nothing resolves.
    """
    if not isinstance(capture_content, dict):
        return []
    ioi_root = capture_content.get("ioi_root")
    if not isinstance(ioi_root, str) or not ioi_root or not os.path.isdir(ioi_root):
        return []

    sequences = capture_content.get("sequences") or {}
    if not isinstance(sequences, dict) or not sequences:
        return []
    first_seq_key = sorted(sequences.keys())[0]
    first_seq = sequences[first_seq_key]
    seq_name = first_seq.get("ioi") if isinstance(first_seq, dict) else None
    if not seq_name:
        return []
    seq_dir = os.path.join(ioi_root, str(seq_name))
    if not os.path.isdir(seq_dir):
        return []

    declared_cams = capture_content.get("cams") or []
    if isinstance(declared_cams, list) and declared_cams:
        cams = [str(c) for c in declared_cams]
    else:
        cams = sorted(
            entry for entry in _safe_listdir(seq_dir)
            if not entry.startswith(".")
            and os.path.isdir(os.path.join(seq_dir, entry))
        )

    candidates: list[dict] = []
    for cam in cams:
        cam_dir = os.path.join(seq_dir, cam)
        if not os.path.isdir(cam_dir):
            continue
        seen = 0
        for dirpath, _, files in os.walk(cam_dir, followlinks=False):
            picked = None
            for fname in sorted(files):
                seen += 1
                if seen >= max_walk_entries:
                    break
                ext = os.path.splitext(fname)[1].lower()
                if ext in _THUMBNAIL_EXTENSIONS:
                    picked = os.path.join(dirpath, fname)
                    break
            if picked:
                candidates.append({"cam": cam, "path": picked})
                break  # one per cam is enough
    return candidates


def find_input_thumbnail(
    capture_content: dict | None,
    *,
    max_walk_entries: int = 50,
) -> str | None:
    """Return a representative input frame for a capture.

    Layout: <ioi_root>/<sequence_name>/<camera_name>/<frame.png>.
    We pick the first sequence, then the first camera that resolves to
    an image — this is what the user instinctively pictures when they
    think of a capture (the actual scene from the first camera angle),
    and it doesn't change between pipeline runs.

    Camera resolution is layered:
      1) `cams` list from capture.json if present (the canonical case),
      2) otherwise list the sequence directory itself — every non-hidden
         child directory is treated as a camera.

    The fallback covers real-world capture.jsons that omit `cams`
    entirely, or that use cam folder names not matching the listed ones.
    Returns None if nothing resolves within `max_walk_entries` per cam.
    """
    if not isinstance(capture_content, dict):
        return None
    ioi_root = capture_content.get("ioi_root")
    if not isinstance(ioi_root, str) or not ioi_root or not os.path.isdir(ioi_root):
        return None

    sequences = capture_content.get("sequences") or {}
    if not isinstance(sequences, dict) or not sequences:
        return None

    # First sequence by sorted key — keys are "000", "001", … in the
    # current capture.json convention.
    first_seq_key = sorted(sequences.keys())[0]
    first_seq = sequences[first_seq_key]
    seq_name = first_seq.get("ioi") if isinstance(first_seq, dict) else None
    if not seq_name:
        return None

    seq_dir = os.path.join(ioi_root, str(seq_name))
    if not os.path.isdir(seq_dir):
        return None

    declared_cams = capture_content.get("cams") or []
    if isinstance(declared_cams, list) and declared_cams:
        cams = [str(c) for c in declared_cams]
    else:
        # Fall back: any non-hidden subdir of the sequence dir is a camera.
        cams = sorted(
            entry for entry in _safe_listdir(seq_dir)
            if not entry.startswith(".")
            and os.path.isdir(os.path.join(seq_dir, entry))
        )

    # Try each camera in order; first one with a readable frame wins.
    for cam in cams:
        cam_dir = os.path.join(seq_dir, cam)
        if not os.path.isdir(cam_dir):
            continue
        seen = 0
        for dirpath, _, files in os.walk(cam_dir, followlinks=False):
            for fname in sorted(files):
                seen += 1
                if seen >= max_walk_entries:
                    return None
                ext = os.path.splitext(fname)[1].lower()
                if ext in _THUMBNAIL_EXTENSIONS:
                    return os.path.join(dirpath, fname)
    return None


def find_capture_thumbnail(
    output_root: str,
    output_id: str,
    dataset: str | None,
    *,
    max_walk_entries: int = 200,
) -> str | None:
    """Find a small representative image for a capture's most-recent run.

    Walks `<output_root>/<step>/<output_id>/<dataset>/` for each step in
    priority order, recursing up to `max_walk_entries` files per step,
    and returns the first image path it finds. Returns None if nothing
    is available (silent fallback — the UI shows a placeholder).

    Cap on entries keeps the call cheap on huge run trees: one image is
    enough; we don't need to enumerate the whole output."""
    if not output_root or not output_id:
        return None

    for step in _THUMBNAIL_STEP_PRIORITY:
        step_run = os.path.join(output_root, step, output_id)
        if not os.path.isdir(step_run):
            continue
        # If we know the dataset, scope the search; otherwise walk the
        # whole step run.
        roots_to_try = [os.path.join(step_run, dataset)] if dataset else [step_run]
        for root in roots_to_try:
            if not os.path.isdir(root):
                continue
            seen = 0
            for dirpath, _, files in os.walk(root, followlinks=False):
                for fname in sorted(files):
                    seen += 1
                    if seen >= max_walk_entries:
                        return None  # give up cheaply rather than scanning everything
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in _THUMBNAIL_EXTENSIONS:
                        return os.path.join(dirpath, fname)
    return None


def enumerate_done_sentinels(
    out_dir: str,
    output_id: str,
    dataset_name: str,
    sequences: Iterable[str],
    steps: Iterable[str],
) -> dict[str, list[str]]:
    """Return `{seq_name: [step_name, ...]}` for every (step, seq) pair
    whose DONE sentinel exists under the run group's output tree.

    Powers the New Task form's "what's already done?" hints. The DONE
    sentinel layout matches what the runner writes:
        <out_dir>/<step>/<output_id>/<dataset>/<seq>/DONE
    """
    done: dict[str, list[str]] = {}
    if not out_dir or not output_id:
        return done
    seq_list = list(sequences)
    step_list = list(steps)
    if not seq_list or not step_list:
        return done
    for seq in seq_list:
        hits: list[str] = []
        for step in step_list:
            sentinel = os.path.join(out_dir, step, output_id, dataset_name or "", seq, "DONE")
            if os.path.isfile(sentinel):
                hits.append(step)
        done[seq] = hits
    return done


def humanize_size(n: int) -> str:
    """Render a byte count as a compact human string. Used in the audit
    JSON so the frontend doesn't need to re-implement formatting."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024:
            return f"{size:.0f} {u}" if u == "B" else f"{size:.1f} {u}"
        size /= 1024
    return f"{size:.1f} PB"


def _sanity_check_path_inside(child: str, parent: str) -> bool:
    """True if `child` is contained under `parent` after normalisation."""
    try:
        child_abs = str(Path(child).resolve())
        parent_abs = str(Path(parent).resolve())
        return child_abs == parent_abs or child_abs.startswith(parent_abs + os.sep)
    except (OSError, ValueError):
        return False


# Video-thumbnail extraction for released-dataset captures that ship
# inputs as videos (not per-camera image frames). `find_input_thumbnail`
# above expects an image-frame layout; this is the fallback for the
# video-shipped layout. Lazy on-disk cache keyed by capture_name.

# Priority order when picking which videos* subdir to extract from.
# Compressed variants beat the uncompressed `videos/` to avoid reading
# multi-GB sources just for one frame. `videos_light` ships with the
# iphones datasets and is the smallest of the lot.
_VIDEO_DIR_PRIORITY = ("videos_light", "videos_crf24", "videos", "videos_crf16")


def _pick_video_thumbnail_source(
    capture_content: dict | None,
    capture_root_abs: str | None,
) -> str | None:
    """Resolve `<capture_root>/<first_seq>/<videos_dir>/<first_video>.mp4`
    or None when no candidate is on disk. Pure file-system probe."""
    if not isinstance(capture_content, dict) or not capture_root_abs:
        return None
    if not os.path.isdir(capture_root_abs):
        return None
    sequences = capture_content.get("sequences") or {}
    if not isinstance(sequences, dict) or not sequences:
        return None
    # Pick the first sequence by sequence-name (the user-visible label),
    # matching the convention used by find_input_thumbnail.
    seq_names = []
    for _, val in sequences.items():
        if not isinstance(val, dict):
            continue
        nm = val.get("name") or val.get("ioi")
        if nm:
            seq_names.append(nm)
    if not seq_names:
        return None
    seq_names.sort()
    seq_dir = os.path.join(capture_root_abs, seq_names[0])
    if not os.path.isdir(seq_dir):
        return None
    # First videos* dir in priority order, then first .mp4 alphabetically.
    # Use os.scandir: entry.is_file() reads the cached dirent type (no extra
    # stat() per file like os.path.isfile would), so this stays cheap even on a
    # cold inode cache. We still pick the alphabetically-first .mp4 for a
    # deterministic source, but without N stat() syscalls.
    for sub in _VIDEO_DIR_PRIORITY:
        cand_dir = os.path.join(seq_dir, sub)
        if not os.path.isdir(cand_dir):
            continue
        first = None
        try:
            with os.scandir(cand_dir) as it:
                for entry in it:
                    name = entry.name
                    if name.lower().endswith(".mp4") and (first is None or name < first):
                        if entry.is_file():
                            first = name
        except OSError:
            continue
        if first:
            return os.path.join(cand_dir, first)
    return None


def extract_video_middle_frame(
    video_path: str,
    dest_path: str,
    *,
    quality: int = 85,
) -> str | None:
    """Decode the middle frame of `video_path` with cv2 and write it as a
    JPEG to `dest_path` (atomic-rename so a half-written file can't be
    served). Returns dest_path on success, None on any failure.

    The cv2 import is deferred to the call site so just *importing*
    sync.py stays light — the inference-package import chain pulls in
    plenty of native code already and we don't want to add to it for
    flows that never need thumbnails."""
    try:
        import cv2  # noqa: PLC0415  (deferred for module-import cost)
    except Exception:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target = max(0, count // 2)
        if target > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok and target > 0:
            # Seek failed (some codecs / very short clips); fall back to
            # the first decodable frame.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        # Encode in-memory then atomic-rename via tempfile. `cv2.imwrite`
        # dispatches the encoder on the destination extension, so writing
        # to "<dest>.tmp" would fail with "could not find a writer for the
        # specified extension". `imencode(".jpg", …)` lets us bypass that.
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok2, buf = cv2.imencode(".jpg", frame, params)
        if not ok2 or buf is None:
            return None
        tmp_path = f"{dest_path}.partial.{os.getpid()}"
        try:
            with open(tmp_path, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp_path, dest_path)
        except OSError:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return None
        return dest_path
    except Exception:
        return None
    finally:
        try:
            cap.release()
        except Exception:
            pass


def find_video_thumbnail(
    capture_content: dict | None,
    capture_root_abs: str | None,
    capture_name: str,
    cache_dir: str,
) -> str | None:
    """Lazy on-disk thumbnail for a video-shipped capture.

    Returns the cached JPEG when present, else extracts a fresh middle frame.

    Cache-first by design: resolving the source video
    (``_pick_video_thumbnail_source``) does a cold filesystem stat-sweep of the
    dataset's videos dir, which costs seconds on large datasets when the OS inode
    cache is cold — and this runs for every present example capture on every
    ``/api/captures`` request. So we check the cached JPEG (keyed by
    ``capture_name``, in the small cache dir) *before* touching the dataset at
    all. Source videos are immutable once downloaded, so a present cache entry is
    valid; we no longer re-stat the source for a freshness check on the hot path.
    (Delete the cached JPEG to force a refresh.)"""
    # Sanitize: capture_name comes from a Path stem we control, but defend
    # against any future caller passing user-supplied text.
    safe_name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in capture_name)
    if not safe_name:
        return None
    dest = os.path.join(cache_dir, f"{safe_name}.jpg")
    if os.path.isfile(dest):
        return dest  # cache hit — return without probing the dataset
    # Cache miss: now resolve the source (cheap scandir, see
    # _pick_video_thumbnail_source) and extract.
    src = _pick_video_thumbnail_source(capture_content, capture_root_abs)
    if not src:
        return None
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        return None
    return extract_video_middle_frame(src, dest)
