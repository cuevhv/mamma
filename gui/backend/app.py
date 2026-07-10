"""Flask backend for the MAMMA webtool — vendored under ``gui/`` in
mamma_release. The pipeline runner is the parent repo's
``inference`` package; this app spawns ``runner_main.py`` as a child
process, which in turn calls :func:`inference.runner.run_dag` with a
SQLite-writing ``StatusSink``.

Differences from upstream master:
  * No login: anyone reaching the backend is treated as the OS user
    (`getpass.getuser()`). LDAP, Flask-Login, and SSH-to-cluster paths
    are gone.
  * Jobs run locally via ``runner_main.py`` (thin shim over
    ``inference.runner``), started as a detached child process.
  * Postgres is gone; storage is SQLite under MAMMA_DATA_DIR
    (default: ``gui/var/mamma.sqlite``).
"""
import collections
import getpass
import json
import mimetypes
import os
import re as _re
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

# Make the parent ``mamma_release`` repo importable so ``from inference...``
# works for ``bootstrap_env()`` below. Same bridge ``runner_main.py`` uses.
_REPO_ROOT = Path(__file__).resolve().parents[2]  # backend -> gui -> mamma_release
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from inference.env import bootstrap_env  # noqa: E402

# Populate MAMMA_* defaults and apply .env.local before anything reads env.
bootstrap_env()

# Shared preset/run-config materializer (same code path the CLI's
# ``run --preset … --capture …`` uses) — keeps GUI submissions and CLI
# invocations consistent in how a preset binds to a capture.
from inference.config import materialize_run_config  # noqa: E402

import db  # backend/db.py
from config_io import load_config_file
from objects.processes import ProcessType
from objects.sequences import get_sequences_from_data


# ---------------------------------------------------------------------------
# Config (env-driven; sensible defaults for `python app.py` outside docker)
# ---------------------------------------------------------------------------

LOCAL_USER = os.environ.get("MAMMA_LOCAL_USER") or getpass.getuser()

# Default data dir lives inside the gui/ subdir at ./var/ so users don't
# accumulate state under $HOME and `git clean -fdx` resets the slate.
# Override with MAMMA_DATA_DIR (or MAMMA_INTERFACE_DIR / MAMMA_DB_PATH for
# finer control).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # backend/ -> gui/
# Repo root = the parent of gui/. Used to render capture data paths as
# `./data/...` for display, instead of the absolute path or the JSON-
# relative `../../../data/...` form stored in capture.json.
_REPO_ROOT = _PROJECT_ROOT.parent
_DEFAULT_DATA_DIR = Path(os.environ.get("MAMMA_DATA_DIR") or _PROJECT_ROOT / "var").expanduser()


def _display_capture_path(capture_root_abs: str | None) -> str:
    """Render the capture's data root for the UI: repo-root-relative
    ``./data/...`` if the path is under the repo, else the absolute path.
    Empty string when the capture has no resolvable root."""
    if not capture_root_abs:
        return ""
    try:
        rel = os.path.relpath(capture_root_abs, _REPO_ROOT)
    except ValueError:
        # Different drives on Windows. Fall back to absolute.
        return capture_root_abs
    if rel.startswith(".."):
        return capture_root_abs
    return "./" + rel.replace(os.sep, "/")

# Where the webtool reads task templates / capture jsons / writes generated
# task configs. On master this used to be a docker bind mount at
# /project/mamma/interface; outside docker we let the user choose.
MOUNT_POINT = os.environ.get("MAMMA_INTERFACE_DIR", str(_DEFAULT_DATA_DIR / "interface"))

# Where pipeline outputs land when the form's "Output dir" is empty or
# relative. Absolute paths typed into the form override this entirely.
# Defaults to the parent repo's top-level ``output/`` so the artifacts a
# GUI run produces sit next to the artifacts produced by the CLI runner
# (``python -m inference run ...``) — single shared output tree.
DEFAULT_OUTPUT_DIR = os.environ.get("MAMMA_OUTPUT_DIR", str(_REPO_ROOT / "output"))
# Where the GUI persists the frozen run config it built for each submitted
# run. One JSON per task_id; the runner re-reads it by path. Previously
# task_jsons/task_config_<id>.json — renamed 2026-05-24 alongside the
# preset/run-config vocabulary split (see scripts/migrate_to_presets.py).
RUN_CONFIG_DIR = os.path.join(MOUNT_POINT, "run_configs")
DEFAULT_PRESET_PATH = os.environ.get(
    "MAMMA_DEFAULT_PRESET",
    os.path.join(MOUNT_POINT, "samples", "presets", "task_template.json"),
)

# Read-only directories the GUI scans (in addition to MOUNT_POINT) for
# example captures and tasks shipped with the repo. Entries here surface
# in listing endpoints tagged with ``source: "example"``; the GUI must
# refuse writes/deletes on these paths and offer "Save a writable copy"
# instead. All examples live under ``<repo>/configs/examples/``.
_EXAMPLES_ROOT = _REPO_ROOT / "configs" / "examples"
_EXAMPLE_CAPTURES_DIR = _EXAMPLES_ROOT / "captures"
_EXAMPLE_PRESETS_DIRS = (
    _EXAMPLES_ROOT / "presets",          # full-scale presets
    _EXAMPLES_ROOT / "presets" / "quick", # 4-cam, 30-frame variety smoke presets
)

# Lazy on-disk cache for auto-extracted thumbnails of video-shipped
# captures (released datasets). One JPEG per capture; regenerated when
# the source video's mtime advances. Lives under MAMMA_DATA_DIR so
# `rm -rf gui/var/` wipes it along with the rest of the GUI state.
_THUMB_CACHE_DIR = str(_DEFAULT_DATA_DIR / "thumb_cache")


def _list_example_captures():
    """Scan configs/examples/captures/ and return capture-listing dicts.

    Mirrors the shape that ``get_captures`` builds from the DB, with
    ``source: "example"`` and a deterministic ``id`` derived from the
    filename. Missing fields default to empty so the front-end can
    render the row even without a DB-side history.
    """
    out = []
    if not _EXAMPLE_CAPTURES_DIR.is_dir():
        return out
    for path in sorted(_EXAMPLE_CAPTURES_DIR.glob("*.json")) + sorted(_EXAMPLE_CAPTURES_DIR.glob("*.yaml")):
        try:
            content = load_config_file(str(path))
        except (OSError, ValueError):
            continue
        sequences = content.get("sequences") or {}
        # Sequence name extraction: prefer the new "name" key, fall
        # back to legacy "ioi" so old example files still surface.
        seq_names = sorted(
            n for n in (
                (v.get("name") or v.get("ioi") or "") if isinstance(v, dict) else ""
                for v in sequences.values()
            ) if n
        )
        cams = list(content.get("cams") or [])
        try:
            ctime = path.stat().st_mtime
        except OSError:
            ctime = 0
        capture_root_abs = _resolve_capture_root_abs(str(path), content)
        released_present = capture_root_abs is not None and os.path.isdir(capture_root_abs)
        # Lazy auto-thumbnail: middle frame of the first camera's video.
        # No-op if the capture has no `videos*` dir on disk or cv2 fails —
        # the card falls back to the standard placeholder icon.
        import sync as _pipeline_sync  # local import: same module the
                                       # detail endpoint already lazy-loads
        thumbnail_path = _pipeline_sync.find_video_thumbnail(
            content, capture_root_abs, path.stem, _THUMB_CACHE_DIR,
        ) if released_present else None
        out.append({
            "id": f"example:{path.stem}",
            "captureName": path.stem,
            "jsonPath": str(path),
            "seqNames": seq_names,
            "cams": cams,
            "datasetName": None,
            "outputDir": content.get("capture_root") or content.get("ioi_root") or "",
            "dataPath": _display_capture_path(capture_root_abs),
            "status": "example",
            "processes": [],
            "createdAt": ctime,
            "taskCount": 0,
            "thumbnailPath": thumbnail_path,
            "source": "example",
            "releasedDataPresent": released_present,
        })
    return out


def _is_readonly_path(abs_path: str) -> bool:
    """True if ``abs_path`` lives under a read-only root (shipped examples).

    Mutation endpoints (PUT/DELETE) call this and reject the request so
    examples can't be accidentally overwritten by a curious user with
    DevTools open.
    """
    if not abs_path:
        return False
    candidate = os.path.normpath(abs_path)
    for root in (_EXAMPLE_CAPTURES_DIR, *_EXAMPLE_PRESETS_DIRS):
        try:
            root_str = str(root.resolve())
        except OSError:
            continue
        if os.path.commonpath([candidate, root_str]) == root_str:
            return True
    return False


def _list_example_presets():
    """Scan every read-only presets dir for shipped example presets.

    The current set is ``configs/examples/presets/`` (full-scale) plus
    ``configs/examples/presets/quick/`` (4-cam, 30-frame smoke variants).
    Each entry's ``name`` is namespaced by the parent directory
    (``presets/<stem>`` or ``quick/<stem>``) so quick + full variants
    don't collide in the picker.
    """
    out = []
    for root in _EXAMPLE_PRESETS_DIRS:
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            name = path.name
            if not (name.endswith(".yaml") or name.endswith(".yml") or name.endswith(".json")):
                continue
            try:
                cfg = load_config_file(str(path))
            except (OSError, ValueError):
                cfg = {}
            g = cfg.get("global") or {}
            display = g.get("display_name") or path.stem
            description = g.get("description") or ""
            # Namespace by directory so quick + full tasks coexist.
            entry_name = f"{root.name}/{path.stem}"
            out.append({
                "name": entry_name,
                "path": str(path),
                "displayName": display,
                "description": description,
                "isUser": False,
                "source": "example",
            })
    return out


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

# Data-readiness panel (Home → "Body Models & Weights" card).
# Self-contained module: registers /api/data/readiness/* routes.
from data_readiness import register_routes as _register_data_readiness  # noqa: E402
_register_data_readiness(app)

# Dataset library (Home → "MAMMA Datasets" card). Self-contained module:
# registers /api/datasets/* routes. See dataset_downloader.py.
from dataset_downloader import register_routes as _register_dataset_downloader  # noqa: E402
_register_dataset_downloader(app)

# Exporter tab (SMPL-X -> npz/FBX/ABC/BVH/USD). Self-contained module:
# registers /api/exporter/* routes (readiness, Blender/add-on downloads, export).
from exporter import register_routes as _register_exporter  # noqa: E402
_register_exporter(app)


# ---------------------------------------------------------------------------
# Task queue + coordinator
#
# Why: `POST /api/tasks` used to spawn the runner subprocess inline,
# meaning two concurrent Submit clicks → two parallel runners
# fighting for one GPU. The queue serializes spawns; the
# `concurrency_limit` setting (stored in SQLite, default 1) caps how
# many runners may be alive at once. A single coordinator thread is
# the only code path that calls `subprocess.Popen` for a runner.
#
# Within-task sequence dispatch is unchanged — the runner itself
# still iterates seq_names sequentially (`inference/engines.py:32`
# "Single-threaded by design").
# ---------------------------------------------------------------------------

class TaskQueue:
    """Process-local task queue serializing runner_main subprocesses.

    Lives for the lifetime of the Flask process. Each `POST /api/tasks`
    appends a (task_id, args) tuple; the coordinator thread pops in FIFO
    order, waits until the running set is under the current DB
    concurrency limit, and then `Popen`s the runner. When a runner
    exits, the coordinator is notified and re-checks the limit.

    On Flask boot we hydrate the queue from any DB tasks still in the
    `Queued` process-status state — so in-flight queue entries survive a
    backend restart even though the in-memory deque is lost.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        # (task_id, kwargs-for-_spawn_runner)
        self._pending: "collections.deque[tuple[int, dict]]" = collections.deque()
        # task_id -> Popen instance for tasks whose runner is currently alive.
        self._running: dict[int, subprocess.Popen] = {}
        self._coordinator: threading.Thread | None = None

    # ---- public API used by routes -----------------------------------

    def enqueue(self, *, task_id: int, task_json_path: str, output_id: str,
                output_dir: str, jobs_log_dir: str) -> None:
        """Schedule a task for runner spawn. Returns immediately.

        The task's processes were already inserted in DB with status
        'Queued' (see ``db.create_entry``), so this just hands the id
        plus the spawn arguments to the coordinator."""
        spawn_kwargs = {
            "task_json_path": task_json_path,
            "output_id": output_id,
            "output_dir": output_dir,
            "jobs_log_dir": jobs_log_dir,
        }
        with self._cond:
            self._pending.append((task_id, spawn_kwargs))
            self._cond.notify_all()
        print(f"---> Enqueued task_id={task_id} (queue depth now {self.queue_depth()})")

    def cancel_queued(self, task_id: int) -> bool:
        """Drop a still-queued task from the queue. Returns True if
        the task was found and dropped; False if it had already been
        popped (in which case the caller should use the stop-running
        codepath)."""
        with self._cond:
            for i, (tid, _kwargs) in enumerate(self._pending):
                if tid == task_id:
                    del self._pending[i]
                    self._cond.notify_all()
                    return True
            return False

    def notify_limit_changed(self) -> None:
        """Called by `PUT /api/settings/concurrency` so the coordinator
        re-checks the limit immediately (otherwise a bump from 1->2
        wouldn't take effect until a task happens to finish)."""
        with self._cond:
            self._cond.notify_all()

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._pending)

    def queue_position(self, task_id: int) -> int | None:
        """1-indexed FIFO position of a queued task, or None if not
        queued. Used by the frontend to render 'Queued · #2'."""
        with self._lock:
            for i, (tid, _) in enumerate(self._pending):
                if tid == task_id:
                    return i + 1
            return None

    def running_pids(self) -> dict[int, int]:
        with self._lock:
            return {tid: proc.pid for tid, proc in self._running.items()}

    # ---- coordinator + spawn ----------------------------------------

    def start_coordinator(self) -> None:
        """Hydrate the queue from DB then start the coordinator thread.
        Idempotent — safe to call once at module import."""
        if self._coordinator is not None:
            return
        try:
            ids = db.tasks_with_all_processes_status("Queued")
        except Exception as e:
            print(f"---> WARN: could not hydrate task queue from DB: {e}")
            ids = []
        if ids:
            print(f"---> Hydrating task queue with {len(ids)} pre-existing queued task(s)")
            for tid in ids:
                kwargs = self._reconstruct_spawn_kwargs(tid)
                if kwargs is None:
                    print(f"---> WARN: dropping queued task_id={tid} — config no longer on disk")
                    db.cancel_queued_task(tid)
                    continue
                with self._lock:
                    self._pending.append((tid, kwargs))
        t = threading.Thread(target=self._run, name="task-queue-coordinator", daemon=True)
        t.start()
        self._coordinator = t

    @staticmethod
    def _reconstruct_spawn_kwargs(task_id: int) -> dict | None:
        """Build the spawn kwargs for a hydrated queue entry from
        on-disk state. Returns None if the run config is missing."""
        try:
            with db.create_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT task_json_path, output_path, output_id FROM tasks WHERE task_id = ?",
                    (task_id,),
                )
                row = cur.fetchone()
            if not row or not row["task_json_path"] or not os.path.isfile(row["task_json_path"]):
                return None
            cfg = load_config_file(row["task_json_path"]) or {}
            g = cfg.get("global") or {}
            return {
                "task_json_path": row["task_json_path"],
                "output_id": row["output_id"] or "",
                "output_dir": row["output_path"] or g.get("out_dir") or DEFAULT_OUTPUT_DIR,
                "jobs_log_dir": g.get("jobs_log_dir") or os.path.join(row["output_path"] or DEFAULT_OUTPUT_DIR, "logs"),
            }
        except Exception as e:
            print(f"---> WARN: _reconstruct_spawn_kwargs(task_id={task_id}) failed: {e}")
            return None

    def _run(self) -> None:
        """Coordinator loop: pop next queued task, wait until under the
        current concurrency limit, spawn its runner."""
        while True:
            with self._cond:
                while not self._pending:
                    self._cond.wait()
                # Peek-don't-pop until we have a slot — otherwise a
                # popped-but-not-yet-spawned task wouldn't show in the
                # queue depth, and the next Submit click would get a
                # misleading "Queued · #1" for a moment.
                while len(self._running) >= db.get_concurrency_limit():
                    self._cond.wait()
                if not self._pending:
                    continue
                task_id, kwargs = self._pending.popleft()
            try:
                proc = self._spawn_runner(task_id=task_id, **kwargs)
            except Exception as e:
                print(f"---> ERROR: failed to spawn runner for task_id={task_id}: {e}")
                # Mark the task cancelled so it doesn't sit in a wedged
                # 'Queued' state forever.
                try:
                    db.cancel_queued_task(task_id)
                except Exception:
                    pass
                continue
            with self._cond:
                self._running[task_id] = proc
            threading.Thread(
                target=self._wait_and_release,
                args=(task_id, proc),
                name=f"task-{task_id}-waiter",
                daemon=True,
            ).start()

    def _spawn_runner(self, *, task_id: int, task_json_path: str, output_id: str,
                      output_dir: str, jobs_log_dir: str) -> subprocess.Popen:
        """The one and only `subprocess.Popen` site for runners. Same
        argv as the pre-queue code; flipped processes Queued -> Waiting
        in DB just before launch so the runner's status-sink takes over
        cleanly."""
        runner_log = os.path.join(jobs_log_dir, LOCAL_USER, str(task_id), "runner.log")
        os.makedirs(os.path.dirname(runner_log), exist_ok=True)
        log_fp = open(runner_log, "ab", buffering=0)
        # Flip Queued -> Waiting BEFORE the Popen so the runner can
        # immediately mark its first step Running without racing the
        # transition.
        db.bump_task_processes_from_queued_to_waiting(task_id)
        proc = subprocess.Popen(
            [sys.executable, "-m", "runner_main",
             "--task", task_json_path,
             "--task-id", str(task_id),
             "--out-tag", str(output_id),
             "--log-tag", str(task_id)],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            start_new_session=True,
        )
        db.set_task_runner_pid(task_id, proc.pid)
        print(f"---> Started runner pid={proc.pid} for task_id={task_id} (log: {runner_log})")
        return proc

    def _wait_and_release(self, task_id: int, proc: subprocess.Popen) -> None:
        """Block on the runner subprocess; once it exits, release the
        slot so the coordinator can spawn the next queued task."""
        try:
            proc.wait()
        except Exception as e:
            print(f"---> WARN: wait() failed for task_id={task_id}: {e}")
        with self._cond:
            self._running.pop(task_id, None)
            self._cond.notify_all()
        print(f"---> Runner exited for task_id={task_id} (rc={proc.returncode})")


_task_queue = TaskQueue()


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------

@app.route("/api/captures", methods=["GET"])
def get_captures():
    captures_data = db.get_all_captures()
    # Lazy import — only this route uses these helpers.
    import sync as _sync
    response = []
    for c in captures_data:
        latest = c.get("latest_task") or {}

        # Thumbnail resolution priority:
        #   1) user-set ``thumbnail`` field on the capture JSON
        #   2) the raw footage (image frames OR a mid-frame from videos_*/)
        #   3) the latest run's ma_vis preview output
        # ma_masks / ma_2d / ma_3d outputs are intentionally excluded —
        # a person-segmentation heatmap or landmark map isn't what a
        # user expects when scanning the captures list.
        thumbnail_path = None
        cams: list[str] = []
        capture_json_content: dict | None = None
        capture_json_path = c.get("capture_json_path")
        if capture_json_path and os.path.isfile(capture_json_path):
            try:
                capture_json_content = load_config_file(capture_json_path)
                # 1) Honour a user-set `thumbnail` override (set via the
                # picker on the Captures table).
                user_thumb = (capture_json_content or {}).get("thumbnail")
                if isinstance(user_thumb, str) and user_thumb and os.path.isfile(user_thumb):
                    thumbnail_path = user_thumb
                else:
                    # 2a) Image-frame footage (legacy ioi_root schema).
                    thumbnail_path = _sync.find_input_thumbnail(capture_json_content)
                    # 2b) Videos footage — extract a mid-frame. Covers
                    # released-dataset captures (capture_root + videos_*).
                    if not thumbnail_path:
                        capture_root_abs = _resolve_capture_root_abs(
                            capture_json_path, capture_json_content,
                        )
                        if capture_root_abs and os.path.isdir(capture_root_abs):
                            stem = os.path.splitext(os.path.basename(capture_json_path))[0]
                            thumbnail_path = _sync.find_video_thumbnail(
                                capture_json_content, capture_root_abs,
                                stem, _THUMB_CACHE_DIR,
                            )
                # Surface cams for the Captures table. Capture.json may not
                # declare a `cams` list; fall back to scanning the first
                # sequence dir on disk to pick them up so the UI shows real
                # camera names instead of an empty cell.
                if isinstance(capture_json_content, dict):
                    declared = capture_json_content.get("cams") or []
                    if isinstance(declared, list) and declared:
                        cams = [str(x) for x in declared]
                    else:
                        cams = _scan_cams_for_first_sequence(capture_json_content)
            except (OSError, ValueError):
                pass

        # 3) Output-side fallback: scan the latest run's ma_vis preview
        # (and only ma_vis — see _THUMBNAIL_STEP_PRIORITY in sync.py).
        if not thumbnail_path and latest.get("output_id") and latest.get("task_json_path"):
            task_root = latest.get("output_path") or DEFAULT_OUTPUT_DIR
            dataset = None
            try:
                tjp = latest.get("task_json_path")
                if tjp and os.path.isfile(tjp):
                    task_cfg = load_config_file(tjp)
                    g = (task_cfg or {}).get("global") or {}
                    task_root = _sync.resolve_output_root(task_cfg, task_root)
                    dataset = (g.get("dataset_name") or "").strip() or None
            except (OSError, ValueError):
                pass
            thumbnail_path = _sync.find_capture_thumbnail(
                task_root, latest["output_id"], dataset,
            )
        # Pull dataset_name from the most-recent task config when present;
        # the Results card shows it as a small caption beneath the name.
        dataset_name = None
        try:
            tjp = latest.get("task_json_path")
            if tjp and os.path.isfile(tjp):
                g = (load_config_file(tjp) or {}).get("global") or {}
                dn = (g.get("dataset_name") or "").strip()
                if dn:
                        dataset_name = dn
        except (OSError, ValueError):
            pass

        # Match the example-row schema so Home's data-presence dot works
        # for DB-row captures too. Without this the DB entry shadows the
        # example entry (it appears first in the response), and the
        # frontend reads `undefined` -> "not downloaded" even when the
        # data is on disk.
        released_present = False
        cap_root_abs = None
        if capture_json_path and capture_json_content is not None:
            cap_root_abs = _resolve_capture_root_abs(
                capture_json_path, capture_json_content,
            )
            released_present = cap_root_abs is not None and os.path.isdir(cap_root_abs)

        response.append({
            "id": str(c["capture_id"]),
            "captureName": c["capture_name"],
            "jsonPath": c["capture_json_path"],
            "seqNames": c["seq_names"],
            "cams": cams,
            "datasetName": dataset_name,
            "outputDir": c["ioi_root"],
            "dataPath": _display_capture_path(cap_root_abs),
            "status": c["status"],
            "processes": c["processes"],
            "createdAt": c["created_at"],
            "taskCount": c["task_count"],
            # Surface latest-task hints so the listing can render thumbnails
            # and "last run X ago" without a second round-trip.
            "thumbnailPath": thumbnail_path,
            "lastTaskAt": latest.get("created_at") if latest else None,
            "source": "user",
            "releasedDataPresent": released_present,
        })
    # Append shipped examples (read-only). Front-end shows them with a
    # badge / section divider and disables write actions for them.
    response.extend(_list_example_captures())
    return jsonify(response)


def _scan_cams_for_first_sequence(capture_content: dict) -> list[str]:
    """Best-effort camera list when capture.json has no `cams` field.

    Walks `<ioi_root>/<first_sequence>/` and treats each non-hidden subdir
    as a camera, mirroring the on-disk convention. Returns sorted names so
    the Captures UI is deterministic even when camera ordering on disk is
    arbitrary."""
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
    try:
        return sorted(
            entry for entry in os.listdir(seq_dir)
            if not entry.startswith(".") and os.path.isdir(os.path.join(seq_dir, entry))
        )
    except OSError:
        return []


def _resolve_capture_root_abs(capture_json_path: str | None, content: dict | None) -> str | None:
    """Return the absolute path of the capture's on-disk data root, or None.

    Captures generated by the released-dataset capture generator declare a
    relative ``capture_root`` (e.g. ``../../../data/mamma_markerless_dance``);
    user-imported captures declare an absolute ``ioi_root``. We accept
    either, resolving relatives against the JSON's parent dir."""
    if not isinstance(content, dict):
        return None
    raw = content.get("capture_root") or content.get("ioi_root")
    if not isinstance(raw, str) or not raw:
        return None
    if os.path.isabs(raw):
        cand = raw
    elif capture_json_path:
        cand = os.path.normpath(os.path.join(os.path.dirname(capture_json_path), raw))
    else:
        return None
    cand = os.path.abspath(cand)
    return cand if os.path.isdir(cand) else None


def _resolve_capture_thumbnail(capture_json_path, content, latest_task):
    """Best-effort absolute thumbnail path for a capture, or None.

    Mirrors the priority used by the captures-list route (``get_captures``):
      1) user-set ``thumbnail`` override on the capture JSON,
      2) raw input footage (image frames),
      3) a mid-frame extracted from ``videos_*/`` footage,
      4) the latest run's ma_vis preview output.
    ``content`` is the already-parsed capture JSON (may be None); ``latest_task``
    is the most-recent task dict from ``get_capture_details`` (may be None)."""
    import sync as _sync
    thumb = None
    if isinstance(content, dict) and capture_json_path:
        try:
            user_thumb = content.get("thumbnail")
            if isinstance(user_thumb, str) and user_thumb and os.path.isfile(user_thumb):
                return user_thumb
            thumb = _sync.find_input_thumbnail(content)
            if not thumb:
                cap_root_abs = _resolve_capture_root_abs(capture_json_path, content)
                if cap_root_abs and os.path.isdir(cap_root_abs):
                    stem = os.path.splitext(os.path.basename(capture_json_path))[0]
                    thumb = _sync.find_video_thumbnail(
                        content, cap_root_abs, stem, _THUMB_CACHE_DIR,
                    )
        except (OSError, ValueError):
            thumb = None
    if not thumb and latest_task and latest_task.get("output_id") and latest_task.get("task_json_path"):
        task_root = latest_task.get("output_path") or DEFAULT_OUTPUT_DIR
        dataset = None
        try:
            tjp = latest_task.get("task_json_path")
            if tjp and os.path.isfile(tjp):
                task_cfg = load_config_file(tjp)
                g = (task_cfg or {}).get("global") or {}
                task_root = _sync.resolve_output_root(task_cfg, task_root)
                dataset = (g.get("dataset_name") or "").strip() or None
        except (OSError, ValueError):
            pass
        try:
            thumb = _sync.find_capture_thumbnail(task_root, latest_task["output_id"], dataset)
        except (OSError, ValueError):
            thumb = None
    return thumb


def _example_capture_details(capture_name: str) -> dict | None:
    """Load a shipped-example capture's details from disk when the DB
    has no row for it (example captures aren't persisted). Returns the
    same shape as ``db.get_capture_details`` so the route handler can
    treat both paths uniformly."""
    if not _EXAMPLE_CAPTURES_DIR.is_dir():
        return None
    for ext in (".json", ".yaml", ".yml"):
        cand = _EXAMPLE_CAPTURES_DIR / f"{capture_name}{ext}"
        if cand.is_file():
            path = str(cand)
            break
    else:
        return None
    try:
        content = load_config_file(path)
    except (OSError, ValueError):
        return None
    sequences = []
    for _, val in (content.get("sequences") or {}).items():
        if not isinstance(val, dict):
            continue
        name = val.get("name") or val.get("ioi")
        if name:
            sequences.append({"name": name, "path": None})
    sequences.sort(key=lambda s: s["name"])
    return {
        "capture_name": capture_name,
        "capture_json_path": path,
        "ioi_root": content.get("capture_root") or content.get("ioi_root"),
        "sequences": sequences,
        "tasks": [],
    }


@app.route("/api/captures/<capture_name>", methods=["GET"])
def get_capture_detail(capture_name):
    details = db.get_capture_details(capture_name)
    if not details:
        # Example captures aren't in the DB — synthesize details from
        # the shipped JSON so the Results tab can still browse their
        # on-disk content.
        details = _example_capture_details(capture_name)
    if not details:
        return jsonify({"error": "Capture not found"}), 404

    tasks_formatted = []
    for t in details["tasks"]:
        status = "Completed"
        process_names = set()
        sequence_names = set()
        for p in t["processes"]:
            if p.get("process"):
                process_names.add(p["process"])
            if p.get("sequence"):
                sequence_names.add(p["sequence"])
            normalized = (p.get("status") or "").lower()
            if "fail" in normalized:
                status = "Failed"
            elif status != "Failed" and normalized not in ("completed", "done", "val_completed"):
                status = "Running"

        # Read dataset_name from the per-task generated task.json so the
        # file explorer can build the real output path
        # (<output_path>/<step>/<output_id>/<dataset_name>/<seq>/).
        dataset_name = None
        try:
            dataset_name = load_config_file(t["task_json_path"]).get("global", {}).get("dataset_name")
        except (OSError, ValueError, KeyError, TypeError):
            pass

        tasks_formatted.append({
            "id": str(t["task_id"]),
            "status": status,
            "startedAt": t["created_at"],
            "user": t["username"],
            "processes": sorted(process_names),
            "sequenceNames": sorted(sequence_names),
            "outputPath": t.get("output_path"),
            "outputId": t.get("output_id"),
            "datasetName": dataset_name,
        })

    # Essential capture metadata for the Results-page info card. Read straight
    # from the capture JSON (same source the captures-list route uses), so it
    # stays best-effort — a missing/unparseable JSON just yields empties and the
    # card omits those stats rather than failing the whole request.
    cams: list[str] = []
    cam_fps = None
    calib = None
    data_path = None
    content = None
    capture_json_path = details.get("capture_json_path")
    if capture_json_path and os.path.isfile(capture_json_path):
        try:
            content = load_config_file(capture_json_path)
            if isinstance(content, dict):
                declared = content.get("cams") or []
                if isinstance(declared, list) and declared:
                    cams = [str(x) for x in declared]
                else:
                    cams = _scan_cams_for_first_sequence(content)
                cam_fps = content.get("cam_fps")
                calib_val = content.get("calib")
                if isinstance(calib_val, str) and calib_val:
                    calib = os.path.basename(calib_val)
                cap_root_abs = _resolve_capture_root_abs(capture_json_path, content)
                data_path = _display_capture_path(cap_root_abs)
        except (OSError, ValueError):
            content = None

    # Same thumbnail priority as the captures list (user override -> input
    # footage -> mid-frame from videos -> latest ma_vis preview), so the
    # Results info card matches the card the user clicked to get here.
    latest_task = details["tasks"][0] if details.get("tasks") else None
    thumbnail_path = _resolve_capture_thumbnail(capture_json_path, content, latest_task)

    # Sequences come straight from the DB — no input-dir probing. The Results
    # view renders only local pipeline outputs and never consumed the old
    # `releasedSections` field (which scanned each sequence's input root — slow
    # when that root is on a network FS, e.g. a released dataset symlinked to the
    # cluster). If browsing released source assets is ever wanted, fetch it lazily
    # per sequence on expand rather than eagerly here.
    return jsonify({
        "captureName": details["capture_name"],
        "tasks": tasks_formatted,
        "sequences": details["sequences"],
        "cams": cams,
        "camFps": cam_fps,
        "calib": calib,
        "dataPath": data_path,
        "captureJsonPath": capture_json_path,
        "ioiRoot": details.get("ioi_root"),
        "thumbnailPath": thumbnail_path,
    })


@app.route("/api/captures/generate-json", methods=["POST"])
def generate_capture_json():
    data = request.json
    ioi_root_input = (data.get("ioiRoot") or "").strip().replace("\\", "/")
    calib_input = (data.get("calib") or "").strip().replace("\\", "/")
    output_name = (data.get("outputName") or "").strip()

    if not ioi_root_input or not calib_input:
        return jsonify({"error": "ioiRoot and calib are required"}), 400

    if not os.path.isdir(ioi_root_input):
        return jsonify({"error": f"ioi_root directory not found: {ioi_root_input}"}), 404

    # Belt-and-suspenders: validate the calibration BEFORE writing
    # anything. Clients should preflight first via
    # /api/captures/preflight, but a stale or hand-crafted POST mustn't
    # land a capture.json that points at a malformed YAML.
    from capture.calibration import (  # noqa: PLC0415
        load_calibration, detect_up_axis, normalize_up_axis, CalibrationError,
    )
    try:
        _calib = load_calibration(calib_input)
    except (FileNotFoundError, CalibrationError) as exc:
        return jsonify({"error": f"Invalid calibration: {exc}"}), 400
    # Up-axis: honor an explicit choice from the form, else auto-detect. Either
    # way it's stamped into capture.json so the pipeline + viewers all use it.
    _raw_up = str(data.get("upAxis") or "auto").strip().lower()
    try:
        up_axis_value = (normalize_up_axis(_raw_up)
                         if _raw_up and _raw_up != "auto"
                         else detect_up_axis(_calib)[0])
    except CalibrationError:
        up_axis_value = detect_up_axis(_calib)[0]

    if not output_name:
        output_name = os.path.basename(ioi_root_input.rstrip("/"))

    sequences = {}
    try:
        entries = sorted(
            e for e in os.listdir(ioi_root_input)
            if os.path.isdir(os.path.join(ioi_root_input, e)) and e != "logs"
        )
        for i, entry in enumerate(entries):
            sequences[f"{i:03}"] = {"ioi": entry}
    except Exception as e:
        return jsonify({"error": f"Error scanning ioi_root: {e}"}), 500

    if not entries:
        return jsonify({"error": "No sequences found under ioi_root"}), 400

    # Detect cameras + the videos subdir from the first sequence's actual
    # layout. Earlier this route listed every subdir of the first sequence
    # as a "camera", which incorrectly included `meta/`, `preview/`,
    # `videos_light/` etc. and double-counted basenames when both
    # `videos/` and `videos_light/` existed. The shared
    # _detect_seq_layout helper picks the first canonical videos subdir
    # at level-1 only (or falls back to per-camera image dirs).
    first_entry_path = os.path.join(ioi_root_input, entries[0])
    layout, cams, videos_subdir = _detect_seq_layout(first_entry_path)
    if not cams:
        return jsonify({
            "error": (
                f"Could not detect cameras in the first sequence "
                f"'{entries[0]}'. Expected either a videos/* subdir of "
                f".mp4 files or per-camera image directories."
            ),
        }), 400


    capture_json_data = {
        "ioi_root": ioi_root_input,
        "calib": calib_input,
        "cam_fps": 30,
        "up_axis": up_axis_value,
        "cams": cams,
        "sequences": sequences,
    }
    # Record the detected videos subdir whenever the layout is videos.
    # This is the signal the inference layer uses to tell a video
    # capture from a per-camera image-dir capture (which omits it): see
    # inference/config.py._derive_videos_dir. Always writing it — even
    # for the "videos_crf24" default — keeps that videos-vs-images
    # decision unambiguous for GUI-generated captures (which carry
    # ioi_root, not capture_root).
    if layout == "videos" and videos_subdir:
        capture_json_data["videos_subdir"] = videos_subdir

    capture_json_dir = os.path.join(MOUNT_POINT, "capture_jsons")
    os.makedirs(capture_json_dir, exist_ok=True)
    out_path = os.path.join(capture_json_dir, f"{output_name}.json")

    # Reject silent overwrites: a previous "Save capture info" run with
    # the same name produced a JSON we would otherwise stomp. The
    # frontend prompts the user; on confirm it re-POSTs with
    # `overwrite: true` and we proceed to overwrite-with-reconcile.
    overwrite = bool(data.get("overwrite"))
    if not overwrite and os.path.isfile(out_path):
        return jsonify({
            "error": f"A capture named '{output_name}' already exists.",
            "code": "name_in_use",
            "existingName": output_name,
        }), 409

    try:
        with open(out_path, "w") as f:
            json.dump(capture_json_data, f, indent=2)
    except Exception as e:
        return jsonify({"error": f"Failed to write capture JSON: {e}"}), 500

    # Insert / update the captures row and reconcile sequences from the
    # just-written JSON. Idempotent on first-create; on overwrite this
    # also refreshes ioi_root + capture_name on the existing row and
    # prunes orphan sequence rows with no associated processes.
    db.save_capture_with_sequences(output_name, ioi_root_input, out_path)

    relative_path = os.path.relpath(out_path, MOUNT_POINT)
    return jsonify({
        "message": "Capture JSON created successfully",
        "outputName": output_name,
        "path": relative_path,
        "sequenceCount": len(sequences),
    }), 201


# ─── Preflight check (read-only) ─────────────────────────────────────────
#
# The New Task form's "Create capture" step uses this to render live
# validation badges as the user types. It runs the same probes the
# /generate-json route runs, but never writes to disk or touches the DB
# and always returns 200 (unless the request body itself is malformed).
# Two independent sides — `footage` and `calibration` — so the UI can
# show two badges that flip green independently.

def _preflight_footage_empty(error: str | None) -> dict:
    return {
        "ok": False, "error": error,
        "sequences": 0, "cameras": 0,
        "layout": None, "firstSequence": None,
        "sequenceNames": [], "cameraNames": [],
    }


def _preflight_calibration_empty(error: str | None) -> dict:
    return {
        "ok": False, "error": error,
        "cameraCount": 0, "distortionModels": [],
        "cameraNames": [],
    }


# Canonical videos subdir names in priority order. Probed at level-1
# (no recursion) so a sequence dir that contains *both* "videos/" and
# "videos_light/" (typical for the iPhone release) doesn't double-count,
# and preview clips like "preview/overlay_grid.mp4" or
# "preview/overlay/<cam>_overlay.mp4" aren't mistaken for cameras.
_VIDEOS_SUBDIRS = ("videos", "videos_light", "videos_crf24", "videos_crf16")


def _detect_seq_layout(seq_dir: str) -> tuple:
    """Detect (layout, sorted_camera_names, videos_subdir) for a sequence dir.

    Returns (None, [], None) when neither a canonical videos subdir nor
    an image-camera subdir layout is detectable.
    """
    from capture.discovery import find_video_files, find_image_cam_dirs  # noqa: PLC0415
    for sub in _VIDEOS_SUBDIRS:
        cand = os.path.join(seq_dir, sub)
        if not os.path.isdir(cand):
            continue
        files = find_video_files(cand)
        if files:
            cams = sorted({os.path.splitext(os.path.basename(v))[0] for v in files})
            return "videos", cams, sub
    cam_dirs = find_image_cam_dirs(seq_dir)
    if cam_dirs:
        cams = sorted(os.path.basename(d) for d in cam_dirs)
        return "images", cams, None
    return None, [], None


def _preflight_footage(ioi_root: str) -> dict:
    if not os.path.isdir(ioi_root):
        return _preflight_footage_empty(f"Directory not found: {ioi_root}")
    try:
        entries = sorted(
            e for e in os.listdir(ioi_root)
            if os.path.isdir(os.path.join(ioi_root, e)) and e != "logs"
        )
    except OSError as exc:
        return _preflight_footage_empty(f"Cannot scan directory: {exc}")
    if not entries:
        return _preflight_footage_empty("No sequences found in this directory")

    first_seq = entries[0]
    first_seq_path = os.path.join(ioi_root, first_seq)

    try:
        layout, cams, _videos_subdir = _detect_seq_layout(first_seq_path)
    except OSError as exc:
        return _preflight_footage_empty(f"Layout probe failed: {exc}")

    if layout is None:
        return {
            "ok": False,
            "error": (
                f"First sequence '{first_seq}' contains neither a "
                "videos/ subdir with .mp4 files nor per-camera image "
                "subdirs — does this look right?"
            ),
            "sequences": len(entries), "cameras": 0,
            "layout": None, "firstSequence": first_seq,
            "sequenceNames": [], "cameraNames": [],
        }

    return {
        "ok": True, "error": None,
        "sequences": len(entries),
        "cameras": len(cams),
        "layout": layout,
        "firstSequence": first_seq,
        # Full lists so the New Task form can populate Step 3's
        # sequence + camera dropdowns directly from preflight, without
        # an explicit "Save Capture Info" round-trip.
        "sequenceNames": entries,
        "cameraNames": cams,
    }


def _preflight_calibration(calib: str) -> dict:
    from capture.calibration import load_calibration, CalibrationError  # noqa: PLC0415
    try:
        calibration = load_calibration(calib)
    except FileNotFoundError as exc:
        return _preflight_calibration_empty(str(exc))
    except CalibrationError as exc:
        return _preflight_calibration_empty(str(exc))
    except Exception as exc:  # noqa: BLE001 — surface any unexpected parse failure
        return _preflight_calibration_empty(f"{type(exc).__name__}: {exc}")

    distortion_models = sorted({c.distortion_model for c in calibration.cameras.values()})
    return {
        "ok": True, "error": None,
        "cameraCount": len(calibration.cameras),
        "distortionModels": distortion_models,
        "cameraNames": sorted(calibration.cameras.keys()),
    }


@app.route("/api/captures/preflight", methods=["POST"])
def preflight_capture_json():
    data = request.json or {}
    ioi_root = (data.get("ioiRoot") or "").strip().replace("\\", "/")
    calib = (data.get("calib") or "").strip().replace("\\", "/")

    if not ioi_root and not calib:
        return jsonify({"error": "ioiRoot and/or calib required"}), 400

    footage = (
        _preflight_footage(ioi_root)
        if ioi_root else _preflight_footage_empty(None)
    )
    calibration = (
        _preflight_calibration(calib)
        if calib else _preflight_calibration_empty(None)
    )
    return jsonify({"footage": footage, "calibration": calibration})


def _resolve_calib_arg(data: dict) -> tuple[str | None, str | None]:
    """Resolve a calibration path from a request body. Accepts ``calibPath``
    (+ optional ``baseDir``) or ``captureJsonPath`` (reads its ``calib`` field).
    Relative paths are anchored to ``baseDir`` / the capture.json's dir, mirroring
    the pipeline. Returns ``(resolved_abs_path, error_message)`` — one is None."""
    calib = (data.get("calibPath") or "").strip().replace("\\", "/")
    base_dir = (data.get("baseDir") or "").strip().replace("\\", "/")
    capture_json = (data.get("captureJsonPath") or "").strip()
    if capture_json and not calib:
        if not os.path.isfile(capture_json):
            return None, f"capture json not found: {capture_json}"
        try:
            content = load_config_file(capture_json) or {}
        except (OSError, ValueError) as e:
            return None, f"could not read capture json: {e}"
        calib = str(content.get("calib") or "").strip()
        if not calib:
            return None, "capture json has no 'calib' field"
        base_dir = os.path.dirname(os.path.abspath(capture_json))
    if not calib:
        return None, "calibPath or captureJsonPath is required"
    resolved = calib
    if not os.path.isabs(calib):
        resolved = os.path.normpath(os.path.join(base_dir, calib)) if base_dir \
            else os.path.abspath(calib)
    if not os.path.exists(resolved):
        # Echo what the user actually entered. Only show the resolved absolute
        # path when it was anchored to a meaningful base (the capture.json dir);
        # otherwise a relative input would surface the backend's CWD, which is
        # irrelevant and confusing.
        shown = resolved if (base_dir and not os.path.isabs(calib)) else calib
        return None, f"calibration path not found: {shown}"
    return resolved, None


@app.route("/api/calib/inspect", methods=["POST"])
def calib_inspect_route():
    """Lightweight calibration validation for the live status line under the
    calib input. Always returns 200 with ``{ok, ...}`` so the frontend treats a
    bad file as a field error, not a network failure."""
    resolved, err = _resolve_calib_arg(request.json or {})
    if err:
        return jsonify({"ok": False, "error": err})
    try:
        from capture.calibration import load_calibration, detect_up_axis
        cal = load_calibration(resolved)
        up_axis, up_conf = detect_up_axis(cal)
        return jsonify({
            "ok": True,
            "cameraCount": len(cal.cameras),
            "cameraNames": sorted(cal.cameras),
            "distortionModels": sorted({c.distortion_model for c in cal.cameras.values()}),
            "sourceFormat": cal.source_format,
            "upAxis": up_axis,                       # auto-detected world up-axis
            "upAxisConfidence": round(up_conf, 3),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/calib/preview", methods=["POST"])
def calib_preview_route():
    """Build a small camera-rig .rrd from a calibration so the user can eyeball
    their multi-view setup and confirm the world2cam/cam2world convention (a
    correct rig has cameras around the scene, looking inward). Returns the .rrd
    path, which the GUI opens in its existing Rerun viewer."""
    data = request.json or {}
    resolved, err = _resolve_calib_arg(data)
    if err:
        return jsonify({"error": err}), 400
    try:
        from capture.calibration import (
            load_calibration, detect_up_axis, normalize_up_axis,
        )
        import calib_preview  # gui/backend/calib_preview.py
        # Resolve the up-axis: explicit signed value, else auto-detect.
        raw = str(data.get("upAxis") or "auto").strip().lower()
        if raw and raw != "auto":
            axis = normalize_up_axis(raw)
        else:
            axis = detect_up_axis(load_calibration(resolved))[0]
        out_dir = os.path.join(MOUNT_POINT, "calib_previews")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.basename(resolved.rstrip("/")) or "calib"
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)
        # Axis in the filename → each up-axis is a distinct file, so the viewer
        # reloads when the user switches axes. "-y" → "neg_y" to keep it tidy.
        tag = axis.replace("-", "neg_")
        out_path = os.path.join(out_dir, f"{safe}__{tag}up.rrd")
        n = calib_preview.write_calib_rrd(resolved, out_path, up_axis=axis)
        return jsonify({"rrdPath": out_path, "cameraCount": n, "upAxis": axis})
    except Exception as e:
        return jsonify({"error": f"Could not build camera-rig preview: {e}"}), 400


def _safe_capture_json_path(rel_or_abs_path: str) -> str | None:
    """Resolve a user-supplied capture-json path to an absolute path.

    Trust model mirrors the existing `/api/files/*` endpoints: the
    backend binds to loopback by default (`MAMMA_BIND_HOST=127.0.0.1`),
    so the caller already has the same filesystem permissions as the
    Flask user. Absolute paths are accepted as-is — many users keep
    captures on a shared mount outside `MAMMA_INTERFACE_DIR`.

    Relative paths are still anchored inside `MOUNT_POINT` after
    normalisation, blocking `../foo.json` traversal. Non-.json suffixes
    are rejected. Returns None for any rejected input."""
    if not rel_or_abs_path:
        return None
    if os.path.isabs(rel_or_abs_path):
        candidate = os.path.normpath(rel_or_abs_path)
    else:
        candidate = os.path.normpath(os.path.join(MOUNT_POINT, rel_or_abs_path))
        # Relative inputs must still resolve inside MOUNT_POINT after
        # normalisation — collapses `..` traversal.
        mount = os.path.normpath(MOUNT_POINT)
        try:
            if os.path.commonpath([candidate, mount]) != mount:
                return None
        except ValueError:
            return None
    if not candidate.endswith(".json"):
        return None
    return candidate


@app.route("/api/captures/run-groups", methods=["GET"])
def capture_run_groups():
    """List existing run groups (distinct `output_id` values) for a
    capture, with a per-(seq, step) "is it already done?" map. The New
    Task form uses this to pre-fill / auto-update Run-steps when the
    user reuses an Output ID — see runner DONE-sentinel skip logic in
    inference/runner.py."""
    rel = request.args.get("path", "")
    abs_path = _safe_capture_json_path(rel)
    if abs_path is None:
        return jsonify({"error": "Invalid capture path"}), 400
    if not os.path.isfile(abs_path):
        return jsonify({"error": "Capture file not found"}), 404

    capture = db.get_capture_by_json_path(abs_path)
    if capture is None:
        # No tasks have ever touched this capture — return empty list.
        return jsonify({"runGroups": []})

    try:
        capture_content = load_config_file(abs_path)
    except (OSError, ValueError):
        capture_content = {}
    sequences = sorted(get_sequences_from_data(capture_content))
    all_steps = list(_ProcessType.__members__.keys())

    raw_groups = db.get_tasks_grouped_by_output_id(capture["capture_id"])

    result = []
    for g in raw_groups:
        # Resolve out_dir + dataset_name from the latest task config when
        # available — covers custom out_dirs that aren't the default tree.
        out_dir = g.get("latestOutputPath") or _audit_output_root()
        dataset_name = ""
        try:
            tjp = g.get("latestTaskJsonPath")
            if tjp and os.path.isfile(tjp):
                cfg = load_config_file(tjp)
                gcfg = (cfg or {}).get("global") or {}
                out_dir = _pipeline_sync.resolve_output_root(cfg, out_dir)
                dataset_name = (gcfg.get("dataset_name") or "").strip()
        except (OSError, ValueError):
            pass

        steps_done = _pipeline_sync.enumerate_done_sentinels(
            out_dir, g["outputId"], dataset_name, sequences, all_steps,
        )
        result.append({
            "outputId": g["outputId"],
            "submissions": g["submissions"],
            "lastSubmittedAt": g["lastSubmittedAt"],
            "outputDir": out_dir,
            "datasetName": dataset_name,
            "stepsDone": steps_done,
        })

    return jsonify({"runGroups": result})


@app.route("/api/captures/thumbnail-candidates", methods=["GET"])
def capture_thumbnail_candidates():
    """List one representative frame per camera for a capture's first
    sequence — backs the "edit thumbnail" picker on the Captures table.

    Also surfaces the currently-active thumbnail (override or auto) so
    the modal can highlight it."""
    rel = request.args.get("path", "")
    abs_path = _safe_capture_json_path(rel)
    if abs_path is None:
        return jsonify({"error": "Invalid capture path"}), 400
    if not os.path.isfile(abs_path):
        return jsonify({"error": "Capture file not found"}), 404
    try:
        content = load_config_file(abs_path)
    except (OSError, ValueError) as e:
        return jsonify({"error": f"Failed to read capture: {e}"}), 500

    candidates = _pipeline_sync.list_input_thumbnail_candidates(content)
    user_override = content.get("thumbnail") if isinstance(content, dict) else None
    auto_detected = _pipeline_sync.find_input_thumbnail(content)
    return jsonify({
        "candidates": candidates,
        "userOverride": user_override if isinstance(user_override, str) else None,
        "autoDetected": auto_detected,
    })


@app.route("/api/captures/json", methods=["GET"])
def read_capture_json():
    """Return the raw parsed content of a capture.json so the manage UI can
    edit it. The frontend addresses captures by their relative path."""
    rel = request.args.get("path", "")
    abs_path = _safe_capture_json_path(rel)
    if abs_path is None:
        return jsonify({"error": "Invalid capture path"}), 400
    if not os.path.isfile(abs_path):
        return jsonify({"error": "Capture file not found"}), 404
    try:
        content = load_config_file(abs_path)
    except (OSError, ValueError) as e:
        return jsonify({"error": f"Failed to read capture: {e}"}), 500
    stat = os.stat(abs_path)
    return jsonify({
        # Echo back the absolute path so save+delete round-trip cleanly,
        # even when the file lives outside MAMMA_INTERFACE_DIR (a relative
        # path with `../` traversal would fail the relative-path
        # containment check on the way back in).
        "path": abs_path,
        "absolutePath": abs_path,
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "content": content,
    })


@app.route("/api/captures/json", methods=["PUT"])
def update_capture_json():
    """Overwrite a capture.json with edited content. Also syncs the captures
    row's ioi_root + display name with the new content so listings stay
    consistent."""
    data = request.json or {}
    rel = data.get("path", "")
    content = data.get("content")
    if content is None or not isinstance(content, dict):
        return jsonify({"error": "Missing or invalid `content`"}), 400
    abs_path = _safe_capture_json_path(rel)
    if abs_path is None:
        return jsonify({"error": "Invalid capture path"}), 400
    if not os.path.isfile(abs_path):
        return jsonify({"error": "Capture file not found"}), 404
    if _is_readonly_path(abs_path):
        return jsonify({
            "error": "This capture is a shipped example (read-only). "
                     "Use 'Save a writable copy' to edit it.",
        }), 403

    try:
        with open(abs_path, "w") as f:
            json.dump(content, f, indent=2)
    except OSError as e:
        return jsonify({"error": f"Failed to write capture: {e}"}), 500

    # Mirror the new ioi_root onto the captures row. Display name is derived
    # from the file basename and stays stable; only ioi_root changes here.
    new_ioi_root = (content.get("ioi_root") or "").strip() or None
    if new_ioi_root:
        db.update_capture_metadata(abs_path, ioi_root=new_ioi_root)

    stat = os.stat(abs_path)
    return jsonify({
        "ok": True,
        "path": abs_path,
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
    })


@app.route("/api/captures/db", methods=["DELETE"])
def delete_capture_db_row():
    """Remove a capture from the **database only**. Cascades sequences,
    tasks, and processes for that capture. **Does not touch any files**:
    the capture.json itself, every task_json, every output under
    `<output_path>/<step>/<output_id>/`, and every log under
    `<jobs_log_dir>/<user>/<task_id>/` are all left in place.

    Intent: hide a capture from the Results grid / Tasks table without
    losing artifacts. The user can resubmit the same capture.json later
    and the runner will re-create rows; existing DONE sentinels still
    short-circuit finished work.

    Refuses if any task's runner subprocess is still alive — stop those
    first so we don't orphan a process the DB no longer knows about.
    """
    rel = request.args.get("path", "")
    abs_path = _safe_capture_json_path(rel)
    if abs_path is None:
        return jsonify({"error": "Invalid capture path"}), 400

    cap_row = db.get_capture_by_json_path(abs_path)
    if not cap_row:
        return jsonify({"error": "Capture not found in database"}), 404

    # Liveness probe on each task's runner before we cascade. We hit the
    # tasks table directly via the helper since `get_capture_by_json_path`
    # only returns the captures row.
    for t in db.get_all_task_minimal():
        if t.get("captureJsonPath") != abs_path:
            continue
        # The minimal helper doesn't expose runnerPid; refetch per id.
        full = db.get_task_by_id(int(t["taskId"]))
        pid = full.get("runnerPid") if full else None
        if not pid:
            continue
        try:
            os.kill(int(pid), 0)
            return jsonify({
                "error": (
                    f"Task #{t['taskId']} (runner pid={pid}) is still running. "
                    "Stop it from the Tasks tab first, then delete the capture."
                ),
            }), 409
        except ProcessLookupError:
            pass  # stale pid, safe to drop

    result = db.force_delete_capture_by_path(abs_path)
    if not result["deleted"]:
        return jsonify({"error": "Capture not found in database"}), 404
    return jsonify({
        "ok": True,
        "filesDeleted": False,
        "captureJsonPath": abs_path,
    })


@app.route("/api/captures/json", methods=["DELETE"])
def delete_capture_json():
    """Remove a capture.json file. The captures DB row is removed only when
    no tasks reference it — this preserves historical task readability when
    a still-used capture is deleted by mistake.
    """
    rel = request.args.get("path", "")
    abs_path = _safe_capture_json_path(rel)
    if abs_path is None:
        return jsonify({"error": "Invalid capture path"}), 400
    if not os.path.isfile(abs_path):
        return jsonify({"error": "Capture file not found"}), 404
    if _is_readonly_path(abs_path):
        return jsonify({
            "error": "This capture is a shipped example (read-only); "
                     "cannot be deleted.",
        }), 403

    task_count = db.delete_capture_by_path(abs_path)

    try:
        os.remove(abs_path)
    except OSError as e:
        return jsonify({"error": f"Failed to delete capture: {e}"}), 500

    return jsonify({
        "ok": True,
        "tasksAffected": task_count,
        "dbRowKept": task_count > 0,
    })


@app.route("/api/captures/list-jsons", methods=["GET"])
def list_capture_jsons():
    capture_json_dir = os.path.join(MOUNT_POINT, "capture_jsons")
    if not os.path.isdir(capture_json_dir):
        return jsonify([])
    try:
        files = sorted(f for f in os.listdir(capture_json_dir) if f.endswith(".json"))
        result = []
        for f in files:
            full_path = os.path.join(capture_json_dir, f)
            rel_path = os.path.relpath(full_path, MOUNT_POINT)
            stat = os.stat(full_path)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            result.append({"name": f[:-5], "filename": f, "path": rel_path, "modified": modified})
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/captures/parse-sequences", methods=["POST"])
def get_sequences_from_file():
    data = request.json
    rel_path = data.get("captureJsonPath", "")
    if not rel_path:
        return jsonify({"error": "No path provided"}), 400

    capture_json_path = os.path.join(MOUNT_POINT, rel_path)
    if not os.path.exists(capture_json_path):
        return jsonify({"error": f"File not found at {capture_json_path}"}), 404

    try:
        data = load_config_file(capture_json_path)
        sequences_list = sorted(get_sequences_from_data(data))
        cameras = None
        cams_attr = data.get("cams")
        if isinstance(cams_attr, list):
            cameras = [str(cam) for cam in cams_attr if str(cam).strip()] or None
        return jsonify({"sequences": sequences_list, "cameras": cameras})
    except Exception as e:
        return jsonify({"error": f"Error parsing capture file: {e}"}), 500


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def _deep_merge(base, override):
    """Recursively merge `override` into `base` in place.

    Nested dicts are recursed into; lists and scalars in `override` replace
    whatever was in `base` (so users can fully redefine a flags array or
    bind path list, not just append). Returns the mutated `base`.
    """
    if not isinstance(override, dict) or not isinstance(base, dict):
        return override if override is not None else base
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


@app.route("/api/tasks/preview-commands", methods=["POST"])
def preview_task_commands():
    """Resolve the EXACT per-step shell command the runner would execute for
    the current form selection — WITHOUT creating a task, writing a run
    config, or running anything.

    Same code path as a real run (``materialize_run_config`` -> ``get_builder``
    -> ``engines.build_command``), so it's an accurate confirmation that the
    user's flags landed. Per-step errors (e.g. a missing MAMMA_* asset or an
    unresolved --weights) are returned inline so one bad step doesn't blank the
    whole preview (and it doubles as a pre-flight check).
    """
    import shlex as _shlex
    from inference.steps import get_builder
    from inference.engines import build_command
    from inference.runner import enabled_steps as _enabled_steps, topo_order, resolve_seq_names

    data = request.json or {}
    capture_json_path = os.path.join(MOUNT_POINT, data.get("captureJsonPath", ""))
    preset_path = (
        DEFAULT_PRESET_PATH
        if not data.get("taskJsonPath")
        else os.path.join(MOUNT_POINT, data["taskJsonPath"])
    )
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, data.get("outputDir", ""))
    seq_names = data.get("seqNames")
    cameras = data.get("cameras", [])
    processes = data.get("processes", [])
    output_id = (data.get("outputId") or "").strip()
    task_overrides = data.get("taskOverrides") or {}
    sequence_major = bool(data.get("sequenceMajor", False))

    try:
        proc_enum = [ProcessType[p] for p in processes]
    except KeyError as e:
        return jsonify({"error": f"Invalid process type: {e}"}), 400
    if not capture_json_path or not preset_path or not seq_names:
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        task_data = materialize_run_config(
            preset_path,
            capture_json_path,
            seq_names=seq_names,
            cam_names=cameras,
            out_dir=output_dir,
            username=LOCAL_USER,
            enabled_steps=[p.name for p in proc_enum],
            overrides=task_overrides or None,
            sequence_major=sequence_major,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to materialize run config: {e}"}), 400

    # Representative sequence + output tag. The output_id is only known after
    # task creation (it defaults to the DB row id when left blank), and it only
    # affects the <tag> path segment in --out/--out_path — show a clearly
    # symbolic placeholder so the user isn't surprised.
    try:
        resolved = resolve_seq_names(task_data)
    except Exception:
        resolved = seq_names
    seq_name = (resolved or seq_names)[0]
    tag = output_id or "<output_id>"

    g = task_data.get("global", {})
    commands = {}
    for step_name in topo_order(task_data, _enabled_steps(task_data)):
        step_cfg = task_data.get(step_name, {})
        try:
            builder = get_builder(step_name, step_cfg, g, tag)
            cmd, cwd = build_command(builder, seq_name)
            commands[step_name] = {
                "command": " ".join(_shlex.quote(c) for c in cmd),
                "cwd": cwd,
                "engine": builder.engine,
            }
        except Exception as e:  # missing asset / unresolved weights / bad engine
            commands[step_name] = {"error": str(e)}

    return jsonify({
        "commands": commands,
        "seqName": seq_name,
        "outputIdPlaceholder": None if output_id else tag,
    })


@app.route("/api/tasks", methods=["POST"])
def start_task():
    data = request.json or {}
    capture_json_path = os.path.join(MOUNT_POINT, data.get("captureJsonPath", ""))
    preset_path = (
        DEFAULT_PRESET_PATH
        if not data.get("taskJsonPath")
        else os.path.join(MOUNT_POINT, data["taskJsonPath"])
    )
    # Empty or relative form value → land in the repo-local default
    # (./var/output/<form value>). Absolute paths pass through (os.path.join
    # discards earlier components when a later one is absolute).
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, data.get("outputDir", ""))
    seq_names = data.get("seqNames")
    cameras = data.get("cameras", [])
    processes = data.get("processes", [])
    output_id = data.get("outputId") or None
    # Deep-partial of the preset structure carrying user edits. Applied on
    # top of the loaded preset before the form-level fields land, so the
    # saved run_<id>.json reflects exactly what runs.
    task_overrides = data.get("taskOverrides") or {}
    # Sequence dispatch order: True = sequence-major (finish each seq
    # end-to-end before the next); False (default) = step-major
    # (today's behaviour: finish each step across all seqs first).
    sequence_major = bool(data.get("sequenceMajor", False))

    try:
        processes = [ProcessType[p] for p in processes]
    except KeyError as e:
        return jsonify({"error": f"Invalid process type: {e}"}), 400

    if not capture_json_path or not preset_path or not seq_names or not output_dir:
        return jsonify({"error": "Missing required parameters"}), 400

    # Build the bound run config via the shared materializer (same code
    # path the CLI's ``run --preset … --capture …`` uses).
    try:
        os.makedirs(RUN_CONFIG_DIR, exist_ok=True)
        capture_data = load_config_file(capture_json_path)
        task_data = materialize_run_config(
            preset_path,
            capture_json_path,
            seq_names=seq_names,
            cam_names=cameras,
            out_dir=output_dir,
            username=LOCAL_USER,
            enabled_steps=[p.name for p in processes],
            overrides=task_overrides or None,
            sequence_major=sequence_major,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to generate run configuration: {e}"}), 500

    status, task_id, task_json_final_path, output_id = db.create_entry(
        capture_json_path,
        seq_names,
        output_dir,
        processes,
        username=LOCAL_USER,
        task_json_server_dir=RUN_CONFIG_DIR,
        capture_content=capture_data,
        task_content=task_data,
        task_template_path=preset_path,
        output_id=output_id,
        preset_path=preset_path,
    )
    if status != "Entry created successfully":
        return jsonify({"error": status}), 500

    try:
        with open(task_json_final_path, "w") as f:
            json.dump(task_data, f, indent=2)
        print(f"Wrote task config to {task_json_final_path}")
    except Exception as e:
        return jsonify({"error": f"Failed to write task configuration: {e}"}), 500

    # Enqueue for the task coordinator to spawn when a runner slot is
    # free (default: one task at a time; raise via the Run-mode toggle
    # in the Tasks page if you have multiple GPUs). The runner subprocess
    # is no longer spawned inline — `_task_queue.enqueue` schedules it,
    # and the coordinator thread is the sole place where Popen happens.
    _task_queue.enqueue(
        task_id=task_id,
        task_json_path=task_json_final_path,
        output_id=output_id,
        output_dir=output_dir,
        jobs_log_dir=task_data["global"].get("jobs_log_dir", os.path.join(output_dir, "logs")),
    )

    return jsonify({"message": "Task queued", "taskId": task_id}), 201


@app.route("/api/tasks/history", methods=["GET"])
def get_task_history():
    try:
        tasks = db.get_all_tasks_with_processes()
        # Decorate any task whose runner hasn't been spawned yet with
        # its 1-indexed queue position. Frontend renders "Queued · #N"
        # when this is set. Tasks not in the queue get None.
        for t in tasks:
            try:
                tid = int(t.get("taskId") or t.get("task_id") or 0)
            except (TypeError, ValueError):
                continue
            if tid:
                pos = _task_queue.queue_position(tid)
                if pos is not None:
                    t["queuePosition"] = pos
        return jsonify(tasks)
    except Exception as e:
        print(f"Error fetching task history: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Task presets — curated task.json templates the UI surfaces in a dropdown
# and previews to the user. Convention: $MAMMA_INTERFACE_DIR/samples/presets/*.json.
# Each preset can carry an optional `global.display_name` and
# `global.description`. The digest endpoint returns a structured summary
# of what the preset will run so the form can render it without parsing.
# ---------------------------------------------------------------------------

_PRESETS_DIR = os.path.join(MOUNT_POINT, "samples", "presets")
_USER_SUBDIR = "user"  # presets saved by users land in samples/presets/user/
_PRESET_STEP_ORDER = ["ma_cap", "ma_masks", "ma_2d", "ma_3d", "ma_vis"]
_PRESET_META_KEYS = {  # not part of any step's argv — used to filter the "extra" flags display
    "submit_cfg", "engine", "enabled", "dependencies", "script",
    "sif_path", "docker_image", "repo_path", "flags",
}
_USER_PRESET_NAME_RE = _re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$')


def _safe_preset_name(name: str) -> str | None:
    """Validate a preset identifier.

    Accepts either a bare basename (curated) or `user/<basename>` (user-saved).
    Returns the canonicalised name (with the user/ prefix preserved when present)
    or None if the name is unsafe — protects against `..` traversal, absolute
    paths, and silly extensions.
    """
    if not name:
        return None
    parts = name.split("/")

    def _ok(bare: str) -> bool:
        return bool(bare) and not bare.startswith(".") and not bare.endswith(".json") \
               and "\\" not in bare and ".." not in bare

    if len(parts) == 1:
        return parts[0] if _ok(parts[0]) else None
    if len(parts) == 2 and parts[0] == _USER_SUBDIR and _ok(parts[1]):
        return f"{_USER_SUBDIR}/{parts[1]}"
    return None


def _preset_path(safe_name: str) -> str:
    return os.path.join(_PRESETS_DIR, safe_name + ".json")


def _example_preset_path(name: str) -> str | None:
    """Resolve `<dir-stem>/<file-stem>` to a path under
    _EXAMPLE_PRESETS_DIRS, mirroring the names emitted by
    _list_example_presets. Returns None if the name isn't a
    valid example preset reference. Accepted dir-stems:
    ``presets`` (full-scale) and ``quick`` (smoke variants).
    """
    if "/" not in name or ".." in name or "\\" in name:
        return None
    for root in _EXAMPLE_PRESETS_DIRS:
        if not root.is_dir():
            continue
        prefix = f"{root.name}/"
        if name.startswith(prefix):
            stem = name[len(prefix):]
            if not stem or "/" in stem:
                return None
            for ext in (".yaml", ".yml", ".json"):
                cand = root / f"{stem}{ext}"
                if cand.is_file():
                    return str(cand)
    return None


def _read_preset(name: str) -> tuple[str, dict] | None:
    # Curated / user-saved presets live in the writable area.
    safe = _safe_preset_name(name)
    if safe is not None:
        path = _preset_path(safe)
        if os.path.isfile(path):
            try:
                return path, load_config_file(path)
            except (OSError, ValueError):
                return None
    # Shipped examples (read-only) — namespaced as `<dir>/<stem>`.
    ex_path = _example_preset_path(name)
    if ex_path is not None:
        try:
            return ex_path, load_config_file(ex_path)
        except (OSError, ValueError):
            return None
    return None


def _preset_summary(name: str, path: str, *, is_user: bool) -> dict:
    """Build the dict the listing endpoint returns for one preset."""
    display_name = name
    description = ""
    try:
        cfg = load_config_file(path)
        g = cfg.get("global") or {}
        display_name = g.get("display_name") or name
        description = g.get("description") or ""
    except (OSError, ValueError):
        # Bad file — still list it; the digest endpoint will report the error.
        pass
    return {
        "name": name,
        "displayName": display_name,
        "description": description,
        "path": path,
        "isUser": is_user,
        "source": "user" if is_user else "example",
    }


@app.route("/api/task-presets", methods=["GET"])
def list_task_presets():
    """List both curated and user-saved task presets, plus repo examples."""
    out = []
    # Curated: flat .json files at the top of samples/presets/ (writable root).
    if os.path.isdir(_PRESETS_DIR):
        for fname in sorted(os.listdir(_PRESETS_DIR)):
            full = os.path.join(_PRESETS_DIR, fname)
            if fname.endswith(".json") and os.path.isfile(full):
                out.append(_preset_summary(fname[:-5], full, is_user=False))
        # User-saved: nested in samples/presets/user/.
        user_dir = os.path.join(_PRESETS_DIR, _USER_SUBDIR)
        if os.path.isdir(user_dir):
            for fname in sorted(os.listdir(user_dir)):
                full = os.path.join(user_dir, fname)
                if fname.endswith(".json") and os.path.isfile(full):
                    out.append(_preset_summary(f"{_USER_SUBDIR}/{fname[:-5]}", full, is_user=True))
    # Shipped examples (read-only) from <repo>/configs/examples/presets/.
    out.extend(_list_example_presets())
    return jsonify(out)


@app.route("/api/task-presets", methods=["POST"])
def save_task_preset():
    """Save a user-owned preset by applying overrides on top of a source preset.

    Body: { sourceName, newName, overrides?, displayName?, description? }.
    Lands the file at $MAMMA_INTERFACE_DIR/samples/presets/user/<newName>.json
    and returns the new preset's listing entry. Refuses to overwrite either
    an existing user preset or a curated one with the same name.
    """
    data = request.json or {}
    source_name = data.get("sourceName") or ""
    new_name = (data.get("newName") or "").strip()
    overrides = data.get("overrides") or {}
    display_name = (data.get("displayName") or "").strip()
    description = (data.get("description") or "").strip()

    if not _USER_PRESET_NAME_RE.match(new_name):
        return jsonify({"error": "Invalid name. Use letters, digits, underscore, or dash (1–64 chars)."}), 400

    loaded = _read_preset(source_name)
    if loaded is None:
        return jsonify({"error": f"Source preset '{source_name}' not found"}), 404
    _, cfg = loaded

    if overrides:
        _deep_merge(cfg, overrides)

    cfg.setdefault("global", {})
    if display_name:
        cfg["global"]["display_name"] = display_name
    if description:
        cfg["global"]["description"] = description

    user_dir = os.path.join(_PRESETS_DIR, _USER_SUBDIR)
    try:
        os.makedirs(user_dir, exist_ok=True)
    except OSError as e:
        return jsonify({"error": f"Failed to create user presets dir: {e}"}), 500

    new_path = os.path.join(user_dir, new_name + ".json")
    if os.path.exists(new_path):
        return jsonify({"error": f"A user preset named '{new_name}' already exists"}), 409
    if os.path.exists(os.path.join(_PRESETS_DIR, new_name + ".json")):
        return jsonify({"error": f"Name '{new_name}' collides with a curated preset"}), 409

    try:
        with open(new_path, "w") as f:
            json.dump(cfg, f, indent=2)
    except OSError as e:
        return jsonify({"error": f"Failed to write preset: {e}"}), 500

    return jsonify(_preset_summary(f"{_USER_SUBDIR}/{new_name}", new_path, is_user=True)), 201


@app.route("/api/task-presets/<path:name>", methods=["DELETE"])
def delete_task_preset(name):
    """Delete a user-saved preset. Curated presets are read-only and cannot
    be deleted via the API — they're shipped with the tool."""
    safe = _safe_preset_name(name)
    if safe is None:
        return jsonify({"error": "Invalid preset name"}), 400
    if not safe.startswith(_USER_SUBDIR + "/"):
        return jsonify({"error": "Only user-saved presets can be deleted"}), 403
    path = _preset_path(safe)
    if not os.path.isfile(path):
        return jsonify({"error": "Preset not found"}), 404
    try:
        os.remove(path)
    except OSError as e:
        return jsonify({"error": f"Failed to delete preset: {e}"}), 500
    return jsonify({"ok": True}), 200


@app.route("/api/task-presets/<path:name>/digest", methods=["GET"])
def get_task_preset_digest(name):
    """Structured summary of what a preset will run, for read-only preview."""
    loaded = _read_preset(name)
    if loaded is None:
        return jsonify({"error": f"Preset '{name}' not found or unreadable"}), 404
    path, cfg = loaded
    g = cfg.get("global") or {}

    # Steps in canonical order, but skip ones not in the preset (lets future
    # branch additions just add entries without breaking older presets).
    steps_out = []
    for step_name in _PRESET_STEP_ORDER:
        s = cfg.get(step_name)
        if not isinstance(s, dict):
            continue
        # Surface "extra" config (config_path / weights / ma_*_dir / etc.)
        # that's neither one of the step-meta keys nor the engine/sif fields.
        extras = {k: v for k, v in s.items() if k not in _PRESET_META_KEYS}
        steps_out.append({
            "name": step_name,
            "enabled": bool(s.get("enabled")),
            "engine": (s.get("engine") or "conda").lower(),
            "condaEnv": s.get("conda_env") or g.get("conda_env") or "mamma",
            "sifPath": s.get("sif_path") or "",
            "dockerImage": s.get("docker_image") or "",
            "script": s.get("script") or "",
            "repoPath": s.get("repo_path") or "",
            "flags": list(s.get("flags") or []),
            "dependencies": list(s.get("dependencies") or []),
            "extras": extras,
        })

    return jsonify({
        "name": name,
        "path": path,
        "displayName": g.get("display_name") or name,
        "description": g.get("description") or "",
        "global": {
            "datasetName": g.get("dataset_name") or "",
            "condaEnv": g.get("conda_env") or "mamma",
            "bind": list(g.get("bind") or []),
            # Run frame window. ma_cap turns these into --start/--end and bakes
            # them into the per-camera NPZ that downstream steps inherit (see
            # inference/steps/base.py). null = unset = process every frame.
            "startFrame": g.get("start_frame"),
            "endFrame": g.get("end_frame"),
            # Signed world up axis; consumed by ma_3d + ma_vis. None = default (z).
            "upAxis": g.get("up_axis"),
        },
        "steps": steps_out,
    })


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------

@app.route("/api/processes/active", methods=["GET"])
def get_active_processes():
    try:
        return jsonify(db.get_all_active_tasks_with_processes())
    except Exception as e:
        print(f"Error fetching active tasks: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/processes/<int:process_id>/stop", methods=["POST"])
def stop_process(process_id):
    process = db.get_process_by_id(process_id)
    if not process:
        return jsonify({"error": f"Process {process_id} not found"}), 404
    stoppable = ["Running", "Queued", "Pending", "PENDING", "Retrying"]
    if process.get("status") not in stoppable:
        return jsonify({"error": f"Process cannot be stopped (status: {process.get('status')})"}), 400
    db.set_process_status(process_id, "Cancelled")
    return jsonify({"message": f"Process {process_id} marked Cancelled"})


@app.route("/api/processes/<int:process_id>", methods=["DELETE"])
def remove_process(process_id):
    return jsonify({"message": f"Process {process_id} removed"})


@app.route("/api/tasks/<int:task_id>", methods=["DELETE"])
def delete_task(task_id):
    """Drop a task row + its processes from the DB. **Does not touch
    files on disk** — outputs under `<output_path>/<step>/<output_id>/`
    and logs under `<jobs_log_dir>/<user>/<task_id>/` are left intact,
    so the user can resubmit against the same `output_id` and pick up
    DONE sentinels exactly as before.

    The intent is "I don't want this run cluttering the Tasks table",
    not "wipe the run." For an actual cleanup the user can `rm -rf` the
    output dir manually.

    Refuses to delete a task whose runner process is still alive so we
    don't leak an unmanaged subprocess. Stop the task first."""
    task = db.get_task_by_id(task_id)
    if not task:
        return jsonify({"error": f"Task {task_id} not found"}), 404

    # A queued task has no runnerPid yet, but the coordinator holds its spawn
    # kwargs in memory and would still launch it after the DB rows are gone —
    # an invisible GPU run with no Tasks row and no Stop button. Drop it from
    # the queue first (no-op if already popped), same as stop_task does.
    _task_queue.cancel_queued(task_id)

    pid = task.get("runnerPid")
    if pid:
        try:
            os.kill(int(pid), 0)  # signal 0 = liveness probe
            return jsonify({
                "error": (
                    f"Task {task_id}'s runner (pid={pid}) is still running. "
                    "Stop it first, then delete."
                ),
            }), 409
        except ProcessLookupError:
            pass  # stale pid, safe to drop
        except Exception as e:
            print(f"Error checking runner pid={pid}: {e}")

    removed = db.delete_task_with_processes(task_id)
    if not removed:
        return jsonify({"error": f"Task {task_id} not found"}), 404
    return jsonify({"message": f"Task {task_id} removed from database", "filesDeleted": False})


@app.route("/api/tasks/<int:task_id>/sequence", methods=["DELETE"])
def delete_task_sequence(task_id):
    """Remove a single sequence from a task — the Tasks table renders one
    row per (task, sequence), so the row-level delete must scope to that
    sequence, not the whole task. Drops only this task's processes for the
    named sequence; if it was the task's last sequence, the empty task row
    goes too. Like the whole-task delete, **files on disk are untouched**.

    Refuses while the task's runner is still alive: one subprocess runs all
    of a task's sequences, so its DB rows are mutating — stop the task first."""
    seq_name = (request.args.get("name") or "").strip()
    if not seq_name:
        return jsonify({"error": "missing sequence name (?name=...)"}), 400
    task = db.get_task_by_id(task_id)
    if not task:
        return jsonify({"error": f"Task {task_id} not found"}), 404

    # A queued task has no runnerPid, but the coordinator will still run ALL
    # its sequences (the task JSON on disk drives enumeration) — deleting one
    # sequence's rows now would leave that sequence running invisibly. Unlike
    # the whole-task delete we can't just cancel_queued (that would cancel the
    # task's *other* sequences too), so refuse until the task is stopped.
    if _task_queue.queue_position(task_id) is not None:
        return jsonify({
            "error": (
                f"Task {task_id} is still queued; its sequences have not run "
                "yet. Stop the task first, then delete."
            ),
        }), 409

    pid = task.get("runnerPid")
    if pid:
        try:
            os.kill(int(pid), 0)  # signal 0 = liveness probe
            return jsonify({
                "error": (
                    f"Task {task_id}'s runner (pid={pid}) is still running. "
                    "Stop it first, then delete."
                ),
            }), 409
        except ProcessLookupError:
            pass  # stale pid, safe to drop
        except Exception as e:
            print(f"Error checking runner pid={pid}: {e}")

    removed, task_removed = db.delete_sequence_from_task(task_id, seq_name)
    if not removed:
        return jsonify({"error": f"Sequence '{seq_name}' not found on task {task_id}"}), 404
    return jsonify({
        "message": f"Removed sequence '{seq_name}' from task {task_id}",
        "taskRemoved": task_removed,
        "filesDeleted": False,
    })


@app.route("/api/tasks/<int:task_id>/stop", methods=["POST"])
def stop_task(task_id):
    task = db.get_task_by_id(task_id)
    if not task:
        return jsonify({"error": f"Task {task_id} not found"}), 404

    # If the task hasn't been spawned yet (still in the in-memory
    # queue), drop it from the queue first. This is a no-op when the
    # coordinator has already popped it.
    dropped_from_queue = _task_queue.cancel_queued(task_id)

    pid = task.get("runnerPid")
    if pid and not dropped_from_queue:
        try:
            os.kill(int(pid), signal.SIGTERM)
            print(f"Sent SIGTERM to runner pid={pid} for task {task_id}")
        except ProcessLookupError:
            print(f"Runner pid={pid} for task {task_id} already exited")
        except Exception as e:
            print(f"Error signaling runner pid={pid}: {e}")

    cancelled = db.cancel_all_task_processes(task_id)
    return jsonify({
        "message": f"Task {task_id} stopped",
        "runnerPid": pid,
        "cancelledProcesses": cancelled,
        "droppedFromQueue": dropped_from_queue,
    })


@app.route("/api/tasks/<int:task_id>/restart", methods=["POST"])
def restart_task(task_id):
    """Re-enqueue a stopped task. The runner's existing DONE-sentinel
    logic handles resume — Completed (step, seq) pairs are skipped via
    the on-disk DONE marker, so restart effectively continues from
    where the previous run failed or was cancelled.

    Refuses when:
      * The task no longer exists in DB (404).
      * Any of its processes is currently Running or Queued (409). Use
        Stop first if you want to forcibly restart.
      * The on-disk run config (run_<id>.json) is missing (409).
    """
    task = db.get_task_by_id(task_id)
    if not task:
        return jsonify({"error": f"Task {task_id} not found"}), 404

    processes = db.get_processes_for_task(task_id) or []
    statuses = {(p.get("status") or "").lower() for p in processes}
    if any("running" in s for s in statuses):
        return jsonify({"error": "Task is currently running. Stop it first if you want to restart."}), 409
    if any(s == "queued" for s in statuses):
        return jsonify({"error": "Task is already queued."}), 409

    # Find the run config the previous attempt used. The coordinator
    # re-uses it verbatim so resume runs against exactly the same
    # preset/capture binding.
    spawn_kwargs = _task_queue._reconstruct_spawn_kwargs(task_id)
    if spawn_kwargs is None:
        return jsonify({"error": "Run config not found on disk; can't restart."}), 409

    affected = db.reset_non_completed_to_queued(task_id)
    if affected == 0:
        # Nothing to re-run — task was already fully Completed.
        return jsonify({
            "message": f"Task {task_id} is already fully completed; nothing to restart.",
            "resetCount": 0,
        })

    _task_queue.enqueue(task_id=task_id, **spawn_kwargs)
    return jsonify({
        "message": f"Task {task_id} re-queued for restart",
        "resetCount": affected,
    })


@app.route("/api/steps/<step_name>/flags", methods=["GET"])
def get_step_flags(step_name):
    """Cached argparse `--help` catalogue for one pipeline step. The
    preset edit form uses this to show "available flags" alongside the
    user's current flag list. ``?refresh=true`` skips the disk cache.

    See gui/backend/help_cache.py for the cache + parser. First request
    per step is slow (~1-5s for the script's heavy imports); subsequent
    requests are instant. The cache invalidates by script mtime.
    """
    import help_cache  # local import to keep module-top fast
    force = (request.args.get("refresh") or "").lower() in ("1", "true", "yes")
    try:
        data = help_cache.get_flags(step_name, force_refresh=force)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(data)


@app.route("/api/steps/settings", methods=["GET"])
def get_steps_settings():
    """Curated per-step "common settings" schema for the preset editor's
    friendly widgets (toggle/select/slider/number).

    Static manifest + config-recipe choices resolved from disk. The full
    argparse surface stays available via ``/api/steps/<step>/flags``. See
    gui/backend/step_settings.py.
    """
    import step_settings  # local import to keep module-top fast
    return jsonify(step_settings.build_settings())


@app.route("/api/steps/<step_name>/recipe", methods=["GET"])
def get_step_recipe(step_name):
    """Raw contents of a config-recipe YAML shown in a step's dropdown
    (ma_2d ``config_path`` / ma_3d ``config_file``), for the read-only "peek"
    viewer. Path-safe: only files that are actual dropdown options are served.
    """
    import step_settings  # local import to keep module-top fast
    rel = request.args.get("path") or ""
    try:
        path, text = step_settings.read_recipe(step_name, rel)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (FileNotFoundError, OSError) as e:
        return jsonify({"error": str(e)}), 404
    return jsonify({"path": path, "text": text})


@app.route("/api/settings/concurrency", methods=["GET"])
def get_concurrency_setting():
    n = db.get_concurrency_limit()
    return jsonify({
        "limit": n,
        "mode": "sequential" if n <= 1 else "parallel",
    })


@app.route("/api/settings/concurrency", methods=["PUT"])
def set_concurrency_setting():
    data = request.json or {}
    try:
        n = int(data.get("limit", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "limit must be an integer"}), 400
    if n < 1 or n > 8:
        return jsonify({"error": "limit must be in [1, 8]"}), 400
    db.set_concurrency_limit(n)
    # Wake the coordinator so a bump (1->2) takes effect immediately
    # for already-queued tasks. A drop (2->1) takes effect on the next
    # natural transition (next runner exit).
    _task_queue.notify_limit_changed()
    print(f"---> concurrency_limit set to {n}")
    return jsonify({
        "limit": n,
        "mode": "sequential" if n <= 1 else "parallel",
    })


# ---------------------------------------------------------------------------
# Example-data download. Spawns data/download_example.sh as a background
# subprocess and exposes a status endpoint the frontend polls. Lets users
# fetch the no-login demo sequence (mamma_example) with one click.
# ---------------------------------------------------------------------------

_EXAMPLE_SCRIPT = _REPO_ROOT / "data" / "download_example.sh"
_EXAMPLE_VIDEOS_DIR = (
    _REPO_ROOT / "data" / "mamma_example" / "pushing_and_lifting_from_ground" / "videos"
)
_EXAMPLE_CAMS = ("A001", "B001", "C001", "D001")
_EXAMPLE_LOG = _DEFAULT_DATA_DIR / "logs" / "example_download.log"

_example_download_lock = threading.Lock()
_example_download = {
    "process": None,       # subprocess.Popen | None
    "started_at": None,    # ISO timestamp of the most recent run
    "last_error": None,    # last failure message, cleared on next start
}


def _example_data_present() -> bool:
    for cam in _EXAMPLE_CAMS:
        f = _EXAMPLE_VIDEOS_DIR / f"{cam}.mp4"
        try:
            if not f.is_file() or f.stat().st_size == 0:
                return False
        except OSError:
            return False
    return True


def _example_log_tail(max_lines: int = 8) -> list[str]:
    if not _EXAMPLE_LOG.is_file():
        return []
    try:
        with open(_EXAMPLE_LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            raw = f.read().decode("utf-8", errors="replace")
        return [ln for ln in raw.splitlines() if ln.strip()][-max_lines:]
    except OSError:
        return []


def _example_state_snapshot() -> dict:
    proc = _example_download["process"]
    running = proc is not None and proc.poll() is None
    # Reap finished process and capture error if data still missing
    if proc is not None and not running:
        rc = proc.returncode
        _example_download["process"] = None
        if rc != 0 and not _example_data_present():
            _example_download["last_error"] = (
                f"download script exited with code {rc} — see {_EXAMPLE_LOG}"
            )
    present = _example_data_present()
    if running:
        state = "downloading"
    elif present:
        state = "ready"
        _example_download["last_error"] = None
    elif _example_download["last_error"]:
        state = "error"
    else:
        state = "missing"
    return {
        "state": state,
        "tail": _example_log_tail(),
        "error": _example_download["last_error"],
        "started_at": _example_download["started_at"],
    }


@app.route("/api/example/status", methods=["GET"])
def get_example_status():
    with _example_download_lock:
        return jsonify(_example_state_snapshot())


@app.route("/api/example/download", methods=["POST"])
def trigger_example_download():
    with _example_download_lock:
        snap = _example_state_snapshot()
        if snap["state"] == "downloading":
            return jsonify(snap), 409
        if snap["state"] == "ready":
            return jsonify(snap)
        if not _EXAMPLE_SCRIPT.is_file():
            return jsonify({"error": f"missing script: {_EXAMPLE_SCRIPT}"}), 500
        _EXAMPLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        # Truncate the log so the tail reflects only this run
        log_fp = open(_EXAMPLE_LOG, "wb")
        try:
            proc = subprocess.Popen(
                ["bash", str(_EXAMPLE_SCRIPT)],
                cwd=str(_REPO_ROOT),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as e:
            log_fp.close()
            _example_download["last_error"] = f"could not spawn script: {e}"
            return jsonify(_example_state_snapshot()), 500
        # Popen owns the fd now; closing here is fine because Popen dup'd it
        log_fp.close()
        _example_download["process"] = proc
        _example_download["started_at"] = datetime.utcnow().isoformat() + "Z"
        _example_download["last_error"] = None
        return jsonify(_example_state_snapshot())


# ---------------------------------------------------------------------------
# File serving (used by the frontend's log viewers, image previews, etc.)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sync — DB ↔ filesystem reconciliation
# ---------------------------------------------------------------------------

import sync as _pipeline_sync  # noqa: E402  (kept module-local for clarity)
from objects.processes import ProcessType as _ProcessType  # noqa: E402


def _audit_output_root() -> str:
    """Where the runner lands outputs by default. The audit walks this
    tree to find filesystem-only runs."""
    return DEFAULT_OUTPUT_DIR


# ---------------------------------------------------------------------------
# Rerun viewer (.rrd files)
# ---------------------------------------------------------------------------
#
# Why subprocess + native viewer instead of embedding Rerun's web viewer:
# the WebViewer loads the whole .rrd into browser memory, which chokes on
# the 1GB+ files our pipeline routinely produces. The native Rust/WGPU
# viewer handles GB-scale recordings out of the box. Since the webtool
# binds to loopback by default (single-user-local), spawning a GUI
# process on the same machine the user is looking at works naturally.

@app.route("/api/rrd/file.rrd", methods=["GET"])
def serve_rrd_file():
    """Stream a `.rrd` file inline (no Content-Disposition: attachment) so
    the embedded `@rerun-io/web-viewer` can fetch it. `conditional=True`
    enables HTTP Range responses, which the viewer uses to progressively
    pull data — critical for files that exceed a few hundred MB.

    The route path itself ends in `.rrd` because Rerun's WASM viewer
    inspects the URL path (sans query string) to determine the file type
    — without the extension it loads but shows the welcome screen
    instead of the recording. The actual file location is taken from the
    `path=` query string so we don't have to URL-encode an absolute path
    into the route."""
    file_path = (request.args.get("path") or "").strip()
    if not file_path:
        return jsonify({"error": "path is required"}), 400
    if not file_path.lower().endswith(".rrd"):
        return jsonify({"error": "Only .rrd files are supported"}), 400
    if not os.path.isfile(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 404
    return send_file(file_path, mimetype="application/octet-stream", conditional=True)


def _find_rerun_binary() -> str | None:
    """Locate the `rerun` CLI. Resolution order:

      1. `$MAMMA_RERUN_BIN` if set — explicit override always wins.
      2. `rerun` on `$PATH`.
      3. `rerun` inside any conda/mamba env under common install roots
         (`~/miniforge3`, `~/miniconda3`, `~/anaconda3`, `~/mambaforge`).
         This is what makes the native viewer "just work" when the
         backend runs in `mamma` (no rerun-sdk by default) but the user
         has it installed in another env like `mv-rerun`.

    Returns the absolute path or `None` if no candidate is found.
    """
    import shutil
    from glob import glob

    override = os.environ.get("MAMMA_RERUN_BIN")
    if override:
        return override if os.path.isfile(override) else None

    on_path = shutil.which("rerun")
    if on_path:
        return on_path

    home = os.path.expanduser("~")
    for root in ("miniforge3", "mambaforge", "miniconda3", "anaconda3"):
        for candidate in glob(os.path.join(home, root, "envs", "*", "bin", "rerun")):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def _rerun_data_dir() -> str:
    """Native Rerun viewer's per-user data dir (holds the saved `blueprints/`)."""
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/rerun")
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming")
        return os.path.join(base, "rerun", "data")
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "rerun")


def _reset_rrd_blueprint(rrd_path: str) -> int:
    """Delete the native viewer's saved blueprint(s) for this `.rrd`'s app_id so it
    reopens with a fresh layout instead of a cached one. The viewer keys blueprints
    by application_id; our viz uses ``MAMMA · <seq>`` (post-fix) or the legacy
    ``MAMMA visualization``. We confirm which id the recording actually carries, then
    delete the `.rbl` whose contents reference it. Returns the number removed."""
    try:
        with open(rrd_path, "rb") as fh:
            head = fh.read(1 << 16)  # app_id lives in the StoreInfo near the start
    except OSError:
        return 0
    seq = os.path.basename(os.path.dirname(rrd_path))
    app_id = next((c for c in (f"MAMMA · {seq}", "MAMMA visualization")
                   if c.encode("utf-8") in head), None)
    if not app_id:
        return 0
    bp_dir = os.path.join(_rerun_data_dir(), "blueprints")
    if not os.path.isdir(bp_dir):
        return 0
    needle = app_id.encode("utf-8")
    removed = 0
    for name in os.listdir(bp_dir):
        if not name.endswith(".rbl"):
            continue
        fp = os.path.join(bp_dir, name)
        try:
            with open(fp, "rb") as fh:
                if needle in fh.read():
                    os.remove(fp)
                    removed += 1
        except OSError:
            pass
    return removed


@app.route("/api/rrd/open", methods=["POST"])
def open_rrd():
    """Launch the native Rerun viewer with the given .rrd file. Detaches
    immediately so the request returns even though the viewer keeps running.

    With ``reset_layout`` true, first delete the viewer's saved blueprint for this
    recording's app_id so it opens with a fresh layout (the "fresh layout" action).
    """
    data = request.json or {}
    path = (data.get("path") or "").strip()
    if not path:
        return jsonify({"error": "path is required"}), 400
    if not path.lower().endswith(".rrd"):
        return jsonify({"error": "Only .rrd files are supported"}), 400
    if not os.path.isfile(path):
        return jsonify({"error": f"File not found: {path}"}), 404

    layout_reset = _reset_rrd_blueprint(path) if data.get("reset_layout") else 0

    rerun_bin = _find_rerun_binary()
    if not rerun_bin:
        return jsonify({
            "error": (
                "`rerun` not found. Install with `pip install rerun-sdk` "
                "in a conda env, or set MAMMA_RERUN_BIN to the absolute "
                "path of the rerun binary (e.g. "
                "$CONDA_PREFIX/envs/mamma/bin/rerun)."
            ),
        }), 500
    try:
        proc = subprocess.Popen(
            [rerun_bin, path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach from this Flask process
        )
    except OSError as e:
        return jsonify({"error": f"Failed to launch {rerun_bin}: {e}"}), 500
    _reap_detached(proc)

    return jsonify({"ok": True, "pid": proc.pid, "path": path,
                    "binary": rerun_bin, "layout_reset": layout_reset})


# Players we try in priority order when the frontend asks us to open a
# file that the browser can't decode (e.g. HEVC clips from iPhones).
# `xdg-open` / `open` defer to the user's default app — best UX since
# the user has likely already set one. The direct-binary fallbacks
# cover headless dev machines without a desktop session.
_NATIVE_VIDEO_PLAYERS = (
    "xdg-open",  # Linux dispatcher → user's default app
    "open",      # macOS dispatcher
    "mpv",
    "vlc",
    "mplayer",
    "ffplay",
)


def _find_native_video_player() -> str | None:
    override = os.environ.get("MAMMA_VIDEO_PLAYER")
    if override and shutil.which(override):
        return override
    for name in _NATIVE_VIDEO_PLAYERS:
        path = shutil.which(name)
        if path:
            return path
    return None


@app.route("/api/files/open-native", methods=["POST"])
def open_file_native():
    """Spawn a native viewer for a file the browser can't open.

    Single-user local-trust model — same as `/api/rrd/open`. The viewer
    runs on the same host as the Flask backend (which, in this app, is
    the user's workstation). Detaches via `start_new_session` so the
    spawned process outlives the HTTP request."""
    data = request.json or {}
    path = (data.get("path") or "").strip()
    if not path:
        return jsonify({"error": "path is required"}), 400
    if not os.path.isfile(path):
        return jsonify({"error": f"File not found: {path}"}), 404
    player = _find_native_video_player()
    if not player:
        return jsonify({
            "error": (
                "No native viewer found. Install one of: "
                f"{', '.join(_NATIVE_VIDEO_PLAYERS)}; or set "
                "MAMMA_VIDEO_PLAYER to the binary you prefer."
            ),
        }), 500
    try:
        proc = subprocess.Popen(
            [player, path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        return jsonify({"error": f"Failed to launch {player}: {e}"}), 500
    _reap_detached(proc)
    return jsonify({"ok": True, "pid": proc.pid, "path": path, "binary": player})


@app.route("/api/sync/audit", methods=["GET"])
def sync_audit():
    """Compare filesystem outputs with the DB. Returns the discrepancy
    report the Database tab renders.

    The output_id is the join key: same string as `tasks.output_id` and
    as the directory under `<output_root>/<step>/`. Anything that lives
    on disk under that name but is missing from the tasks table is a
    "filesystem-only" run; anything in the tasks table whose output
    dirs no longer exist is "orphan in DB"."""
    output_root = _audit_output_root()
    all_steps = list(_ProcessType.__members__.keys())

    fs_runs = _pipeline_sync.scan_output_dirs(output_root, all_steps)
    db_tasks = db.get_all_task_minimal()
    db_by_output_id = {t["outputId"]: t for t in db_tasks if t.get("outputId")}

    fs_only = []
    for output_id, info in fs_runs.items():
        if output_id in db_by_output_id:
            continue
        fs_only.append({
            "outputId": info["outputId"],
            "outputDir": info["outputDir"],
            "dataset": info["dataset"] or "",
            "steps": info["steps"],
            "sequences": info["sequences"],
            "sizeBytes": info["sizeBytes"],
            "sizeHuman": _pipeline_sync.humanize_size(info["sizeBytes"]),
            # If the GUI was used, the saved per-task config sits here:
            "guessedTaskJsonPath": os.path.join(MOUNT_POINT, "run_configs", f"run_{output_id}.json"),
        })

    db_only = []
    for t in db_tasks:
        steps = [s for s in (t.get("steps") or []) if s]
        if not steps:
            continue
        # The task could store its own out_dir in the task_json; if we can
        # read that, prefer it. Fall back to the global default.
        task_root = output_root
        try:
            if t.get("taskJsonPath") and os.path.isfile(t["taskJsonPath"]):
                task_root = _pipeline_sync.resolve_output_root(
                    load_config_file(t["taskJsonPath"]), output_root)
        except (OSError, ValueError):
            pass
        if not _pipeline_sync.task_outputs_exist(task_root, t.get("outputId") or "", steps):
            db_only.append({
                "taskId": t["taskId"],
                "captureName": t["captureName"],
                "outputId": t.get("outputId"),
                "outputPath": t.get("outputPath"),
                "createdAt": t.get("createdAt"),
                "expectedDir": os.path.join(task_root, steps[0], t.get("outputId") or ""),
                "steps": steps,
            })

    return jsonify({
        "outputRoot": output_root,
        "summary": {
            "dbTasks": len(db_tasks),
            "fsRuns": len(fs_runs),
            "fsOnly": len(fs_only),
            "dbOnly": len(db_only),
        },
        "filesystemOnly": sorted(fs_only, key=lambda r: r["outputId"]),
        "databaseOnly": db_only,
    })


@app.route("/api/sync/import-task", methods=["POST"])
def sync_import_task():
    """Register a CLI-run task into the DB.

    Two body shapes are accepted, picked by which fields are present:

    Body A — task.json-first (legacy; emitted by older callers):
      { "taskJsonPath": "/path/to/task_or_run_config.json",
        "outputId":     "<tag>",            # optional, falls back to task.json
        "statusPolicy": "infer"|"completed" }
      Reads capture_json, dataset_name, out_dir, and enabled steps OUT
      of the task.json. The original code path; no behaviour change.

    Body B — capture-first (primary path used by the new
    "Import previous run" modal in Tasks):
      { "captureJsonPath": "/path/to/capture.json",   # required
        "outputDir":       "/path/to/output",         # required (parent of <step>/)
        "outputId":        "<tag>",                   # required
        "datasetName":     "<dataset>",               # required
        "steps":           ["ma_cap", ...],           # optional; defaults to disk scan
        "username":        "alice",                   # optional, falls back to LOCAL_USER
        "presetPath":      null,                      # optional, audit-only
        "taskJsonPath":    null,                      # optional, audit-only string
        "statusPolicy":    "infer"|"completed" }
      Synthesizes a minimal task_content dict so the shared persistence
      helper accepts it. No task.json file needs to exist on disk.
    """
    data = request.json or {}
    task_json_path_in = (data.get("taskJsonPath") or "").strip()
    capture_json_path_in = (data.get("captureJsonPath") or "").strip()

    # Branch routing: body-A wins when an actual task.json file path is
    # supplied. The capture-first form (body B) is keyed by the
    # captureJsonPath being non-empty AND no task.json on disk. We treat
    # taskJsonPath as audit-only metadata in body B (string pointer that
    # doesn't have to exist).
    use_body_a = bool(task_json_path_in) and os.path.isfile(task_json_path_in)

    # === Pre-flight: gather (capture_json_path, capture_content, ===
    # === task_content, output_root, output_id, dataset, steps).  ===

    if use_body_a:
        try:
            task_content = load_config_file(task_json_path_in)
        except (OSError, ValueError) as e:
            return jsonify({"error": f"Failed to parse task.json: {e}"}), 400

        g = (task_content.get("global") or {}) if isinstance(task_content, dict) else {}
        capture_json_path = g.get("capture_json")
        if not capture_json_path:
            return jsonify({"error": "task.json is missing global.capture_json"}), 400
        if not os.path.isfile(capture_json_path):
            return jsonify({"error": f"capture_json referenced by the task does not exist: {capture_json_path}"}), 404
        try:
            capture_content = load_config_file(capture_json_path)
        except (OSError, ValueError) as e:
            return jsonify({"error": f"Failed to parse capture.json: {e}"}), 400

        output_root = _pipeline_sync.resolve_output_root(task_content, _audit_output_root())
        output_id = (data.get("outputId") or "").strip() or g.get("output_id") or g.get("out_tag")
        if not output_id:
            return jsonify({"error": "Could not determine output_id; pass it in the body."}), 400

        requested_steps = []
        for step_name in _ProcessType.__members__.keys():
            cfg = task_content.get(step_name) if isinstance(task_content, dict) else None
            if isinstance(cfg, dict) and cfg.get("enabled"):
                requested_steps.append(step_name)
        if not requested_steps:
            return jsonify({"error": "No enabled steps found in task.json"}), 400

        dataset_from_task = g.get("dataset_name")
        task_json_path_stored = task_json_path_in
        username = LOCAL_USER
        preset_path = None
    else:
        # body-B
        if not capture_json_path_in:
            return jsonify({"error": "captureJsonPath (or a valid taskJsonPath) is required"}), 400
        if not os.path.isfile(capture_json_path_in):
            return jsonify({"error": f"capture.json not found: {capture_json_path_in}"}), 404
        try:
            capture_content = load_config_file(capture_json_path_in)
        except (OSError, ValueError) as e:
            return jsonify({"error": f"Failed to parse capture.json: {e}"}), 400

        output_root = (data.get("outputDir") or "").strip()
        if not output_root:
            return jsonify({"error": "outputDir is required for capture-first imports"}), 400
        output_id = (data.get("outputId") or "").strip()
        if not output_id:
            return jsonify({"error": "outputId is required for capture-first imports"}), 400
        dataset_from_task = (data.get("datasetName") or "").strip() or None

        # Step list: prefer body, else disk-scan, else fail.
        raw_steps = data.get("steps")
        if isinstance(raw_steps, list) and raw_steps:
            requested_steps = [s for s in raw_steps if s in _ProcessType.__members__]
        else:
            requested_steps = []
        if not requested_steps:
            # disk-scan fallback: ask sync.scan_output_dirs which steps
            # actually exist for this output_id and use that list.
            all_steps = list(_ProcessType.__members__.keys())
            fs_runs = _pipeline_sync.scan_output_dirs(output_root, all_steps)
            entry = fs_runs.get(output_id) or {}
            requested_steps = [s for s in (entry.get("steps") or []) if s in _ProcessType.__members__]
        if not requested_steps:
            return jsonify({"error": "Could not determine any valid steps for this output; pass `steps` explicitly."}), 400

        # Username / preset / task-json: optional audit metadata.
        username = (data.get("username") or "").strip() or LOCAL_USER
        preset_path = (data.get("presetPath") or None) or None
        task_json_path_stored = (data.get("taskJsonPath") or "").strip() or _imported_task_json_placeholder(output_id)
        capture_json_path = capture_json_path_in

        # Synthesize a minimal task_content. The downstream persistence
        # helper only reads `<step>.sif_file` and `<step>.enabled`; the
        # global block carries the audit fields the rest of the GUI
        # reads back from the DB row.
        task_content = {
            "global": {
                "version": 1.0,
                "username": username,
                "capture_json": capture_json_path,
                "dataset_name": dataset_from_task,
                "output_id": output_id,
                "out_dir": output_root,
            },
            **{s: {"enabled": True} for s in requested_steps},
        }

    # === Common tail: sequence discovery → dup check → import. ===

    fs_runs = _pipeline_sync.scan_output_dirs(output_root, requested_steps)
    fs_entry = fs_runs.get(output_id)
    if fs_entry and fs_entry.get("sequences"):
        seq_names = list(fs_entry["sequences"])
        dataset = fs_entry.get("dataset") or dataset_from_task
    else:
        # Reconstruct from task.json's seq_ids (body A only) or fall
        # back to "all sequences in capture.json" (body B).
        if use_body_a:
            g = (task_content.get("global") or {}) if isinstance(task_content, dict) else {}
            seq_ids = set(int(s) for s in (g.get("seq_ids") or []) if str(s).isdigit())
            seq_names = []
            for sid, sval in (capture_content.get("sequences") or {}).items():
                try:
                    if int(sid) in seq_ids:
                        name = (sval or {}).get("ioi") if isinstance(sval, dict) else None
                        if name:
                            seq_names.append(name)
                except (TypeError, ValueError):
                    pass
        else:
            seq_names = list(get_sequences_from_data(capture_content))
        dataset = dataset_from_task

    if not seq_names:
        return jsonify({"error": "Could not determine sequences for this run — neither outputs on disk nor capture.json resolve."}), 400

    # Refuse to re-import what's already there.
    existing = db.get_task_by_output_id(output_id)
    if existing:
        return jsonify({
            "error": f"A task with output_id '{output_id}' is already in the DB (task #{existing['task_id']}).",
            "code": "already_imported",
            "existingTaskId": existing['task_id'],
        }), 409

    policy = (data.get("statusPolicy") or "infer").strip().lower()
    if policy == "completed":
        # Use "Done" — the same terminal-success tag the live runner writes —
        # so imported runs are labelled consistently with runs started here.
        status_provider = "Done"
    else:
        def status_provider(step, seq, _root=output_root, _oid=output_id, _ds=dataset):
            return _pipeline_sync.infer_step_status(_root, _oid, _ds, step, seq)

    process_enums = [_ProcessType[s] for s in requested_steps]

    status, task_id, _, resolved_output_id = db.import_cli_task(
        capture_json_path=capture_json_path,
        capture_content=capture_content,
        task_content=task_content,
        seq_names=seq_names,
        processes=process_enums,
        output_dir=output_root,
        output_id=output_id,
        username=username,
        task_json_path=task_json_path_stored,
        process_status=status_provider,
        preset_path=preset_path,
    )
    if status != "Entry created successfully":
        return jsonify({"error": status}), 500

    return jsonify({
        "ok": True,
        "taskId": task_id,
        "outputId": resolved_output_id,
        "sequences": sorted(seq_names),
        "steps": requested_steps,
        "statusPolicy": policy,
    }), 201


def _imported_task_json_placeholder(output_id: str) -> str:
    """Audit-only string for body-B imports where the user didn't pass
    a real task.json. The file doesn't have to exist; it's just a
    stable label in the captures-list / Tasks-history view."""
    fname = f"imported_run_{output_id}.json"
    return os.path.join(MOUNT_POINT, "run_configs", fname)


@app.route("/api/sync/discover-runs", methods=["POST"])
def sync_discover_runs():
    """Given a task.json path, walk that task's `global.out_dir` and
    return every output_id directory found there. The Database tab uses
    this when the user wants to import a run whose outputs live OUTSIDE
    the default `MAMMA_OUTPUT_DIR` (so the global audit missed it), or
    when they don't remember which `--out-tag` they used.

    Body: { taskJsonPath }
    Returns: { taskJsonPath, outputRoot, datasetName, runs: [
        { outputId, dataset, steps, sequences, sizeBytes, sizeHuman, alreadyInDb }
    ]}
    """
    data = request.json or {}
    task_json_path = (data.get("taskJsonPath") or "").strip()
    if not task_json_path:
        return jsonify({"error": "taskJsonPath is required"}), 400
    if not os.path.isfile(task_json_path):
        return jsonify({"error": f"task.json not found: {task_json_path}"}), 404

    try:
        task_content = load_config_file(task_json_path)
    except (OSError, ValueError) as e:
        return jsonify({"error": f"Failed to parse task.json: {e}"}), 400
    if not isinstance(task_content, dict):
        return jsonify({"error": "task.json is malformed"}), 400

    g = task_content.get("global") or {}
    output_root = _pipeline_sync.resolve_output_root(task_content, _audit_output_root())

    # Walk every declared step (enabled or not). The user may have run
    # only a subset; we want to discover anything that exists on disk.
    declared_steps = [
        step_name for step_name in _ProcessType.__members__.keys()
        if isinstance(task_content.get(step_name), dict)
    ]
    if not declared_steps:
        return jsonify({"error": "task.json has no recognised step blocks"}), 400

    fs_runs = _pipeline_sync.scan_output_dirs(output_root, declared_steps)

    # Optional dataset filter: scope to runs that wrote into this task's
    # declared dataset. Skipped if the task.json doesn't declare one.
    target_dataset = (g.get("dataset_name") or "").strip() or None

    # Mark runs already registered so the UI can show "Already imported".
    db_tasks = db.get_all_task_minimal()
    db_output_ids = {t.get("outputId") for t in db_tasks if t.get("outputId")}

    runs = []
    for output_id, info in fs_runs.items():
        if target_dataset and info.get("dataset") and info["dataset"] != target_dataset:
            continue
        runs.append({
            "outputId": info["outputId"],
            "outputDir": info["outputDir"],
            "dataset": info["dataset"] or "",
            "steps": info["steps"],
            "sequences": info["sequences"],
            "sizeBytes": info["sizeBytes"],
            "sizeHuman": _pipeline_sync.humanize_size(info["sizeBytes"]),
            "alreadyInDb": output_id in db_output_ids,
        })

    return jsonify({
        "taskJsonPath": task_json_path,
        "outputRoot": output_root,
        "datasetName": target_dataset,
        "runs": sorted(runs, key=lambda r: r["outputId"]),
    })


@app.route("/api/sync/task/<int:task_id>", methods=["DELETE"])
def sync_delete_task(task_id):
    """Remove a tasks row + cascade its processes. Used by the Database
    tab's orphan-cleanup action."""
    deleted = db.delete_task_with_processes(task_id)
    if not deleted:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({"ok": True, "taskId": task_id})


@app.route("/api/tasks/<int:task_id>/config-path", methods=["GET"])
def get_task_config_path(task_id):
    """Resolve the absolute path of a task's saved config so the file viewer
    can open it. The config itself is fetched via /api/files/content.

    Returns both the frozen run config path and (when available) the
    source preset path the GUI used to build it. ``presetPath`` is null
    for legacy rows (submitted before the preset_path column existed)
    and for CLI-imported tasks.
    """
    path = db.get_task_json_path(task_id)
    if not path:
        return jsonify({"error": "Task not found"}), 404
    if not os.path.isfile(path):
        return jsonify({"error": f"Run config file missing on disk: {path}"}), 404
    preset = db.get_preset_path(task_id)
    preset_exists = bool(preset and os.path.isfile(preset))
    return jsonify({
        "path": path,
        "presetPath": preset if preset_exists else None,
    })


def _explain_missing_log(file_path):
    """When a `.out/.err` path doesn't exist on disk, try to infer
    *why* and produce a friendly synthetic body the frontend can show.

    Common cause: the runner short-circuited that (step, seq) because a
    DONE sentinel from a previous submission was already present (see
    `inference/runner.py` skip path). The DB row was created with the
    output-path-that-would-have-been-written, but the engine never ran,
    so no output file exists. From the user's POV that looks like a
    broken link — actually it's correct behavior of partial-resume.

    Returns (content_str, note_kind) when we can explain, else (None, None).
    """
    proc = db.find_process_by_log_path(file_path)
    if not proc:
        return None, None

    step = proc.get("step") or ""
    seq = proc.get("seq_name") or ""
    output_id = proc.get("output_id") or ""
    output_path = proc.get("output_path") or ""
    task_id = proc.get("task_id")
    status = proc.get("status") or ""

    dataset = ""
    tjp = proc.get("task_json_path") or ""
    if tjp and os.path.isfile(tjp):
        try:
            tcfg = load_config_file(tjp)
            dataset = (tcfg.get("global", {}).get("dataset_name")
                       or tcfg.get("dataset_name") or "")
        except Exception:
            pass

    done_path = ""
    done_exists = False
    if output_path and step and output_id and dataset and seq:
        done_path = os.path.join(output_path, step, output_id, dataset, seq, "DONE")
        done_exists = os.path.isfile(done_path)

    if done_exists:
        body = (
            f"# No output file for {step} on {seq} (task #{task_id}).\n"
            f"#\n"
            f"# This step was skipped at runtime because a DONE sentinel\n"
            f"# already existed at:\n"
            f"#   {done_path}\n"
            f"#\n"
            f"# That means a previous submission against output_id\n"
            f"# `{output_id}` had already produced this (step, sequence) pair.\n"
            f"# The runner reused those outputs instead of re-running, so no\n"
            f"# new stdout/stderr was written this time.\n"
            f"#\n"
            f"# To force a re-run (and get fresh output), delete the DONE file:\n"
            f"#   rm {done_path}\n"
            f"# then resubmit. The runner will execute the step and write\n"
            f"# to:\n"
            f"#   {file_path}\n"
        )
        return body, "skipped-via-done"

    # No DONE sentinel and no output file: the run probably never executed
    # (e.g. cancelled before reaching this step, or upstream dependency
    # failed).
    body = (
        f"# No output file at:\n"
        f"#   {file_path}\n"
        f"#\n"
        f"# The runner never wrote one for {step} on {seq}\n"
        f"# (task #{task_id}, status: {status}). This usually means the\n"
        f"# step never started — check upstream steps for failures, or\n"
        f"# the task may have been cancelled before reaching this step.\n"
    )
    return body, "never-ran"


@app.route("/api/files/content", methods=["GET"])
def get_file_content():
    file_path = request.args.get("path")
    if not file_path:
        return jsonify({"error": "Path parameter is required"}), 400
    try:
        with open(file_path, "r", errors="replace") as f:
            return jsonify({"content": f.read()})
    except FileNotFoundError:
        # Before giving up, try to explain *why* the file is missing.
        # Logs vanish in expected ways (skipped via DONE, step never ran)
        # and a synthetic-content 200 is much friendlier than a raw 404.
        explained, kind = _explain_missing_log(file_path)
        if explained is not None:
            return jsonify({"content": explained, "note": kind, "synthetic": True})
        return jsonify({"error": f"File not found: {file_path}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/files/download", methods=["GET"])
def download_file():
    file_path = request.args.get("path")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)


@app.route("/api/visualizations", methods=["GET"])
def get_visualizations():
    capture = request.args.get("capture")
    sequence = request.args.get("sequence")
    camera = request.args.get("camera")
    subject = request.args.get("subject")
    vis_type = request.args.get("type")
    return jsonify({
        "path": f"/data/visualizations/{capture}/{sequence}/{vis_type}/{camera}/subject_{subject}.mp4",
        "url": "http://localhost:8000/static/placeholder.mp4",
        "fileType": "video",
    })


def _safe_full_path(rel_path: str) -> str | None:
    """Resolve a path arg from the file APIs.

    - Absolute paths (e.g. /path/to/output/...) pass through
      unchanged. The runner writes outputs wherever task.json's `out_dir`
      points, which is typically outside MAMMA_INTERFACE_DIR.
    - Relative paths are joined with MAMMA_INTERFACE_DIR and required to
      stay under it (path-traversal guard for the legacy in-mount layout).

    Single-user local tool by design — absolute paths trust the OS-level
    permissions of the user running the backend.
    """
    if not rel_path:
        return None
    rel_path = rel_path.replace("\\", "/")
    if os.path.isabs(rel_path):
        return os.path.normpath(rel_path)
    full_path = os.path.normpath(os.path.join(MOUNT_POINT, rel_path.lstrip("/")))
    if not full_path.startswith(os.path.abspath(MOUNT_POINT)):
        return None
    return full_path


@app.route("/api/files/list", methods=["GET"])
def list_files():
    rel_path = request.args.get("path", "")
    if not rel_path:
        return jsonify({"error": "Path parameter is required"}), 400
    full_path = _safe_full_path(rel_path)
    if full_path is None:
        return jsonify({"error": "Invalid path"}), 400
    if not os.path.isdir(full_path):
        return jsonify({"error": "Path does not exist"}), 404
    try:
        dirs, files = [], []
        with os.scandir(full_path) as it:
            for entry in sorted(it, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower())):
                if entry.is_dir(follow_symlinks=False):
                    dirs.append({"name": entry.name})
                else:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    files.append({"name": entry.name, "size": size})
        # Echo the resolved absolute path so the UI can offer a "copy path"
        # affordance (users paste it into their OS file explorer).
        return jsonify({"dirs": dirs, "files": files, "absPath": full_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/files/stream", methods=["GET"])
def stream_file():
    rel_path = request.args.get("path", "")
    if not rel_path:
        return jsonify({"error": "Path parameter is required"}), 400
    full_path = _safe_full_path(rel_path)
    if full_path is None:
        return jsonify({"error": "Invalid path"}), 400
    if not os.path.isfile(full_path) or not full_path.lower().endswith(".mp4"):
        return jsonify({"error": "File not found or not an MP4"}), 404
    return send_file(full_path, mimetype="video/mp4", conditional=True)


@app.route("/api/files/html", methods=["GET"])
def serve_html_file():
    """Serve an HTML output inline (no Content-Disposition: attachment)
    so the frontend can embed it in a sandboxed `<iframe>`. The iframe
    on the frontend uses `sandbox="allow-scripts"` (no `allow-same-origin`),
    which puts the loaded HTML into an opaque origin — its JS can run
    for interactivity (Plotly/Bokeh widgets) but cannot reach any
    cookies or hit our API as the user.

    Best for self-contained HTML (Plotly/Bokeh/pandas-profiling style,
    where assets are inlined). HTML that references sibling CSS/JS/images
    via relative paths will have those resources fail to load — that's a
    known limitation; we'll add a path-based serving route if it becomes
    necessary."""
    rel_path = request.args.get("path", "")
    if not rel_path:
        return jsonify({"error": "Path parameter is required"}), 400
    full_path = _safe_full_path(rel_path)
    if full_path is None:
        return jsonify({"error": "Invalid path"}), 400
    if not os.path.isfile(full_path):
        return jsonify({"error": "File not found"}), 404
    lower = full_path.lower()
    if not (lower.endswith(".html") or lower.endswith(".htm")):
        return jsonify({"error": "Only .html / .htm files are supported"}), 400
    return send_file(full_path, mimetype="text/html", conditional=True)


# `.npy` dtype string → bytes per element. The dtype strings come from
# numpy's array-protocol typestrings (the `descr` field in a `.npy`
# header); we only enumerate the fixed-width primitives our pipeline
# routinely produces. Object/string dtypes (`|O`, `<U…`, `|S…`) are
# variable- or unknown-width here and we report `None` for itemsize.
_NPY_DTYPE_ITEMSIZE = {
    "|b1": 1, "|i1": 1, "|u1": 1,
    "<i2": 2, "<u2": 2, "<f2": 2, ">i2": 2, ">u2": 2, ">f2": 2,
    "<i4": 4, "<u4": 4, "<f4": 4, ">i4": 4, ">u4": 4, ">f4": 4,
    "<i8": 8, "<u8": 8, "<f8": 8, ">i8": 8, ">u8": 8, ">f8": 8,
    "<c8": 8, ">c8": 8,
    "<c16": 16, ">c16": 16,
}

_NPY_DTYPE_FRIENDLY = {
    "|b1": "bool", "|i1": "int8", "|u1": "uint8",
    "<i2": "int16", "<u2": "uint16", "<f2": "float16",
    "<i4": "int32", "<u4": "uint32", "<f4": "float32",
    "<i8": "int64", "<u8": "uint64", "<f8": "float64",
    "<c8": "complex64", "<c16": "complex128",
}


def _read_npy_header(stream):
    """Parse a `.npy` file's header from an open binary stream and return
    the descr/shape dict. Pure stdlib; spec is documented at
    numpy.lib.format. Reads ~10 + header_len bytes — never the array
    body — so this stays cheap on multi-GB files.

    Raises ValueError for any malformed magic / version / header dict.
    """
    import ast
    import struct

    magic = stream.read(10)
    if len(magic) < 10 or magic[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file (bad magic)")
    major, minor = magic[6], magic[7]
    if major == 1:
        # 2-byte little-endian header length
        header_len = struct.unpack("<H", magic[8:10])[0]
    elif major in (2, 3):
        # 4-byte little-endian header length; the 2 bytes we already read
        # at positions 8-9 are the low half — re-read 2 more.
        extra = stream.read(2)
        if len(extra) < 2:
            raise ValueError("truncated v2/v3 header")
        header_len = struct.unpack("<I", magic[8:10] + extra)[0]
    else:
        raise ValueError(f"unsupported .npy version {major}.{minor}")

    raw = stream.read(header_len)
    if len(raw) < header_len:
        raise ValueError("truncated header dict")
    encoding = "utf-8" if major == 3 else "latin-1"
    header_text = raw.decode(encoding).strip()
    # The header dict is a Python literal — `ast.literal_eval` is safe.
    try:
        info = ast.literal_eval(header_text)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"unparseable header: {e}")
    if not isinstance(info, dict) or "descr" not in info or "shape" not in info:
        raise ValueError("header missing descr/shape")
    return info


@app.route("/api/files/npz-meta", methods=["GET"])
def serve_npz_meta():
    """Return a lightweight summary of a `.npz` archive: every contained
    `.npy` array's name, shape, and dtype. Reads only the per-member
    headers (first ~256 bytes each) — never the array bodies — so this
    is instant even for multi-GB files. Powers the click-to-inspect
    affordance in the Outputs explorer.

    Pure stdlib (`zipfile` + `ast.literal_eval`) so this endpoint
    works in any env — no numpy dep just for the metadata view.
    """
    import zipfile

    rel_path = request.args.get("path", "")
    if not rel_path:
        return jsonify({"error": "Path parameter is required"}), 400
    full_path = _safe_full_path(rel_path)
    if full_path is None:
        return jsonify({"error": "Invalid path"}), 400
    if not os.path.isfile(full_path):
        return jsonify({"error": "File not found"}), 404
    if not full_path.lower().endswith(".npz"):
        return jsonify({"error": "Only .npz files are supported"}), 400

    try:
        file_size = os.path.getsize(full_path)
    except OSError as e:
        return jsonify({"error": f"Cannot stat file: {e}"}), 500

    arrays = []
    try:
        with zipfile.ZipFile(full_path) as zf:
            for info in zf.infolist():
                name = info.filename
                if not name.lower().endswith(".npy"):
                    continue
                key = name[:-4]  # drop ".npy"
                entry = {
                    "name": key,
                    "compressedSize": info.compress_size,
                }
                try:
                    with zf.open(info) as member:
                        hdr = _read_npy_header(member)
                except Exception as e:
                    entry["error"] = str(e)
                    arrays.append(entry)
                    continue
                descr = hdr.get("descr")
                shape = list(hdr.get("shape") or ())
                # `descr` can be a list of (name, type) for structured
                # dtypes — we don't try to summarize those here; just
                # surface the raw descr and skip itemsize math.
                dtype_str = descr if isinstance(descr, str) else str(descr)
                entry["shape"] = shape
                entry["dtype"] = _NPY_DTYPE_FRIENDLY.get(dtype_str, dtype_str)
                if isinstance(descr, str):
                    itemsize = _NPY_DTYPE_ITEMSIZE.get(descr)
                    if itemsize is not None:
                        n = 1
                        for d in shape:
                            n *= int(d)
                        entry["sizeBytes"] = n * itemsize
                arrays.append(entry)
    except zipfile.BadZipFile as e:
        return jsonify({"error": f"Not a valid .npz archive: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to read archive: {e}"}), 500

    return jsonify({
        "path": full_path,
        "fileSize": file_size,
        "arrays": arrays,
    })


@app.route("/api/files/image", methods=["GET"])
def serve_image_file():
    rel_path = request.args.get("path", "")
    if not rel_path:
        return jsonify({"error": "Path parameter is required"}), 400
    full_path = _safe_full_path(rel_path)
    if full_path is None:
        return jsonify({"error": "Invalid path"}), 400
    if not os.path.isfile(full_path):
        return jsonify({"error": "File not found"}), 404
    mime_type, _ = mimetypes.guess_type(full_path)
    if not mime_type or not mime_type.startswith("image/"):
        return jsonify({"error": "Only image files are supported"}), 400
    return send_file(full_path, mimetype=mime_type, conditional=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _reap_detached(proc):
    """Reap one fire-and-forget child (native viewer / video player / example
    downloader) via a dedicated wait() thread, so it never lingers as a zombie.

    Deliberately NOT a global SIGCHLD ``waitpid(-1)`` handler: that races every
    other thread's ``Popen.wait()``/``poll()`` (task runners, the exporter, the
    dataset downloader, the example script). When the handler won the race,
    the loser got ECHILD, which CPython maps to returncode 0 — a *failed*
    Blender export or download was then reported as success. Runners are
    reaped by TaskQueue._wait_and_release; other subprocesses by their own
    wait()/poll(); only detached children need this helper."""
    threading.Thread(target=proc.wait, daemon=True).start()


# ---------------------------------------------------------------------------
# Production static-serve fallback. When gui/frontend/build/ exists (after
# `npm run build`), Flask serves it at "/" so users get the bundled UI on
# the same origin as the API — no Vite needed. Flask's routing table picks
# /api/* routes first, so this catch-all is safe to register last.
# In dev, `gui/scripts/dev.sh` runs Vite on :3000 with a /api proxy and
# this block is skipped because the build dir won't exist.
# ---------------------------------------------------------------------------

_BUILD_DIR = _PROJECT_ROOT / "frontend" / "build"
if _BUILD_DIR.is_dir():
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def _serve_frontend(path: str):
        # Unknown /api/* routes should 404, not silently fall through to
        # the SPA index — otherwise an API typo returns 200 + HTML and
        # the frontend renders nothing while looking like it worked.
        if path.startswith("api/") or path == "api":
            return jsonify({"error": "Not Found", "path": "/" + path}), 404
        candidate = _BUILD_DIR / path if path else None
        if candidate and candidate.is_file():
            return send_file(candidate)
        return send_file(_BUILD_DIR / "index.html")


# ---------------------------------------------------------------------------
# Initialise process-level state at module import time. Runs both for
# `python app.py` (dev mode via Werkzeug) AND for `waitress-serve app:app`
# (prod mode) — waitress does not execute the __main__ block, so anything
# that *must* happen on startup has to live up here.
# All steps below are idempotent:
#   * `CREATE TABLE IF NOT EXISTS` for the SQLite schema,
#   * `signal.signal()` overwrites whatever handler was there.
# ---------------------------------------------------------------------------

db.initialize_database()
db.test_database_connection()
# NOTE: no SIGCHLD handler on purpose. Every child is reaped by an owner:
# task runners by TaskQueue._wait_and_release, exporter/downloader/example
# subprocesses by their own wait()/poll(), detached viewer launches by
# _reap_detached. A global waitpid(-1) handler used to race those wait()s
# and turn failures into ECHILD -> returncode 0 (reported as success).
# Start the task-queue coordinator. Hydrates from DB (any rows still
# in 'Queued' status from a previous Flask process), then spawns
# runners one at a time (or N, per the concurrency_limit setting).
#
# In Flask debug mode the auto-reloader runs this module in two
# processes (a parent watcher and a child worker). Without this guard
# both processes would hydrate the same Queued rows and double-spawn
# their runners, which races on GPU memory and the output tree. Only
# the worker (or the single process when debug is off) should run
# the coordinator. WERKZEUG_RUN_MAIN=true marks the worker; absent
# means either the parent watcher or a non-reloader run.
_debug_enabled = os.environ.get("MAMMA_DEBUG", "1") not in ("0", "false", "False", "")
_is_reloader_worker_or_single = (not _debug_enabled) or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
if _is_reloader_worker_or_single:
    _task_queue.start_coordinator()
else:
    print("---> Skipping queue coordinator in reloader parent (will start in worker)")
print(f"---> MOUNT_POINT={MOUNT_POINT}")
print(f"---> LOCAL_USER={LOCAL_USER}")


if __name__ == "__main__":
    # Bind to loopback by default. Combined with the file APIs that accept
    # arbitrary absolute paths, binding to 0.0.0.0 would let anyone on the
    # network read this user's filesystem via /api/files/*. Override
    # explicitly via MAMMA_BIND_HOST=0.0.0.0 if you really mean it.
    bind_host = os.environ.get("MAMMA_BIND_HOST", "127.0.0.1")
    bind_port = int(os.environ.get("MAMMA_BIND_PORT", "8000"))
    debug = os.environ.get("MAMMA_DEBUG", "1") not in ("0", "false", "False", "")
    print(f"---> Listening on http://{bind_host}:{bind_port} (debug={debug})")
    app.run(host=bind_host, port=bind_port, debug=debug)
