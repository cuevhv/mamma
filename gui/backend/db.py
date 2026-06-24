"""SQLite-backed database layer for the cvpr_release branch.

Replaces the postgres + psycopg2 module that lived in `database.py`.
Public API mirrors the old module so callers in `app.py` only need an
import-path change. HTCondor/DAGMan-specific helpers (e.g. log parsing)
have been removed; the local pipeline runner writes statuses directly
via `set_process_status`.

Storage: ${MAMMA_DB_PATH:-${MAMMA_DATA_DIR:-~/.mamma}/mamma.sqlite}
"""
import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from config_io import load_config_file
from objects.processes import ProcessType
from objects.sequences import get_sequences_from_data


_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # backend/ -> repo root


def _db_path() -> Path:
    explicit = os.environ.get("MAMMA_DB_PATH")
    if explicit:
        return Path(explicit).expanduser()
    data_dir_env = os.environ.get("MAMMA_DATA_DIR")
    data_dir = Path(data_dir_env).expanduser() if data_dir_env else _PROJECT_ROOT / "var"
    return data_dir / "mamma.sqlite"


def create_connection():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        timeout=30.0,
    )
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def _parse_ts(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _iso_utc(value):
    """Serialize a stored ``created_at`` as an explicit-UTC ISO-8601 string.

    SQLite's ``CURRENT_TIMESTAMP`` records naive UTC; with ``PARSE_DECLTYPES``
    the value comes back as a naive ``datetime``. Rendering it with ``str()``
    drops any timezone marker (``'2026-06-11 08:06:54'``), and the browser's
    ``Date.parse`` then reads that as the *viewer's local* time, skewing
    "X ago" displays by the local UTC offset. Emitting an explicit ``Z``
    (``'2026-06-11T08:06:54Z'``) makes the instant unambiguous on the wire,
    so every endpoint and page agrees.

    Accepts a ``datetime`` or a stored string; returns ``None`` for empty
    input and falls back to ``str(value)`` for anything unparseable.
    """
    if value is None or value == "":
        return None
    dt = value if isinstance(value, datetime) else _parse_ts(value)
    if dt is None:
        return str(value)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def initialize_database():
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS captures (
                capture_id INTEGER PRIMARY KEY AUTOINCREMENT,
                capture_name TEXT,
                ioi_root TEXT,
                capture_json_path TEXT
            );"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS sequences (
                sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                capture_id INTEGER REFERENCES captures(capture_id) ON DELETE CASCADE,
                sequence_name TEXT,
                sequence_path TEXT
            );"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
                task_id INTEGER PRIMARY KEY AUTOINCREMENT,
                capture_id INTEGER REFERENCES captures(capture_id) ON DELETE CASCADE,
                username TEXT,
                task_json_path TEXT,
                output_path TEXT,
                output_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dag_job_id TEXT,
                preset_path TEXT
            );"""
        )
        # Idempotent migration for installs where the tasks table predates
        # the preset_path column (added 2026-05-24 alongside the
        # preset/run-config vocabulary split).
        existing_cols = {r["name"] for r in cur.execute("PRAGMA table_info(tasks)").fetchall()}
        if "preset_path" not in existing_cols:
            cur.execute("ALTER TABLE tasks ADD COLUMN preset_path TEXT")
        # Key-value settings store. Holds runtime knobs that survive
        # Flask restarts — currently just the task-queue concurrency
        # limit (default 1). Add new keys by writing them; consumers
        # read with sensible defaults if the row is missing.
        cur.execute(
            """CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            );"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS processes (
                process_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER REFERENCES tasks(task_id) ON DELETE CASCADE,
                sequence_id INTEGER REFERENCES sequences(sequence_id),
                process TEXT,
                process_mapping TEXT,
                validation_mapping TEXT,
                cluster_job_id TEXT,
                sif_file TEXT,
                sub_file TEXT,
                sh_file TEXT,
                log_file TEXT,
                out_file TEXT,
                err_file TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                ended_at TIMESTAMP,
                num_frames INTEGER,
                num_cameras INTEGER
            );"""
        )
        # Idempotent migration for installs whose processes table predates the
        # execution-timing columns (added 2026-06-17). started_at is stamped on
        # the Running transition, ended_at on a terminal one; per-step duration
        # = ended_at - started_at, measured in the runner outside the step
        # subprocess (zero pipeline overhead). Old rows keep NULLs (no timing).
        proc_cols = {r["name"] for r in cur.execute("PRAGMA table_info(processes)").fetchall()}
        if "started_at" not in proc_cols:
            cur.execute("ALTER TABLE processes ADD COLUMN started_at TIMESTAMP")
        if "ended_at" not in proc_cols:
            cur.execute("ALTER TABLE processes ADD COLUMN ended_at TIMESTAMP")
        # Idempotent migration (added 2026-06-24): actual frame/camera counts,
        # written on the ma_cap process row once ma_cap finishes for a
        # (task, sequence). Old rows / non-ma_cap rows keep NULLs (shown as "—").
        if "num_frames" not in proc_cols:
            cur.execute("ALTER TABLE processes ADD COLUMN num_frames INTEGER")
        if "num_cameras" not in proc_cols:
            cur.execute("ALTER TABLE processes ADD COLUMN num_cameras INTEGER")
        conn.commit()
        print(f"---> SQLite database initialized at {_db_path()}.")


def test_database_connection():
    try:
        with create_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cur.fetchall()]
            print(f"\n---> Database Connection Test: SUCCESS. Tables: {tables}\n")
            return True
    except Exception as e:
        print(f"\n---> Database Connection Test: FAILED. Error: {e}\n")
        return False


def _step_log_paths(task_content, username, log_tag, step_name, seq_name):
    """Compute per-(step, sequence) stdout/stderr file paths from the task
    config.

    Mirrors the directory layout used by the old HTCondor jobs so the
    frontend's log viewers keep working without changes:
        <jobs_log_dir>/<user>/<log_tag>/<step>/<seq>.{out,err}
    """
    jobs_log_dir = (task_content or {}).get("global", {}).get("jobs_log_dir", "")
    if not jobs_log_dir:
        return ("", "")
    base = os.path.join(jobs_log_dir, str(username), str(log_tag), step_name)
    return (
        os.path.join(base, f"{seq_name}.out"),
        os.path.join(base, f"{seq_name}.err"),
    )


def _derive_capture_name(capture_json_path):
    """Display-friendly capture name. Basename alone collides for generic
    files like `capture.json`, so fall back to the parent dir name."""
    basename = os.path.basename(capture_json_path).replace(".json", "")
    parent = os.path.basename(os.path.dirname(capture_json_path))
    if basename.lower() in ("capture", "default", "") and parent:
        return parent
    return basename or parent


def _persist_task_rows(
    *,
    capture_json_path,
    capture_content,
    task_content,
    seq_names,
    processes,
    output_dir,
    output_id,
    username,
    final_task_json_path,
    preset_path=None,
    process_status="Waiting",
):
    """Insert captures (upsert) + tasks + sequences + processes for one
    submission. Shared between `create_entry` (live GUI, all processes
    Waiting) and `import_cli_task` (statuses inferred from the filesystem).

    `process_status` is either:
      - a string applied to every (step, seq), or
      - a callable `(step_name, seq_name) -> status` for per-cell control.

    `final_task_json_path` is the path the tasks row should record. Pass
    None to use the legacy `<task_jsons_dir>/task_config_<task_id>.json`
    layout — but the caller must then supply `task_json_server_dir` via
    the parent helper.
    """
    capture_name = _derive_capture_name(capture_json_path)
    if not capture_name:
        return "Issues with capture name", None, None, None

    ioi_root = capture_content.get("ioi_root", "") if isinstance(capture_content, dict) else ""

    with create_connection() as conn:
        cur = conn.cursor()

        # Match captures by file path (stable) rather than display name.
        cur.execute(
            "SELECT capture_id, capture_name FROM captures WHERE capture_json_path = ?",
            (capture_json_path,),
        )
        row = cur.fetchone()
        if row:
            capture_id = row["capture_id"]
            # Upgrade legacy generic names if a better one is now available.
            if (row["capture_name"] or "").lower() in ("capture", "default") and capture_name not in ("capture", "default"):
                cur.execute(
                    "UPDATE captures SET capture_name = ? WHERE capture_id = ?",
                    (capture_name, capture_id),
                )
        else:
            cur.execute(
                "INSERT INTO captures (capture_name, ioi_root, capture_json_path) VALUES (?, ?, ?)",
                (capture_name, ioi_root, capture_json_path),
            )
            capture_id = cur.lastrowid

        cur.execute(
            "INSERT INTO tasks (capture_id, username, task_json_path, output_path, output_id, preset_path) VALUES (?, ?, ?, ?, ?, ?)",
            (capture_id, username, final_task_json_path or "PENDING_CREATION", output_dir, output_id, preset_path),
        )
        task_id = cur.lastrowid
        if output_id is None:
            output_id = str(task_id)
            cur.execute("UPDATE tasks SET output_id = ? WHERE task_id = ?", (output_id, task_id))

        # Register sequences (the runner writes per-step outputs at
        # <output_dir>/<step>/<output_id>/<dataset>/<seq>/, so we don't
        # mkdir these — they're a logical hint only).
        all_seq_names = set(get_sequences_from_data(capture_content))
        # The import path may pass sequences that exist on disk but aren't
        # in capture.json. Register them anyway so processes can attach.
        all_seq_names.update(seq_names or [])
        seq_name_to_id = {}
        for seq_name in all_seq_names:
            seq_path = os.path.join(output_dir, seq_name)
            cur.execute(
                "SELECT sequence_id FROM sequences WHERE capture_id = ? AND sequence_name = ?",
                (capture_id, seq_name),
            )
            existing = cur.fetchone()
            if existing:
                seq_name_to_id[seq_name] = existing["sequence_id"]
            else:
                cur.execute(
                    "INSERT INTO sequences (capture_id, sequence_name, sequence_path) VALUES (?, ?, ?)",
                    (capture_id, seq_name, seq_path),
                )
                seq_name_to_id[seq_name] = cur.lastrowid

        status_fn = process_status if callable(process_status) else (lambda _step, _seq: process_status)
        for seq_name in seq_names:
            sequence_id = seq_name_to_id.get(seq_name)
            for process in processes:
                if process not in ProcessType:
                    conn.rollback()
                    return f"Invalid process type: {process}", None, None, None
                step_name = process.value if isinstance(process, ProcessType) else process
                step_cfg = (task_content or {}).get(step_name, {}) if isinstance(task_content, dict) else {}
                sif_file = step_cfg.get("sif_path", "") or ""
                out_file, err_file = _step_log_paths(
                    task_content, username, task_id, step_name, seq_name
                )
                # log_file column is retained in the schema for backwards
                # compatibility with existing DBs but always written as
                # NULL — the runner never produced a separate .log file
                # (it only writes .out and .err).
                cur.execute(
                    """INSERT INTO processes (
                        task_id, sequence_id, process, process_mapping, validation_mapping,
                        cluster_job_id, sif_file, sub_file, sh_file,
                        log_file, out_file, err_file, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                    (
                        task_id,
                        sequence_id,
                        step_name,
                        f"{step_name}_{seq_name}",
                        None,
                        None,
                        sif_file,
                        "",
                        "",
                        None,
                        out_file,
                        err_file,
                        status_fn(step_name, seq_name),
                    ),
                )

        conn.commit()
        return "Entry created successfully", task_id, final_task_json_path, output_id


def create_entry(
    capture_json_path,
    seq_names,
    output_dir,
    processes,
    username,
    task_json_server_dir,
    capture_content=None,
    task_content=None,
    task_template_path=None,
    output_id=None,
    preset_path=None,
):
    """Insert DB rows for a freshly-submitted GUI run. All processes start
    in `Queued` — the in-process task coordinator pops them off the queue
    one (or N) at a time, flips them to `Waiting`, and spawns the runner
    subprocess. The runner then drives the per-process lifecycle to
    Running -> Completed/Failed as before.

    ``preset_path`` records the source preset the GUI used to build this
    run config — used to surface "preset: <name>" lineage in the Runs
    table. Pass None when no preset is involved (e.g. CLI imports)."""
    try:
        if capture_content is None:
            capture_content = load_config_file(capture_json_path)
        if task_content is None:
            if task_template_path:
                task_content = load_config_file(task_template_path)
            else:
                return "Task content or template path required", None, None, None
    except Exception as e:
        print(f"Error reading config file: {e}")
        return "Error reading config file", None, None, None

    # First call inserts the row with a placeholder task_json_path; we
    # then update it to the resolved per-task copy once we know the id.
    status, task_id, _, output_id = _persist_task_rows(
        capture_json_path=capture_json_path,
        capture_content=capture_content,
        task_content=task_content,
        seq_names=seq_names,
        processes=processes,
        output_dir=output_dir,
        output_id=output_id,
        username=username,
        final_task_json_path=None,
        preset_path=preset_path,
        process_status="Queued",
    )
    if status != "Entry created successfully":
        return status, None, None, None

    final_task_json_path = os.path.join(task_json_server_dir, f"run_{task_id}.json")
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE tasks SET task_json_path = ? WHERE task_id = ?",
            (final_task_json_path, task_id),
        )
        conn.commit()

    capture_name = _derive_capture_name(capture_json_path)
    print(f"---> Created entry for capture '{capture_name}' task_id={task_id} output_id={output_id}")
    return "Entry created successfully", task_id, final_task_json_path, output_id


def import_cli_task(
    *,
    capture_json_path,
    capture_content,
    task_content,
    seq_names,
    processes,
    output_dir,
    output_id,
    username,
    task_json_path,
    process_status,
    preset_path=None,
):
    """Insert a tasks row for a CLI-run task that already produced
    filesystem outputs. The caller has read the task.json and decided
    which sequences/processes to register; we just write the rows.

    `task_json_path` is the path to the user's source task.json (we
    record it so the GUI's "View task config" still resolves).

    `process_status` may be a string ("Completed") or a callable that
    returns a status per (step, seq) — typically wired to
    `pipeline.sync.infer_step_status`.
    """
    return _persist_task_rows(
        capture_json_path=capture_json_path,
        capture_content=capture_content,
        task_content=task_content,
        seq_names=seq_names,
        processes=processes,
        output_dir=output_dir,
        output_id=output_id,
        username=username,
        final_task_json_path=task_json_path,
        preset_path=preset_path,
        process_status=process_status,
    )


def get_task_by_output_id(output_id):
    """Look up a task by its output_id (the directory name on disk).
    Returns None if no match."""
    if output_id is None or output_id == "":
        return None
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT t.task_id, t.username, t.created_at, t.output_path, t.output_id,
                      t.task_json_path, c.capture_name, c.capture_json_path
               FROM tasks t
               JOIN captures c ON t.capture_id = c.capture_id
               WHERE t.output_id = ?""",
            (output_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def delete_task_with_processes(task_id):
    """Drop a tasks row + cascade its processes. Sequences are shared
    across tasks, so they're left alone. Returns True if a row was
    actually removed."""
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT task_id FROM tasks WHERE task_id = ?", (task_id,))
        if cur.fetchone() is None:
            return False
        # Processes cascade automatically via the schema's ON DELETE CASCADE.
        cur.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        conn.commit()
        return True


def delete_sequence_from_task(task_id, seq_name):
    """Drop just the processes for one (task, sequence) pair.

    The Tasks table renders one row per (task, sequence), so the row-level
    delete must scope to a single sequence — not the whole task. We delete
    only this task's processes for the named sequence; the shared `sequences`
    row is left alone (other tasks may reference it). Restricting the delete
    to `processes.task_id = ?` keeps it safe even if the same sequence_name
    exists under another capture — this task only owns its own capture's rows.

    If that was the task's last remaining sequence, the now-empty task row is
    removed too, so an orphan task with zero processes doesn't linger.

    Returns ``(removed, task_removed)``: whether any process row matched, and
    whether the parent task row was also dropped as a result.
    """
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """DELETE FROM processes
                 WHERE task_id = ?
                   AND sequence_id IN (
                       SELECT sequence_id FROM sequences WHERE sequence_name = ?
                   )""",
            (task_id, seq_name),
        )
        removed = cur.rowcount > 0
        task_removed = False
        if removed:
            cur.execute("SELECT COUNT(*) AS n FROM processes WHERE task_id = ?", (task_id,))
            if cur.fetchone()["n"] == 0:
                cur.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
                task_removed = cur.rowcount > 0
        conn.commit()
        return removed, task_removed


def get_all_task_minimal():
    """Compact listing of every task, used by the audit endpoint to
    detect orphans (DB rows whose outputs are gone). Avoids the full
    process-row hydration that `get_all_tasks_with_processes` does."""
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT t.task_id, t.username, t.created_at, t.output_path, t.output_id,
                      t.task_json_path, t.preset_path, c.capture_name, c.capture_json_path,
                      (SELECT GROUP_CONCAT(DISTINCT p.process)
                         FROM processes p WHERE p.task_id = t.task_id) AS step_csv
               FROM tasks t
               JOIN captures c ON t.capture_id = c.capture_id
               ORDER BY t.task_id DESC"""
        )
        out = []
        for r in cur.fetchall():
            out.append({
                "taskId": str(r["task_id"]),
                "username": r["username"],
                "createdAt": _iso_utc(r["created_at"]),
                "outputPath": r["output_path"],
                "outputId": r["output_id"],
                "taskJsonPath": r["task_json_path"],
                "presetPath": r["preset_path"],
                "captureName": r["capture_name"],
                "captureJsonPath": r["capture_json_path"],
                "steps": (r["step_csv"] or "").split(",") if r["step_csv"] else [],
            })
        return out


def set_task_runner_pid(task_id, pid):
    """Stores the local runner PID for a task. The DB column is named
    `dag_job_id` for legacy reasons (HTCondor-era schema) but is
    repurposed in cvpr_release as the OS PID of `pipeline.cli.run`."""
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE tasks SET dag_job_id = ? WHERE task_id = ?", (str(pid), task_id))
        conn.commit()
        print(f"---> Updated runner PID for task {task_id} to {pid}")


# Backwards-compatible alias for any code path still using the old name.
update_task_dag_job_id = set_task_runner_pid


def get_unfinished_tasks_info():
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT DISTINCT t.task_id, t.username, t.dag_job_id, t.created_at
               FROM tasks t
               JOIN processes p ON t.task_id = p.task_id
               WHERE p.status NOT IN ('Completed','Failed','val_Completed','val_Failed','Done','Cancelled')"""
        )
        return [
            {
                "task_id": r["task_id"],
                "username": r["username"],
                "dag_job_id": r["dag_job_id"],
                "created_at": _iso_utc(r["created_at"]) if r["created_at"] else "",
            }
            for r in cur.fetchall()
        ]


def is_task_created_within_minute(task_id):
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT created_at FROM tasks WHERE task_id = ?", (task_id,))
        row = cur.fetchone()
        if row is None:
            return None
        created_at = _parse_ts(row["created_at"])
        if created_at is None:
            return False
        return (datetime.now() - created_at) < timedelta(minutes=1)


_TERMINAL_STATUSES = {"Done", "Completed", "Failed", "Cancelled"}


def set_process_status(process_id, status, pid=None):
    """Update the status of a single process. The local runner calls this.
    `pid` is the OS PID of the engine subprocess; stored under the
    legacy `cluster_job_id` column.

    The same write also stamps execution-time markers so the GUI can show
    per-step durations at no extra pipeline cost (this UPDATE already fires on
    every transition): ``started_at`` on the ``Running`` transition (reset each
    time, so a forced re-run doesn't inherit a stale start) and ``ended_at`` on
    any terminal transition — including ``Failed``/``Cancelled``, so the UI can
    show time-spent-before-failure. A skipped/cached step emits ``Done`` with no
    preceding ``Running``, leaving ``started_at`` NULL (rendered as "cached",
    never a negative duration)."""
    sets = ["status = ?"]
    params = [status]
    if pid is not None:
        sets.append("cluster_job_id = ?")
        params.append(str(pid))
    if status == "Running":
        sets.append("started_at = CURRENT_TIMESTAMP")
    elif status in _TERMINAL_STATUSES:
        sets.append("ended_at = CURRENT_TIMESTAMP")
    params.append(process_id)
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE processes SET {', '.join(sets)} WHERE process_id = ?",
            params,
        )
        conn.commit()
        return cur.rowcount > 0


# Alias used by app.py.
update_process_status = set_process_status


def set_process_metadata(process_id, num_frames=None, num_cameras=None):
    """Store the actual frame/camera counts on a process row.

    Written on the ma_cap row once that step finishes for a (task, sequence),
    so the Tasks table can show real counts without reading any files. Both
    args are optional; only the provided ones are updated. Returns True if a
    row was updated."""
    sets = []
    params = []
    if num_frames is not None:
        sets.append("num_frames = ?")
        params.append(int(num_frames))
    if num_cameras is not None:
        sets.append("num_cameras = ?")
        params.append(int(num_cameras))
    if not sets:
        return False
    params.append(process_id)
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE processes SET {', '.join(sets)} WHERE process_id = ?",
            params,
        )
        conn.commit()
        return cur.rowcount > 0


def get_processes_for_task(task_id):
    """Return all (process_id, step, sequence_name, status) rows for a task.

    Used by the local runner to walk the planned work.
    """
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT p.process_id, p.process, s.sequence_name, p.status,
                      p.out_file, p.err_file
               FROM processes p
               JOIN sequences s ON p.sequence_id = s.sequence_id
               WHERE p.task_id = ?
               ORDER BY p.process_id ASC""",
            (task_id,),
        )
        return [
            {
                "process_id": r["process_id"],
                "process": r["process"],
                "sequence_name": r["sequence_name"],
                "status": r["status"],
                "out_file": r["out_file"],
                "err_file": r["err_file"],
            }
            for r in cur.fetchall()
        ]


def get_active_processes_for_user(username):
    unfinished = get_unfinished_tasks_info()
    user_ids = [t["task_id"] for t in unfinished if t["username"] == username]
    if not user_ids:
        return []
    return _list_processes_for_task_ids(user_ids)


def _list_processes_for_task_ids(task_ids):
    placeholders = ",".join("?" for _ in task_ids)
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""SELECT p.process_id, p.task_id, c.capture_name, s.sequence_name, p.process,
                       p.status, p.created_at, p.cluster_job_id, p.out_file,
                       p.err_file
                FROM processes p
                JOIN tasks t ON p.task_id = t.task_id
                JOIN captures c ON t.capture_id = c.capture_id
                JOIN sequences s ON p.sequence_id = s.sequence_id
                WHERE t.task_id IN ({placeholders})
                ORDER BY p.created_at DESC""",
            task_ids,
        )
        return [
            {
                "processId": str(r["process_id"]),
                "taskId": str(r["task_id"]),
                "captureName": r["capture_name"],
                "sequenceName": r["sequence_name"],
                "processType": r["process"],
                "status": r["status"],
                "createdAt": _iso_utc(r["created_at"]),
                "pid": r["cluster_job_id"],  # legacy column name; surface as generic pid
                "outFile": r["out_file"],
                "errFile": r["err_file"],
            }
            for r in cur.fetchall()
        ]


def get_all_active_tasks_with_processes():
    unfinished = get_unfinished_tasks_info()
    if not unfinished:
        return []
    task_ids = [t["task_id"] for t in unfinished]
    info_map = {t["task_id"]: t for t in unfinished}
    placeholders = ",".join("?" for _ in task_ids)

    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""SELECT p.process_id, p.task_id, c.capture_name, c.capture_json_path,
                       s.sequence_name, p.process,
                       p.status, p.created_at, p.started_at, p.ended_at,
                       p.cluster_job_id, p.out_file,
                       p.err_file, t.username
                FROM processes p
                JOIN tasks t ON p.task_id = t.task_id
                JOIN captures c ON t.capture_id = c.capture_id
                JOIN sequences s ON p.sequence_id = s.sequence_id
                WHERE t.task_id IN ({placeholders})
                ORDER BY p.task_id DESC, p.created_at DESC""",
            task_ids,
        )
        tasks = {}
        for r in cur.fetchall():
            tid = str(r["task_id"])
            if tid not in tasks:
                info = info_map.get(int(tid), {})
                tasks[tid] = {
                    "taskId": tid,
                    "captureName": r["capture_name"],
                    "captureJsonPath": r["capture_json_path"],
                    "username": r["username"],
                    "createdAt": info.get("created_at", _iso_utc(r["created_at"])),
                    "runnerPid": info.get("dag_job_id"),  # legacy column name
                    "processes": [],
                }
            tasks[tid]["processes"].append(
                {
                    "processId": str(r["process_id"]),
                    "sequenceName": r["sequence_name"],
                    "processType": r["process"],
                    "status": r["status"],
                    "createdAt": _iso_utc(r["created_at"]),
                    "startedAt": _iso_utc(r["started_at"]),
                    "endedAt": _iso_utc(r["ended_at"]),
                    "pid": r["cluster_job_id"],
                    "outFile": r["out_file"],
                    "errFile": r["err_file"],
                }
            )
        return list(tasks.values())


def add_process_info(task_id, mapping_content):
    """Kept for API compatibility. The local runner already populates
    process_mapping at create_entry time, so this is a no-op pass-through
    for any externally provided mapping that wants to override it."""
    if not mapping_content:
        return
    with create_connection() as conn:
        cur = conn.cursor()
        for key, sequence_name in mapping_content.items():
            is_validation = "_validate_" in key
            if is_validation:
                process_name = key.split("_validate_")[0]
                col = "validation_mapping"
            else:
                process_name = key.rsplit("_", 1)[0]
                col = "process_mapping"
            cur.execute(
                f"""UPDATE processes SET {col} = ?
                    WHERE task_id = ?
                      AND process = ?
                      AND sequence_id IN (SELECT sequence_id FROM sequences WHERE sequence_name = ?)""",
                (key, task_id, process_name, sequence_name),
            )
        conn.commit()


_TERMINAL_OK = ("Completed", "Done", "val_Completed")
_TERMINAL_BAD = ("Failed", "val_Failed", "Cancelled")


def _capture_status_from_processes(statuses):
    """Roll up many process statuses into one capture-level summary."""
    if not statuses:
        return "Pending"
    if any("Running" in s for s in statuses):
        return "Running"
    if any(s in _TERMINAL_BAD for s in statuses):
        return "Failed"
    if all(s in _TERMINAL_OK for s in statuses):
        return "Completed"
    # "Queued" only when every process is still queued (i.e. the task
    # hasn't been popped off the in-memory queue yet). Mixed Queued +
    # Waiting collapses to plain Pending so the row doesn't flicker
    # status as the coordinator transitions it.
    if all(s == "Queued" for s in statuses):
        return "Queued"
    return "Pending"


# Set of process statuses considered "queued" — for the runner-side
# transition where Queued processes get bumped to Waiting before
# the runner subprocess starts executing them.
_PROCESS_QUEUED_STATUSES = {"Queued"}


def get_concurrency_limit() -> int:
    """Task-queue concurrency cap. Defaults to 1 (sequential)."""
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key = 'concurrency_limit'")
        row = cur.fetchone()
    if not row or row["value"] is None:
        return 1
    try:
        n = int(row["value"])
    except (TypeError, ValueError):
        return 1
    return max(1, min(n, 8))


def set_concurrency_limit(n: int) -> None:
    """Persist the task-queue concurrency cap. Clamped to [1, 8]."""
    n = max(1, min(int(n), 8))
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO settings (key, value) VALUES ('concurrency_limit', ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(n),),
        )
        conn.commit()


def tasks_with_all_processes_status(status: str) -> list[int]:
    """Return task_ids whose every process has the given status,
    ordered by created_at ascending (FIFO).

    Used to hydrate the in-memory task queue from DB on Flask restart:
    rows still marked "Queued" repopulate the queue in submission order.
    """
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT t.task_id FROM tasks t
               WHERE t.task_id NOT IN (
                   SELECT task_id FROM processes WHERE status IS NOT ?
               )
               AND t.task_id IN (SELECT task_id FROM processes WHERE status = ?)
               ORDER BY t.created_at ASC""",
            (status, status),
        )
        return [r["task_id"] for r in cur.fetchall()]


def bump_task_processes_from_queued_to_waiting(task_id: int) -> None:
    """Coordinator → runner handoff: flip all of a task's processes
    from Queued to Waiting just before the runner subprocess spawns.
    Idempotent; rows that aren't Queued are left alone."""
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE processes SET status = 'Waiting'"
            " WHERE task_id = ? AND status = 'Queued'",
            (task_id,),
        )
        conn.commit()


def reset_non_completed_to_queued(task_id: int) -> int:
    """Restart-prep: flip every non-Completed / non-Done process of a
    task back to Queued so the coordinator can pick it up again. Done
    processes stay Done so the runner's DONE-sentinel skip kicks in,
    making resume cheap.

    Returns the number of processes that flipped — useful for the
    route handler to tell whether anything was actually reset (zero
    means "task was already fully Completed, restart was a no-op")."""
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE processes SET status = 'Queued'"
            " WHERE task_id = ?"
            "   AND status NOT IN ('Completed', 'val_Completed', 'Done')",
            (task_id,),
        )
        affected = cur.rowcount
        conn.commit()
    return affected


def cancel_queued_task(task_id: int) -> bool:
    """Mark every Queued process for this task as Cancelled.

    Returns True if any rows actually flipped (i.e. the task really was
    queued, not already running/finished). False if no-op — useful for
    the DELETE /api/tasks/<id>/queue route to tell whether the queue
    cancel landed in time."""
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE processes SET status = 'Cancelled'"
            " WHERE task_id = ? AND status = 'Queued'",
            (task_id,),
        )
        affected = cur.rowcount
        conn.commit()
    return affected > 0


def get_latest_task_for_capture(capture_id):
    """Return (output_path, output_id, task_json_path, created_at) for the
    most-recent task tied to this capture, or None if no tasks exist.
    Used by the captures listing to compute thumbnail paths."""
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT output_path, output_id, task_json_path, created_at
               FROM tasks WHERE capture_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (capture_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_all_captures():
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT capture_id, capture_name, ioi_root, capture_json_path
               FROM captures ORDER BY capture_id DESC"""
        )
        rows = cur.fetchall()
        out = []
        for r in rows:
            cid = r["capture_id"]
            cur.execute("SELECT sequence_name FROM sequences WHERE capture_id = ?", (cid,))
            seq_names = [s["sequence_name"] for s in cur.fetchall()]
            cur.execute("SELECT COUNT(*) AS c FROM tasks WHERE capture_id = ?", (cid,))
            task_count = cur.fetchone()["c"]
            cur.execute(
                """SELECT output_path, output_id, task_json_path, created_at
                   FROM tasks WHERE capture_id = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (cid,),
            )
            latest_task = cur.fetchone()
            created_at = _iso_utc(latest_task["created_at"]) if latest_task else "N/A"
            # Normalize the latest-task dict's timestamp too, so lastTaskAt
            # goes out as ISO-UTC ('...Z') instead of relying on jsonify's
            # RFC-1123 'GMT' rendering — keeps every endpoint on one format.
            latest_task_dict = dict(latest_task) if latest_task else None
            if latest_task_dict is not None:
                latest_task_dict["created_at"] = _iso_utc(latest_task_dict.get("created_at"))

            # Roll up actual process statuses across all of this capture's
            # tasks instead of the placeholder "Pending" the original schema
            # returned. Same for the distinct process types.
            cur.execute(
                """SELECT DISTINCT p.process FROM processes p
                   JOIN tasks t ON p.task_id = t.task_id
                   WHERE t.capture_id = ?""",
                (cid,),
            )
            processes = sorted(p["process"] for p in cur.fetchall() if p["process"])
            cur.execute(
                """SELECT p.status FROM processes p
                   JOIN tasks t ON p.task_id = t.task_id
                   WHERE t.capture_id = ?""",
                (cid,),
            )
            statuses = [s["status"] for s in cur.fetchall() if s["status"]]
            status = _capture_status_from_processes(statuses)

            out.append(
                {
                    "capture_id": cid,
                    "capture_name": r["capture_name"],
                    "ioi_root": r["ioi_root"],
                    "capture_json_path": r["capture_json_path"],
                    "seq_names": seq_names,
                    "task_count": task_count,
                    "created_at": created_at,
                    "status": status,
                    "processes": processes,
                    # Latest task hints — frontend uses these to render
                    # "last run X ago" + a thumbnail. Path resolution is
                    # done in app.py so the DB layer stays filesystem-free.
                    "latest_task": latest_task_dict,
                }
            )
        return out


def get_capture_details(capture_name):
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT capture_id, capture_json_path, ioi_root FROM captures WHERE capture_name = ?",
            (capture_name,),
        )
        capture_row = cur.fetchone()
        if not capture_row:
            return None
        capture_id = capture_row["capture_id"]

        cur.execute(
            """SELECT sequence_name, sequence_path FROM sequences
               WHERE capture_id = ? ORDER BY sequence_name""",
            (capture_id,),
        )
        sequences = [{"name": r["sequence_name"], "path": r["sequence_path"]} for r in cur.fetchall()]

        cur.execute(
            """SELECT t.task_id, t.username, t.created_at, t.output_path, t.output_id, t.task_json_path
               FROM tasks t
               WHERE t.capture_id = ? ORDER BY t.created_at DESC""",
            (capture_id,),
        )
        task_rows = cur.fetchall()

        tasks = []
        for tr in task_rows:
            tid = tr["task_id"]
            cur.execute(
                """SELECT p.process, p.status, s.sequence_name FROM processes p
                   LEFT JOIN sequences s ON p.sequence_id = s.sequence_id
                   WHERE p.task_id = ?""",
                (tid,),
            )
            procs = [
                {"process": p["process"], "status": p["status"], "sequence": p["sequence_name"]}
                for p in cur.fetchall()
            ]
            tasks.append(
                {
                    "task_id": tid,
                    "username": tr["username"],
                    "created_at": _iso_utc(tr["created_at"]),
                    "processes": procs,
                    "output_path": tr["output_path"],
                    "output_id": tr["output_id"],
                    "task_json_path": tr["task_json_path"],
                }
            )
        return {
            "capture_name": capture_name,
            "capture_json_path": capture_row["capture_json_path"],
            "ioi_root": capture_row["ioi_root"],
            "sequences": sequences,
            "tasks": tasks,
        }


def save_capture_with_sequences(capture_name, ioi_root, capture_json_path):
    """Insert or update the captures row, then reconcile its sequences
    against the current capture.json.

    Idempotent. The (capture_json_path) column is the natural key; if a
    row already exists for that path we UPDATE its ``capture_name`` and
    ``ioi_root`` (so an overwrite-create from the New Task form doesn't
    leave stale metadata in the DB). Otherwise we INSERT a new row.

    For the sequences:
      * Names present in the JSON are upserted into the sequences table.
      * Orphan sequence rows (in DB but no longer in the JSON) are
        deleted **only when no ``processes.sequence_id`` references
        them** — this preserves task history. A sequence that was used
        by a now-removed branch of the capture stays as a ghost row
        rather than dangling-FK-ing the processes table.

    Best-effort on the JSON parse: if loading fails we still commit the
    captures-row insert/update (the file is on disk; the user may fix
    and re-try)."""
    if not capture_json_path:
        return
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT capture_id FROM captures WHERE capture_json_path = ?",
            (capture_json_path,),
        )
        existing = cur.fetchone()
        if existing:
            capture_id = existing["capture_id"]
            cur.execute(
                "UPDATE captures SET capture_name = ?, ioi_root = ? WHERE capture_id = ?",
                (capture_name, ioi_root, capture_id),
            )
        else:
            cur.execute(
                "INSERT INTO captures (capture_name, ioi_root, capture_json_path) VALUES (?, ?, ?)",
                (capture_name, ioi_root, capture_json_path),
            )
            capture_id = cur.lastrowid

        try:
            capture_content = load_config_file(capture_json_path) or {}
            new_seq_names = set(get_sequences_from_data(capture_content))
        except Exception:
            new_seq_names = set()

        cur.execute(
            "SELECT sequence_id, sequence_name FROM sequences WHERE capture_id = ?",
            (capture_id,),
        )
        existing_seqs = {r["sequence_name"]: r["sequence_id"] for r in cur.fetchall()}

        # Insert any name in the JSON that's not already in the DB.
        for seq_name in new_seq_names:
            if seq_name in existing_seqs:
                continue
            seq_path = os.path.join(ioi_root or "", seq_name) if ioi_root else seq_name
            cur.execute(
                "INSERT INTO sequences (capture_id, sequence_name, sequence_path) VALUES (?, ?, ?)",
                (capture_id, seq_name, seq_path),
            )

        # Delete orphan rows that no longer appear in the JSON, but only
        # those with no associated processes — task history wins.
        for orphan_name, orphan_id in existing_seqs.items():
            if orphan_name in new_seq_names:
                continue
            cur.execute(
                "SELECT 1 FROM processes WHERE sequence_id = ? LIMIT 1",
                (orphan_id,),
            )
            if cur.fetchone() is None:
                cur.execute(
                    "DELETE FROM sequences WHERE sequence_id = ?",
                    (orphan_id,),
                )
        conn.commit()


# Back-compat alias: callers that imported the old name keep working.
# Both functions reconcile sequences, so behaviour matches the spirit
# of the original "make sure the row exists" contract.
upsert_capture_skeleton = save_capture_with_sequences


def delete_capture_by_path(capture_json_path):
    """Delete the captures row matching this path. Sequences cascade via FK;
    tasks do not (we want their saved task_jsons readable post-delete).

    Returns the task_count at the time of deletion (0 means a clean delete).
    """
    if not capture_json_path:
        return 0
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT capture_id FROM captures WHERE capture_json_path = ?",
            (capture_json_path,),
        )
        row = cur.fetchone()
        if row is None:
            return 0
        cid = row["capture_id"]
        cur.execute("SELECT COUNT(*) AS c FROM tasks WHERE capture_id = ?", (cid,))
        task_count = cur.fetchone()["c"]
        # If there are tasks, leave the captures row alone so historical task
        # rows keep resolving via JOINs. Only delete when no tasks reference it.
        if task_count == 0:
            cur.execute("DELETE FROM captures WHERE capture_id = ?", (cid,))
            conn.commit()
        return task_count


def find_process_by_log_path(path):
    """Look up a process by either of its out/err file paths. Used to
    enrich missing-log responses: if the runner skipped a (step, seq)
    via DONE sentinels, no output was written even though the row's
    out_file/err_file column points at a plausible-looking path. Knowing
    which (task, step, seq, output_id, output_path, dataset) the path
    corresponds to lets us check the DONE sentinel and tell the user
    "this was skipped" instead of just "file not found."

    Returns None if no row matches.
    """
    if not path:
        return None
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT p.process AS step, s.sequence_name AS seq_name, p.status,
                      t.task_id, t.output_id, t.output_path,
                      t.task_json_path, t.created_at
               FROM processes p
               JOIN sequences s ON p.sequence_id = s.sequence_id
               JOIN tasks t ON p.task_id = t.task_id
               WHERE p.out_file = ? OR p.err_file = ?
               LIMIT 1""",
            (path, path),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def force_delete_capture_by_path(capture_json_path):
    """Drop the captures row + all its descendants (sequences, tasks,
    processes) regardless of whether tasks reference it. The schema's
    ON DELETE CASCADE on captures→sequences and captures→tasks (and
    tasks→processes) makes this a single delete.

    Returns the runner PIDs of tasks that were still alive at deletion
    time so the caller can decide whether to refuse / signal them.
    Caller is responsible for not touching any files on disk; this is
    purely a DB operation.
    """
    if not capture_json_path:
        return {"deleted": False, "runnerPids": []}
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT capture_id FROM captures WHERE capture_json_path = ?",
            (capture_json_path,),
        )
        row = cur.fetchone()
        if row is None:
            return {"deleted": False, "runnerPids": []}
        cid = row["capture_id"]
        cur.execute(
            "SELECT task_id, dag_job_id FROM tasks WHERE capture_id = ? AND dag_job_id IS NOT NULL",
            (cid,),
        )
        runner_pids = [
            {"taskId": str(r["task_id"]), "pid": str(r["dag_job_id"])}
            for r in cur.fetchall()
        ]
        cur.execute("DELETE FROM captures WHERE capture_id = ?", (cid,))
        conn.commit()
        return {"deleted": True, "runnerPids": runner_pids}


def get_capture_by_json_path(capture_json_path):
    """Look up the capture row by absolute json path (the stable PK).
    Returns None if nothing matches."""
    if not capture_json_path:
        return None
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT capture_id, capture_name, ioi_root, capture_json_path FROM captures WHERE capture_json_path = ?",
            (capture_json_path,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_tasks_grouped_by_output_id(capture_id):
    """Group `tasks` rows by `output_id` for one capture and return a
    summary suitable for the run-groups UI: submission count, latest
    timestamp, plus the most-recent task's task_json_path/output_path
    (used to resolve the run group's actual output_dir + dataset_name).

    Sorted by latest submission desc."""
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT output_id, COUNT(*) AS submissions, MAX(created_at) AS last_at
               FROM tasks
               WHERE capture_id = ? AND output_id IS NOT NULL AND output_id != ''
               GROUP BY output_id
               ORDER BY last_at DESC""",
            (capture_id,),
        )
        groups = []
        for row in cur.fetchall():
            output_id = row["output_id"]
            cur.execute(
                """SELECT task_id, task_json_path, output_path
                   FROM tasks
                   WHERE capture_id = ? AND output_id = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (capture_id, output_id),
            )
            latest = cur.fetchone()
            groups.append({
                "outputId": output_id,
                "submissions": row["submissions"],
                "lastSubmittedAt": _iso_utc(row["last_at"]),
                "latestTaskId": str(latest["task_id"]) if latest else None,
                "latestTaskJsonPath": latest["task_json_path"] if latest else None,
                "latestOutputPath": latest["output_path"] if latest else None,
            })
        return groups


def get_task_json_path(task_id):
    """Return the saved run-config path for a task id.

    Post-2026-05-24 these live at
    ``$MAMMA_INTERFACE_DIR/run_configs/run_<id>.json``; pre-migration
    rows can still point at the old ``task_jsons/task_config_<id>.json``
    layout (the migration script rewrites these in bulk). Returns None
    when no such task exists.
    """
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT task_json_path FROM tasks WHERE task_id = ?", (task_id,))
        row = cur.fetchone()
        return row["task_json_path"] if row else None


def get_preset_path(task_id):
    """Return the source preset path recorded for a task, or None.

    NULL for legacy rows submitted before the preset_path column existed,
    and for CLI imports that didn't carry preset lineage.
    """
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT preset_path FROM tasks WHERE task_id = ?", (task_id,))
        row = cur.fetchone()
        return row["preset_path"] if row else None


def update_capture_metadata(capture_json_path, capture_name=None, ioi_root=None):
    """Sync the captures row's display name / ioi_root with the file content
    after an edit, so listings stay consistent with what's on disk."""
    if not capture_json_path:
        return
    with create_connection() as conn:
        cur = conn.cursor()
        if capture_name is not None and ioi_root is not None:
            cur.execute(
                "UPDATE captures SET capture_name = ?, ioi_root = ? WHERE capture_json_path = ?",
                (capture_name, ioi_root, capture_json_path),
            )
        elif capture_name is not None:
            cur.execute(
                "UPDATE captures SET capture_name = ? WHERE capture_json_path = ?",
                (capture_name, capture_json_path),
            )
        elif ioi_root is not None:
            cur.execute(
                "UPDATE captures SET ioi_root = ? WHERE capture_json_path = ?",
                (ioi_root, capture_json_path),
            )
        conn.commit()


def get_capture_info(capture_id):
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT capture_name, ioi_root, capture_json_path FROM captures WHERE capture_id = ?",
            (capture_id,),
        )
        r = cur.fetchone()
        if not r:
            return None
        return {
            "capture_name": r["capture_name"],
            "ioi_root": r["ioi_root"],
            "capture_json_path": r["capture_json_path"],
        }


def get_process_by_id(process_id):
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT p.process_id, p.task_id, p.cluster_job_id, p.status, p.process,
                      t.username, c.capture_name, s.sequence_name
               FROM processes p
               JOIN tasks t ON p.task_id = t.task_id
               JOIN captures c ON t.capture_id = c.capture_id
               JOIN sequences s ON p.sequence_id = s.sequence_id
               WHERE p.process_id = ?""",
            (process_id,),
        )
        r = cur.fetchone()
        if not r:
            return None
        return {
            "processId": str(r["process_id"]),
            "taskId": str(r["task_id"]),
            "pid": r["cluster_job_id"],
            "status": r["status"],
            "processType": r["process"],
            "username": r["username"],
            "captureName": r["capture_name"],
            "sequenceName": r["sequence_name"],
        }


def get_task_by_id(task_id):
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT t.task_id, t.username, t.dag_job_id, c.capture_name
               FROM tasks t JOIN captures c ON t.capture_id = c.capture_id
               WHERE t.task_id = ?""",
            (task_id,),
        )
        r = cur.fetchone()
        if not r:
            return None
        return {
            "taskId": str(r["task_id"]),
            "username": r["username"],
            "runnerPid": r["dag_job_id"],
            "captureName": r["capture_name"],
        }


def get_all_tasks_with_processes():
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT p.process_id, p.task_id, c.capture_name, c.capture_json_path,
                      s.sequence_name, p.process, p.status,
                      p.created_at, p.started_at, p.ended_at, p.cluster_job_id, p.sif_file,
                      p.out_file, p.err_file, p.num_frames, p.num_cameras,
                      t.username, t.created_at AS task_created_at, t.preset_path
               FROM processes p
               JOIN tasks t ON p.task_id = t.task_id
               JOIN captures c ON t.capture_id = c.capture_id
               JOIN sequences s ON p.sequence_id = s.sequence_id
               ORDER BY t.created_at DESC, s.sequence_name, p.created_at"""
        )
        tasks = {}
        for r in cur.fetchall():
            tid = str(r["task_id"])
            if tid not in tasks:
                tasks[tid] = {
                    "taskId": tid,
                    "captureName": r["capture_name"],
                    "captureJsonPath": r["capture_json_path"],
                    "presetPath": r["preset_path"],
                    "username": r["username"],
                    "createdAt": _iso_utc(r["task_created_at"]),
                    "sequences": {},
                }
            seq = r["sequence_name"]
            if seq not in tasks[tid]["sequences"]:
                tasks[tid]["sequences"][seq] = {
                    "seqName": seq, "processes": [],
                    "numFrames": None, "numCameras": None,
                }
            # Frame/camera counts are written on the ma_cap row; lift them to the
            # sequence level so the Tasks table can show one value per row.
            if r["num_frames"] is not None:
                tasks[tid]["sequences"][seq]["numFrames"] = r["num_frames"]
            if r["num_cameras"] is not None:
                tasks[tid]["sequences"][seq]["numCameras"] = r["num_cameras"]
            tasks[tid]["sequences"][seq]["processes"].append(
                {
                    "processId": str(r["process_id"]),
                    "processType": r["process"],
                    "status": r["status"] or "Unknown",
                    "pid": r["cluster_job_id"] or "",
                    "userId": r["username"],
                    "createdAt": _iso_utc(r["created_at"]),
                    "startedAt": _iso_utc(r["started_at"]),
                    "endedAt": _iso_utc(r["ended_at"]),
                    "imagePath": r["sif_file"] or "",
                    "outFile": r["out_file"] or "",
                    "errFile": r["err_file"] or "",
                }
            )
        result = []
        for t in tasks.values():
            t["sequences"] = list(t["sequences"].values())
            result.append(t)
        return result


def cancel_all_task_processes(task_id):
    with create_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """UPDATE processes SET status = 'Cancelled'
               WHERE task_id = ?
                 AND status NOT IN ('Completed','Failed','val_Completed','val_Failed','Done','Cancelled')""",
            (task_id,),
        )
        conn.commit()
        return cur.rowcount
