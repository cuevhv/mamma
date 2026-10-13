"""Re-sync cluster-task board rows from the cluster's durable status files.

The live tracker (runner_main's monitor loop) is a mortal local process: a
backend restart, a reboot, or an out-of-band DAG resubmission leaves the run
progressing on the cluster with nobody streaming its status back, and the
board freezes on the last state it heard. The cluster
keeps the whole truth in ``dag/<task_id>/status/`` across resubmissions, so
replaying it through the same SqliteSink the tracker uses restores the rows.

Used by the backend's startup pass (every stale cluster task) and by
``POST /api/tasks/<id>/reconcile`` (the task's "Re-sync from cluster"
button).
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import db  # noqa: E402  (cwd-on-path convention, like the rest of gui/backend)
from sinks import SqliteSink  # noqa: E402


def reconcile_task(task_id: int) -> dict:
    """Replay one cluster task's status files into its board rows.

    Returns the summary from :func:`inference.cluster.monitor.reconcile`
    plus ``task_id``, or ``{"error": ...}`` for a non-cluster task.
    """
    _backend, profile_name = db.get_task_backend(task_id)
    if not profile_name:
        return {"task_id": task_id, "error": "not a cluster task"}

    from inference.cluster import profile as profile_mod
    from inference.cluster import remote as remote_mod
    from inference.cluster.monitor import reconcile

    prof = profile_mod.load_profile(profile_name)
    tr = remote_mod.make_transport(prof)
    ws = (prof.workspace_root or "").rstrip("/")
    dag_dir = f"{ws}/dag/{task_id}"
    res = reconcile(tr, f"{dag_dir}/status", f"{dag_dir}/{task_id}.dag.lock",
                    SqliteSink(task_id))
    if res.get("alive") is False:
        # Cells that never wrote a status line (steps that never started)
        # exist only as board rows; the replay cannot see them, so sweep
        # whatever is still non-terminal now that the DAG is verifiably dead.
        terminal = {"Done", "Completed", "Failed", "Cancelled",
                    "val_Completed", "val_Failed"}
        extra = 0
        for row in db.get_processes_for_task(task_id):
            if row["status"] not in terminal:
                db.set_process_status(row["process_id"], "Cancelled")
                extra += 1
        res["downgraded"] = res.get("downgraded", 0) + extra
    res["task_id"] = task_id
    return res


def reconcile_stale_tasks() -> list:
    """Startup pass: re-sync every cluster task with non-terminal rows.

    One task's failure (deleted dag dir, unreachable profile) never blocks
    the others or the backend's startup.
    """
    out = []
    for t in db.get_unfinished_tasks_info():
        tid = t["task_id"]
        _backend, profile_name = db.get_task_backend(tid)
        if not profile_name:
            continue  # local runs are owned by their own live runner
        try:
            out.append(reconcile_task(tid))
        except Exception as e:  # noqa: BLE001
            out.append({"task_id": tid, "error": str(e)})
    return out
