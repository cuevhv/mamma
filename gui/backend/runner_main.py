"""Runner entry point spawned by the GUI's Flask app.

This is a thin shim that:

1. Adds the parent repo (``mamma_release/``) to ``sys.path`` so the
   ``inference`` package imports cleanly.
2. Calls :func:`inference.env.bootstrap_env` to populate ``MAMMA_*``
   defaults and apply any ``.env.local`` override.
3. Parses argv from the Flask layer.
4. Builds a :class:`SqliteSink` bound to the submitted ``task_id`` so
   the runner's status transitions land in ``gui/var/mamma.sqlite``.
5. Hands off to :func:`inference.runner.run_dag` (or
   :func:`inference.runner.run_step` when ``--step`` is set).

Invoked from ``gui/backend/app.py`` as::

    python -m runner_main --task <path> --task-id <int> \
        --out-tag <output_id> --log-tag <task_id>

with ``cwd=gui/backend`` so module imports (``db``, ``sinks``,
``objects.*``) resolve via the cwd-on-path convention the GUI already
uses.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make ``inference`` importable from the parent repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.env import bootstrap_env  # noqa: E402

bootstrap_env()

from inference.runner import run_dag, run_step  # noqa: E402
from sinks import SqliteSink  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GUI-side runner shim.")
    p.add_argument("--task", required=True, help="Path to a task.json / task.yaml.")
    p.add_argument("--task-id", type=int, required=True, dest="task_id",
                   help="DB task_id; the SqliteSink writes statuses against it.")
    p.add_argument("--out-tag", default=None, dest="out_tag",
                   help="Tag used in output/<step>/<tag>/... (typically task.output_id).")
    p.add_argument("--log-tag", default=None, dest="log_tag",
                   help="Tag used in jobs_log_dir/<user>/<tag>/... (typically str(task_id)).")
    p.add_argument("--step", default=None,
                   help="If set, run a single step instead of the full DAG.")
    p.add_argument("--force", action="store_true",
                   help="Ignore DONE sentinels and re-run every (step, seq).")
    p.add_argument("--cluster-profile", default=None, dest="cluster_profile",
                   help="If set, submit to this remote cluster profile and track "
                        "it, instead of running the pipeline locally.")
    return p.parse_args()


def _run_cluster(args, sink) -> int:
    """Submit the run to a remote cluster and reconcile its status into the DB.

    Runs on the GUI host (the client): sync code, upload the run-config, submit
    the DAG over SSH, then poll status.jsonl + condor_q through the SAME
    SqliteSink a local run uses, so the frontend renders it identically. SIGTERM
    (from the GUI's Stop) condor_rm's the DAG before exiting."""
    import json
    import signal
    import tempfile
    import time

    from inference.cluster import profile as profile_mod
    from inference.cluster import remote as remote_mod
    from inference.cluster.monitor import ClusterMonitor

    prof = profile_mod.load_profile(args.cluster_profile)
    print(f"[cluster] submitting task {args.task_id} to profile "
          f"'{prof.name}' ({prof.username}@{prof.host})", flush=True)

    # Place outputs + per-cell logs under the cluster workspace (alongside the
    # run_configs/ and dag/ dirs submit_remote creates), not inside the synced
    # repo checkout. The GUI can't know the workspace path at submit time, but
    # the profile does, so we rewrite the run-config here before it is uploaded.
    task_path = args.task
    ws = (prof.workspace_root or "").rstrip("/")
    if ws:
        with open(args.task) as _f:
            _cfg = json.load(_f)
        _g = _cfg.setdefault("global", {})
        # Safety net for configs from other sources: RELATIVE paths (the
        # local default tree the cluster cannot see) are anchored at the
        # workspace. ABSOLUTE paths are the author's explicit choice (a
        # user-picked shared dir, or start_task's resolved root) and pass
        # through untouched (this rewrite used to clobber an explicitly
        # given absolute output dir).
        rewritten = False
        if not os.path.isabs(str(_g.get("out_dir") or "")):
            _g["out_dir"] = f"{ws}/output"
            rewritten = True
        if not os.path.isabs(str(_g.get("jobs_log_dir") or "")):
            _g["jobs_log_dir"] = f"{ws}/logs"
            rewritten = True
        if rewritten:
            _fd, task_path = tempfile.mkstemp(
                prefix=f"mamma_cluster_{args.task_id}_", suffix=".json")
            with os.fdopen(_fd, "w") as _f:
                json.dump(_cfg, _f, indent=2)

    handle = remote_mod.submit_remote(
        prof, task_path,
        out_tag=str(args.out_tag), log_tag=str(args.log_tag),
    )
    print(f"[cluster] DAG {handle.dag_cluster_id}, {len(handle.node_to_seq)} nodes, "
          f"status log {handle.status_jsonl}", flush=True)

    from inference.cluster.scheduler import get_scheduler
    sched = get_scheduler(handle.scheduler)
    transport = remote_mod.make_transport(prof)
    stop = {"flag": False}

    def _on_cancel(_signum, _frame):
        """Intentional Stop (SIGUSR1 from the GUI): cancel the remote DAG."""
        stop["flag"] = True
        if handle.dag_cluster_id:
            cancel = sched.cancel_cmd(handle.dag_cluster_id)
            print(f"[cluster] stop requested; {' '.join(cancel)}", flush=True)
            try:
                transport.run(cancel)
            except Exception as e:  # noqa: BLE001
                print(f"[cluster] cancel failed: {e}", flush=True)

    def _on_detach(_signum, _frame):
        """Incidental SIGTERM/SIGINT (shutdown, reboot, kill): the DAG lives
        in HTCondor independent of this client, so detach WITHOUT cancelling
        (task 135 post-mortem: an OS-update reboot condor_rm'd a healthy
        run). Stop it from the GUI (SIGUSR1) or condor_rm to really cancel."""
        stop["flag"] = True
        print("[cluster] SIGTERM/SIGINT: detaching; the DAG keeps running "
              "on the cluster (use the GUI Stop to cancel it)", flush=True)

    signal.signal(signal.SIGUSR1, _on_cancel)
    signal.signal(signal.SIGTERM, _on_detach)
    signal.signal(signal.SIGINT, _on_detach)

    monitor = ClusterMonitor(
        transport, handle.dag_cluster_id, handle.status_jsonl,
        handle.node_to_seq, sink, scheduler=sched,
        dag_lock=(handle.dag_path + ".lock") if handle.dag_path else None,
    )
    while not stop["flag"]:
        monitor.tick()
        if monitor.finished():
            monitor.tick()  # final drain of status.jsonl
            break
        time.sleep(10.0)
    # Cells killed out-of-band (condor_rm, preemption) never stream a
    # terminal state; downgrade them so the board cannot show a dead run
    # as Running forever.
    stuck = monitor.finalize_missing()
    if stuck:
        print(f"[cluster] marked {stuck} unfinished cell(s) Cancelled", flush=True)
    print(f"[cluster] task {args.task_id} finished (stopped={stop['flag']})", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    sink = SqliteSink(args.task_id)
    if args.cluster_profile:
        return _run_cluster(args, sink)
    if args.step:
        return run_step(
            args.task,
            args.step,
            out_tag=args.out_tag,
            log_tag=args.log_tag,
            sink=sink,
            force=args.force,
        )
    return run_dag(
        args.task,
        out_tag=args.out_tag,
        log_tag=args.log_tag,
        sink=sink,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
