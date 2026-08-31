"""``python -m inference submit-cluster`` — generate + submit an HTCondor DAG.

Phase 0 entry point: run this ON a cluster login node with a materialized
run-config. It renders one node per ``(step, sequence)`` via
:mod:`inference.cluster.htcondor` and submits with
``condor_submit_dag[_bid]``. The client-side SSH transport (a later phase)
invokes this same command remotely.

Use ``--dry-run`` to generate the DAG artifacts without submitting (works with
no HTCondor installed, e.g. for inspection or CI).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .. import config
from ..cluster import htcondor
from ..cluster.scheduler import SCHEDULERS, get_scheduler
from ..env import bootstrap_env, repo_root
from .run import _configure_logging


def _write_handle(work_dir: str, bundle: "htcondor.DagBundle", status_jsonl: str,
                  scheduler_name: str) -> str:
    """Persist a machine-readable handle the client fetches after submit.

    Single source of truth for the DAG id, the node->seq map, and which
    scheduler submitted the run (computed here on the cluster, not re-derived
    by the client)."""
    handle = {
        "dag_cluster_id": bundle.dag_cluster_id,
        "dag_path": bundle.dag_path,
        "status_jsonl": status_jsonl,
        "work_dir": work_dir,
        "node_to_seq": bundle.node_to_seq,
        "scheduler": scheduler_name,
    }
    path = os.path.join(work_dir, "handle.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(handle, f, indent=2)
    return path


def _resolve_username(global_cfg: dict) -> str:
    return (
        global_cfg.get("username")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or "local"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m inference submit-cluster",
        description="Generate and submit an HTCondor DAG for a run-config.",
    )
    p.add_argument("--task", required=True,
                   help="Path to a materialized run-config (.json / .yaml)")
    p.add_argument("--out-tag", default="local")
    p.add_argument("--log-tag", default=None, help="Defaults to --out-tag")
    p.add_argument("--work-dir", default=None,
                   help="Where to write .sh/.sub/.dag + status.jsonl. "
                        "Default: <global.jobs_log_dir>/<user>/<log-tag>/dag")
    p.add_argument("--repo-dir", default=None,
                   help="Repo root on the execute node the wrapper cd's into "
                        "(default: this repo root)")
    p.add_argument("--activate", default="",
                   help="Shell snippet to activate the runner env on the node "
                        "(e.g. 'conda activate mamma')")
    p.add_argument("--python", default="python",
                   help="Execute-node python interpreter (e.g. 'python3')")
    p.add_argument("--status-jsonl", default=None,
                   help="Status stream the poller reads. A directory (the "
                        "default: <work-dir>/status) holds one file per "
                        "(step, seq) cell: safe for concurrent nodes on a "
                        "shared FS. A path ending in .jsonl keeps the legacy "
                        "single shared file")
    p.add_argument("--scheduler", choices=sorted(SCHEDULERS), default="htcondor",
                   help="Batch scheduler adapter to submit with")
    p.add_argument("--flavor", choices=["bid", "standard"], default="bid")
    p.add_argument("--bid", type=int, default=100)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--dry-run", action="store_true",
                   help="Generate the DAG artifacts but do not submit")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def main(argv=None) -> None:
    bootstrap_env()
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        task_cfg = config.load_run_config(args.task)
    except (FileNotFoundError, ValueError, config.TaskConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    log_tag = args.log_tag or args.out_tag
    global_cfg = task_cfg.get("global", {})
    user = _resolve_username(global_cfg)

    work_dir = args.work_dir
    if work_dir is None:
        base = os.path.expanduser(os.path.expandvars(
            global_cfg.get("jobs_log_dir", "") or "~/.mamma/logs"
        ))
        work_dir = os.path.join(base, user, log_tag, "dag")
    work_dir = os.path.abspath(work_dir)
    status_jsonl = args.status_jsonl or os.path.join(work_dir, "status")

    opts = htcondor.CondorOpts(
        flavor=args.flavor,
        bid_price=args.bid,
        retries=args.retries,
        repo_dir=os.path.abspath(args.repo_dir) if args.repo_dir else str(repo_root()),
        activate=args.activate,
        remote_task_path=os.path.abspath(args.task),
        status_jsonl=status_jsonl,
        python=args.python,
    )

    sched = get_scheduler(args.scheduler)
    try:
        bundle = sched.emit(task_cfg, args.out_tag, log_tag, opts, work_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"DAG written: {bundle.dag_path}")
    print(f"Nodes: {len(bundle.node_to_seq)}   status.jsonl: {status_jsonl}")

    if args.dry_run:
        _write_handle(work_dir, bundle, status_jsonl, sched.name)
        print("dry-run: not submitting.")
        return

    try:
        sched.submit(bundle, opts)
    except FileNotFoundError:
        _write_handle(work_dir, bundle, status_jsonl, sched.name)
        print(
            f"error: {sched.name} submit command not found. Are you on a "
            "cluster login node? (re-run with --dry-run to only generate "
            "the DAG)",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"error: {sched.name} submit failed (exit {e.returncode}): "
              f"{(e.stderr or '').strip()}", file=sys.stderr)
        sys.exit(1)

    _write_handle(work_dir, bundle, status_jsonl, sched.name)
    print(f"Submitted. DAG cluster id: {bundle.dag_cluster_id}")


if __name__ == "__main__":
    main()
