"""``python -m inference submit-remote`` — submit a run to a remote cluster.

Client-side entry: loads a cluster profile, materializes / uploads the
run-config, syncs code, submits the DAG on the login node over the transport,
and (with ``--watch``) tails the run to completion, reconciling per-cell status
to stdout. Headless; the GUI drives the same :mod:`inference.cluster.remote`
and :mod:`inference.cluster.monitor` code.
"""
from __future__ import annotations

import argparse
import sys
import time

from .. import config
from ..cluster import profile as profile_mod
from ..cluster import remote as remote_mod
from ..cluster.monitor import ClusterMonitor
from ..env import bootstrap_env
from ..status import StatusSink
from .run import _configure_logging, _persist_temp


class _PrintSink(StatusSink):
    """Always-visible per-cell status line (independent of logging config)."""

    def update(self, step_name, seq_name, status, pid=None):
        print(f"[{status:<9}] {step_name}[{seq_name}]", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m inference submit-remote",
        description="Submit a pipeline run to a remote HTCondor cluster.",
    )
    p.add_argument("--profile", required=True,
                   help="Cluster profile name (in ~/.config/mamma/clusters) or a path")
    p.add_argument("--task", default=None,
                   help="Path to a materialized run-config. Mutually exclusive with --cfg/--capture.")
    p.add_argument("--cfg", "--preset", dest="cfg", default=None,
                   help="Preset to materialize with --capture")
    p.add_argument("--capture", default=None, help="Capture JSON (requires --cfg)")
    p.add_argument("--out-tag", default="cluster")
    p.add_argument("--log-tag", default=None, help="Defaults to --out-tag")
    p.add_argument("--bid", type=int, default=None, help="Override the profile bid price")
    p.add_argument("--retries", type=int, default=None)
    p.add_argument("--no-sync", action="store_true", help="Skip the code rsync")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate the DAG remotely but do not submit")
    p.add_argument("--watch", action="store_true",
                   help="Tail the run to completion, printing per-cell status")
    p.add_argument("--interval", type=float, default=5.0,
                   help="Poll interval in seconds for --watch (default 5)")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def _resolve_task_path(args) -> str:
    if args.task and (args.cfg or args.capture):
        raise SystemExit("error: --task is mutually exclusive with --cfg/--capture")
    if args.task:
        # Validate on read (load_run_config validates).
        config.load_run_config(args.task)
        return args.task
    if args.cfg and args.capture:
        cfg = config.materialize_run_config(args.cfg, args.capture)
        config.validate(cfg)
        return _persist_temp(cfg)
    raise SystemExit("error: provide --task <run-config>, or --cfg <preset> --capture <capture.json>")


def main(argv=None) -> None:
    bootstrap_env()
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        prof = profile_mod.load_profile(args.profile)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        task_path = _resolve_task_path(args)
    except (FileNotFoundError, ValueError, config.TaskConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Submitting to cluster '{prof.name}' ({prof.username}@{prof.host}) ...")
    try:
        handle = remote_mod.submit_remote(
            prof, task_path,
            out_tag=args.out_tag, log_tag=args.log_tag,
            bid=args.bid, retries=args.retries,
            dry_run=args.dry_run, do_sync=not args.no_sync,
        )
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  work dir:   {handle.work_dir}")
    print(f"  nodes:      {len(handle.node_to_seq)}")
    print(f"  DAG id:     {handle.dag_cluster_id}")
    print(f"  status log: {handle.status_jsonl}")

    if args.dry_run:
        print("dry-run: DAG generated remotely, not submitted.")
        return
    if not args.watch:
        print("Submitted. Re-run with --watch to follow, or track from the GUI.")
        return
    if not handle.dag_cluster_id:
        print("warning: no DAG id parsed; cannot watch.", file=sys.stderr)
        return

    from ..cluster.scheduler import get_scheduler
    tr = remote_mod.make_transport(prof)
    mon = ClusterMonitor(tr, handle.dag_cluster_id, handle.status_jsonl,
                         handle.node_to_seq, _PrintSink(),
                         scheduler=get_scheduler(handle.scheduler),
                         dag_lock=(handle.dag_path + ".lock") if handle.dag_path else None)
    print("Watching (Ctrl-C to stop watching; the run keeps going) ...")
    try:
        while True:
            mon.tick()
            if mon.finished():
                mon.tick()  # final drain
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped watching. The DAG continues on the cluster.")
        return
    print("Run finished on the cluster.")


if __name__ == "__main__":
    main()
