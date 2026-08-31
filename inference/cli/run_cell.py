"""``python -m inference run-cell`` — execute ONE ``(step, sequence)`` cell.

This is the unit a single HTCondor DAG node runs. Its execution path is
identical to the DAG walker's inner loop (``engines.dispatch``); the cluster
just invokes it once per node instead of looping, so a cluster cell runs
byte-identically to the same cell locally.

``--seq`` is a resolved sequence NAME (the emitter already ran
``resolve_seq_names``), so this does no capture lookup.
"""
from __future__ import annotations

import argparse
import sys

from .. import runner
from ..env import bootstrap_env
from ..status import make_sink
from .run import _configure_logging


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m inference run-cell",
        description="Run a single (step, sequence) cell (one DAG node).",
    )
    p.add_argument("--task", required=True, help="Path to task.json / run-config")
    p.add_argument("--step", required=True, choices=runner.ALL_STEPS,
                   help="Which step to run")
    p.add_argument("--seq", required=True,
                   help="Sequence NAME (already resolved, not a seq_id)")
    p.add_argument("--out-tag", default=None)
    p.add_argument("--log-tag", default=None)
    p.add_argument("--force", action="store_true",
                   help="Run even if a DONE sentinel exists")
    p.add_argument("--status-jsonl", default=None,
                   help="Append status transitions here (the poller tails this)")
    p.add_argument("--cell-id", default="",
                   help="Sub-cell identifier for camera fan-out (namespaces "
                        "the DONE sentinel); empty for a whole-sequence cell")
    p.add_argument("--cell-flags", default="",
                   help="Extra step flags for this cell (camera fan-out "
                        "passes --cam_names/--cam_init here)")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def main(argv=None) -> None:
    bootstrap_env()
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    sink = make_sink(args.status_jsonl)
    rc = runner.run_cell(
        args.task,
        args.step,
        args.seq,
        out_tag=args.out_tag,
        log_tag=args.log_tag,
        sink=sink,
        force=args.force,
        cell_id=args.cell_id or None,
        cell_flags=args.cell_flags or None,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
