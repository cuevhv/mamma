"""Module entry point: ``python -m inference <command> [options]``.

This is the canonical, cross-platform way to invoke the MAMMA local
pipeline runner without installing anything. Each subcommand re-exports
the existing CLI module under :mod:`inference.cli`; this file is just a
thin dispatcher.

Subcommands::

    python -m inference run        --task my.json --out-tag run01 -v
    python -m inference run-step   --task my.json --step ma_2d -v
"""
from __future__ import annotations

import sys

from . import __version__
from .assets import dump_env_example
from .env import bootstrap_env

USAGE = """\
usage: python -m inference <command> [options]

MAMMA local pipeline runner.

commands:
  run                Run the full pipeline DAG end-to-end
  run-step           Run a single pipeline step
  run-cell           Run a single (step, sequence) cell (one DAG node)
  submit-cluster     Generate + submit an HTCondor DAG (run on a login node)
  submit-remote      Submit a run to a remote cluster over SSH + track it
  cluster            Manage cluster profiles (list / show / add / test)
  doctor             Pre-flight validator (env, step reqs, optional task file)
  dump-env-example   Print/write a `.env.example` template from the registry

Use `python -m inference <command> --help` for command-specific options.

global options:
  -h, --help  Show this help message and exit
  --version   Print version and exit
"""

_DUMP_ENV_USAGE = """\
usage: python -m inference dump-env-example [-o FILE]

Render a `.env.example` template from inference/assets.py:ASSETS.
Writes to stdout by default; use ``-o FILE`` to write to disk.

options:
  -o, --output FILE   Write the template to FILE instead of stdout
  -h, --help          Show this help message and exit
"""


def _dump_env_example_main(argv):
    out = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            sys.stdout.write(_DUMP_ENV_USAGE)
            sys.exit(0)
        if a in ("-o", "--output"):
            if i + 1 >= len(argv):
                sys.stderr.write(f"error: {a} requires a path argument\n")
                sys.exit(2)
            out = argv[i + 1]
            i += 2
            continue
        sys.stderr.write(f"error: unknown option {a!r}\n\n")
        sys.stderr.write(_DUMP_ENV_USAGE)
        sys.exit(2)
    body = dump_env_example()
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(body)
    else:
        sys.stdout.write(body)


def main(argv=None) -> None:
    bootstrap_env()
    argv = sys.argv[1:] if argv is None else list(argv)

    if not argv or argv[0] in {"-h", "--help"}:
        sys.stdout.write(USAGE)
        sys.exit(0)
    if argv[0] == "--version":
        sys.stdout.write(f"mamma_pipeline {__version__}\n")
        sys.exit(0)

    cmd, rest = argv[0], argv[1:]
    # Lazy per-command imports keep the cluster-side commands (submit-cluster,
    # run-cell) from pulling the client-only CLIs (submit-remote, cluster),
    # which depend on yaml / the local env. That lets those two run on a bare
    # cluster python3 with no third-party packages installed.
    if cmd == "run":
        from .cli import run
        run.main(rest)
    elif cmd == "run-step":
        from .cli import run_step
        run_step.main(rest)
    elif cmd == "run-cell":
        from .cli import run_cell
        run_cell.main(rest)
    elif cmd == "submit-cluster":
        from .cli import submit_cluster
        submit_cluster.main(rest)
    elif cmd == "submit-remote":
        from .cli import submit_remote
        submit_remote.main(rest)
    elif cmd == "cluster":
        from .cli import cluster
        cluster.main(rest)
    elif cmd == "doctor":
        from .cli import doctor
        doctor.main(rest)
    elif cmd == "dump-env-example":
        _dump_env_example_main(rest)
    else:
        sys.stderr.write(f"error: unknown command {cmd!r}\n\n")
        sys.stderr.write(USAGE)
        sys.exit(2)


if __name__ == "__main__":
    main()
