"""MAMMA local pipeline runner.

Run from the repo root::

    python -m inference.cli.run --task my_task.json -v

Or import the runner from a script::

    from inference import runner
    runner.run_dag("path/to/task.json")
"""
__version__ = "1.1.0"

from . import runner  # noqa: F401  (re-export)
