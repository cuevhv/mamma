"""Status sinks for the runner.

Two sinks ship with the package:

* :class:`PrintSink` — human-readable lines on stdout.
* :class:`JsonlSink` — one JSON object per status transition appended to a
  file. Easy to consume from CI, ``jq``, or external monitoring.

Custom sinks (e.g. a database writer) just subclass :class:`StatusSink` and
override :meth:`update`.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Iterable, Optional, TextIO

log = logging.getLogger(__name__)


class StatusSink:
    """Write status updates for ``(step, sequence)`` cells of one task."""

    def update(
        self,
        step_name: str,
        seq_name: str,
        status: str,
        pid: Optional[int] = None,
    ) -> None:
        raise NotImplementedError

    def cancel_remaining(
        self, step_names: Iterable[str], seq_names: Iterable[str]
    ) -> None:
        for step_name in step_names:
            for seq in seq_names:
                self.update(step_name, seq, "Cancelled")

    def record_counts(
        self,
        step_name: str,
        seq_name: str,
        num_frames: Optional[int],
        num_cameras: Optional[int],
    ) -> None:
        """Persist actual frame/camera counts for a (step, sequence).

        No-op by default; the GUI's DB-backed sink overrides this to store the
        counts the runner derives after ma_cap. Keeping it on the base means the
        runner can call it unconditionally without caring which sink is wired."""


class PrintSink(StatusSink):
    """Print a one-line status update to stdout (plus the package logger)."""

    def update(self, step_name, seq_name, status, pid=None):
        extra = f" pid={pid}" if pid else ""
        log.info("status: %s[%s] -> %s%s", step_name, seq_name, status, extra)


class JsonlSink(StatusSink):
    """Append one JSON line per transition to ``path``.

    Each line:

        {"ts": <unix_seconds>, "step": "<name>", "seq": "<name>",
         "status": "<state>", "pid": <int|null>}

    The file is opened in line-buffered append mode so external readers
    (``tail -f``, CI log parsers) see updates as they happen.
    """

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._fp: TextIO = open(path, "a", buffering=1)

    def update(self, step_name, seq_name, status, pid=None):
        record = {
            "ts": time.time(),
            "step": step_name,
            "seq": seq_name,
            "status": status,
            "pid": pid,
        }
        self._fp.write(json.dumps(record) + "\n")
        log.info("status: %s[%s] -> %s%s", step_name, seq_name, status,
                 f" pid={pid}" if pid else "")

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


def make_sink(jsonl_path: Optional[str] = None) -> StatusSink:
    """Build a :class:`PrintSink` (default) or a :class:`JsonlSink`."""
    if jsonl_path:
        return JsonlSink(jsonl_path)
    return PrintSink()
