"""SQLite-backed StatusSink for the runner.

When a GUI-submitted task runs, ``runner_main.py`` builds a ``SqliteSink``
bound to ``task_id`` and hands it to :func:`inference.runner.run_dag`.
The runner emits ``update(step, seq, status, pid)`` for each cell and we
write the corresponding row in the ``processes`` table.

The sink keeps an in-memory ``(step, seq) -> process_id`` index built
from ``db.get_processes_for_task`` so we don't re-query on every
transition. Inserts are done at submit time by ``db.create_entry``; the
runner only updates.
"""
from __future__ import annotations

from typing import Optional

from inference.status import StatusSink

import db  # gui/backend/db.py — on sys.path because cwd is gui/backend/


class SqliteSink(StatusSink):
    """Write status updates to the project's SQLite DB via ``db.py``."""

    def __init__(self, task_id: int) -> None:
        self.task_id = task_id
        self._index: dict[tuple[str, str], int] = {}
        for row in db.get_processes_for_task(task_id):
            self._index[(row["process"], row["sequence_name"])] = row["process_id"]

    def update(
        self,
        step_name: str,
        seq_name: str,
        status: str,
        pid: Optional[int] = None,
    ) -> None:
        pid_arg = str(pid) if pid is not None else None
        proc_id = self._index.get((step_name, seq_name))
        if proc_id is None:
            print(
                f"[status] WARN: no process row for {step_name}[{seq_name}] "
                f"in task {self.task_id}"
            )
            return
        db.set_process_status(proc_id, status, pid=pid_arg)
        extra = f" pid={pid}" if pid else ""
        print(f"[status] {step_name}[{seq_name}] -> {status}{extra}", flush=True)

    def record_counts(
        self,
        step_name: str,
        seq_name: str,
        num_frames: Optional[int],
        num_cameras: Optional[int],
    ) -> None:
        proc_id = self._index.get((step_name, seq_name))
        if proc_id is None:
            return
        db.set_process_metadata(proc_id, num_frames=num_frames, num_cameras=num_cameras)
        print(
            f"[status] {step_name}[{seq_name}] counts frames={num_frames} "
            f"cameras={num_cameras}",
            flush=True,
        )
