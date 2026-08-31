"""Map an HTCondor job state to a GUI Process status, for jobs still IN the queue.

``condor_q`` lists only in-flight jobs; Completed jobs leave the queue
immediately, so success / failure is NOT read here. The pipeline's own
``status.jsonl`` (written by ``run-cell`` via
:class:`inference.status.JsonlSink`) is authoritative for
``Running`` / ``Done`` / ``Failed`` / ``Cancelled``. This table only
classifies queued jobs: ``Queued`` (idle, waiting for a slot) versus a stuck
``Held`` job that must be surfaced.

Kept as a pure function so the poller can be tested without a cluster.
"""
from __future__ import annotations

from typing import Optional, Tuple

# HTCondor JobStatus attribute codes.
IDLE = 1
RUNNING = 2
REMOVED = 3
COMPLETED = 4
HELD = 5
TRANSFERRING_OUTPUT = 6
SUSPENDED = 7


def map_condor_state(
    job_status: int,
    hold_reason: Optional[str] = None,
) -> Optional[Tuple[str, Optional[str]]]:
    """Return ``(gui_status, note)`` for an in-queue job, or ``None`` to defer
    to the ``status.jsonl`` / DONE ground truth.

    ``note`` carries a human-readable reason (e.g. a HELD reason) or ``None``.
    """
    if job_status == IDLE:
        return ("Queued", None)
    if job_status in (RUNNING, TRANSFERRING_OUTPUT, SUSPENDED):
        return ("Running", None)
    if job_status == HELD:
        return ("Failed", f"HTCondor HELD: {hold_reason or 'unknown reason'}")
    if job_status == REMOVED:
        return ("Cancelled", None)
    # COMPLETED and anything else: the output check in status.jsonl decides.
    return None
