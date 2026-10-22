"""Reconcile a remote HTCondor DAG's state into a StatusSink.

The clean seam: ``run-cell`` writes ``status.jsonl`` via
:class:`inference.status.JsonlSink` with exactly the ``{step, seq, status}``
records a local run produces (plus ``event: counts`` records carrying the
ma_cap frame/camera counts). So the cluster and a local run emit an identical
status stream; the monitor just replays the remote JSONL through the same sink,
and uses the scheduler adapter's queue view (condor_q for HTCondor) only to
classify jobs still IN the queue (Idle vs Held), because completed jobs leave
the queue and cannot be read there. The adapter is injected (see
:mod:`inference.cluster.scheduler`), so a new scheduler needs no monitor
changes.

Transport and sink are injected, so this runs headless (a PrintSink and a
LocalTransport in tests) and in the GUI (a SqliteSink and an SshTransport)
without change. The class is synchronous; the GUI wraps :meth:`tick` in a
thread.
"""
from __future__ import annotations

import json
import logging
import re
import shlex
from typing import Dict, Optional, Set, Tuple

from ..status import StatusSink
from .scheduler import Scheduler, get_scheduler
from .transport import Transport

_STEP_RE = re.compile(r"^(ma_[a-z0-9]+)")


def _node_step(node: str) -> str:
    """Step name from a DAG node name.

    Plain nodes are ``<step>_<idx>``; camera fan-out adds
    ``<step>_init_<idx>``, ``<step>_<CAM>_<idx>`` and
    ``<step>_finalize_<idx>``, so stripping one ``_<suffix>`` is not
    enough. Steps follow the ``ma_<word>`` shape.
    """
    m = _STEP_RE.match(node)
    return m.group(1) if m else node.rsplit("_", 1)[0]

log = logging.getLogger(__name__)

_TERMINAL = {"Done", "Failed", "Cancelled"}


class ClusterMonitor:
    def __init__(
        self,
        transport: Transport,
        dag_cluster_id: Optional[str],
        status_jsonl: str,
        node_to_seq: Dict[str, str],
        sink: StatusSink,
        scheduler: Optional[Scheduler] = None,
        dag_lock: Optional[str] = None,
    ) -> None:
        self.tr = transport
        self.dag = str(dag_cluster_id) if dag_cluster_id else ""
        self.status_jsonl = status_jsonl
        self.node_to_seq = node_to_seq            # {"ma_2d_3": "<seq>"}
        self.sink = sink
        self.sched = scheduler or get_scheduler(None)
        # Two status protocols: a DIRECTORY of per-cell files (one writer
        # each; the fix for shared-append tearing on NFS/Lustre) read in full
        # with line-level dedup, or the legacy single shared .jsonl tailed
        # incrementally.
        self._dir_mode = not str(status_jsonl or "").endswith(".jsonl")
        self._seen_lines: Set[str] = set()
        self._offset = 0
        self._terminal: Set[Tuple[str, str]] = set()
        self._seen: Set[Tuple[str, str]] = set()   # every (step, seq) observed
        # DAGMan's .dag.lock on the shared FS: exists exactly while DAGMan
        # runs, removed on any clean exit. The authoritative liveness
        # cross-check against schedds whose condor_q serves a CACHE that can
        # come back empty under load while cells are still running.
        self.dag_lock = dag_lock or ""
        self._lock_override_streak = 0
        # finished() debouncing: right after submit the job may not be
        # visible in the queue yet (bid-scheduler latency), and a busy schedd
        # can return an empty queue view transiently. Track whether we ever
        # saw the job alive and how many consecutive probes said "gone".
        self._seen_alive = False
        self._gone_streak = 0

    # --- one reconciliation pass -----------------------------------------
    def tick(self) -> None:
        self._tick_jsonl()
        self._tick_scheduler()

    def _tick_jsonl(self) -> None:
        """The status stream is ground truth for what actually ran."""
        if self._dir_mode:
            res = self.tr.run([
                "bash", "-lc",
                f"cat {shlex.quote(self.status_jsonl)}/*.jsonl 2>/dev/null || true",
            ])
            if not res.ok:
                return
            records = []
            for line in (res.stdout or "").splitlines():
                line = line.strip()
                if not line or line in self._seen_lines:
                    continue
                self._seen_lines.add(line)
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            # Re-establish global order across the per-cell files.
            records.sort(key=lambda r: r.get("ts") or 0)
            for r in records:
                self._apply_record(r)
            return

        text, new_off = self.tr.tail(self.status_jsonl, self._offset)
        if not text:
            self._offset = new_off
            return
        if not text.endswith("\n"):
            # Consume only complete lines; a partially flushed tail line
            # would otherwise be skipped AND stepped over, losing the event.
            cut = text.rfind("\n") + 1
            if cut == 0:
                return
            text = text[:cut]
            new_off = self._offset + len(text.encode("utf-8", "replace"))
        self._offset = new_off
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn record (shared-file append collision)
            self._apply_record(r)

    def _apply_record(self, r: dict) -> None:
        step, seq = r.get("step"), r.get("seq")
        if not step or not seq:
            return
        if r.get("event") == "counts":
            self.sink.record_counts(step, seq, r.get("num_frames"),
                                    r.get("num_cameras"))
            return
        status = r.get("status")
        if not status:
            return
        self._seen.add((step, seq))
        self.sink.update(step, seq, status, pid=r.get("pid"))
        if status in _TERMINAL:
            self._terminal.add((step, seq))
        else:
            # A DAGMan retry, or the next fan-out cell of this (step, seq),
            # reopens it for the queue view and finalize_missing.
            self._terminal.discard((step, seq))

    def _tick_scheduler(self) -> None:
        """The scheduler's queue view fills in-flight scheduling state for
        cells the JSONL has not yet moved to a terminal state (Queued while
        idle, Failed while Held)."""
        if not self.dag:
            return
        res = self.tr.run(self.sched.queue_cmd(self.dag))
        if not res.ok or not (res.stdout or "").strip():
            return
        for node, status, note in self.sched.parse_queue(res.stdout):
            seq = self.node_to_seq.get(node)
            if not seq:
                continue
            step = _node_step(node)
            if (step, seq) in self._terminal:      # JSONL already decided this cell
                continue
            self._seen.add((step, seq))
            self.sink.update(step, seq, status)
            if note:
                log.warning("%s[%s]: %s", step, seq, note)

    def finalize_missing(self, status: str = "Cancelled") -> int:
        """Mark cells that never reached a terminal state.

        Call once the run has verifiably ended: a cell still Running or
        Queued at that point was killed out-of-band (condor_rm from a
        terminal, a preemption without requeue) and will never write its
        own terminal line, so the board would show it Running forever.
        Returns the number of cells downgraded.
        """
        stuck = sorted(self._seen - self._terminal)
        for step, seq in stuck:
            self.sink.update(step, seq, status)
            self._terminal.add((step, seq))
        return len(stuck)

    # Consecutive "gone" probes needed before declaring the run finished.
    # After the job was seen alive, 2 (one blip forgiven). Never seen alive:
    # 6 (grace for submit-to-queue visibility latency: the DAG may not be in
    # condor_q yet right after condor_submit_dag_bid returns, and a single
    # probe would stop tracking a run that keeps executing on the cluster).
    GONE_STREAK_SEEN = 2
    GONE_STREAK_UNSEEN = 6
    # Queue says gone but the DAG lock is still on disk: keep polling this
    # many extra probes (~5 min at the GUI cadence) before trusting the
    # queue anyway (covers a crashed DAGMan leaving a stale lock behind).
    LOCK_OVERRIDE_PROBES = 30

    # --- termination ------------------------------------------------------
    def finished(self) -> bool:
        """True once the run has verifiably ended: left the queue AND, when a
        DAG lock path is known, DAGMan's lock is gone from the shared FS."""
        if not self.dag:
            return True
        res = self.tr.run(self.sched.alive_cmd(self.dag))
        if not res.ok:
            return False              # transport/schedd hiccup: keep polling
        if self.sched.parse_alive(res.stdout or ""):
            self._seen_alive = True
            self._gone_streak = 0
            self._lock_override_streak = 0
            return False
        self._gone_streak += 1
        needed = self.GONE_STREAK_SEEN if self._seen_alive else self.GONE_STREAK_UNSEEN
        if self._gone_streak < needed:
            return False
        if not self.dag_lock:
            return True
        try:
            lock_alive = self.tr.exists(self.dag_lock)
        except Exception:  # noqa: BLE001 - transport hiccup: keep polling
            return False
        if not lock_alive:
            return True
        # Queue view empty but DAGMan still holds its lock: the schedd's
        # cached condor_q is stale. Keep tracking.
        self._lock_override_streak += 1
        if self._lock_override_streak == 1:
            log.warning("queue reports the DAG gone but %s still exists; "
                        "trusting the lock and continuing to poll", self.dag_lock)
        return self._lock_override_streak >= self.LOCK_OVERRIDE_PROBES


def reconcile(transport: Transport, status_dir: str, dag_lock: str,
              sink: StatusSink) -> dict:
    """One-shot re-sync of a run's board rows from the cluster's durable state.

    The live tracker (runner_main's monitor loop) is a mortal local process:
    a backend restart, reboot, or an out-of-band DAG resubmission leaves the
    run progressing on the cluster with nobody streaming its status back, and
    the board freezes on the last state it heard. The
    per-cell status files under ``dag/<task>/status/`` survive all of that,
    including resubmissions, so replaying them through the same sink restores
    the truth. When DAGMan's lock is gone (nothing will write further), cells
    still non-terminal afterwards are downgraded to Cancelled.

    Returns ``{"alive", "seen", "downgraded"}``.
    """
    try:
        have_status = transport.exists(status_dir)
    except Exception:  # noqa: BLE001
        have_status = False
    if not have_status:
        # A just-submitted task has no dag dir yet (and a cleaned-up one has
        # lost it): there is nothing to replay, and finalizing rows on absent
        # evidence would cancel a run that is about to start.
        return {"alive": None, "seen": 0, "downgraded": 0,
                "note": "status dir absent; left untouched"}
    mon = ClusterMonitor(transport, None, status_dir, {}, sink)
    mon._tick_jsonl()
    alive = False
    if dag_lock:
        try:
            alive = transport.exists(dag_lock)
        except Exception:  # noqa: BLE001 - transport hiccup: don't finalize on doubt
            alive = True
    downgraded = 0 if alive else mon.finalize_missing()
    return {"alive": alive, "seen": len(mon._seen), "downgraded": downgraded}
