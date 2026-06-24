"""DAG walker.

Topo-sorts the steps enabled in a ``task.json`` by their ``dependencies``
list, walks them in order, runs each ``(step, sequence)`` pair via the
configured engine, and streams status updates to a :class:`StatusSink`.

This is the headless counterpart to the cluster runner in
``mamma_apptainer/submit_dag.py``.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from . import engines
from .capture_loader import CaptureCfgJsonLoader
from .status import StatusSink, make_sink
from .steps import get_builder

# Body branch only on cvpr_release. Face branch (ma_face / ma_smirk /
# ma_flame2smplx / ma_blender) is omitted pending upstream public release.
ALL_STEPS = ["ma_cap", "ma_masks", "ma_2d", "ma_3d", "ma_vis"]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cooperative cancellation
# ---------------------------------------------------------------------------

class _CancelFlag:
    def __init__(self) -> None:
        self.set: bool = False

    def __bool__(self) -> bool:
        return self.set


CANCEL = _CancelFlag()


# Seconds to wait for an engine subprocess to honour SIGTERM before
# escalating to SIGKILL. A CUDA-wedged python process inside a kernel
# can ignore SIGTERM until the kernel returns; SIGKILL is enforced by
# the OS and guarantees the GPU context is reclaimed. Long enough to
# let a well-behaved engine flush logs and release CUDA cleanly, short
# enough that the next queued task isn't held up by a hung engine.
_CANCEL_KILL_GRACE_SECONDS = 5.0


def _install_sigterm_handler() -> None:
    def handler(signum, frame):
        log.warning("received signal %s; cancelling", signum)
        CANCEL.set = True
        # Kill the in-flight engine child (whole process tree, since each
        # is launched in its own session). Without this, SIGTERM would
        # only set the flag and the running ML script would run to
        # completion before the runner moved on.
        if engines.kill_current(signal.SIGTERM):
            log.warning("sent SIGTERM to engine subprocess group")
            threading.Thread(
                target=_escalate_to_sigkill,
                daemon=True,
                name="cancel-escalate",
            ).start()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _escalate_to_sigkill() -> None:
    """If the engine subprocess hasn't died within the grace window,
    escalate to SIGKILL. Run from a daemon thread so the signal handler
    itself never blocks."""
    time.sleep(_CANCEL_KILL_GRACE_SECONDS)
    if engines.kill_current(signal.SIGKILL):
        log.warning(
            "escalated to SIGKILL: engine ignored SIGTERM for %ss "
            "(likely CUDA-wedged; GPU memory now released forcibly)",
            _CANCEL_KILL_GRACE_SECONDS,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_task_config(task_path: str) -> dict:
    # Delegate to the suffix-dispatching loader so ``.yaml`` run-config
    # files work end-to-end through the CLI (not just smoke tests).
    from . import config
    return config.load_run_config(task_path)


def enabled_steps(task_cfg: dict) -> List[str]:
    return [s for s in ALL_STEPS if task_cfg.get(s, {}).get("enabled")]


def topo_order(task_cfg: dict, steps: Iterable[str]) -> List[str]:
    """Topological sort honoring each step's ``dependencies`` list."""
    sorter: TopologicalSorter = TopologicalSorter()
    steps = list(steps)
    enabled_set = set(steps)
    for s in steps:
        deps = task_cfg.get(s, {}).get("dependencies", []) or []
        # Only honor deps that are also enabled; missing deps would
        # otherwise stall the sort indefinitely.
        deps = [d for d in deps if d in enabled_set]
        sorter.add(s, *deps)
    try:
        return list(sorter.static_order())
    except CycleError as e:
        raise RuntimeError(f"Cycle detected in step dependencies: {e}")


def resolve_seq_names(task_cfg: dict) -> List[str]:
    g = task_cfg.get("global", {})
    capture_json = os.path.expanduser(os.path.expandvars(g.get("capture_json", "")))
    if capture_json and not os.path.isabs(capture_json):
        # Anchor relative capture_json to repo root, mirroring StepBuilder._expand.
        from .env import repo_root
        capture_json = str(repo_root() / capture_json)
    if not capture_json or not os.path.exists(capture_json):
        raise FileNotFoundError(f"capture_json not found: {capture_json!r}")
    loader = CaptureCfgJsonLoader(capture_json)
    seq_ids = g.get("seq_ids", [])
    if not seq_ids:
        return loader.all_seq_names()
    return loader.get_seq_names(seq_ids)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_step(
    task_path: str,
    step_name: str,
    *,
    out_tag: Optional[str] = None,
    log_tag: Optional[str] = None,
    sink: Optional[StatusSink] = None,
    force: bool = False,
) -> int:
    """Run one step across all configured sequences. Returns 0 on success."""
    task_cfg = load_task_config(task_path)
    if not task_cfg.get(step_name, {}).get("enabled"):
        log.warning("step %r is not enabled in task config; running anyway", step_name)
    seq_names = resolve_seq_names(task_cfg)
    if not seq_names:
        log.warning("no sequences resolved; nothing to do")
        return 0

    if sink is None:
        sink = make_sink()

    return _run_step_inner(task_cfg, step_name, seq_names, out_tag, log_tag, sink, force)


def _ma_cap_counts(seq_dir: str) -> Tuple[Optional[int], Optional[int]]:
    """Derive (num_frames, num_cameras) from a finished ma_cap output dir.

    ma_cap writes ``<seq_dir>/gt/<cam>.npz`` (one per camera) plus a
    ``global.npz``; each camera NPZ carries ``frame_start``/``frame_end``.
    Returns ``(None, None)`` on any problem — this is best-effort metadata."""
    import glob
    npzs = glob.glob(os.path.join(seq_dir, "gt", "*.npz"))
    if not npzs:
        npzs = glob.glob(os.path.join(seq_dir, "*.npz"))
    cams = [p for p in npzs if os.path.basename(p).lower() != "global.npz"]
    num_cameras = len(cams) or None
    num_frames = None
    if cams:
        try:
            import numpy as np
            # allow_pickle=False: we only read the plain int frame_start/end
            # members, never the object arrays in the archive — keep it safe.
            d = np.load(cams[0], allow_pickle=False)
            if "frame_start" in d and "frame_end" in d:
                num_frames = max(0, int(d["frame_end"]) - int(d["frame_start"]))
        except Exception as e:  # noqa: BLE001 — metadata must never break a run
            log.warning("could not read frame range from %s: %s", cams[0], e)
    return num_frames, num_cameras


def _record_ma_cap_counts(builder, step_name: str, seq: str, sink: StatusSink) -> None:
    """After ma_cap finishes for a (step, seq), persist the actual frame/camera
    counts via the sink. No-op for other steps; the default sink ignores it.
    Wrapped so a metadata hiccup can never fail the pipeline run."""
    if step_name != "ma_cap":
        return
    # Skip the output scan entirely for sinks that don't persist counts
    # (CLI/headless PrintSink/JsonlSink) — no point reading NPZs to no-op.
    if type(sink).record_counts is StatusSink.record_counts:
        return
    try:
        seq_dir = os.path.join(builder.step_out_dir(with_dataset=True), seq)
        frames, cams = _ma_cap_counts(seq_dir)
        if frames is not None or cams is not None:
            sink.record_counts(step_name, seq, frames, cams)
    except Exception as e:  # noqa: BLE001
        log.warning("%s[%s] could not record counts: %s", step_name, seq, e)


def run_dag(
    task_path: str,
    *,
    out_tag: Optional[str] = None,
    log_tag: Optional[str] = None,
    sink: Optional[StatusSink] = None,
    force: bool = False,
) -> int:
    """Run the full DAG of enabled steps in dependency order."""
    _install_sigterm_handler()
    task_cfg = load_task_config(task_path)
    steps = enabled_steps(task_cfg)
    if not steps:
        log.warning("no enabled steps; nothing to do")
        return 0
    order = topo_order(task_cfg, steps)
    seq_names = resolve_seq_names(task_cfg)
    if not seq_names:
        log.warning("no sequences resolved; nothing to do")
        return 0

    if sink is None:
        sink = make_sink()

    log.info("order: %s", order)
    log.info("sequences: %s", seq_names)

    # Choose the loop nesting based on the per-task `sequence_major`
    # flag. Default False = step-major (today's behaviour: finish
    # every step across all sequences before advancing). True =
    # sequence-major (finish each sequence end-to-end before the
    # next). The two share engines + status sink; only the iteration
    # order and failure-cascade scope differ.
    seq_major = bool((task_cfg.get("global") or {}).get("sequence_major", False))
    if seq_major:
        log.info("dispatch mode: sequence-major")
        return _run_dag_seq_major(task_cfg, order, seq_names, out_tag, log_tag, sink, force)

    log.info("dispatch mode: step-major (default)")
    failed: List[str] = []
    for step_name in order:
        if CANCEL:
            sink.cancel_remaining([step_name], seq_names)
            continue

        # If a dependency failed (or was cancelled because *its* dependency
        # failed earlier), skip this step too. Treat cancelled steps as
        # "blocked" for downstream-dep purposes — otherwise a 3-deep DAG
        # like ma_masks->ma_2d->ma_3d would have ma_3d try to run even
        # when ma_masks failed and ma_2d got cascaded into Cancelled.
        deps = task_cfg.get(step_name, {}).get("dependencies", []) or []
        if any(d in failed for d in deps):
            log.warning("skipping %s: dependency failed/cancelled", step_name)
            for s in seq_names:
                sink.update(step_name, s, "Cancelled")
            failed.append(step_name)
            continue

        rc = _run_step_inner(task_cfg, step_name, seq_names, out_tag, log_tag, sink, force)
        if rc != 0:
            failed.append(step_name)

    if failed:
        log.error("DAG completed with failures: %s", failed)
        return 1
    log.info("DAG completed successfully")
    return 0


def _run_dag_seq_major(
    task_cfg: dict,
    order: List[str],
    seq_names: List[str],
    out_tag: Optional[str],
    log_tag: Optional[str],
    sink: StatusSink,
    force: bool,
) -> int:
    """Sequence-major DAG: finish each sequence end-to-end (all of its
    steps, in order) before moving to the next sequence.

    Failure cascade is **per-sequence**: if seq A's ma_masks fails, the
    rest of seq A's downstream steps are Cancelled, but seq B starts
    fresh from ma_cap. Contrast with step-major where one sequence's
    ma_cap failure blocks every sequence's ma_masks. The per-sequence
    isolation here matches the natural user mental model — 'this
    sequence broke; the others may still finish.'
    """
    overall_failed_steps: set[str] = set()
    tag = out_tag or "local"
    # Pre-build a step_name -> builder map (cheap; avoids re-doing it
    # per sequence in the inner loop).
    step_builders: dict[str, "StepBuilder"] = {}
    for step_name in order:
        step_cfg = task_cfg.get(step_name, {})
        global_cfg = task_cfg.get("global", {})
        step_builders[step_name] = get_builder(step_name, step_cfg, global_cfg, tag)

    for seq in seq_names:
        if CANCEL:
            for step_name in order:
                sink.update(step_name, seq, "Cancelled")
            continue
        seq_failed: set[str] = set()
        for step_name in order:
            if CANCEL:
                sink.update(step_name, seq, "Cancelled")
                continue

            deps = task_cfg.get(step_name, {}).get("dependencies", []) or []
            if any(d in seq_failed for d in deps):
                log.warning("skipping %s[%s]: dependency failed/cancelled", step_name, seq)
                sink.update(step_name, seq, "Cancelled")
                seq_failed.add(step_name)
                continue

            builder = step_builders[step_name]
            global_cfg = task_cfg.get("global", {})

            done = builder.done_file(seq)
            if not force and os.path.exists(done):
                log.info("%s[%s] DONE present at %s; skipping", step_name, seq, done)
                sink.update(step_name, seq, "Done")
                _record_ma_cap_counts(builder, step_name, seq, sink)
                continue

            out_path, err_path = _log_paths(global_cfg, log_tag or tag, step_name, seq)
            sink.update(step_name, seq, "Running", pid=os.getpid())
            try:
                rc = engines.dispatch(builder, seq, out_path, err_path)
            except Exception as e:
                log.exception("%s[%s] engine error: %s", step_name, seq, e)
                sink.update(step_name, seq, "Failed")
                seq_failed.add(step_name)
                overall_failed_steps.add(step_name)
                continue

            if rc == 0:
                try:
                    Path(done).parent.mkdir(parents=True, exist_ok=True)
                    Path(done).touch()
                except OSError as e:
                    log.warning("%s[%s] could not write DONE sentinel at %s: %s",
                                step_name, seq, done, e)
                sink.update(step_name, seq, "Done")
                _record_ma_cap_counts(builder, step_name, seq, sink)
            elif CANCEL:
                sink.update(step_name, seq, "Cancelled")
                log.warning("%s[%s] cancelled by signal (exit=%s)", step_name, seq, rc)
            else:
                sink.update(step_name, seq, "Failed")
                seq_failed.add(step_name)
                overall_failed_steps.add(step_name)
                log.error("%s[%s] exit=%s -- check %s", step_name, seq, rc, err_path)

    if overall_failed_steps:
        log.error("DAG (seq-major) completed with failures: %s", sorted(overall_failed_steps))
        return 1
    log.info("DAG (seq-major) completed successfully")
    return 0


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _run_step_inner(
    task_cfg: dict,
    step_name: str,
    seq_names: List[str],
    out_tag: Optional[str],
    log_tag: Optional[str],
    sink: StatusSink,
    force: bool,
) -> int:
    step_cfg = task_cfg.get(step_name, {})
    global_cfg = task_cfg.get("global", {})
    tag = out_tag or "local"
    builder = get_builder(step_name, step_cfg, global_cfg, tag)

    failures = 0
    for seq in seq_names:
        if CANCEL:
            sink.update(step_name, seq, "Cancelled")
            continue

        # DONE-sentinel skip. ``--force`` (CLI) bypasses this.
        done = builder.done_file(seq)
        if not force and os.path.exists(done):
            log.info("%s[%s] DONE present at %s; skipping", step_name, seq, done)
            sink.update(step_name, seq, "Done")
            _record_ma_cap_counts(builder, step_name, seq, sink)
            continue

        out_path, err_path = _log_paths(global_cfg, log_tag or tag, step_name, seq)
        sink.update(step_name, seq, "Running", pid=os.getpid())
        try:
            rc = engines.dispatch(builder, seq, out_path, err_path)
        except Exception as e:
            log.exception("%s[%s] engine error: %s", step_name, seq, e)
            sink.update(step_name, seq, "Failed")
            failures += 1
            continue

        if rc == 0:
            # The cluster .sh wrappers wrote DONE on success; the per-step
            # python scripts don't, so we do it ourselves so re-runs skip
            # already-done (step, seq) work.
            try:
                Path(done).parent.mkdir(parents=True, exist_ok=True)
                Path(done).touch()
            except OSError as e:
                log.warning("%s[%s] could not write DONE sentinel at %s: %s",
                            step_name, seq, done, e)
            sink.update(step_name, seq, "Done")
            _record_ma_cap_counts(builder, step_name, seq, sink)
        elif CANCEL:
            # Non-zero exit because we signalled the child during a stop.
            # That's a cancel, not a failure.
            sink.update(step_name, seq, "Cancelled")
            log.warning("%s[%s] cancelled by signal (exit=%s)", step_name, seq, rc)
        else:
            sink.update(step_name, seq, "Failed")
            failures += 1
            log.error("%s[%s] exit=%s -- check %s", step_name, seq, rc, err_path)

    return 0 if failures == 0 else 1


def _log_paths(
    global_cfg: dict,
    tag: str,
    step_name: str,
    seq_name: str,
) -> Tuple[str, str]:
    """Per-(step, seq) stdout/stderr path pair."""
    jobs_log_dir = os.path.expanduser(os.path.expandvars(
        global_cfg.get("jobs_log_dir", "")
    ))
    if not jobs_log_dir:
        # Fallback to a runtime tmp dir under MAMMA_DATA_DIR so we don't
        # write to surprising locations.
        jobs_log_dir = str(
            Path(os.environ.get("MAMMA_DATA_DIR", "~/.mamma")).expanduser() / "logs"
        )
    user = global_cfg.get("username") or os.environ.get("USER") or "local"
    base = os.path.join(jobs_log_dir, user, tag, step_name)
    return (
        os.path.join(base, f"{seq_name}.out"),
        os.path.join(base, f"{seq_name}.err"),
    )
