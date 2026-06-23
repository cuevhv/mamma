"""Engine adapters: how to actually execute a step's python invocation.

Three engines, all with the same shape::

    run(builder, seq_name, out_path, err_path) -> int

The runner picks an engine per-step based on ``step_cfg["engine"]``
(default ``conda``).

Each engine spawns the child in its own process group so the runner can
kill the entire tree (engine wrapper + bash + python) on cancellation.

POSIX-only: relies on :func:`os.killpg` and ``start_new_session``.
"""
from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

from .steps.base import StepBuilder

log = logging.getLogger(__name__)


# Tracks the currently-running engine subprocess so the runner's SIGTERM
# handler can kill it. Single-threaded by design — the runner is serial.
_current_lock = threading.Lock()
_current_proc: Optional[subprocess.Popen] = None


# Env-var prefixes that the runner translates into explicit --flag
# arguments at the subprocess boundary (see inference/steps/ma_3d.py).
# Stripping them here makes the contract one-way: subprocesses receive
# paths via argv, never via inherited environment. Prevents an inner
# module's stale os.environ.get(...) from silently picking up a value
# the user didn't intend.
_STRIPPED_PREFIXES = ("MAMMA_",)


def _child_env() -> dict:
    """Return os.environ with the ``MAMMA_*`` keys removed."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(k.startswith(p) for p in _STRIPPED_PREFIXES)
    }


def current_proc() -> Optional[subprocess.Popen]:
    return _current_proc


def kill_current(sig: int = signal.SIGTERM) -> bool:
    """Signal the entire process group of the currently-running engine child.

    Returns ``True`` if a signal was sent, ``False`` if no child was active.
    Safe to call from a signal handler.
    """
    proc = _current_proc
    if proc is None or proc.poll() is not None:
        return False
    try:
        os.killpg(os.getpgid(proc.pid), sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _ensure_log_dir(*paths: str) -> None:
    for p in paths:
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)


def _open_logs(out_path: str, err_path: str):
    """Open stdout/stderr file handles for the child."""
    _ensure_log_dir(out_path, err_path)
    out_f = open(out_path, "ab", buffering=0) if out_path else None
    err_f = open(err_path, "ab", buffering=0) if err_path else None
    return out_f, err_f


def _run(cmd: List[str], cwd: Optional[str], out_f, err_f) -> int:
    """Run cmd in its own process group, streaming output. Returns exit code."""
    global _current_proc
    log.info("cwd=%s", cwd or os.getcwd())
    log.info("cmd=%s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=cwd or None,
        stdout=out_f or None,
        stderr=err_f or None,
        # Own session/process group so SIGTERM via os.killpg propagates
        # to every descendent (engine wrapper + bash + python).
        start_new_session=True,
        # Paths flow via argv only; MAMMA_* env keys do not leak to the
        # child. See _child_env() above for the rationale.
        env=_child_env(),
    )
    with _current_lock:
        _current_proc = proc
    try:
        return proc.wait()
    finally:
        with _current_lock:
            _current_proc = None


def run_conda(
    builder: StepBuilder,
    seq_name: str,
    out_path: str,
    err_path: str,
) -> int:
    """Run a step via conda, or via the active interpreter if conda is absent.

    The public docs and shipped presets historically default to
    ``engine: conda``. For lightweight local installs that use a plain venv
    instead, fall back to ``sys.executable`` when ``conda`` is not on PATH.
    This preserves the config surface while keeping standalone installs usable.
    """
    argv = builder.build_argv(seq_name)
    if shutil.which("conda"):
        cmd = [
            "conda",
            "run",
            "-n",
            builder.conda_env,
            "--no-capture-output",
            "--live-stream",
            "python",
            *argv,
        ]
    else:
        log.warning(
            "conda executable not found; falling back to current interpreter %s "
            "for step %r",
            sys.executable,
            builder.step_name,
        )
        cmd = [sys.executable, *argv]
    out_f, err_f = _open_logs(out_path, err_path)
    try:
        return _run(cmd, builder.host_cwd(), out_f, err_f)
    finally:
        if out_f:
            out_f.close()
        if err_f:
            err_f.close()


def run_apptainer(
    builder: StepBuilder,
    seq_name: str,
    out_path: str,
    err_path: str,
) -> int:
    """``apptainer run [--nv] --bind <list> <sif> <argv>``.

    ``--nv`` is only passed when the step asks for a GPU
    (``submit_cfg.gpus > 0``). Passing ``--nv`` to a CPU-only step makes
    apptainer inject the host's GL libs into the container and can fail with
    a glibc-version mismatch when the container's glibc is older.
    """
    sif = builder.sif_path
    if not sif:
        raise RuntimeError(
            f"Step {builder.step_name!r} engine=apptainer but sif_path is empty"
        )
    binds: List[str] = []
    for b in builder.binds():
        binds += ["--bind", b]
    needs_gpu = int((builder.step_cfg.get("submit_cfg") or {}).get("gpus", 0) or 0) > 0
    nv_flag = ["--nv"] if needs_gpu else []
    argv = builder.build_argv(seq_name)
    cmd = ["apptainer", "run", *nv_flag, *binds, sif, *argv]
    out_f, err_f = _open_logs(out_path, err_path)
    try:
        return _run(cmd, None, out_f, err_f)
    finally:
        if out_f:
            out_f.close()
        if err_f:
            err_f.close()


def run_docker(
    builder: StepBuilder,
    seq_name: str,
    out_path: str,
    err_path: str,
) -> int:
    """``docker run --rm --gpus all -v <list> -w /repo <image> python <argv>``."""
    image = builder.docker_image
    if not image:
        raise RuntimeError(
            f"Step {builder.step_name!r} engine=docker but docker_image is empty"
        )
    volumes: List[str] = []
    for b in builder.binds():
        volumes += ["-v", b]
    argv = builder.build_argv(seq_name)
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-w",
        builder.container_cwd(),
        *volumes,
        image,
        "python",
        *argv,
    ]
    out_f, err_f = _open_logs(out_path, err_path)
    try:
        return _run(cmd, None, out_f, err_f)
    finally:
        if out_f:
            out_f.close()
        if err_f:
            err_f.close()


ENGINES = {
    "conda": run_conda,
    "apptainer": run_apptainer,
    "docker": run_docker,
}


def dispatch(
    builder: StepBuilder,
    seq_name: str,
    out_path: str,
    err_path: str,
) -> int:
    """Look up the configured engine for ``builder`` and run it."""
    engine = builder.engine
    if engine not in ENGINES:
        raise ValueError(
            f"Unknown engine {engine!r} for step {builder.step_name!r}. "
            f"Supported: {sorted(ENGINES)}"
        )
    return ENGINES[engine](builder, seq_name, out_path, err_path)
