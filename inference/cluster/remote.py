"""Remote submission: sync code, upload the run-config, submit the DAG.

Ties a :class:`~inference.cluster.profile.ClusterProfile` to a
:class:`~inference.cluster.transport.Transport` and drives the login-node
``submit-cluster`` command remotely. The heavy lifting (DAG generation) stays
in :mod:`inference.cluster.htcondor`, invoked over the transport; this module
is just the client-side choreography.
"""
from __future__ import annotations

import json
import os
import posixpath
import shlex
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..env import repo_root
from .profile import ClusterProfile
from .resources import apply_resources
from .transport import LocalTransport, SshTransport, Transport


@dataclass
class RemoteHandle:
    """What the client keeps to track a submitted run."""

    profile_name: str
    dag_cluster_id: Optional[str]
    status_jsonl: str
    dag_path: str
    work_dir: str
    remote_task: str
    node_to_seq: Dict[str, str] = field(default_factory=dict)
    scheduler: str = "htcondor"        # adapter that submitted the run


def make_transport(profile: ClusterProfile) -> Transport:
    if profile.transport == "local":
        return LocalTransport()
    if profile.transport == "ssh":
        return SshTransport(
            host=profile.host,
            username=profile.username,
            port=profile.port,
            ssh_opts=profile.ssh_opts,
        )
    raise ValueError(f"unknown transport {profile.transport!r}")


def _itemized_changes(stdout: str) -> List[str]:
    """File-level changes from ``rsync --itemize-changes`` output.

    Directory-attribute and timestamp-only lines are skipped; what remains
    is content the sync transfers, creates, or deletes."""
    out = []
    for line in (stdout or "").splitlines():
        line = line.rstrip()
        if len(line) < 12 or " " not in line:
            continue
        code, path = line.split(None, 1)
        if code.startswith("*deleting"):
            out.append(path)
        elif code[0] in "<>c" and len(code) > 1 and code[1] == "f":
            out.append(path)
    return out


def sync_code(tr: Transport, profile: ClusterProfile,
              local_repo: Optional[str] = None,
              dry_run: bool = False) -> List[str]:
    """Sync the local tree to the cluster repo dir (or, with ``dry_run``,
    only compare). Returns the file paths that changed."""
    local_repo = local_repo or str(repo_root())
    res = tr.sync_dir(local_repo, profile.resolved_repo_dir(),
                      profile.sync_excludes, dry_run=dry_run)
    if not res.ok:
        raise RuntimeError(f"code sync failed: {res.stderr.strip() or res.returncode}")
    return _itemized_changes(res.stdout)


def probe_runner_python(tr: Transport) -> Optional[str]:
    """Find the interpreter for MAMMA's job wrapper on the login node.

    Prefers ``python3``, falls back to ``python``; returns the bare command
    name, or None when neither exists (the preflight surfaces that)."""
    res = tr.run(["bash", "-lc", "command -v python3 || command -v python"])
    lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
    if res.ok and lines:
        return posixpath.basename(lines[-1])
    return None


def resolve_runner_python(tr: Transport, profile: ClusterProfile) -> str:
    """An explicit ``profile.python`` wins; empty means auto-detect."""
    return profile.python or probe_runner_python(tr) or "python3"


def build_remote_submit_argv(
    profile: ClusterProfile,
    remote_task: str,
    out_tag: str,
    log_tag: str,
    work_dir: str,
    status_jsonl: str,
    *,
    bid: Optional[int] = None,
    retries: Optional[int] = None,
    dry_run: bool = False,
    python: Optional[str] = None,
) -> List[str]:
    """The ``<python> -m inference submit-cluster ...`` argv to run on the login node."""
    py = python or profile.python or "python3"
    argv = [
        py, "-m", "inference", "submit-cluster",
        "--task", remote_task,
        "--out-tag", out_tag,
        "--log-tag", log_tag,
        "--work-dir", work_dir,
        "--repo-dir", profile.resolved_repo_dir(),
        "--activate", profile.activate,
        "--python", py,
        "--status-jsonl", status_jsonl,
        "--scheduler", profile.scheduler or "htcondor",
        "--flavor", profile.flavor,
        "--bid", str(bid if bid is not None else profile.bid_price),
        "--retries", str(retries if retries is not None else profile.retries),
    ]
    if dry_run:
        argv.append("--dry-run")
    return argv


def build_remote_submit_script(
    profile: ClusterProfile, submit_argv: List[str]
) -> str:
    """Wrap the submit argv in a login-shell snippet: activate + cd repo + submit."""
    lines: List[str] = ["set -e"]
    if profile.activate:
        lines.append(profile.activate)
    lines.append(f"cd {shlex.quote(profile.resolved_repo_dir())}")
    lines.append(shlex.join(submit_argv))
    return "\n".join(lines) + "\n"


def submit_remote(
    profile: ClusterProfile,
    task_path: str,
    *,
    out_tag: str,
    log_tag: Optional[str] = None,
    bid: Optional[int] = None,
    retries: Optional[int] = None,
    dry_run: bool = False,
    do_sync: bool = True,
    local_repo: Optional[str] = None,
    transport: Optional[Transport] = None,
) -> RemoteHandle:
    """Sync code, upload the run-config, submit the DAG, and return a handle.

    ``transport`` may be injected (tests use a fake / LocalTransport); otherwise
    it is built from the profile.
    """
    log_tag = log_tag or out_tag
    tr = transport or make_transport(profile)

    ws = profile.workspace_root.rstrip("/")
    if not ws:
        raise ValueError("profile.workspace_root is required for remote submit")
    work_dir = posixpath.join(ws, "dag", log_tag)
    # Directory of per-cell status files (one writer each); a shared .jsonl
    # gets torn by concurrent appends from different nodes on NFS/Lustre.
    status_jsonl = posixpath.join(work_dir, "status")
    remote_task = posixpath.join(ws, "run_configs", f"{log_tag}.json")

    if do_sync:
        changed = sync_code(tr, profile, local_repo)
        if changed:
            sample = ", ".join(changed[:3]) + (", ..." if len(changed) > 3 else "")
            print(f"[cluster] code sync: {len(changed)} file(s) updated on "
                  f"'{profile.name}' ({sample})", flush=True)
        else:
            print(f"[cluster] code sync: cluster copy already matches the "
                  f"local tree", flush=True)

    tr.mkdirs(posixpath.dirname(remote_task))
    baked = _bake_task_config(task_path, profile)
    try:
        tr.put(baked, remote_task)
    finally:
        try:
            os.remove(baked)
        except OSError:
            pass

    submit_argv = build_remote_submit_argv(
        profile, remote_task, out_tag, log_tag, work_dir, status_jsonl,
        bid=bid, retries=retries, dry_run=dry_run,
        python=resolve_runner_python(tr, profile),
    )
    res = tr.run_script(build_remote_submit_script(profile, submit_argv))
    if not res.ok:
        raise RuntimeError(
            f"remote submit-cluster failed (exit {res.returncode}):\n"
            f"{res.stderr.strip()}\n{res.stdout.strip()}"
        )

    # submit-cluster writes handle.json in work_dir; fetch it (single source of
    # truth for the DAG id and node->seq map, computed on the cluster).
    handle = _fetch_handle(tr, work_dir)
    return RemoteHandle(
        profile_name=profile.name,
        dag_cluster_id=handle.get("dag_cluster_id"),
        status_jsonl=handle.get("status_jsonl", status_jsonl),
        dag_path=handle.get("dag_path", ""),
        work_dir=work_dir,
        remote_task=remote_task,
        node_to_seq=handle.get("node_to_seq", {}),
        scheduler=handle.get("scheduler") or profile.scheduler or "htcondor",
    )


def _bake_task_config(task_path: str, profile: ClusterProfile) -> str:
    """Resolve per-step scheduler resources into a temp copy of the run-config.

    The task file stays free of cluster concerns; the uploaded frozen copy
    carries an explicit ``submit_cfg`` per step (defaults <- profile.resources
    <- the task's own submit_cfg, see :mod:`inference.cluster.resources`), so
    the emitter and the execute-node ``--nv`` decision both see final values.
    Always writes JSON: the cluster side parses it with a bare stdlib python.
    """
    with open(task_path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError:
        import yaml  # client-side dependency (profiles already require it)
        cfg = yaml.safe_load(text)
    cfg = apply_resources(cfg, profile.resources)
    fd, path = tempfile.mkstemp(prefix="mamma_task_baked_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return path


def _fetch_handle(tr: Transport, work_dir: str) -> dict:
    remote_handle = posixpath.join(work_dir, "handle.json")
    fd, local = tempfile.mkstemp(prefix="mamma_handle_", suffix=".json")
    os.close(fd)
    try:
        tr.get(remote_handle, local)
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        try:
            os.remove(local)
        except OSError:
            pass
