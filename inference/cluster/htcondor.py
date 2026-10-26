"""HTCondor DAGMan emitter.

Renders a run-config into ``.sh`` + ``.sub`` + ``.dag`` and (optionally)
submits it. One DAG node per ``(step, sequence)``; each node runs a tiny
wrapper that re-enters the runner for a single cell::

    <step>.sh $1(seq)  ->  python -m inference run-cell --step <step> --seq $1
                              -> engines.dispatch(builder, seq)   [same as local]
                                 -> apptainer run <sif> <script> ...

Because the node wrapper defers to ``run-cell`` -> ``engines.dispatch``, no
per-step argv logic lives here: it stays in :mod:`inference.steps`. This is
the same code path as a local run, so no apptainer command is hand-written
per step.

Graph shape (enabled steps, topo order) and sequence resolution are reused
from :mod:`inference.runner`, so a DAG matches what ``run_dag`` would walk.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..runner import enabled_steps, resolve_seq_names, topo_order
from .resources import DEFAULT_STEP_RESOURCES, resolve_submit_cfg

log = logging.getLogger(__name__)


@dataclass
class CondorOpts:
    """Everything the emitter needs that is not in the run-config itself.

    The path fields are absolute locations **on the machine that will run the
    jobs** (the cluster). In Phase 0, ``submit-cluster`` runs on the login node
    and fills these from the local repo; later the SSH client fills them from
    the cluster profile.
    """

    flavor: str = "bid"          # "bid" -> condor_submit_dag_bid ; "standard" -> condor_submit_dag
    bid_price: int = 100
    retries: int = 1
    repo_dir: str = ""           # repo root the node wrapper cd's into
    activate: str = ""           # shell snippet to activate the runner env on the node
    remote_task_path: str = ""   # run-config the node wrapper passes to run-cell
    status_jsonl: str = ""       # status stream the poller reads: a DIRECTORY
                                 # (one file per cell; safe for concurrent
                                 # nodes) or a legacy .jsonl shared file
    python: str = "python"       # execute-node interpreter (e.g. "python3" on many clusters)


@dataclass
class DagBundle:
    """The generated artifacts and the handle needed to track the run."""

    dag_path: str
    node_to_seq: Dict[str, str] = field(default_factory=dict)   # {"ma_2d_3": "<seq>"}
    dag_cluster_id: Optional[str] = None                        # set by submit()


# ---------------------------------------------------------------------------
# per-artifact renderers (pure; unit-tested)
# ---------------------------------------------------------------------------

def _q(path: str) -> str:
    """Quote a DAG path token if it contains whitespace."""
    return f'"{path}"' if (path and any(c.isspace() for c in path)) else path


def render_node_sh(step: str, opts: CondorOpts, out_tag: str, log_tag: str) -> str:
    """One wrapper per step; ``$1`` is the sequence name. Re-enters the runner
    for a single cell so cluster and local execution cannot drift.

    Status protocol: when ``opts.status_jsonl`` is a DIRECTORY (no ``.jsonl``
    suffix), each cell writes its own ``<seq>.<step>.jsonl`` inside it: one
    writer per file. Concurrent cells on different nodes must never share an
    append target: NFS/Lustre appends from multiple clients tear and
    overwrite each other. A ``.jsonl`` path keeps the
    legacy single shared file.
    """
    activate = f"{opts.activate}\n" if opts.activate else ""
    if opts.status_jsonl.endswith(".jsonl"):
        mkdir_line = ""
        status_arg = shlex.quote(opts.status_jsonl)
    else:
        d = shlex.quote(opts.status_jsonl)
        mkdir_line = f"mkdir -p {d}\n"
        status_arg = f'{d}/"$SEQ".{step}"${{CELL:+.$CELL}}".jsonl'
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'SEQ="$1"\n'
        '# Camera fan-out cells pass a cell id + extra step flags; plain\n'
        '# nodes pass empty strings.\n'
        'CELL="${2:-}"\n'
        'CELL_FLAGS="${3:-}"\n'
        "# Temp spills (the ma_masks mask store etc.) belong in the job's\n"
        "# request_disk-sized scratch, not the node's shared /tmp (ENOSPC).\n"
        "# Condor usually sets TMPDIR already; this pins it even\n"
        "# when only _CONDOR_SCRATCH_DIR is provided.\n"
        'TMPDIR="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"\n'
        "export TMPDIR\n"
        f"{activate}"
        f"{mkdir_line}"
        f"cd {shlex.quote(opts.repo_dir)}\n"
        f"exec {opts.python} -m inference run-cell "
        f"--task {shlex.quote(opts.remote_task_path)} "
        f"--step {step} "
        '--seq "$SEQ" '
        f"--out-tag {shlex.quote(out_tag)} "
        f"--log-tag {shlex.quote(log_tag)} "
        f"--status-jsonl {status_arg} "
'--cell-id "$CELL" --cell-flags "$CELL_FLAGS"\n'
    )


def render_sub(step: str, sh_path: str, log_dir: str, submit_cfg: dict) -> str:
    """HTCondor submit description for one step (the DAG supplies ``seq_arg``).

    GPU / CUDA requirement lines are emitted only when the step asks for a GPU
    (``submit_cfg.gpus > 0``), so CPU-only steps like ma_cap do not carry a
    spurious ``CUDAGlobalMemoryMb`` requirement.
    """
    gpus = int(submit_cfg.get("gpus", 0) or 0)
    lines = [
        "universe = vanilla",
        f"executable = {sh_path}",
        'arguments = "$(seq_arg) $(cell_id) \'$(cell_flags)\'"',
        f"output = {log_dir}/$(seq_arg)_$(Cluster)_$(Process).out",
        f"error  = {log_dir}/$(seq_arg)_$(Cluster)_$(Process).err",
        f"log    = {log_dir}/$(seq_arg)_$(Cluster)_$(Process).log",
        f"request_cpus = {submit_cfg.get('cpus', 4)}",
        f"request_memory = {submit_cfg.get('memory', 16000)}",
        # Explicit MB unit: request_disk's bare-number unit is KiB, unlike
        # request_memory's MB. Sizes the scratch dir the node wrapper points
        # TMPDIR at (mask spills etc.), so temp usage never lands on /tmp.
        f"request_disk = {submit_cfg.get('disk', 8000)}MB",
        f"max_materialize = {submit_cfg.get('max_jobs', 20)}",
    ]
    reqs = []
    if gpus > 0:
        lines.append(f"request_gpus = {gpus}")
        min_cuda = submit_cfg.get("min_cuda_memory")
        if min_cuda:
            reqs.append(f"TARGET.CUDAGlobalMemoryMb > {min_cuda}")
        # Keep the GPU architecture inside what the execution env has kernels
        # for, else "no kernel image is available": torch wheels stop at a
        # newest arch (cu124: 9.0), and compiled extensions start at the
        # oldest arch they were built for (the MAMMA image: 8.0). Per profile.
        max_cc = submit_cfg.get("max_cuda_capability")
        if max_cc:
            reqs.append(f"TARGET.CUDACapability <= {max_cc}")
        min_cc = submit_cfg.get("min_cuda_capability")
        if min_cc:
            reqs.append(f"TARGET.CUDACapability >= {min_cc}")
    # Operational blocklist for wedged hosts: a fast-failing or hanging node
    # keeps grabbing queued cells and can exhaust a node's retries. CPU steps
    # land on the same hosts, so it applies to every step. Set in the profile
    # under ``resources.default.exclude_machines``; remove entries once the
    # host is fixed.
    for machine in submit_cfg.get("exclude_machines") or []:
        reqs.append(f'TARGET.Machine =!= "{machine}"')
    if reqs:
        lines.append("requirements = " + " && ".join(f"({r})" for r in reqs))
    lines.append("queue 1")  # single_var: the DAG's VARS supplies seq_arg per node
    return "\n".join(lines) + "\n"


def render_dag(
    jobs: List[dict],
    edges: List[Tuple[List[str], str]],
    maxjobs: Dict[str, int],
    retries: int,
) -> str:
    """JOB / VARS / CATEGORY / MAXJOBS / RETRY / PARENT-CHILD.

    ``jobs``: dicts with ``name``, ``sub``, ``category`` and ``vars`` (macro
    dict; every node carries ``seq_arg``/``cell_id``/``cell_flags`` so one
    .sub per step serves plain and camera-fanout nodes alike). ``edges`` are
    explicit ``(parents, child)`` pairs; the caller wires them index-wise for
    the plain pipeline and fans ma_masks out per camera when asked.
    """
    out = ["# Auto-generated by inference.cluster.htcondor", "# JOB definitions"]
    for j in jobs:
        out.append(f"JOB {j['name']} {_q(j['sub'])}")
        macros = " ".join(f'{k}="{v}"' for k, v in j["vars"].items())
        out.append(f"VARS {j['name']} {macros}")
        out.append(f"CATEGORY {j['name']} {j['category']}")
    if maxjobs:
        out.append("# Max concurrent jobs per step")
        for cat, m in maxjobs.items():
            out.append(f"MAXJOBS {cat} {m}")
    if retries > 0:
        out.append("# Retries")
        for j in jobs:
            out.append(f"RETRY {j['name']} {retries}")
    out.append("# Dependencies")
    for parents, child in edges:
        if parents:
            out.append(f"PARENT {' '.join(parents)} CHILD {child}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# emit + submit
# ---------------------------------------------------------------------------

def emit_dag(
    task_cfg: dict,
    out_tag: str,
    log_tag: str,
    opts: CondorOpts,
    work_dir: str,
) -> DagBundle:
    """Render the full DAG for a run-config. Pure filesystem writes, no submit.

    Reuses the runner's own graph resolution (``enabled_steps`` + ``topo_order``)
    and sequence resolution (``resolve_seq_names``), so the DAG's shape matches
    what ``run_dag`` would walk locally.
    """
    steps = topo_order(task_cfg, enabled_steps(task_cfg))
    if not steps:
        raise ValueError("no enabled steps in run-config; nothing to submit")
    seqs = resolve_seq_names(task_cfg)
    if not seqs:
        raise ValueError("no sequences resolved from the run-config")

    os.makedirs(work_dir, exist_ok=True)
    jobs: List[dict] = []
    edges: List[Tuple[List[str], str]] = []
    node_to_seq: Dict[str, str] = {}
    maxjobs: Dict[str, int] = {}
    step_set = set(steps)
    sub_paths: Dict[str, str] = {}
    # step -> idx -> node names downstream steps must wait on
    tails: Dict[str, Dict[int, List[str]]] = {}

    fanout_cams, fanout_init = _resolve_fanout(task_cfg, "ma_masks")
    fan2d_cams, _ = _resolve_fanout(task_cfg, "ma_2d")

    for step in steps:
        scfg = task_cfg.get(step, {})
        # Resource requests come from the per-step defaults, overridable by the
        # step's own submit_cfg (usually pre-baked by remote.submit_remote from
        # the cluster profile). A run-config with no submit_cfg at all still
        # gets GPUs on its GPU steps.
        submit_cfg = resolve_submit_cfg(step, scfg)
        if DEFAULT_STEP_RESOURCES.get(step, {}).get("gpus", 0) > 0 \
                and int(submit_cfg.get("gpus", 0) or 0) == 0:
            log.warning(
                "step %s resolved to gpus=0 (explicit override); it will be "
                "scheduled on a CPU-only node and run without --nv", step,
            )
        deps = [d for d in (scfg.get("dependencies") or []) if d in step_set]
        maxjobs[step] = int(submit_cfg.get("max_jobs", 20))

        log_dir = os.path.join(work_dir, "logs", step)
        os.makedirs(log_dir, exist_ok=True)
        sh_path = os.path.join(work_dir, f"{step}.sh")
        sub_path = os.path.join(work_dir, f"{step}.sub")
        Path(sh_path).write_text(render_node_sh(step, opts, out_tag, log_tag))
        os.chmod(sh_path, 0o755)
        Path(sub_path).write_text(render_sub(step, sh_path, log_dir, submit_cfg))
        sub_paths[step] = sub_path
        tails[step] = {}

        fan_this = step == "ma_masks" and fanout_cams is not None
        fan_2d = step == "ma_2d" and fan2d_cams is not None
        for idx, seq in enumerate(seqs):
            parents = [n for d in deps for n in tails[d].get(idx, [])]
            if not (fan_this or fan_2d):
                name = f"{step}_{idx}"
                jobs.append({"name": name, "sub": sub_path, "category": step,
                             "vars": {"seq_arg": seq, "cell_id": "",
                                      "cell_flags": ""}})
                node_to_seq[name] = seq
                edges.append((parents, name))
                tails[step][idx] = [name]
                continue
            if fan_2d:
                # ma_2d camera fan-out: cells are fully independent (one NPZ
                # per camera, no shared state), so there is no init anchor
                # and no finalize barrier; downstream steps wait on every
                # cell.
                cam_nodes = []
                for cam in fan2d_cams:
                    name = f"{step}_{cam}_{idx}"
                    jobs.append({"name": name, "sub": sub_path,
                                 "category": step,
                                 "vars": {"seq_arg": seq, "cell_id": cam,
                                          "cell_flags": f"--cam_names {cam}"}})
                    node_to_seq[name] = seq
                    edges.append((parents, name))
                    cam_nodes.append(name)
                tails[step][idx] = cam_nodes
                continue
            # Camera fan-out: one init node defining identities, then one
            # node per remaining camera (each resumes the init cache and
            # processes a single view), then one finalize node that skips
            # the per-camera caches and runs the rig-vote identity repair
            # and the collage exactly once over the full rig.
            # Cells must not run them: concurrent passes would race on renames,
            # and a per-cell collage is rebuilt by every later cell anyway.
            init_name = f"{step}_init_{idx}"
            jobs.append({"name": init_name, "sub": sub_path, "category": step,
                         "vars": {"seq_arg": seq, "cell_id": "init",
                                  "cell_flags": f"--cam_names {fanout_init} "
                                                f"--cam_init {fanout_init} "
                                                f"--rig_vote false --skip_collage"}})
            node_to_seq[init_name] = seq
            edges.append((parents, init_name))
            cam_names = []
            for cam in fanout_cams:
                if cam == fanout_init:
                    continue
                name = f"{step}_{cam}_{idx}"
                jobs.append({"name": name, "sub": sub_path, "category": step,
                             "vars": {"seq_arg": seq, "cell_id": cam,
                                      "cell_flags": f"--cam_names {fanout_init} "
                                                    f"{cam} --cam_init {fanout_init} "
                                                    f"--rig_vote false --skip_collage"}})
                node_to_seq[name] = seq
                edges.append(([init_name], name))
                cam_names.append(name)
            final_name = f"{step}_finalize_{idx}"
            jobs.append({"name": final_name, "sub": sub_path, "category": step,
                         "vars": {"seq_arg": seq, "cell_id": "finalize",
                                  "cell_flags": f"--cam_init {fanout_init}"}})
            node_to_seq[final_name] = seq
            edges.append((cam_names or [init_name], final_name))
            tails[step][idx] = [final_name]

    dag_path = os.path.join(work_dir, f"{log_tag}.dag")
    Path(dag_path).write_text(render_dag(jobs, edges, maxjobs, opts.retries))
    return DagBundle(dag_path=dag_path, node_to_seq=node_to_seq)


def _flags_cam_init(flags) -> Optional[str]:
    """Last --cam_init in a step's flags list (argparse semantics), or None.
    Flags entries may hold several shell-style tokens each."""
    found = None
    tokens = [t for f in (flags or []) for t in str(f).split()]
    for i, t in enumerate(tokens):
        if t == "--cam_init" and i + 1 < len(tokens):
            found = tokens[i + 1]
        elif t.startswith("--cam_init="):
            found = t.split("=", 1)[1]
    return found


def _resolve_fanout(task_cfg: dict, step: str = "ma_masks"):
    """(cam_names, init_cam) when ``step`` camera fan-out is enabled, else
    (None, None). Requires an explicit global camera list; the init camera
    is the step flags' --cam_init (where presets and the GUI put it), else
    ``global.cam_init``, else the list head (the pipeline's own default).
    ma_2d callers ignore the init camera (its cells share no state)."""
    scfg = task_cfg.get(step, {}) or {}
    if not scfg.get("fanout_cameras"):
        return None, None
    g = task_cfg.get("global", {}) or {}
    cams = g.get("cam_names") or []
    if len(cams) < 2:
        log.warning("fanout_cameras ignored: global.cam_names must list "
                    ">=2 cameras explicitly")
        return None, None
    init = _flags_cam_init(scfg.get("flags")) or g.get("cam_init") or cams[0]
    if init not in cams:
        cams = [init] + list(cams)
    return list(cams), init


def build_submit_cmd(dag_path: str, opts: CondorOpts) -> List[str]:
    """The condor submit argv for ``dag_path`` (split out so it is testable)."""
    if opts.flavor == "bid":
        return ["condor_submit_dag_bid", str(opts.bid_price), dag_path]
    return ["condor_submit_dag", dag_path]


def submit(bundle: DagBundle, opts: CondorOpts) -> DagBundle:
    """Submit the DAG and parse the DAGMan cluster id for later status re-attach.

    Runs ``condor_submit_dag[_bid]`` in the current environment (login node in
    Phase 0; the SSH transport's ``.run()`` later). Raises ``FileNotFoundError``
    if condor is not on PATH, and ``subprocess.CalledProcessError`` on a
    non-zero submit.
    """
    proc = subprocess.run(
        build_submit_cmd(bundle.dag_path, opts),
        check=True, capture_output=True, text=True,
    )
    m = re.search(r"submitted to cluster (\d+)", proc.stdout or "")
    bundle.dag_cluster_id = m.group(1) if m else None
    return bundle
