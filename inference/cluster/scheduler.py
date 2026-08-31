"""The scheduler seam: one adapter per batch scheduler, one registry.

Everything else in the cluster backend is scheduler-neutral: the execution
unit is ``run-cell`` per (step, sequence) with DONE sentinels, ground-truth
status flows through ``status.jsonl``, resources are expressed in neutral
units (cpus / memory / gpus / min_cuda_memory / max_jobs), and transport /
profiles / sync / baking never mention a scheduler. What remains, and what an
adapter provides, is exactly five operations:

1. ``emit``    - render the submission artifacts for a run-config
2. ``submit``  - submit them and learn the scheduler's job/DAG id
               (runs ON the login node; uses subprocess directly)
3. queue       - ``queue_cmd`` + ``parse_queue``: classify jobs still in the
               queue (Queued / Running / Failed-held), keyed by node name
               (client side; commands run through the Transport)
4. cancel      - ``cancel_cmd``
5. liveness    - ``alive_cmd`` + ``parse_alive``: is the run still in the
               scheduler at all (the monitor's finished() probe)

Adding a scheduler = one new adapter class here (or its own module) and a
registry entry; the pipeline, transport, monitor, GUI, and CLIs pick it up
via ``get_scheduler``. Stdlib-only: the cluster side imports this on a bare
``python3``.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from . import htcondor
from .status_map import map_condor_state


class Scheduler:
    """Base adapter. ``name`` identifies it in profiles / handle.json;
    ``flavors`` (optional) lists submit variants the GUI can offer."""

    name: str = ""
    flavors: Tuple[str, ...] = ()

    # --- login-node side (used by submit-cluster) --------------------------
    def emit(self, task_cfg: dict, out_tag: str, log_tag: str,
             opts, work_dir: str):
        raise NotImplementedError

    def submit(self, bundle, opts):
        """Submit ``bundle`` and set its scheduler job/DAG id. Runs on the
        login node (subprocess), not through the Transport."""
        raise NotImplementedError

    # --- client side (argv builders + parsers, run through the Transport) --
    def submit_host_cmd(self) -> List[str]:
        """Preflight probe: is this login node a submit host for me?"""
        raise NotImplementedError

    def queue_cmd(self, job_id: str) -> List[str]:
        raise NotImplementedError

    def parse_queue(self, raw: str) -> List[Tuple[str, str, Optional[str]]]:
        """``(node_name, gui_status, note)`` per job still in the queue."""
        raise NotImplementedError

    def cancel_cmd(self, job_id: str) -> List[str]:
        raise NotImplementedError

    def alive_cmd(self, job_id: str) -> List[str]:
        raise NotImplementedError

    def parse_alive(self, raw: str) -> bool:
        raise NotImplementedError


class HTCondorScheduler(Scheduler):
    """HTCondor DAGMan (with the MPI-IS ``bid`` variant). Delegates the
    rendering / submission to :mod:`inference.cluster.htcondor`."""

    name = "htcondor"
    flavors = ("bid", "standard")

    def emit(self, task_cfg, out_tag, log_tag, opts, work_dir):
        return htcondor.emit_dag(task_cfg, out_tag, log_tag, opts, work_dir)

    def submit(self, bundle, opts):
        return htcondor.submit(bundle, opts)

    def submit_host_cmd(self) -> List[str]:
        return ["bash", "-lc",
                "command -v condor_submit_dag_bid || command -v condor_submit_dag"]

    def queue_cmd(self, job_id: str) -> List[str]:
        return ["condor_q", str(job_id), "-json",
                "-attributes", "DAGNodeName,JobStatus,HoldReason"]

    def parse_queue(self, raw: str) -> List[Tuple[str, str, Optional[str]]]:
        try:
            jobs = json.loads(raw)
        except json.JSONDecodeError:
            return []
        out: List[Tuple[str, str, Optional[str]]] = []
        for job in jobs:
            node = job.get("DAGNodeName")
            if not node:
                continue
            mapped = map_condor_state(int(job.get("JobStatus", 0)),
                                      job.get("HoldReason"))
            if mapped is None:
                continue
            status, note = mapped
            out.append((node, status, note))
        return out

    def cancel_cmd(self, job_id: str) -> List[str]:
        return ["condor_rm", str(job_id)]

    def alive_cmd(self, job_id: str) -> List[str]:
        return ["condor_q", str(job_id), "-format", "%d\\n", "ClusterId"]

    def parse_alive(self, raw: str) -> bool:
        return bool((raw or "").strip())


SCHEDULERS: Dict[str, Scheduler] = {
    "htcondor": HTCondorScheduler(),
}


def get_scheduler(name: Optional[str]) -> Scheduler:
    """Look up an adapter by name; empty/None means the default (htcondor)."""
    key = (name or "htcondor").strip().lower()
    if key not in SCHEDULERS:
        raise ValueError(
            f"unknown scheduler {name!r}; available: {sorted(SCHEDULERS)}"
        )
    return SCHEDULERS[key]
