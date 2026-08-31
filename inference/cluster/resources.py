"""Per-step cluster resource resolution (what to request from HTCondor).

Separates *what to run* (the task / run-config) from *what to ask the
scheduler for* (cpus, memory, gpus, CUDA floor, per-step job cap). Resolution
order, later wins:

1. ``DEFAULT_STEP_RESOURCES`` below (safe, live-validated per-step baseline;
   in particular every GPU step requests a GPU, so a bare task file can never
   silently land on a CPU-only node again),
2. the cluster profile's ``resources`` map (per-cluster overrides, the
   ``scheduler_opts.resources`` block from the design doc). The special key
   ``"default"`` applies to EVERY step before the step's own entry (e.g. the
   GUI's "Max parallel jobs" field writes ``default.max_jobs``),
3. the step's own ``submit_cfg`` in the task file (per-run override, kept for
   backward compatibility; no longer required).

The resolved block is baked into the run-config that is uploaded to the
cluster (``remote.submit_remote``), so the execute-node side (``engines``,
which reads ``submit_cfg.gpus`` to decide apptainer ``--nv``) and the emitter
(``htcondor.emit_dag``) both see explicit values, and the frozen uploaded
config records exactly what was requested.

Stdlib-only: ``emit_dag`` imports this on a bare cluster ``python3``.
"""
from __future__ import annotations

from typing import Dict, Optional

# Baseline requests per pipeline step. Values are the live-validated ones
# (smoke run on the MPI-IS cluster; lifted from the legacy
# mamma_apptainer/configs/tasks blocks, memory sized for multi-camera
# full-length sequences). Steps absent from this table (unknown or
# site-specific) fall back to render_sub's own defaults.
DEFAULT_STEP_RESOURCES: Dict[str, dict] = {
    # ``disk`` (MB) sizes the job's scratch dir (HTCondor request_disk); the
    # node wrapper routes TMPDIR there so temp spills are billed against it
    # instead of the node's small shared /tmp (task 152 post-mortem: the
    # ma_masks disk-backed mask store is ~0.8 MB/frame/person at 2056x1504,
    # so a long multi-person camera needs several GB of temp space).
    "ma_cap":   {"cpus": 4, "memory": 16000, "gpus": 0, "max_jobs": 20,
                 "disk": 4000},
    "ma_masks": {"cpus": 8, "memory": 64000, "gpus": 1,
                 "min_cuda_memory": 24000, "max_jobs": 10, "disk": 20000},
    "ma_2d":    {"cpus": 8, "memory": 32000, "gpus": 1, "max_jobs": 20,
                 "disk": 6000},
    "ma_3d":    {"cpus": 8, "memory": 48000, "gpus": 1,
                 "min_cuda_memory": 24000, "max_jobs": 20, "disk": 6000},
    "ma_vis":   {"cpus": 8, "memory": 32000, "gpus": 1, "max_jobs": 10,
                 "disk": 10000},
}


def resolve_submit_cfg(
    step: str,
    step_cfg: Optional[dict] = None,
    profile_resources: Optional[Dict[str, dict]] = None,
) -> dict:
    """Merge defaults <- profile ``default`` bucket <- profile per-step
    entry <- the step's own ``submit_cfg``."""
    merged = dict(DEFAULT_STEP_RESOURCES.get(step, {}))
    prof = profile_resources or {}
    merged.update(prof.get("default") or {})
    merged.update(prof.get(step) or {})
    merged.update((step_cfg or {}).get("submit_cfg") or {})
    return merged


def apply_resources(
    task_cfg: dict,
    profile_resources: Optional[Dict[str, dict]] = None,
) -> dict:
    """Return a copy of ``task_cfg`` with every step's ``submit_cfg`` resolved.

    Steps are the dict-valued top-level keys other than ``global``. The copy
    is shallow except for the step dicts that gain a resolved ``submit_cfg``.
    """
    out = dict(task_cfg)
    for key, val in task_cfg.items():
        if key == "global" or not isinstance(val, dict):
            continue
        step = dict(val)
        step["submit_cfg"] = resolve_submit_cfg(key, val, profile_resources)
        out[key] = step
    return out
