"""Cluster profiles: durable, reusable connection + submission settings.

A profile is set up once (host, credentials strategy, remote workspace, env,
submission flavor) and then referenced by name per run. Stored as YAML files
under ``$XDG_CONFIG_HOME/mamma/clusters/<name>.yaml`` (default
``~/.config/mamma/clusters``), so they are easy to share and version. No
secret is ever stored here; authentication is delegated to the user's SSH
setup (agent / ``~/.ssh/config`` / keys).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import yaml

# Never sync these to the cluster: outputs / datasets / git / images are large
# or wrong-on-the-cluster; .env.local carries local asset paths. The directory
# entries are ANCHORED with a leading "/" so they match only the repo-root dir,
# not same-named nested dirs (e.g. landmarks/configs/.../data/ must still sync;
# an unanchored "data/" would wrongly drop it). Unanchored patterns (*.pyc,
# __pycache__/, *.sif) intentionally match at any depth.
DEFAULT_SYNC_EXCLUDES: List[str] = [
    "/output/", "/data/", "/.git/", ".env.local",
    "/gui/frontend/node_modules/", "*.sif", "__pycache__/", "*.pyc", "*.egg-info/",
]


def clusters_dir() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "mamma", "clusters")


@dataclass
class ClusterProfile:
    """Everything needed to submit to and track one cluster.

    Per-step resource requests live here (``resources``), not in the task
    file: the task file says what to run, the profile says what to ask the
    scheduler for on this cluster. Asset paths stay with the cluster's own
    environment.
    """

    name: str
    host: str
    username: str = ""
    port: int = 22
    transport: str = "ssh"                     # "ssh" | "local"
    workspace_root: str = ""                   # per-user dir on the cluster (shared FS)
    repo_dir: str = ""                         # default: <workspace_root>/repo
    activate: str = ""                         # shell snippet run before MAMMA's job wrapper
                                               # on cluster nodes (e.g. "module load python");
                                               # NOT the pipeline env (that lives in the engine)
    python: str = ""                           # interpreter for the job wrapper on cluster
                                               # nodes; empty = auto-detect on the login node
                                               # (python3 first, then python)
    # --- execution environment (REQUIRED to submit) --------------------------
    # How steps execute on this cluster and where the env lives. Orthogonal to
    # the scheduler: any engine can ride any scheduler. `engine` picks the
    # type; the matching field locates it (cluster-side path/name):
    #   apptainer -> sif_path,  conda -> conda_env,  docker -> docker_image.
    # Empty engine on old profiles is read as apptainer when sif_path is set.
    engine: str = ""                           # "apptainer" | "conda" | "docker"
    sif_path: str = ""                         # .sif on the cluster's shared FS
    conda_env: str = ""                        # env name available on execute nodes
    docker_image: str = ""                     # image resolvable on execute nodes
    scheduler: str = "htcondor"                # batch scheduler adapter name
                                               # (see inference.cluster.scheduler)
    flavor: str = "bid"                        # htcondor: "bid" | "standard"
    bid_price: int = 100                       # htcondor bid flavor only
    retries: int = 1
    ssh_opts: List[str] = field(default_factory=list)          # e.g. ["-J", "bastion"]
    sync_excludes: List[str] = field(default_factory=lambda: list(DEFAULT_SYNC_EXCLUDES))
    # Per-step scheduler requests overriding inference.cluster.resources
    # defaults, e.g. {"ma_masks": {"memory": 128000, "min_cuda_memory": 40000}}.
    # Keys: cpus, memory (MB), gpus, min_cuda_memory (MB), max_cuda_capability
    # (cap GPU arch to what the env's torch build supports), max_jobs, nvccli.
    # The special entry "default" applies to every step before its own entry
    # (the GUI's "Max parallel jobs" field writes default.max_jobs here).
    resources: Dict[str, dict] = field(default_factory=dict)

    def resolved_repo_dir(self) -> str:
        return self.repo_dir or f"{self.workspace_root.rstrip('/')}/repo"

    def execution_env(self) -> tuple:
        """``(engine, location)`` for running steps on this cluster, or a
        ``(None, error_message)`` pair when the profile does not define one.

        Legacy profiles without ``engine`` but with ``sif_path`` resolve to
        apptainer.
        """
        eng = (self.engine or "").strip().lower() or ("apptainer" if self.sif_path else "")
        loc = {"apptainer": self.sif_path, "conda": self.conda_env,
               "docker": self.docker_image}.get(eng, "")
        if not eng:
            return None, (f"cluster profile {self.name!r} does not define its "
                          "execution environment (engine + location)")
        if eng not in ("apptainer", "conda", "docker"):
            return None, (f"cluster profile {self.name!r}: unknown engine {eng!r} "
                          "(expected apptainer, conda, or docker)")
        if not loc:
            field_name = {"apptainer": "sif_path", "conda": "conda_env",
                          "docker": "docker_image"}[eng]
            return None, (f"cluster profile {self.name!r}: engine {eng!r} needs "
                          f"{field_name} set (where the env lives on the cluster)")
        return eng, loc

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(clusters_dir(), f"{self.name}.yaml")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        return path


def _known_fields() -> set:
    return set(ClusterProfile.__dataclass_fields__.keys())


def from_dict(d: dict) -> ClusterProfile:
    known = _known_fields()
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown profile field(s): {sorted(unknown)}")
    if "name" not in d or "host" not in d:
        raise ValueError("profile requires at least 'name' and 'host'")
    return ClusterProfile(**d)


def load_profile(name_or_path: str) -> ClusterProfile:
    """Load a profile by file path or by name (looked up in ``clusters_dir``)."""
    path = name_or_path
    if not os.path.exists(path):
        path = os.path.join(clusters_dir(), f"{name_or_path}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"cluster profile {name_or_path!r} not found "
            f"(looked in {clusters_dir()})"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return from_dict(data)


def list_profiles() -> List[str]:
    d = clusters_dir()
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(d)
        if f.endswith((".yaml", ".yml"))
    )
