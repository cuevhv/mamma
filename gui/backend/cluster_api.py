"""Cluster-profile REST API for the GUI (Flask blueprint).

Thin wrapper over :mod:`inference.cluster.profile` (durable YAML profiles under
``~/.config/mamma/clusters``) plus a connectivity preflight that reuses the
same transport the runner uses. Profiles hold no secrets (auth is delegated to
the user's SSH setup), so responses can return the full profile.

Registered in ``app.py`` via ``app.register_blueprint(cluster_bp)``.
"""
from __future__ import annotations

import shlex

from flask import Blueprint, jsonify, request

from inference.cluster import profile as profile_mod
from inference.cluster import remote as remote_mod
from inference.cluster import scheduler as scheduler_mod

cluster_bp = Blueprint("clusters", __name__)


@cluster_bp.route("/api/schedulers", methods=["GET"])
def list_schedulers():
    """The scheduler adapters this build can submit with. The GUI populates
    its Scheduler dropdown from here, so a new adapter appears without any
    frontend change."""
    return jsonify([
        {"name": s.name, "flavors": list(s.flavors)}
        for s in scheduler_mod.SCHEDULERS.values()
    ])


def _profile_summary(name: str) -> dict:
    p = profile_mod.load_profile(name)
    eng, loc = p.execution_env()
    return {
        "name": p.name,
        "host": p.host,
        "username": p.username,
        "transport": p.transport,
        "flavor": p.flavor,
        "workspaceRoot": p.workspace_root,
        "sifPath": p.sif_path,
        "engine": eng or "",             # "" = env not defined yet
        "envLocation": loc if eng else "",
    }


@cluster_bp.route("/api/clusters", methods=["GET"])
def list_clusters():
    names = profile_mod.list_profiles()
    out = []
    for n in names:
        try:
            out.append(_profile_summary(n))
        except Exception as e:  # noqa: BLE001 — one bad file must not hide the rest
            out.append({"name": n, "error": str(e)})
    return jsonify(out)


@cluster_bp.route("/api/clusters/<name>", methods=["GET"])
def get_cluster(name):
    try:
        return jsonify(profile_mod.load_profile(name).to_dict())
    except FileNotFoundError:
        return jsonify({"error": f"profile {name!r} not found"}), 404


@cluster_bp.route("/api/clusters", methods=["POST"])
def upsert_cluster():
    """Create or overwrite a profile from a JSON body (same fields as the
    ClusterProfile dataclass; camelCase or snake_case both accepted)."""
    data = request.json or {}
    # accept a few camelCase aliases the frontend may send
    alias = {"workspaceRoot": "workspace_root", "repoDir": "repo_dir",
             "bidPrice": "bid_price", "sshOpts": "ssh_opts",
             "syncExcludes": "sync_excludes", "sifPath": "sif_path",
             "condaEnv": "conda_env", "dockerImage": "docker_image"}
    norm = {alias.get(k, k): v for k, v in data.items()}
    try:
        prof = profile_mod.from_dict(norm)
        path = prof.save()
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"message": "saved", "name": prof.name, "path": path}), 201


@cluster_bp.route("/api/clusters/<name>", methods=["DELETE"])
def delete_cluster(name):
    import os
    path = os.path.join(profile_mod.clusters_dir(), f"{name}.yaml")
    if not os.path.exists(path):
        return jsonify({"error": f"profile {name!r} not found"}), 404
    os.remove(path)
    return jsonify({"message": "deleted", "name": name})


@cluster_bp.route("/api/clusters/<name>/connection", methods=["GET"])
def cluster_connection(name):
    """Authentication status for a profile. Read-only: the backend never
    handles credentials. When not connected, the UI shows ``connectCommand``
    for the user to run in a terminal (ssh prompts there; the resulting
    ControlMaster session is then reused by every GUI/CLI call)."""
    try:
        prof = profile_mod.load_profile(name)
    except FileNotFoundError:
        return jsonify({"error": f"profile {name!r} not found"}), 404

    tr = remote_mod.make_transport(prof)
    if tr.connect_argv() is None:  # local transport
        return jsonify({"connected": True, "method": "local",
                        "connectCommand": None})
    if tr.master_alive():
        return jsonify({"connected": True, "method": "session",
                        "connectCommand": None})
    if tr.auth_ok():
        return jsonify({"connected": True, "method": "keys",
                        "connectCommand": None})
    return jsonify({
        "connected": False, "method": None,
        "connectCommand": f"python -m inference cluster connect {shlex.quote(name)}",
    })


@cluster_bp.route("/api/clusters/<name>/test", methods=["POST"])
def test_cluster(name):
    """Connectivity + submit-host preflight. Returns a checklist the UI renders."""
    try:
        prof = profile_mod.load_profile(name)
    except FileNotFoundError:
        return jsonify({"error": f"profile {name!r} not found"}), 404

    tr = remote_mod.make_transport(prof)
    checks = []

    echo = tr.run(["echo", "ok"])
    conn_ok = echo.ok and echo.stdout.strip().endswith("ok")
    checks.append({"label": "connection + auth", "ok": conn_ok,
                   "detail": "" if conn_ok else echo.stderr.strip()[:200]})
    if not conn_ok:
        return jsonify({"profile": name, "ok": False, "checks": checks})

    try:
        sched = scheduler_mod.get_scheduler(prof.scheduler)
    except ValueError as e:
        checks.append({"label": "scheduler", "ok": False, "detail": str(e)})
        return jsonify({"profile": name, "ok": False, "checks": checks})
    submit_host = tr.run(sched.submit_host_cmd())
    checks.append({"label": f"submit host ({sched.name})", "ok": submit_host.ok,
                   "detail": submit_host.stdout.strip().splitlines()[-1]
                             if submit_host.stdout.strip() else ""})

    if prof.workspace_root:
        mk = tr.run(["mkdir", "-p", prof.workspace_root])
        checks.append({"label": "workspace writable", "ok": mk.ok,
                       "detail": prof.workspace_root if mk.ok else mk.stderr.strip()[:200]})
    else:
        checks.append({"label": "workspace writable", "ok": False,
                       "detail": "workspace_root not set"})

    # Execution environment: required, and checkable from the login node.
    eng, loc = prof.execution_env()
    if eng is None:
        checks.append({"label": "execution env", "ok": False, "detail": loc})
    elif eng == "apptainer":
        sif_ok = tr.exists(loc)
        checks.append({"label": "execution env", "ok": sif_ok,
                       "detail": f"apptainer {loc}" if sif_ok
                                 else f"sif not found on cluster: {loc}"})
    else:
        tool = tr.run(["bash", "-lc", f"command -v {eng}"])
        checks.append({"label": "execution env", "ok": tool.ok,
                       "detail": f"{eng} {loc}" if tool.ok
                                 else f"{eng} not found on the login node"})

    if prof.python:
        runner_py = prof.python
        checks.append({"label": "runner python", "ok": True, "detail": runner_py})
    else:
        found = remote_mod.probe_runner_python(tr)
        runner_py = found or "python3"
        checks.append({
            "label": "runner python", "ok": bool(found),
            "detail": f"{runner_py} (auto-detected)" if found
                      else "no python3/python on the login node",
        })

    repo = prof.resolved_repo_dir()
    repo_ok = tr.exists(repo)
    checks.append({"label": "repo present", "ok": repo_ok,
                   "detail": repo if repo_ok else f"{repo} (submit once to sync)"})

    if repo_ok:
        try:
            diff = remote_mod.sync_code(tr, prof, dry_run=True)
            checks.append({"label": "code sync", "ok": True,
                           "detail": "cluster copy matches the local tree"
                           if not diff else
                           f"{len(diff)} file(s) differ; submitting syncs them"})
        except Exception as e:  # noqa: BLE001 - preflight must not fail on this
            checks.append({"label": "code sync", "ok": True,
                           "detail": f"could not compare ({e})"})

    if repo_ok:
        script = (prof.activate + "\n" if prof.activate else "") + \
            f"cd {shlex.quote(repo)}\n{runner_py} -m inference --version"
        py = tr.run_script(script)
        checks.append({"label": "runner env", "ok": py.ok,
                       "detail": (py.stdout.strip() or py.stderr.strip())[-200:]})

    return jsonify({"profile": name, "ok": all(c["ok"] for c in checks), "checks": checks})
