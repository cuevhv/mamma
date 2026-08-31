"""``python -m inference cluster <action>`` — manage cluster profiles.

Actions:
  list                    List saved profiles.
  show <name>             Print a profile.
  add  <name> --host ...  Create / overwrite a profile from flags.
  test <name>             Connectivity + submit-host preflight over the transport.
  remove <name>           Delete a profile.
  connect <name>          One-time interactive login (password / 2FA clusters):
                          ssh prompts in THIS terminal, then the GUI and CLI
                          reuse the authenticated session. Not needed when
                          SSH keys already work.
  status <name>           Show how the profile authenticates right now.
  disconnect <name>       Close the shared session opened by `connect`.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

import yaml

from ..cluster import profile as profile_mod
from ..cluster import remote as remote_mod
from ..cluster import scheduler as scheduler_mod
from ..env import bootstrap_env


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m inference cluster",
        description="Manage remote cluster profiles.",
    )
    sub = p.add_subparsers(dest="action", required=True)

    sub.add_parser("list", help="List saved profiles")

    sp = sub.add_parser("show", help="Print a profile")
    sp.add_argument("name")

    ap = sub.add_parser("add", help="Create / overwrite a profile")
    ap.add_argument("name")
    ap.add_argument("--host", required=True)
    ap.add_argument("--username", default="")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--transport", choices=["ssh", "local"], default="ssh")
    ap.add_argument("--workspace-root", required=True,
                    help="Per-user folder on the cluster's SHARED filesystem; "
                         "the synced repo, run configs, DAG files, logs, and "
                         "all step outputs go under it. Use the path name the "
                         "cluster nodes see; if that storage is also mounted "
                         "locally, prefer the name valid on both (e.g. "
                         "/shared/<user>/... rather than /home/<user>/...) "
                         "so the GUI can open logs/results directly")
    ap.add_argument("--repo-dir", default="")
    ap.add_argument("--activate", default="")
    ap.add_argument("--python", default="",
                    help="Interpreter for MAMMA's job wrapper on cluster nodes "
                         "(not the pipeline env). Default: auto-detect on the "
                         "login node (python3 first, then python)")
    ap.add_argument("--scheduler", choices=sorted(scheduler_mod.SCHEDULERS),
                    default="htcondor",
                    help="Batch scheduler adapter this cluster uses")
    ap.add_argument("--flavor", choices=["bid", "standard"], default="bid")
    ap.add_argument("--bid-price", type=int, default=100)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--ssh-opt", action="append", default=[], dest="ssh_opts",
                    help="Extra ssh token, repeatable (e.g. --ssh-opt -J --ssh-opt bastion)")
    ap.add_argument("--engine", choices=["apptainer", "conda", "docker"],
                    default="apptainer",
                    help="REQUIRED execution environment type on this cluster: "
                         "how steps run there (independent of the scheduler)")
    ap.add_argument("--sif-path", default="",
                    help="Cluster-side .sif path (engine=apptainer)")
    ap.add_argument("--conda-env", default="",
                    help="Conda env name on the execute nodes (engine=conda)")
    ap.add_argument("--docker-image", default="",
                    help="Docker image resolvable on the nodes (engine=docker)")
    ap.add_argument("--max-jobs", type=int, default=None,
                    help="Cap concurrent jobs per pipeline step (scheduler "
                         "MAXJOBS). Default: per-step built-ins (10-20)")

    tp = sub.add_parser("test", help="Preflight a profile's connection")
    tp.add_argument("name")

    rm = sub.add_parser("remove", help="Delete a profile")
    rm.add_argument("name")

    cn = sub.add_parser(
        "connect",
        help="One-time interactive login for password/2FA clusters; the "
             "GUI and CLI then reuse the session")
    cn.add_argument("name")
    cn.add_argument("--persist", default="8h",
                    help="How long ssh keeps the authenticated session "
                         "after the last use (ControlPersist; default 8h)")

    st = sub.add_parser("status", help="Show how the profile authenticates")
    st.add_argument("name")

    dc = sub.add_parser("disconnect",
                        help="Close the shared session opened by `connect`")
    dc.add_argument("name")

    return p


def _do_list() -> int:
    names = profile_mod.list_profiles()
    if not names:
        print(f"No profiles in {profile_mod.clusters_dir()}")
        return 0
    for n in names:
        print(n)
    return 0


def _do_show(name: str) -> int:
    prof = profile_mod.load_profile(name)
    yaml.safe_dump(prof.to_dict(), sys.stdout, sort_keys=False)
    return 0


def _do_add(args) -> int:
    prof = profile_mod.ClusterProfile(
        name=args.name, host=args.host, username=args.username, port=args.port,
        transport=args.transport, workspace_root=args.workspace_root,
        repo_dir=args.repo_dir, activate=args.activate, python=args.python,
        flavor=args.flavor, bid_price=args.bid_price, retries=args.retries,
        ssh_opts=args.ssh_opts, engine=args.engine, sif_path=args.sif_path,
        conda_env=args.conda_env, docker_image=args.docker_image,
        scheduler=args.scheduler,
        resources=({"default": {"max_jobs": args.max_jobs}}
                   if args.max_jobs else {}),
    )
    eng, loc = prof.execution_env()
    if eng is None:
        print(f"error: {loc}", file=sys.stderr)
        return 2
    path = prof.save()
    print(f"Saved profile '{prof.name}' -> {path}  (env: {eng} {loc})")
    return 0


def _do_remove(name: str) -> int:
    path = os.path.join(profile_mod.clusters_dir(), f"{name}.yaml")
    if not os.path.exists(path):
        print(f"error: profile {name!r} not found in {profile_mod.clusters_dir()}",
              file=sys.stderr)
        return 1
    os.remove(path)
    print(f"Removed profile '{name}' ({path})")
    return 0


def _check(label: str, ok: bool, detail: str = "") -> None:
    mark = "OK " if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)


def _do_test(name: str) -> int:
    prof = profile_mod.load_profile(name)
    tr = remote_mod.make_transport(prof)
    print(f"Testing '{prof.name}' ({prof.username}@{prof.host}) ...")

    echo = tr.run(["echo", "ok"])
    _check("connection + auth", echo.ok and echo.stdout.strip() == "ok",
           echo.stderr.strip())
    if not echo.ok:
        print("Cannot reach the cluster; aborting further checks.")
        return 1

    try:
        sched = scheduler_mod.get_scheduler(prof.scheduler)
    except ValueError as e:
        _check("scheduler", False, str(e))
        return 1
    submit_host = tr.run(sched.submit_host_cmd())
    _check(f"submit host ({sched.name})", submit_host.ok, submit_host.stdout.strip())

    ws = prof.workspace_root
    if ws:
        mk = tr.run(["mkdir", "-p", ws])
        _check("workspace writable", mk.ok, ws or mk.stderr.strip())
    else:
        _check("workspace writable", False, "workspace_root not set in profile")

    eng, loc = prof.execution_env()
    if eng is None:
        _check("execution env", False, loc)
    elif eng == "apptainer":
        _check("execution env", tr.exists(loc),
               f"{eng} {loc}" if tr.exists(loc) else f"sif not found on cluster: {loc}")
    else:
        tool = tr.run(["bash", "-lc", f"command -v {eng}"])
        _check("execution env", tool.ok,
               f"{eng} {loc}" if tool.ok else f"{eng} not found on the login node")

    if prof.python:
        runner_py = prof.python
        _check("runner python", True, runner_py)
    else:
        found = remote_mod.probe_runner_python(tr)
        runner_py = found or "python3"
        _check("runner python", bool(found),
               f"{runner_py} (auto-detected)" if found
               else "no python3/python on the login node")

    repo = prof.resolved_repo_dir()
    repo_ok = tr.exists(repo)
    _check("repo present", repo_ok, repo if repo_ok else f"{repo} (run submit-remote to sync)")

    if repo_ok:
        script = ""
        if prof.activate:
            script += prof.activate + "\n"
        script += f"cd {shlex.quote(repo)}\n{runner_py} -m inference --version"
        py = tr.run_script(script)
        _check("runner env (python -m inference)", py.ok,
               py.stdout.strip() or py.stderr.strip())
    return 0


def _do_connect(name: str, persist: str) -> int:
    prof = profile_mod.load_profile(name)
    tr = remote_mod.make_transport(prof)
    argv = tr.connect_argv(persist)
    if argv is None:
        print("Local transport: no authentication involved.")
        return 0
    if tr.master_alive():
        print(f"A session for '{prof.name}' is already active; nothing to do.")
        return 0
    if tr.auth_ok():
        print(f"'{prof.name}' already authenticates via your SSH keys/agent; "
              "no session needed.")
        return 0
    # Foreground ssh with THIS terminal's TTY: OpenSSH does its own prompting
    # (LDAP password, OTP, ...). No credential ever passes through MAMMA. On
    # success the ControlMaster stays alive for --persist of idle time and
    # every GUI/CLI call to this profile reuses it.
    print(f"Authenticating to {prof.username + '@' if prof.username else ''}"
          f"{prof.host} (session kept for {persist} after last use)")
    print(f"  running: {shlex.join(argv)}")
    rc = subprocess.call(argv)
    if rc == 0 and tr.master_alive():
        print("Connected. The GUI and CLI will reuse this session.")
        return 0
    print("Authentication did not complete.", file=sys.stderr)
    return rc or 1


def _do_conn_status(name: str) -> int:
    prof = profile_mod.load_profile(name)
    tr = remote_mod.make_transport(prof)
    if tr.connect_argv() is None:
        print("local transport: no authentication involved")
        return 0
    if tr.master_alive():
        print("authenticated: shared terminal session (ssh ControlMaster)")
        return 0
    if tr.auth_ok():
        print("authenticated: SSH keys/agent")
        return 0
    print(f"not authenticated: run  python -m inference cluster connect {shlex.quote(name)}")
    return 1


def _do_disconnect(name: str) -> int:
    prof = profile_mod.load_profile(name)
    tr = remote_mod.make_transport(prof)
    if not tr.master_alive():
        print("No active session.")
        return 0
    tr.close_master()
    print("Session closed.")
    return 0


def main(argv=None) -> None:
    bootstrap_env()
    args = _build_parser().parse_args(argv)
    try:
        if args.action == "list":
            sys.exit(_do_list())
        if args.action == "show":
            sys.exit(_do_show(args.name))
        if args.action == "add":
            sys.exit(_do_add(args))
        if args.action == "test":
            sys.exit(_do_test(args.name))
        if args.action == "remove":
            sys.exit(_do_remove(args.name))
        if args.action == "connect":
            sys.exit(_do_connect(args.name, args.persist))
        if args.action == "status":
            sys.exit(_do_conn_status(args.name))
        if args.action == "disconnect":
            sys.exit(_do_disconnect(args.name))
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
