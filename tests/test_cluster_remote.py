"""Phase 1 tests: transport, profiles, remote submit choreography, poller.

No real cluster, no SSH, no HTCondor. Uses LocalTransport for real filesystem
primitives and small fakes for the remote submit + condor_q reconciliation.
"""
import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from inference.cluster import profile as profile_mod
from inference.cluster import remote as remote_mod
from inference.cluster.monitor import ClusterMonitor, reconcile
from inference.cluster.transport import CommandResult, LocalTransport, Transport
from inference.status import StatusSink


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #

def test_profile_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    prof = profile_mod.ClusterProfile(
        name="mpi", host="login.x", username="me", workspace_root="/home/me/ws",
        activate="conda activate mamma", flavor="bid", bid_price=77,
    )
    path = prof.save()
    assert path.endswith("clusters/mpi.yaml")
    loaded = profile_mod.load_profile("mpi")
    assert loaded == prof
    assert "mpi" in profile_mod.list_profiles()


def test_profile_resolved_repo_dir_default():
    prof = profile_mod.ClusterProfile(name="p", host="h", workspace_root="/ws/")
    assert prof.resolved_repo_dir() == "/ws/repo"
    prof2 = profile_mod.ClusterProfile(name="p", host="h", workspace_root="/ws",
                                       repo_dir="/custom/repo")
    assert prof2.resolved_repo_dir() == "/custom/repo"


def test_profile_resources_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    prof = profile_mod.ClusterProfile(
        name="res", host="h",
        resources={"ma_masks": {"memory": 128000, "min_cuda_memory": 40000}},
    )
    prof.save()
    loaded = profile_mod.load_profile("res")
    assert loaded.resources == {"ma_masks": {"memory": 128000,
                                             "min_cuda_memory": 40000}}


def test_profile_execution_env():
    # explicit engine + location
    p = profile_mod.ClusterProfile(name="p", host="h", engine="conda", conda_env="mamma")
    assert p.execution_env() == ("conda", "mamma")
    # legacy profile: sif_path set, engine empty -> apptainer inferred
    p = profile_mod.ClusterProfile(name="p", host="h", sif_path="/c/m.sif")
    assert p.execution_env() == ("apptainer", "/c/m.sif")
    # missing location for the chosen engine
    eng, msg = profile_mod.ClusterProfile(name="p", host="h", engine="docker").execution_env()
    assert eng is None and "docker_image" in msg
    # nothing defined at all
    eng, msg = profile_mod.ClusterProfile(name="p", host="h").execution_env()
    assert eng is None and "execution environment" in msg


def test_profile_from_dict_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown profile field"):
        profile_mod.from_dict({"name": "p", "host": "h", "bogus": 1})
    with pytest.raises(ValueError, match="requires at least"):
        profile_mod.from_dict({"host": "h"})


# --------------------------------------------------------------------------- #
# LocalTransport primitives
# --------------------------------------------------------------------------- #

def test_local_transport_run_and_files(tmp_path):
    tr = LocalTransport()
    assert tr.run(["echo", "hi"]).stdout.strip() == "hi"

    src = tmp_path / "a.txt"
    src.write_text("payload")
    dst = tmp_path / "sub" / "b.txt"
    tr.put(str(src), str(dst))
    assert dst.read_text() == "payload"

    back = tmp_path / "c.txt"
    tr.get(str(dst), str(back))
    assert back.read_text() == "payload"

    assert tr.exists(str(dst))
    assert not tr.exists(str(tmp_path / "nope"))


def test_local_transport_auth_hooks_are_noops():
    # Local transport: nothing to authenticate, nothing to connect.
    tr = LocalTransport()
    assert tr.auth_ok() is True
    assert tr.master_alive() is False
    assert tr.connect_argv() is None
    tr.close_master()  # must not raise


def test_ssh_transport_connect_and_probe_argv():
    from inference.cluster.transport import SshTransport
    tr = SshTransport("login.example.org", username="alice", port=2222,
                      ssh_opts=["-J", "bastion"])

    argv = tr.connect_argv(persist="8h")
    # A plain foreground ssh: OpenSSH prompts on the caller's TTY; the
    # authenticated master then lives on OUR control path for 8h idle.
    assert argv[0] == "ssh" and argv[-2:] == ["alice@login.example.org", "true"]
    assert "-J" in argv and "bastion" in argv
    assert ["-p", "2222"] == argv[1:3]
    joined = " ".join(argv)
    assert "ControlPersist=8h" in joined
    assert f"ControlPath={tr.control_path}" in joined

    probe = tr._auth_probe_argv()
    # BatchMode forbids prompting, so the probe can only succeed via
    # keys/agent or a live master, never by asking the user.
    assert "BatchMode=yes" in " ".join(probe)
    assert probe[-1] == "true"

    check = tr._master_ctl_argv("check")
    assert check[:1] == ["ssh"] and "-O" in check and "check" in check
    assert f"ControlPath={tr.control_path}" in " ".join(check)


def test_local_transport_tail_is_incremental(tmp_path):
    tr = LocalTransport()
    f = tmp_path / "status.jsonl"
    # missing file -> empty, offset preserved
    text, off = tr.tail(str(f), 0)
    assert text == "" and off == 0

    f.write_text("line1\n")
    text, off = tr.tail(str(f), 0)
    assert text == "line1\n" and off == 6

    # append; tail from the previous offset yields only the new bytes
    with open(f, "a") as fh:
        fh.write("line2\n")
    text2, off2 = tr.tail(str(f), off)
    assert text2 == "line2\n" and off2 == 12


def test_local_transport_sync_dir(tmp_path):
    tr = LocalTransport()
    src = tmp_path / "src"
    (src / "keep").mkdir(parents=True)
    (src / "keep" / "x.txt").write_text("x")
    (src / "skip").mkdir()
    (src / "skip" / "y.txt").write_text("y")
    dst = tmp_path / "dst"
    res = tr.sync_dir(str(src), str(dst), excludes=["skip/"])
    assert res.ok
    assert (dst / "keep" / "x.txt").read_text() == "x"
    assert not (dst / "skip").exists()


# --------------------------------------------------------------------------- #
# remote submit choreography (fake transport, no real python/condor)
# --------------------------------------------------------------------------- #

def test_build_remote_submit_argv_and_script():
    prof = profile_mod.ClusterProfile(
        name="p", host="h", username="me", workspace_root="/ws",
        activate="conda activate mamma", flavor="bid", bid_price=100,
    )
    argv = remote_mod.build_remote_submit_argv(
        prof, "/ws/run_configs/run7.json", "run7", "run7",
        "/ws/dag/run7", "/ws/dag/run7/status.jsonl", bid=250, retries=2,
    )
    # profile.python is empty (auto) and no resolved value was passed in,
    # so the argv falls back to python3
    assert argv[:4] == ["python3", "-m", "inference", "submit-cluster"]
    assert "--repo-dir" in argv and "/ws/repo" in argv
    assert "--bid" in argv and "250" in argv
    assert "--flavor" in argv and "bid" in argv
    assert "--scheduler" in argv and "htcondor" in argv

    script = remote_mod.build_remote_submit_script(prof, argv)
    assert "conda activate mamma" in script
    assert "cd /ws/repo" in script
    assert "submit-cluster" in script


class _PyProbeTransport(Transport):
    """run() answers only the python probe."""

    def __init__(self, stdout: str, ok: bool = True):
        self._out, self._ok = stdout, ok

    def run(self, argv):
        assert "command -v python3" in argv[-1]
        return CommandResult(0 if self._ok else 1, self._out)


def test_resolve_runner_python():
    prof_auto = profile_mod.ClusterProfile(name="p", host="h")
    prof_set = profile_mod.ClusterProfile(name="p", host="h", python="python3.11")
    # explicit profile value wins, no probe
    assert remote_mod.resolve_runner_python(_PyProbeTransport(""), prof_set) == "python3.11"
    # auto: basename of what the login node reports
    assert remote_mod.resolve_runner_python(
        _PyProbeTransport("/usr/bin/python3\n"), prof_auto) == "python3"
    assert remote_mod.probe_runner_python(_PyProbeTransport("/opt/bin/python\n")) == "python"
    # probe found nothing: falls back to python3 (preflight reports the miss)
    assert remote_mod.probe_runner_python(_PyProbeTransport("", ok=False)) is None
    assert remote_mod.resolve_runner_python(_PyProbeTransport("", ok=False), prof_auto) == "python3"


class _RecTransport(Transport):
    """Records choreography and serves a canned handle.json on get()."""

    def __init__(self, handle: dict):
        self._handle = handle
        self.calls = []

    def sync_dir(self, local_dir, remote_dir, excludes=None, dry_run=False):
        self.calls.append(("sync", remote_dir))
        return CommandResult(0)

    def mkdirs(self, remote_dir):
        self.calls.append(("mkdirs", remote_dir))
        return CommandResult(0)

    def put(self, local_path, remote_path):
        self.calls.append(("put", remote_path))

    def run_script(self, script):
        self.calls.append(("script", script))
        return CommandResult(0, "ok")

    def get(self, remote_path, local_path):
        self.calls.append(("get", remote_path))
        with open(local_path, "w") as f:
            json.dump(self._handle, f)

    def run(self, argv):
        return CommandResult(0)


def test_submit_remote_choreography(tmp_path):
    task = tmp_path / "run.json"
    task.write_text("{}")
    prof = profile_mod.ClusterProfile(
        name="p", host="h", username="me", workspace_root="/ws",
        activate="conda activate mamma",
    )
    canned = {
        "dag_cluster_id": "8421",
        "status_jsonl": "/ws/dag/run7/status.jsonl",
        "dag_path": "/ws/dag/run7/run7.dag",
        "node_to_seq": {"ma_cap_0": "seqA"},
    }
    tr = _RecTransport(canned)
    handle = remote_mod.submit_remote(
        prof, str(task), out_tag="run7", transport=tr,
    )
    kinds = [c[0] for c in tr.calls]
    assert kinds == ["sync", "mkdirs", "put", "script", "get"]
    # run-config uploaded to the expected workspace path
    assert ("put", "/ws/run_configs/run7.json") in tr.calls
    # handle parsed back
    assert handle.dag_cluster_id == "8421"
    assert handle.node_to_seq == {"ma_cap_0": "seqA"}
    assert handle.work_dir == "/ws/dag/run7"


def test_submit_remote_bakes_resources_into_uploaded_config(tmp_path):
    # The task file carries no scheduler concerns; the uploaded copy must have
    # a fully resolved submit_cfg per step (defaults <- profile <- task).
    task = tmp_path / "run.json"
    task.write_text(json.dumps({
        "global": {"out_dir": "/out"},
        "ma_masks": {"enabled": True},
        "ma_3d": {"enabled": True, "submit_cfg": {"gpus": 2}},
    }))
    prof = profile_mod.ClusterProfile(
        name="p", host="h", workspace_root="/ws",
        resources={"ma_masks": {"memory": 128000}},
    )

    class _CapturePut(_RecTransport):
        def __init__(self, handle):
            super().__init__(handle)
            self.uploaded = None

        def put(self, local_path, remote_path):
            with open(local_path) as f:
                self.uploaded = f.read()
            super().put(local_path, remote_path)

    tr = _CapturePut({"dag_cluster_id": "1", "node_to_seq": {}})
    remote_mod.submit_remote(prof, str(task), out_tag="r",
                             transport=tr, do_sync=False)
    cfg = json.loads(tr.uploaded)
    masks = cfg["ma_masks"]["submit_cfg"]
    assert masks["gpus"] == 1              # built-in default
    assert masks["memory"] == 128000       # profile override
    assert cfg["ma_3d"]["submit_cfg"]["gpus"] == 2   # task file wins
    assert "submit_cfg" not in cfg["global"]


def test_submit_remote_requires_workspace_root(tmp_path):
    task = tmp_path / "run.json"
    task.write_text("{}")
    prof = profile_mod.ClusterProfile(name="p", host="h")  # no workspace_root
    with pytest.raises(ValueError, match="workspace_root"):
        remote_mod.submit_remote(prof, str(task), out_tag="x",
                                 transport=_RecTransport({}))


# --------------------------------------------------------------------------- #
# ClusterMonitor reconciliation
# --------------------------------------------------------------------------- #

class _RecSink(StatusSink):
    def __init__(self):
        self.calls = []
        self.counts = []

    def update(self, step_name, seq_name, status, pid=None):
        self.calls.append((step_name, seq_name, status))

    def record_counts(self, step_name, seq_name, num_frames, num_cameras):
        self.counts.append((step_name, seq_name, num_frames, num_cameras))


class _MonTransport(Transport):
    """Serves a growing status stream and a scriptable condor_q."""

    def __init__(self):
        self.status_text = ""            # test appends to this
        self.condorq = "[]"              # -json response
        self.in_queue = True             # finished() flips when False

    def tail(self, path, offset):
        b = self.status_text.encode()
        return b[offset:].decode(), len(b)

    def run(self, argv):
        if argv[0] == "bash" and "cat " in argv[-1]:   # dir-mode status read
            return CommandResult(0, self.status_text)
        if "-json" in argv:
            return CommandResult(0, self.condorq)
        if "-format" in argv:            # finished() probe
            return CommandResult(0, "42\n" if self.in_queue else "")
        return CommandResult(0)

    lock_exists = True                   # .dag.lock presence (dag_lock tests)

    def exists(self, path):
        return self.lock_exists


def _jsonl(*rows):
    return "".join(json.dumps(r) + "\n" for r in rows)


def test_monitor_replays_jsonl_and_maps_condor():
    tr = _MonTransport()
    sink = _RecSink()
    mon = ClusterMonitor(tr, "42", "/ws/status.jsonl",
                         {"ma_cap_0": "S", "ma_masks_0": "S"}, sink)

    # Tick 1: ma_cap runs+done in the JSONL; ma_masks idle in condor_q.
    tr.status_text = _jsonl(
        {"step": "ma_cap", "seq": "S", "status": "Running"},
        {"step": "ma_cap", "seq": "S", "status": "Done"},
    )
    tr.condorq = json.dumps([
        {"DAGNodeName": "ma_masks_0", "JobStatus": 1},   # IDLE -> Queued
    ])
    mon.tick()
    assert ("ma_cap", "S", "Running") in sink.calls
    assert ("ma_cap", "S", "Done") in sink.calls
    assert ("ma_masks", "S", "Queued") in sink.calls

    # Tick 2: ma_masks now Done in the JSONL. Even if condor_q still lists it,
    # the terminal JSONL verdict must win (no overwrite).
    tr.status_text += _jsonl({"step": "ma_masks", "seq": "S", "status": "Done"})
    tr.condorq = json.dumps([{"DAGNodeName": "ma_masks_0", "JobStatus": 2}])  # RUNNING
    sink.calls.clear()
    mon.tick()
    assert ("ma_masks", "S", "Done") in sink.calls
    assert ("ma_masks", "S", "Running") not in sink.calls  # terminal wins

    assert mon.finished() is False   # alive in the queue (marks seen_alive)
    tr.in_queue = False
    assert mon.finished() is False   # one empty probe forgiven (schedd blip)
    assert mon.finished() is True    # two consecutive: verifiably done


def test_monitor_finished_grace_before_first_sighting():
    # A DAG not yet visible in condor_q right after submit (bid-scheduler
    # latency) must not be declared finished by the first empty probes,
    # or the run would go untracked.
    tr = _MonTransport()
    tr.in_queue = False
    mon = ClusterMonitor(tr, "42", "/ws/s.jsonl", {}, _RecSink())
    for _ in range(ClusterMonitor.GONE_STREAK_UNSEEN - 1):
        assert mon.finished() is False
    assert mon.finished() is True
    # a late appearance resets the streak and flips to the seen-alive rule
    tr2 = _MonTransport()
    tr2.in_queue = False
    mon2 = ClusterMonitor(tr2, "42", "/ws/s.jsonl", {}, _RecSink())
    assert mon2.finished() is False
    tr2.in_queue = True
    assert mon2.finished() is False
    tr2.in_queue = False
    assert mon2.finished() is False
    assert mon2.finished() is True


def test_monitor_replays_counts_events(tmp_path):
    # JsonlSink emits the ma_cap counts as an "event: counts" record ...
    from inference.status import JsonlSink
    p = tmp_path / "status.jsonl"
    s = JsonlSink(str(p))
    s.update("ma_cap", "seqA", "Done")
    s.record_counts("ma_cap", "seqA", 1151, 6)
    s.close()
    # ... and the monitor replays it via record_counts, not update, so the
    # GUI's #F/#V columns fill for cluster runs exactly as for local ones.
    tr = _MonTransport()
    tr.status_text = p.read_text()
    sink = _RecSink()
    mon = ClusterMonitor(tr, None, str(p), {}, sink)
    mon._tick_jsonl()
    assert sink.calls == [("ma_cap", "seqA", "Done")]
    assert sink.counts == [("ma_cap", "seqA", 1151, 6)]


def test_monitor_lock_overrides_lying_queue():
    # A cached condor_q can return empty while the DAG runs. With a dag_lock configured,
    # queue-gone alone must NOT finish while the lock is on disk.
    tr = _MonTransport()
    mon = ClusterMonitor(tr, "42", "/ws/s.jsonl", {}, _RecSink(),
                         dag_lock="/ws/dag/7/7.dag.lock")
    assert mon.finished() is False       # alive
    tr.in_queue = False                  # cached queue goes empty, lock stays
    for _ in range(ClusterMonitor.LOCK_OVERRIDE_PROBES - 5):
        assert mon.finished() is False   # lock wins over the lying queue
    # queue flapping back to alive resets every streak
    tr.in_queue = True
    assert mon.finished() is False
    # real completion: queue gone AND the lock removed
    tr.in_queue = False
    tr.lock_exists = False
    assert mon.finished() is False       # gone-debounce probe 1
    assert mon.finished() is True        # probe 2: verifiably done


def test_monitor_lock_stale_override_eventually_finishes():
    # A crashed DAGMan can leave a stale lock; after LOCK_OVERRIDE_PROBES
    # consecutive queue-gone probes the monitor trusts the queue anyway.
    tr = _MonTransport()
    mon = ClusterMonitor(tr, "42", "/ws/s.jsonl", {}, _RecSink(),
                         dag_lock="/ws/dag/7/7.dag.lock")
    assert mon.finished() is False       # seen alive once
    tr.in_queue = False                  # gone forever, lock never removed
    results = [mon.finished() for _ in range(ClusterMonitor.LOCK_OVERRIDE_PROBES + 2)]
    assert results[-1] is True
    assert not any(results[:ClusterMonitor.LOCK_OVERRIDE_PROBES - 1])


def test_monitor_dir_mode_dedups_and_orders():
    # Directory protocol: full read each tick, line-level dedup, records
    # re-ordered globally by ts (per-cell files interleave arbitrarily).
    tr = _MonTransport()
    sink = _RecSink()
    mon = ClusterMonitor(tr, None, "/ws/dag/7/status", {}, sink)  # no .jsonl -> dir
    tr.status_text = _jsonl(
        {"ts": 3, "step": "ma_cap", "seq": "S", "status": "Done"},
        {"ts": 1, "step": "ma_cap", "seq": "S", "status": "Running"},
        {"ts": 5, "step": "ma_cap", "seq": "S", "event": "counts",
         "num_frames": 9, "num_cameras": 2},
    ) + 'eras": 32}\n'                    # a torn fragment must be skipped
    mon._tick_jsonl()
    assert sink.calls == [("ma_cap", "S", "Running"), ("ma_cap", "S", "Done")]
    assert sink.counts == [("ma_cap", "S", 9, 2)]
    # second tick with the same content: everything deduped, nothing reapplied
    sink.calls.clear(); sink.counts.clear()
    mon._tick_jsonl()
    assert sink.calls == [] and sink.counts == []
    # new lines still flow
    tr.status_text += _jsonl({"ts": 7, "step": "ma_masks", "seq": "S", "status": "Running"})
    mon._tick_jsonl()
    assert sink.calls == [("ma_masks", "S", "Running")]


def test_monitor_legacy_file_keeps_partial_lines_for_next_tick():
    # Single-file protocol: a partially flushed final line must be neither
    # applied nor stepped over (the old code advanced the offset past it,
    # silently dropping the event).
    tr = _MonTransport()
    sink = _RecSink()
    mon = ClusterMonitor(tr, None, "/ws/status.jsonl", {}, sink)
    full = _jsonl({"step": "ma_cap", "seq": "S", "status": "Running"})
    partial = '{"step": "ma_cap", "seq": "S", "sta'
    tr.status_text = full + partial
    mon._tick_jsonl()
    assert sink.calls == [("ma_cap", "S", "Running")]
    # the partial line completes; it must be applied exactly once
    tr.status_text = full + '{"step": "ma_cap", "seq": "S", "status": "Done"}\n'
    sink.calls.clear()
    mon._tick_jsonl()
    assert sink.calls == [("ma_cap", "S", "Done")]


def test_monitor_surfaces_held_as_failed():
    tr = _MonTransport()
    sink = _RecSink()
    mon = ClusterMonitor(tr, "42", "/ws/status.jsonl", {"ma_2d_0": "S"}, sink)
    tr.status_text = ""  # nothing ran yet
    tr.condorq = json.dumps([
        {"DAGNodeName": "ma_2d_0", "JobStatus": 5, "HoldReason": "bid too low"},
    ])
    mon.tick()
    assert ("ma_2d", "S", "Failed") in sink.calls


def test_monitor_no_dag_id_is_finished():
    mon = ClusterMonitor(_MonTransport(), None, "/ws/s.jsonl", {}, _RecSink())
    assert mon.finished() is True


def test_itemized_changes_parses_files_only():
    from inference.cluster.remote import _itemized_changes
    out = "\n".join([
        ">f.st...... segmentation/core/pipeline.py",
        "<f+++++++++ inference/cluster/remote.py",
        "cd+++++++++ newdir/",
        ".d..t...... olddir/",
        ".f..t...... touched_only.py",
        "*deleting   gone.py",
    ])
    assert _itemized_changes(out) == [
        "segmentation/core/pipeline.py",
        "inference/cluster/remote.py",
        "gone.py",
    ]


def test_sync_code_reports_and_dry_run_compares(tmp_path):
    from inference.cluster.remote import sync_code
    from inference.cluster.profile import ClusterProfile
    src_repo = tmp_path / "repo"
    src_repo.mkdir()
    (src_repo / "a.py").write_text("one")
    prof = ClusterProfile(name="t", host="", transport="local",
                          workspace_root=str(tmp_path / "ws"),
                          repo_dir=str(tmp_path / "remote_repo"))
    tr = LocalTransport()
    changed = sync_code(tr, prof, local_repo=str(src_repo))
    assert changed == ["a.py"]
    assert sync_code(tr, prof, local_repo=str(src_repo)) == []      # in sync
    (src_repo / "a.py").write_text("two")
    os.utime(src_repo / "a.py", (0, 2_000_000_000))   # defeat size+mtime quick-check
    diff = sync_code(tr, prof, local_repo=str(src_repo), dry_run=True)
    assert diff == ["a.py"]                                          # compare only
    assert (tmp_path / "remote_repo" / "a.py").read_text() == "one"  # untouched
    assert sync_code(tr, prof, local_repo=str(src_repo)) == ["a.py"]


def test_monitor_finalize_missing_downgrades_stuck_cells():
    """A cell killed out-of-band (condor_rm) never streams a terminal state;
    finalize_missing must downgrade it so the board cannot show a dead run
    as Running forever, while leaving properly terminal cells untouched."""
    tr = _MonTransport()
    sink = _RecSink()
    mon = ClusterMonitor(tr, "42", "/ws/status.jsonl",
                         {"ma_cap_0": "S", "ma_masks_init_0": "S"}, sink)
    tr.status_text = _jsonl(
        {"step": "ma_cap", "seq": "S", "status": "Done"},
        {"step": "ma_masks", "seq": "S", "status": "Running"},
    )
    tr.condorq = ""
    mon.tick()
    assert mon.finalize_missing() == 1
    assert ("ma_masks", "S", "Cancelled") in sink.calls
    assert ("ma_cap", "S", "Cancelled") not in sink.calls
    # idempotent: a second pass finds nothing left to downgrade
    assert mon.finalize_missing() == 0


def test_monitor_fanout_node_names_map_to_the_step():
    """Camera fan-out nodes (<step>_init_<idx>, <step>_<CAM>_<idx>,
    <step>_finalize_<idx>) must update the real step row, not a phantom
    "ma_masks_IOI_02" step."""
    tr = _MonTransport()
    sink = _RecSink()
    nodes = {"ma_masks_init_0": "S", "ma_masks_IOI_02_0": "S",
             "ma_masks_finalize_0": "S", "ma_2d_0": "S",
             "ma_2d_IOI_05_0": "S"}
    mon = ClusterMonitor(tr, "42", "/ws/status.jsonl", nodes, sink)
    tr.status_text = ""
    tr.condorq = json.dumps([
        {"DAGNodeName": "ma_masks_IOI_02_0", "JobStatus": 2},
        {"DAGNodeName": "ma_masks_finalize_0", "JobStatus": 1},
        {"DAGNodeName": "ma_2d_0", "JobStatus": 1},
        {"DAGNodeName": "ma_2d_IOI_05_0", "JobStatus": 1},
    ])
    mon.tick()
    steps = {c[0] for c in sink.calls}
    assert steps == {"ma_masks", "ma_2d"}


# --------------------------------------------------------------------------- #
# one-shot reconcile (board re-sync after tracker death / resubmission)
# --------------------------------------------------------------------------- #

def _reconcile_transport(status_text, *, status_dir_exists=True, lock_exists=False):
    tr = _MonTransport()
    tr.status_text = status_text
    paths = {"/ws/dag/7/status": status_dir_exists,
             "/ws/dag/7/7.dag.lock": lock_exists}
    tr.exists = lambda p: paths[p]
    return tr


def test_reconcile_replays_and_finalizes_dead_run():
    """Replay restores completed steps; cells still non-terminal after a
    verified-dead DAG are downgraded (the task-163 freeze)."""
    tr = _reconcile_transport(_jsonl(
        {"ts": 1, "step": "ma_masks", "seq": "S", "status": "Running"},
        {"ts": 2, "step": "ma_masks", "seq": "S", "status": "Done"},
        {"ts": 3, "step": "ma_2d", "seq": "S", "status": "Running"},
    ))
    sink = _RecSink()
    res = reconcile(tr, "/ws/dag/7/status", "/ws/dag/7/7.dag.lock", sink)
    assert res["alive"] is False and res["downgraded"] == 1
    assert ("ma_masks", "S", "Done") in sink.calls
    assert ("ma_2d", "S", "Cancelled") in sink.calls


def test_reconcile_leaves_a_live_run_alone():
    tr = _reconcile_transport(_jsonl(
        {"ts": 1, "step": "ma_masks", "seq": "S", "status": "Running"},
    ), lock_exists=True)
    sink = _RecSink()
    res = reconcile(tr, "/ws/dag/7/status", "/ws/dag/7/7.dag.lock", sink)
    assert res["alive"] is True and res["downgraded"] == 0
    assert ("ma_masks", "S", "Cancelled") not in sink.calls


def test_reconcile_without_status_dir_touches_nothing():
    """A just-submitted task (no dag dir yet) must never get finalized."""
    tr = _reconcile_transport("", status_dir_exists=False)
    sink = _RecSink()
    res = reconcile(tr, "/ws/dag/7/status", "/ws/dag/7/7.dag.lock", sink)
    assert res["downgraded"] == 0 and res["seen"] == 0
    assert sink.calls == []
