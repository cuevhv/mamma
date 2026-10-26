"""Unit tests for the converged HTCondor emitter and single-cell runner.

No cluster, no HTCondor, no GPU, no model weights. Covers:
* the pure renderers (render_node_sh / render_sub / render_dag),
* emit_dag end-to-end against a monkeypatched sequence resolver,
* the pure condor-state -> GUI-status map,
* runner.run_cell's DONE-skip / success / failure paths (engine mocked).
"""
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from inference import runner
from inference.cluster import htcondor, resources, status_map
from inference.status import StatusSink


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _opts(**kw):
    base = dict(
        flavor="bid", bid_price=100, retries=1,
        repo_dir="/cluster/repo", activate="conda activate mamma",
        remote_task_path="/cluster/run_configs/run_7.json",
        status_jsonl="/cluster/logs/status.jsonl",
    )
    base.update(kw)
    return htcondor.CondorOpts(**base)


def _task_cfg():
    """A minimal but realistic run-config: a 3-step chain, apptainer engine."""
    return {
        "global": {"dataset_name": "demo", "out_dir": "/out", "seq_ids": [0, 1]},
        "ma_cap": {
            "enabled": True, "dependencies": [], "engine": "apptainer",
            "submit_cfg": {"cpus": 4, "memory": 16000, "gpus": 0, "max_jobs": 20},
        },
        "ma_masks": {
            "enabled": True, "dependencies": ["ma_cap"], "engine": "apptainer",
            "submit_cfg": {"cpus": 8, "memory": 64000, "gpus": 1,
                           "min_cuda_memory": 24000, "max_jobs": 10},
        },
        "ma_2d": {
            "enabled": True, "dependencies": ["ma_masks"], "engine": "apptainer",
            "submit_cfg": {"cpus": 8, "memory": 32000, "gpus": 1, "max_jobs": 20},
        },
        # disabled: must not appear in the DAG
        "ma_3d": {"enabled": False, "dependencies": ["ma_2d"]},
    }


# --------------------------------------------------------------------------- #
# render_node_sh
# --------------------------------------------------------------------------- #

def test_node_sh_re_enters_runner_with_seq_arg():
    sh = htcondor.render_node_sh("ma_2d", _opts(), out_tag="run7", log_tag="run7")
    assert sh.startswith("#!/bin/bash")
    assert "set -euo pipefail" in sh
    assert 'SEQ="$1"' in sh
    assert "conda activate mamma" in sh
    assert "cd /cluster/repo" in sh
    # No apptainer/argv logic here: it defers to run-cell (that is the point).
    assert "apptainer" not in sh
    assert "python -m inference run-cell" in sh
    assert "--step ma_2d" in sh
    assert '--seq "$SEQ"' in sh
    assert "--status-jsonl /cluster/logs/status.jsonl" in sh
    assert "--task /cluster/run_configs/run_7.json" in sh


def test_node_sh_omits_activate_when_empty():
    sh = htcondor.render_node_sh("ma_cap", _opts(activate=""), "t", "t")
    assert "conda activate" not in sh
    assert "cd /cluster/repo" in sh


def test_node_sh_per_cell_status_files_in_dir_mode():
    # A status DIRECTORY means one file per (step, seq): single writer each,
    # so concurrent nodes never tear a shared append target.
    sh = htcondor.render_node_sh(
        "ma_masks", _opts(status_jsonl="/cluster/dag/7/status"), "t7", "t7")
    assert "mkdir -p /cluster/dag/7/status" in sh
    assert '--status-jsonl /cluster/dag/7/status/"$SEQ".ma_masks"${CELL:+.$CELL}".jsonl' in sh
    # legacy .jsonl path keeps the shared-file behavior (no mkdir)
    sh2 = htcondor.render_node_sh("ma_masks", _opts(), "t7", "t7")
    assert "--status-jsonl /cluster/logs/status.jsonl" in sh2
    assert "mkdir -p" not in sh2


# --------------------------------------------------------------------------- #
# render_sub
# --------------------------------------------------------------------------- #

def test_sub_gpu_step_has_cuda_requirement():
    sub = htcondor.render_sub(
        "ma_masks", "/w/ma_masks.sh", "/w/logs/ma_masks",
        {"cpus": 8, "memory": 64000, "gpus": 1, "min_cuda_memory": 24000, "max_jobs": 10},
    )
    assert "executable = /w/ma_masks.sh" in sub
    assert 'arguments = "$(seq_arg) $(cell_id) \'$(cell_flags)\'"' in sub
    assert "request_cpus = 8" in sub
    assert "request_memory = 64000" in sub
    assert "request_gpus = 1" in sub
    assert "requirements = (TARGET.CUDAGlobalMemoryMb > 24000)" in sub
    assert "max_materialize = 10" in sub
    assert sub.rstrip().endswith("queue 1")


def test_sub_gpu_step_caps_cuda_capability():
    # max_cuda_capability keeps jobs off GPUs newer than the container's
    # torch build (e.g. sm_100 B200 vs a cu124 wheel that stops at sm_90).
    sub = htcondor.render_sub(
        "ma_masks", "/w/ma_masks.sh", "/w/logs/ma_masks",
        {"gpus": 1, "min_cuda_memory": 24000, "max_cuda_capability": 9.0},
    )
    assert ("requirements = (TARGET.CUDAGlobalMemoryMb > 24000) && "
            "(TARGET.CUDACapability <= 9.0)") in sub
    # capability cap alone also works
    sub2 = htcondor.render_sub(
        "ma_2d", "/w/ma_2d.sh", "/w/logs/ma_2d",
        {"gpus": 1, "max_cuda_capability": 9.0},
    )
    assert "requirements = (TARGET.CUDACapability <= 9.0)" in sub2
    assert "CUDAGlobalMemoryMb" not in sub2


def test_sub_gpu_step_floors_cuda_capability():
    # min_cuda_capability keeps jobs off GPUs older than the compiled
    # extensions support (the image builds them for 8.0 and newer).
    sub = htcondor.render_sub(
        "ma_3d", "/w/ma_3d.sh", "/w/logs/ma_3d",
        {"gpus": 1, "max_cuda_capability": 9.0, "min_cuda_capability": 8.0},
    )
    assert ("requirements = (TARGET.CUDACapability <= 9.0) && "
            "(TARGET.CUDACapability >= 8.0)") in sub


def test_sub_gpu_step_excludes_blocklisted_machines():
    """Profile resources.default.exclude_machines keeps cells off wedged
    GPU hosts (a fast-failing node vacuums the queue)."""
    sub = htcondor.render_sub(
        "ma_masks", "/w/ma_masks.sh", "/w/logs/ma_masks",
        {"gpus": 1, "min_cuda_memory": 24000,
         "exclude_machines": ["badnode.example.org"]},
    )
    assert ("requirements = (TARGET.CUDAGlobalMemoryMb > 24000) && "
            '(TARGET.Machine =!= "badnode.example.org")') in sub


def test_sub_cpu_step_also_excludes_blocklisted_machines():
    """CPU steps (ma_cap) land on the same hosts, so the blocklist applies
    to them too; without GPUs no CUDA requirement is added."""
    sub = htcondor.render_sub(
        "ma_cap", "/w/ma_cap.sh", "/w/logs/ma_cap",
        {"gpus": 0, "exclude_machines": ["badnode.example.org"]},
    )
    assert 'requirements = (TARGET.Machine =!= "badnode.example.org")' in sub
    assert "request_gpus" not in sub and "CUDA" not in sub
    plain = htcondor.render_sub("ma_cap", "/w/ma_cap.sh", "/w/logs/ma_cap", {"gpus": 0})
    assert "requirements" not in plain


def test_sub_requests_disk_with_explicit_mb_unit():
    # request_disk's bare-number unit is KiB (unlike request_memory's MB), so
    # the MB suffix is load-bearing; the scratch this sizes is where the node
    # wrapper points TMPDIR (the ma_masks spill can fill the node's /tmp).
    sub = htcondor.render_sub(
        "ma_masks", "/w/ma_masks.sh", "/w/logs/ma_masks",
        {"gpus": 1, "disk": 20000},
    )
    assert "request_disk = 20000MB" in sub
    # default when the resolved cfg carries no disk key
    sub2 = htcondor.render_sub("ma_cap", "/w/s.sh", "/w/l", {"gpus": 0})
    assert "request_disk = 8000MB" in sub2


def test_node_sh_routes_tmpdir_to_condor_scratch():
    sh = htcondor.render_node_sh(
        "ma_masks", _opts(status_jsonl="/w/status"), "out", "log")
    assert 'TMPDIR="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"' in sh
    assert "export TMPDIR" in sh


def test_default_resources_carry_disk():
    from inference.cluster.resources import resolve_submit_cfg
    assert resolve_submit_cfg("ma_masks")["disk"] == 20000
    # per-step profile override wins
    assert resolve_submit_cfg(
        "ma_masks", profile_resources={"ma_masks": {"disk": 40000}}
    )["disk"] == 40000


def test_sub_cpu_step_has_no_gpu_or_cuda_lines():
    # ma_cap is CPU-only: the spurious CUDA requirement the original template
    # always emitted must be absent here.
    sub = htcondor.render_sub(
        "ma_cap", "/w/ma_cap.sh", "/w/logs/ma_cap",
        {"cpus": 4, "memory": 16000, "gpus": 0, "max_jobs": 20},
    )
    assert "request_gpus" not in sub
    assert "CUDAGlobalMemoryMb" not in sub
    assert "request_cpus = 4" in sub


# --------------------------------------------------------------------------- #
# render_dag
# --------------------------------------------------------------------------- #

def test_dag_wires_dependencies_index_wise():
    # two sequences (idx 0,1), a -> b chain
    def job(step, idx, seq, sub):
        return {"name": f"{step}_{idx}", "sub": sub, "category": step,
                "vars": {"seq_arg": seq, "cell_id": "", "cell_flags": ""}}
    jobs = [
        job("ma_cap", 0, "seqA", "/w/ma_cap.sub"),
        job("ma_cap", 1, "seqB", "/w/ma_cap.sub"),
        job("ma_masks", 0, "seqA", "/w/ma_masks.sub"),
        job("ma_masks", 1, "seqB", "/w/ma_masks.sub"),
    ]
    dag = htcondor.render_dag(
        jobs,
        edges=[(["ma_cap_0"], "ma_masks_0"), (["ma_cap_1"], "ma_masks_1")],
        maxjobs={"ma_cap": 20, "ma_masks": 10},
        retries=2,
    )
    assert "JOB ma_cap_0 /w/ma_cap.sub" in dag
    assert 'seq_arg="seqB"' in dag and "VARS ma_masks_1" in dag
    assert "CATEGORY ma_masks_0 ma_masks" in dag
    assert "MAXJOBS ma_masks 10" in dag
    assert "RETRY ma_masks_1 2" in dag
    # index-wise: masks_0 <- cap_0, masks_1 <- cap_1 (never cross-index)
    assert "PARENT ma_cap_0 CHILD ma_masks_0" in dag
    assert "PARENT ma_cap_1 CHILD ma_masks_1" in dag
    assert "PARENT ma_cap_0 CHILD ma_masks_1" not in dag
    # roots have no PARENT line
    assert "CHILD ma_cap_0" not in dag


def test_dag_quotes_paths_with_spaces():
    jobs = [{"name": "ma_cap_0", "sub": "/w s/ma_cap.sub", "category": "ma_cap",
             "vars": {"seq_arg": "s", "cell_id": "", "cell_flags": ""}}]
    dag = htcondor.render_dag(jobs, [], {"ma_cap": 20}, 0)
    assert 'JOB ma_cap_0 "/w s/ma_cap.sub"' in dag
    assert "# Retries" not in dag  # retries=0 -> no RETRY block


# --------------------------------------------------------------------------- #
# emit_dag (end to end, filesystem, seq resolution monkeypatched)
# --------------------------------------------------------------------------- #

def test_emit_dag_writes_all_artifacts(tmp_path, monkeypatch):
    # Avoid needing a real capture file on disk.
    monkeypatch.setattr(htcondor, "resolve_seq_names", lambda cfg: ["seqA", "seqB"])
    work = tmp_path / "dag"
    opts = _opts()
    bundle = htcondor.emit_dag(_task_cfg(), "run7", "run7", opts, str(work))

    # 3 enabled steps x 2 sequences = 6 nodes; ma_3d disabled -> absent.
    assert len(bundle.node_to_seq) == 6
    assert bundle.node_to_seq["ma_2d_1"] == "seqB"
    assert not any(k.startswith("ma_3d") for k in bundle.node_to_seq)

    for step in ("ma_cap", "ma_masks", "ma_2d"):
        assert (work / f"{step}.sh").exists()
        assert (work / f"{step}.sub").exists()
        assert (work / "logs" / step).is_dir()
        # .sh is executable
        assert os.access(work / f"{step}.sh", os.X_OK)

    dag_text = (work / "run7.dag").read_text()
    assert "PARENT ma_cap_0 CHILD ma_masks_0" in dag_text
    assert "PARENT ma_masks_1 CHILD ma_2d_1" in dag_text
    assert bundle.dag_path == str(work / "run7.dag")


def test_resolve_submit_cfg_precedence():
    # Defaults alone: a GPU step requests a GPU with no config at all.
    d = resources.resolve_submit_cfg("ma_masks", {})
    assert d["gpus"] == 1 and d["min_cuda_memory"] == 24000
    # Profile resources override the defaults.
    d = resources.resolve_submit_cfg("ma_masks", {}, {"ma_masks": {"memory": 128000}})
    assert d["memory"] == 128000 and d["gpus"] == 1
    # The task file's own submit_cfg wins over both.
    d = resources.resolve_submit_cfg(
        "ma_masks", {"submit_cfg": {"gpus": 2}}, {"ma_masks": {"gpus": 4}})
    assert d["gpus"] == 2
    # The "default" bucket applies to every step, under per-step entries
    # (the GUI's "Max parallel jobs" field writes default.max_jobs).
    prof = {"default": {"max_jobs": 3}, "ma_2d": {"max_jobs": 7}}
    assert resources.resolve_submit_cfg("ma_masks", {}, prof)["max_jobs"] == 3
    assert resources.resolve_submit_cfg("ma_3d", {}, prof)["max_jobs"] == 3
    assert resources.resolve_submit_cfg("ma_2d", {}, prof)["max_jobs"] == 7
    # Unknown step: only the default bucket, render_sub fallbacks downstream.
    assert resources.resolve_submit_cfg("ma_face", {}, prof) == {"max_jobs": 3}
    assert resources.resolve_submit_cfg("ma_face", {}) == {}


def test_emit_dag_fills_gpu_defaults_for_bare_config(tmp_path, monkeypatch):
    # A run-config with no submit_cfg anywhere (the task-122/123 shape) must
    # still request GPUs for GPU steps and stay CPU-only for ma_cap.
    monkeypatch.setattr(htcondor, "resolve_seq_names", lambda cfg: ["seqA"])
    cfg = _task_cfg()
    for step in ("ma_cap", "ma_masks", "ma_2d"):
        del cfg[step]["submit_cfg"]
    htcondor.emit_dag(cfg, "t", "t", _opts(), str(tmp_path))
    masks_sub = (tmp_path / "ma_masks.sub").read_text()
    assert "request_gpus = 1" in masks_sub
    assert "TARGET.CUDAGlobalMemoryMb > 24000" in masks_sub
    assert "request_memory = 64000" in masks_sub
    cap_sub = (tmp_path / "ma_cap.sub").read_text()
    assert "request_gpus" not in cap_sub


def test_emit_dag_warns_on_explicit_gpus_zero(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(htcondor, "resolve_seq_names", lambda cfg: ["seqA"])
    cfg = _task_cfg()
    cfg["ma_masks"]["submit_cfg"]["gpus"] = 0
    with caplog.at_level("WARNING", logger="inference.cluster.htcondor"):
        htcondor.emit_dag(cfg, "t", "t", _opts(), str(tmp_path))
    assert any("ma_masks" in r.message and "gpus=0" in r.message
               for r in caplog.records)


def test_emit_dag_raises_when_no_enabled_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(htcondor, "resolve_seq_names", lambda cfg: ["seqA"])
    with pytest.raises(ValueError, match="no enabled steps"):
        htcondor.emit_dag({"global": {}}, "t", "t", _opts(), str(tmp_path))


def test_build_submit_cmd_flavors():
    b = htcondor.DagBundle(dag_path="/w/run.dag")
    assert htcondor.build_submit_cmd(b.dag_path, _opts(flavor="bid", bid_price=50)) == \
        ["condor_submit_dag_bid", "50", "/w/run.dag"]
    assert htcondor.build_submit_cmd(b.dag_path, _opts(flavor="standard")) == \
        ["condor_submit_dag", "/w/run.dag"]


# --------------------------------------------------------------------------- #
# scheduler seam (registry + HTCondor adapter)
# --------------------------------------------------------------------------- #

def test_scheduler_registry():
    from inference.cluster import scheduler as sched_mod
    s = sched_mod.get_scheduler(None)          # default
    assert s.name == "htcondor" and "bid" in s.flavors
    assert sched_mod.get_scheduler("HTCondor") is s   # case-insensitive
    with pytest.raises(ValueError, match="unknown scheduler"):
        sched_mod.get_scheduler("slurm")


def test_htcondor_adapter_commands_and_parsers():
    import json as _json
    from inference.cluster.scheduler import get_scheduler
    s = get_scheduler("htcondor")
    assert s.cancel_cmd("42") == ["condor_rm", "42"]
    assert s.queue_cmd("42")[0] == "condor_q"
    assert "command -v condor_submit_dag" in s.submit_host_cmd()[-1]
    assert s.parse_alive("17\n") is True
    assert s.parse_alive("  ") is False
    raw = _json.dumps([
        {"DAGNodeName": "ma_2d_0", "JobStatus": 1},                        # idle
        {"DAGNodeName": "ma_2d_1", "JobStatus": 5, "HoldReason": "bid"},   # held
        {"JobStatus": 2},                                                  # nameless: skipped
    ])
    parsed = s.parse_queue(raw)
    assert ("ma_2d_0", "Queued", None) in parsed
    held = [p for p in parsed if p[0] == "ma_2d_1"]
    assert held and held[0][1] == "Failed" and held[0][2]
    assert s.parse_queue("not json") == []


# --------------------------------------------------------------------------- #
# status_map
# --------------------------------------------------------------------------- #

def test_status_map_in_queue_states():
    assert status_map.map_condor_state(status_map.IDLE) == ("Queued", None)
    assert status_map.map_condor_state(status_map.RUNNING) == ("Running", None)
    assert status_map.map_condor_state(status_map.SUSPENDED) == ("Running", None)
    assert status_map.map_condor_state(status_map.REMOVED) == ("Cancelled", None)
    st, note = status_map.map_condor_state(status_map.HELD, "policy X")
    assert st == "Failed" and "policy X" in note
    # Completed defers to status.jsonl / DONE ground truth.
    assert status_map.map_condor_state(status_map.COMPLETED) is None


# --------------------------------------------------------------------------- #
# runner.run_cell
# --------------------------------------------------------------------------- #

class _RecSink(StatusSink):
    def __init__(self):
        self.calls = []

    def update(self, step_name, seq_name, status, pid=None):
        self.calls.append((step_name, seq_name, status))


class _FakeBuilder:
    def __init__(self, done_path):
        self._done = done_path

    def done_file(self, seq):
        return self._done


def _patch_runcell(monkeypatch, done_path):
    monkeypatch.setattr(runner, "load_task_config", lambda p: {"ma_2d": {}, "global": {}})
    monkeypatch.setattr(runner, "get_builder",
                        lambda *a, **k: _FakeBuilder(done_path))
    monkeypatch.setattr(runner, "_log_paths", lambda *a, **k: ("/dev/null", "/dev/null"))


def test_read_npz_int_stdlib_matches_numpy(tmp_path):
    # The bare-python fallback (cluster nodes without numpy) must read the
    # same scalar ints numpy wrote.
    np = pytest.importorskip("numpy")
    p = tmp_path / "cam.npz"
    np.savez(p, frame_start=np.int64(120), frame_end=np.int64(1271),
             other=np.zeros(3))
    assert runner._read_npz_int(str(p), "frame_start") == 120
    assert runner._read_npz_int(str(p), "frame_end") == 1271
    assert runner._read_npz_int(str(p), "missing") is None
    assert runner._read_npz_int(str(p), "other") is None  # non-scalar/float


def test_run_cell_skips_when_done_exists(tmp_path, monkeypatch):
    done = tmp_path / "DONE"
    done.write_text("")
    _patch_runcell(monkeypatch, str(done))
    # engine must NOT be invoked on the skip path.
    monkeypatch.setattr(runner.engines, "dispatch",
                        lambda *a, **k: pytest.fail("engine ran on DONE-skip"))
    sink = _RecSink()
    rc = runner.run_cell("t.json", "ma_2d", "seqA", sink=sink)
    assert rc == 0
    assert ("ma_2d", "seqA", "Done") in sink.calls


def test_run_cell_success_writes_done(tmp_path, monkeypatch):
    done = tmp_path / "sub" / "DONE"  # parent does not exist yet
    _patch_runcell(monkeypatch, str(done))
    monkeypatch.setattr(runner.engines, "dispatch", lambda *a, **k: 0)
    sink = _RecSink()
    rc = runner.run_cell("t.json", "ma_2d", "seqA", sink=sink)
    assert rc == 0
    assert done.exists()
    assert ("ma_2d", "seqA", "Running") in sink.calls
    assert ("ma_2d", "seqA", "Done") in sink.calls


def test_run_cell_failure_reports_failed(tmp_path, monkeypatch):
    done = tmp_path / "DONE"
    _patch_runcell(monkeypatch, str(done))
    monkeypatch.setattr(runner.engines, "dispatch", lambda *a, **k: 3)
    sink = _RecSink()
    rc = runner.run_cell("t.json", "ma_2d", "seqA", sink=sink)
    assert rc == 3
    assert not done.exists()
    assert ("ma_2d", "seqA", "Failed") in sink.calls


def test_run_cell_engine_exception_is_failed(tmp_path, monkeypatch):
    done = tmp_path / "DONE"
    _patch_runcell(monkeypatch, str(done))

    def _boom(*a, **k):
        raise RuntimeError("cuda gone")

    monkeypatch.setattr(runner.engines, "dispatch", _boom)
    sink = _RecSink()
    rc = runner.run_cell("t.json", "ma_2d", "seqA", sink=sink)
    assert rc == 1
    assert ("ma_2d", "seqA", "Failed") in sink.calls


# --------------------------------------------------------------------------- #
# camera fan-out
# --------------------------------------------------------------------------- #

def _fanout_cfg():
    return {
        "global": {"cam_names": ["IOI_01", "IOI_09", "IOI_17"],
                   "cam_init": "IOI_01", "out_dir": "/o", "jobs_log_dir": "/l"},
        "ma_cap": {"enabled": True, "dependencies": [], "script": "x.py",
                   "repo_path": "capture", "flags": []},
        "ma_masks": {"enabled": True, "dependencies": ["ma_cap"],
                     "script": "y.py", "repo_path": "segmentation",
                     "flags": ["--sam_version sam2"], "fanout_cameras": True,
                     "submit_cfg": {"max_jobs": 10}},
        "ma_2d": {"enabled": True, "dependencies": ["ma_masks"],
                  "script": "z.py", "repo_path": "landmarks", "flags": []},
    }


def test_emit_dag_camera_fanout(tmp_path, monkeypatch):
    monkeypatch.setattr(htcondor, "resolve_seq_names", lambda cfg: ["seqA"])
    opts = htcondor.CondorOpts(repo_dir="/r", remote_task_path="/t.json",
                               status_jsonl="/logs/status", retries=1)
    bundle = htcondor.emit_dag(_fanout_cfg(), "o", "7", opts, str(tmp_path))
    dag = open(bundle.dag_path).read()
    # init node + one node per non-init camera + finalize, all category ma_masks
    assert "JOB ma_masks_init_0" in dag
    assert "JOB ma_masks_IOI_09_0" in dag and "JOB ma_masks_IOI_17_0" in dag
    assert "JOB ma_masks_finalize_0" in dag
    assert "MAXJOBS ma_masks 10" in dag
    # per-cell flags: init runs its camera only; cam cells run init + cam;
    # none runs the rig vote or the collage (the finalize cell does, once)
    assert ('cell_flags="--cam_names IOI_01 --cam_init IOI_01 '
            '--rig_vote false --skip_collage"') in dag
    assert ('cell_flags="--cam_names IOI_01 IOI_09 --cam_init IOI_01 '
            '--rig_vote false --skip_collage"') in dag
    assert 'cell_id="finalize" cell_flags="--cam_init IOI_01"' in dag
    # wiring: ma_cap -> init -> cams -> finalize -> ma_2d
    assert "PARENT ma_cap_0 CHILD ma_masks_init_0" in dag
    assert "PARENT ma_masks_init_0 CHILD ma_masks_IOI_09_0" in dag
    assert "PARENT ma_masks_IOI_09_0 ma_masks_IOI_17_0 CHILD ma_masks_finalize_0" in dag
    assert "PARENT ma_masks_finalize_0 CHILD ma_2d_0" in dag
    # node map covers every fan-out node
    assert bundle.node_to_seq["ma_masks_IOI_17_0"] == "seqA"
    assert bundle.node_to_seq["ma_masks_finalize_0"] == "seqA"


def test_fanout_init_honors_step_flags_cam_init():
    cfg = _fanout_cfg()
    cfg["global"].pop("cam_init")
    cfg["ma_masks"]["flags"] = ["--sam_version sam3", "--cam_init IOI_09"]
    cams, init = htcondor._resolve_fanout(cfg)
    assert init == "IOI_09"
    assert cams[0] == "IOI_01"          # order preserved; init need not be first
    # global.cam_init still wins over the list head when flags carry none
    cfg["ma_masks"]["flags"] = ["--sam_version sam3"]
    cfg["global"]["cam_init"] = "IOI_17"
    _cams, init = htcondor._resolve_fanout(cfg)
    assert init == "IOI_17"


def test_emit_dag_ma_2d_camera_fanout(tmp_path, monkeypatch):
    """ma_2d cells are per-camera independent: no init anchor, no finalize
    barrier, and the downstream step waits on every cell."""
    cfg = _fanout_cfg()
    cfg["ma_2d"]["fanout_cameras"] = True
    cfg["ma_3d"] = {"enabled": True, "dependencies": ["ma_2d"],
                    "script": "w.py", "repo_path": "optimization", "flags": []}
    monkeypatch.setattr(htcondor, "resolve_seq_names", lambda cfg: ["seqA"])
    opts = htcondor.CondorOpts(repo_dir="/r", remote_task_path="/t.json",
                               status_jsonl="/logs/status", retries=1)
    bundle = htcondor.emit_dag(cfg, "o", "7", opts, str(tmp_path))
    dag = open(bundle.dag_path).read()
    for cam in ("IOI_01", "IOI_09", "IOI_17"):
        assert f"JOB ma_2d_{cam}_0" in dag
        assert f'cell_id="{cam}" cell_flags="--cam_names {cam}"' in dag
    assert "ma_2d_init_0" not in dag and "ma_2d_finalize_0" not in dag
    # each cell hangs off the ma_masks tail; ma_3d waits on every cell
    assert "PARENT ma_masks_finalize_0 CHILD ma_2d_IOI_01_0" in dag
    assert ("PARENT ma_2d_IOI_01_0 ma_2d_IOI_09_0 ma_2d_IOI_17_0 "
            "CHILD ma_3d_0") in dag
    assert bundle.node_to_seq["ma_2d_IOI_09_0"] == "seqA"


def test_emit_dag_fanout_off_is_unchanged(tmp_path, monkeypatch):
    cfg = _fanout_cfg()
    cfg["ma_masks"].pop("fanout_cameras")
    monkeypatch.setattr(htcondor, "resolve_seq_names", lambda cfg: ["seqA"])
    opts = htcondor.CondorOpts(repo_dir="/r", remote_task_path="/t.json",
                               status_jsonl="/logs/status", retries=0)
    dag = open(htcondor.emit_dag(cfg, "o", "7", opts, str(tmp_path)).dag_path).read()
    assert "JOB ma_masks_0" in dag and "ma_masks_init_0" not in dag
    assert "PARENT ma_masks_0 CHILD ma_2d_0" in dag
