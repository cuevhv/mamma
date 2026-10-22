"""Apptainer scratch routing: TMPDIR (HTCondor's request_disk-sized job
scratch) must be bound into the container and pinned, or in-container
tempfile silently falls back to the node's small shared /tmp, which the
ma_masks mask spill can fill (ENOSPC)."""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from inference import engines


class _FakeBuilder:
    """Just the surface build_command touches for the apptainer branch."""
    step_name = "ma_masks"
    engine = "apptainer"
    sif_path = "/c/mamma.sif"
    step_cfg = {"submit_cfg": {"gpus": 0}}

    def build_argv(self, seq):
        return ["run_ma_masks.py", "--seq_name", seq]

    def binds(self):
        return ["/data"]

    def container_cwd(self):
        return "/repo/segmentation"


def test_apptainer_binds_condor_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    cmd, _ = engines.build_command(_FakeBuilder(), "seqA")
    assert ["--bind", str(tmp_path)] == cmd[cmd.index("--bind"):cmd.index("--bind") + 2]
    assert f"TMPDIR={tmp_path}" in cmd
    assert "--env" in cmd


def test_apptainer_skips_default_tmp(monkeypatch):
    # /tmp is an apptainer default bind: no extra flags, no env pin.
    monkeypatch.setenv("TMPDIR", "/tmp")
    cmd, _ = engines.build_command(_FakeBuilder(), "seqA")
    assert "--env" not in cmd

    monkeypatch.delenv("TMPDIR", raising=False)
    cmd2, _ = engines.build_command(_FakeBuilder(), "seqA")
    assert "--env" not in cmd2


def test_apptainer_ignores_nonexistent_tmpdir(monkeypatch):
    monkeypatch.setenv("TMPDIR", "/nonexistent/scratch/dir")
    cmd, _ = engines.build_command(_FakeBuilder(), "seqA")
    assert "--env" not in cmd
