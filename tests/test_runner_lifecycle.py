"""Runner lifecycle: forced re-runs and cancellation, with a mocked engine."""
import os
import sys
from pathlib import Path

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from inference import runner  # noqa: E402


class _Sink:
    def __init__(self):
        self.calls = []

    def update(self, step, seq, status, **_k):
        self.calls.append((step, seq, status))

    def cancel_remaining(self, steps, seqs):
        self.calls += [(s, q, "Cancelled") for s in steps for q in seqs]


@pytest.fixture
def dag(tmp_path, monkeypatch):
    """One enabled step, one sequence; returns (cfg, done_path)."""
    cfg = {"global": {}, "ma_2d": {"enabled": True}}
    done = tmp_path / "seqA.DONE"

    class _Builder:
        def done_file(self, seq):
            return str(done)

    monkeypatch.setattr(runner, "load_task_config", lambda p: cfg)
    monkeypatch.setattr(runner, "resolve_seq_names", lambda c: ["seqA"])
    monkeypatch.setattr(runner, "get_builder", lambda *a, **k: _Builder())
    monkeypatch.setattr(runner, "_log_paths", lambda *a, **k: ("/dev/null", "/dev/null"))
    monkeypatch.setattr(runner, "_install_sigterm_handler", lambda: None)
    monkeypatch.setattr(runner.CANCEL, "set", False)
    return cfg, done


@pytest.mark.parametrize("seq_major", [False, True])
def test_failed_forced_rerun_drops_stale_done(dag, monkeypatch, seq_major):
    cfg, done = dag
    cfg["global"]["sequence_major"] = seq_major
    done.touch()  # an earlier successful run
    monkeypatch.setattr(runner.engines, "dispatch", lambda *a: 1)
    assert runner.run_dag("t.json", force=True) == 1
    assert not done.exists()

    # The next plain run must execute the step again, not report Done.
    ran = []
    monkeypatch.setattr(runner.engines, "dispatch", lambda *a: ran.append(1) or 0)
    sink = _Sink()
    assert runner.run_dag("t.json", sink=sink) == 0
    assert ran and ("ma_2d", "seqA", "Running") in sink.calls
    assert done.exists()


def test_failed_forced_run_cell_drops_stale_done(dag, monkeypatch):
    _, done = dag
    done.touch()
    monkeypatch.setattr(runner.engines, "dispatch", lambda *a: 2)
    assert runner.run_cell("t.json", "ma_2d", "seqA", sink=_Sink(), force=True) == 2
    assert not done.exists()


@pytest.mark.parametrize("seq_major", [False, True])
def test_cancelled_dag_is_nonzero_and_flag_resets(dag, monkeypatch, seq_major):
    cfg, _ = dag
    cfg["global"]["sequence_major"] = seq_major

    def _cancel_mid_run(*_a):
        runner.CANCEL.set = True  # what the SIGTERM handler does
        return -15

    monkeypatch.setattr(runner.engines, "dispatch", _cancel_mid_run)
    sink = _Sink()
    assert runner.run_dag("t.json", sink=sink) == runner.EXIT_CANCELLED != 0
    assert ("ma_2d", "seqA", "Cancelled") in sink.calls

    # A later run in the same process starts uncancelled.
    monkeypatch.setattr(runner.engines, "dispatch", lambda *a: 0)
    sink = _Sink()
    assert runner.run_dag("t.json", sink=sink) == 0
    assert ("ma_2d", "seqA", "Done") in sink.calls
