"""scripts/regression_check.py and scripts/benchmark.py helpers."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import regression_check as rc  # noqa: E402


def test_parse_timing_skips_counts_events(tmp_path):
    status = tmp_path / "status.jsonl"
    status.write_text("\n".join(json.dumps(r) for r in [
        {"ts": 1.0, "step": "ma_cap", "seq": "s", "status": "Running", "pid": 1},
        {"ts": 3.5, "step": "ma_cap", "seq": "s", "status": "Done", "pid": None},
        {"ts": 3.6, "step": "ma_cap", "seq": "s", "event": "counts",
         "num_frames": 60, "num_cameras": 4},
    ]) + "\n")
    assert rc._parse_timing(status) == {"ma_cap": 2.5}


def test_one_sided_nan_fails_drift():
    golden = np.random.default_rng(0).normal(size=(4, 10, 2))
    cand = golden.copy()
    cand[1, :3] = np.nan
    mean, _p99, worst = rc._euclidean_drift(golden, cand, "px")
    assert mean == 0.0 and worst == float("inf")


def test_matching_nan_is_skipped():
    golden = np.random.default_rng(0).normal(size=(4, 10, 3))
    golden[0, 0] = np.nan
    assert rc._euclidean_drift(golden, golden.copy(), "mm") == (0.0, 0.0, 0.0)


def test_benchmark_sampler_stops_and_joins():
    import benchmark
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])
    sampler = benchmark._Sampler(proc, interval=0.05)
    sampler.start()
    proc.wait()
    sampler.stop()
    sampler.join(timeout=2)
    assert not sampler.is_alive()
    assert sampler.peak_rss_mb > 0
