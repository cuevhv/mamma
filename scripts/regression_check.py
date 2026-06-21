#!/usr/bin/env python
"""MAMMA output regression harness.

Guards the optimization program (see ``docs/optimization-roadmap.md``)
against silent quality regressions. The repository has no ground-truth
SMPL-X for its in-the-wild example, so "correct" is defined as
*self-consistency vs. the current ``main`` branch*: we snapshot the
pipeline's numeric outputs as a golden reference, then gate every later
change on the drift away from that golden plus a timing budget.

What is compared (per camera / per body, matched by path):

* ``ma_2d`` ``<cam>.npz`` — predicted 2D ``landmarks`` (pixel drift).
* ``ma_3d`` ``verts_joints_body_id-NN.npz`` — ``pred_vertices`` (PVE)
  and ``pred_joints`` (MPJPE), reported in millimetres.

Per-step wall-clock comes from the runner's ``--status-jsonl`` stream
(``Running`` -> ``Done`` timestamps); total wall-clock is measured
around the subprocess.

Usage::

    # 1. Snapshot the current branch as the golden reference (run once on main):
    python scripts/regression_check.py --make-golden \
        --cfg configs/examples/presets/quick.yaml \
        --capture configs/examples/captures/mamma_example.json

    # 2. On a candidate branch, check drift + timing against the golden:
    python scripts/regression_check.py --check \
        --cfg configs/examples/presets/quick.yaml \
        --capture configs/examples/captures/mamma_example.json

    # Re-compare already-produced outputs without re-running the pipeline:
    python scripts/regression_check.py --check --skip-run

Exit code is 0 when every compared array is within tolerance and timing
has not regressed, 1 otherwise. The script is invoked from the repo
root and expects the ``mamma`` conda env to be the active interpreter
(it does not activate one itself).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Default locations. The golden run and the candidate run deliberately
# share one (out-dir, out-tag) so output paths are byte-for-byte stable
# and golden/candidate files match on their path relative to out-dir.
_DEFAULT_GOLDEN = _REPO_ROOT / "tests" / "golden" / "mamma_example"
_DEFAULT_OUT_DIR = _REPO_ROOT / "output" / "regression"
_DEFAULT_OUT_TAG = "reg"
_DEFAULT_CFG = _REPO_ROOT / "configs" / "examples" / "presets" / "quick.yaml"
_DEFAULT_CAPTURE = _REPO_ROOT / "configs" / "examples" / "captures" / "mamma_example.json"


# A comparison target: which step's NPZs to read, the array keys to
# compare, and how to score each key. ``unit`` is "mm" (3D euclidean
# distance scaled to millimetres) or "px" (2D euclidean pixel distance).
@dataclass(frozen=True)
class Target:
    step: str
    glob: str
    keys: Tuple[str, ...]
    unit: str


_TARGETS: Tuple[Target, ...] = (
    Target(step="ma_2d", glob="ma_2d/**/*.npz", keys=("landmarks",), unit="px"),
    Target(
        step="ma_3d",
        glob="ma_3d/**/verts_joints_body_id-*.npz",
        keys=("pred_vertices", "pred_joints"),
        unit="mm",
    ),
)


@dataclass
class Drift:
    relpath: str
    key: str
    unit: str
    mean: float
    p99: float
    max: float
    note: str = ""  # non-empty marks a structural problem (shape/missing)


@dataclass
class Report:
    drifts: List[Drift] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)  # step -> seconds
    total_s: float = 0.0


# --------------------------------------------------------------------------
# Pipeline invocation
# --------------------------------------------------------------------------
def _preflight(cfg: Path, capture: Path) -> None:
    """Fail early with actionable guidance when inputs are absent."""
    missing = [p for p in (cfg, capture) if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing config(s): " + ", ".join(str(p) for p in missing)
        )
    example = _REPO_ROOT / "data" / "mamma_example"
    if not example.exists():
        raise SystemExit(
            f"Example data not found at {example}.\n"
            "Fetch it first:  bash data/download_example.sh"
        )


def _run_pipeline(cfg: Path, capture: Path, out_dir: Path, out_tag: str) -> Report:
    """Run the full DAG into ``out_dir``/``out_tag`` and collect timing."""
    # Steps self-skip when their output .npz already exists, and the runner's
    # --force only clears DONE sentinels — so without a clean slate a check
    # silently reuses stale outputs and reports zero drift. Wipe the tree to
    # guarantee every step recomputes from scratch.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / f"status-{out_tag}.jsonl"
    cmd = [
        sys.executable, "-m", "inference", "run",
        "--cfg", str(cfg),
        "--capture", str(capture),
        "--out-dir", str(out_dir),
        "--out-tag", out_tag,
        "--force",  # never skip via DONE sentinels — we need fresh numbers
        "--status-jsonl", str(status_path),
        "-v",
    ]
    print(f"  $ {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=_REPO_ROOT)
    total = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"Pipeline run failed (exit {proc.returncode}).")
    report = Report(total_s=total, timing=_parse_timing(status_path))
    return report


def _parse_timing(status_path: Path) -> Dict[str, float]:
    """Per-step seconds from Running->Done transitions, summed over seqs."""
    if not status_path.exists():
        return {}
    starts: Dict[Tuple[str, str], float] = {}
    durations: Dict[str, float] = {}
    for line in status_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        key = (rec["step"], rec["seq"])
        if rec["status"] == "Running":
            starts[key] = rec["ts"]
        elif rec["status"] == "Done" and key in starts:
            durations[rec["step"]] = durations.get(rec["step"], 0.0) + (
                rec["ts"] - starts[key]
            )
    return durations


# --------------------------------------------------------------------------
# Output collection & comparison
# --------------------------------------------------------------------------
def _collect(run_root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Map relpath (from run_root) -> {key: array} for every target NPZ.

    Only the keys we score are loaded, keeping the golden snapshot small.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for target in _TARGETS:
        for path in sorted(run_root.glob(target.glob)):
            with np.load(path, allow_pickle=True) as npz:
                arrays = {
                    k: np.asarray(npz[k], dtype=np.float64)
                    for k in target.keys
                    if k in npz.files
                }
            if arrays:
                out[path.relative_to(run_root).as_posix()] = arrays
    return out


def _unit_of(relpath: str) -> str:
    return next((t.unit for t in _TARGETS if relpath.startswith(t.step + "/")), "mm")


def _euclidean_drift(a: np.ndarray, b: np.ndarray, unit: str) -> Optional[Tuple[float, float, float]]:
    """(mean, p99, max) euclidean per-point distance, or None on mismatch.

    The last axis is the coordinate axis (2 for pixels, 3 for metres).
    Metre distances are scaled to millimetres.
    """
    if a.shape != b.shape:
        return None
    dist = np.linalg.norm(a - b, axis=-1)
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    scale = 1000.0 if unit == "mm" else 1.0
    return (
        float(finite.mean() * scale),
        float(np.percentile(finite, 99) * scale),
        float(finite.max() * scale),
    )


def _compare(golden_dir: Path, run_root: Path) -> List[Drift]:
    """Compare every golden array against the candidate run."""
    golden = _collect(golden_dir)
    candidate = _collect(run_root)
    drifts: List[Drift] = []

    for relpath in sorted(set(golden) | set(candidate)):
        unit = _unit_of(relpath)
        if relpath not in candidate:
            drifts.append(Drift(relpath, "*", unit, *(float("nan"),) * 3,
                                note="missing from candidate output"))
            continue
        if relpath not in golden:
            drifts.append(Drift(relpath, "*", unit, *(float("nan"),) * 3,
                                note="not present in golden (new output)"))
            continue
        for key in sorted(set(golden[relpath]) | set(candidate[relpath])):
            ga, ca = golden[relpath].get(key), candidate[relpath].get(key)
            if ga is None or ca is None:
                drifts.append(Drift(relpath, key, unit, *(float("nan"),) * 3,
                                    note="key present on only one side"))
                continue
            scored = _euclidean_drift(ga, ca, unit)
            if scored is None:
                drifts.append(Drift(relpath, key, unit, *(float("nan"),) * 3,
                                    note=f"shape mismatch {ga.shape} vs {ca.shape}"))
            else:
                drifts.append(Drift(relpath, key, unit, *scored))
    return drifts


# --------------------------------------------------------------------------
# Golden snapshot
# --------------------------------------------------------------------------
def _write_golden(run_root: Path, golden_dir: Path, timing: Report) -> int:
    """Persist the compared arrays + timing as the golden reference."""
    if golden_dir.exists():
        shutil.rmtree(golden_dir)
    collected = _collect(run_root)
    for relpath, arrays in collected.items():
        dst = golden_dir / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **arrays)
    (golden_dir / "timing.json").write_text(
        json.dumps({"per_step_s": timing.timing, "total_s": timing.total_s}, indent=2)
    )
    return len(collected)


def _load_golden_timing(golden_dir: Path) -> Dict[str, float]:
    path = golden_dir / "timing.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _print_drifts(drifts: List[Drift], tol: Dict[str, float]) -> bool:
    print("\n  drift vs golden (euclidean per-point):")
    print(f"  {'status':>6}  {'metric':<7} {'mean':>10} {'p99':>10} {'max':>10}  file")
    ok = True
    for d in sorted(drifts, key=lambda x: (x.relpath, x.key)):
        if d.note:
            ok = False
            print(f"  {'FAIL':>6}  {d.key:<7} {'—':>10} {'—':>10} {'—':>10}  "
                  f"{d.relpath}  [{d.note}]")
            continue
        limit = tol["px"] if d.unit == "px" else tol[d.key]
        passed = np.isfinite(d.mean) and d.max <= limit
        ok = ok and passed
        unit = d.unit
        print(f"  {'ok' if passed else 'FAIL':>6}  {d.key:<7} "
              f"{d.mean:>8.3f}{unit:<2} {d.p99:>8.3f}{unit:<2} {d.max:>8.3f}{unit:<2}  "
              f"{d.relpath}")
    return ok


def _print_timing(candidate: Report, golden_timing: Dict[str, float],
                  regress_frac: float) -> bool:
    print("\n  timing (per-step seconds):")
    g_steps = golden_timing.get("per_step_s", {})
    for step in sorted(set(candidate.timing) | set(g_steps)):
        c = candidate.timing.get(step, float("nan"))
        g = g_steps.get(step, float("nan"))
        print(f"    {step:<10} {c:>8.2f}s   (golden {g:>8.2f}s)")
    g_total = golden_timing.get("total_s")
    ok = True
    if g_total:
        budget = g_total * (1.0 + regress_frac)
        ok = candidate.total_s <= budget
        verdict = "ok" if ok else "REGRESSED"
        print(f"    {'total':<10} {candidate.total_s:>8.2f}s   "
              f"(golden {g_total:>8.2f}s, budget {budget:>8.2f}s)  [{verdict}]")
    else:
        print(f"    {'total':<10} {candidate.total_s:>8.2f}s   (no golden timing)")
    print("    note: wall-clock is cache-sensitive (cold vs. warm OS/GPU caches);"
          " a coarse guard — the drift check is the real gate.")
    return ok


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python scripts/regression_check.py",
        description="Snapshot or check MAMMA outputs against a golden reference.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--make-golden", action="store_true",
                      help="Run the pipeline and store its outputs as the golden reference.")
    mode.add_argument("--check", action="store_true",
                      help="Run the pipeline and compare against the golden (default).")
    p.add_argument("--cfg", type=Path, default=_DEFAULT_CFG)
    p.add_argument("--capture", type=Path, default=_DEFAULT_CAPTURE)
    p.add_argument("--golden-dir", type=Path, default=_DEFAULT_GOLDEN)
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    p.add_argument("--out-tag", default=_DEFAULT_OUT_TAG)
    p.add_argument("--skip-run", action="store_true",
                   help="Compare existing outputs in --out-dir without re-running the pipeline.")
    # Tolerances. Defaults are intentionally tight; tune per the noise floor
    # measured by a main-vs-main run (see docs/optimization-roadmap.md).
    p.add_argument("--pred_vertices-tol-mm", dest="tol_pve", type=float, default=1.0,
                   help="Max allowed per-vertex drift (PVE), millimetres.")
    p.add_argument("--pred_joints-tol-mm", dest="tol_mpjpe", type=float, default=1.0,
                   help="Max allowed per-joint drift (MPJPE), millimetres.")
    p.add_argument("--px-tol", dest="tol_px", type=float, default=1.0,
                   help="Max allowed 2D landmark drift, pixels.")
    p.add_argument("--timing-regress-frac", type=float, default=0.25,
                   help="Allowed total wall-clock growth before flagging a regression.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    run_root = args.out_dir

    if not args.skip_run:
        _preflight(args.cfg, args.capture)
        print("Running pipeline ...")
        report = _run_pipeline(args.cfg, args.capture, args.out_dir, args.out_tag)
    else:
        report = Report(timing={}, total_s=0.0)

    if args.make_golden:
        n = _write_golden(run_root, args.golden_dir, report)
        print(f"\nWrote golden reference: {n} file(s) under {args.golden_dir}")
        print(f"  total wall-clock: {report.total_s:.2f}s")
        return 0

    # --check (default)
    if not args.golden_dir.exists():
        raise SystemExit(
            f"No golden reference at {args.golden_dir}. Create one first with "
            "--make-golden on the baseline branch."
        )
    drifts = _compare(args.golden_dir, run_root)
    tol = {"pred_vertices": args.tol_pve, "pred_joints": args.tol_mpjpe, "px": args.tol_px}
    quality_ok = _print_drifts(drifts, tol)
    timing_ok = _print_timing(report, _load_golden_timing(args.golden_dir),
                              args.timing_regress_frac)

    print()
    if quality_ok and timing_ok:
        print("PASS — outputs within tolerance, timing not regressed.")
        return 0
    print("FAIL — see flagged rows above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
