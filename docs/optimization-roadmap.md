# MAMMA efficiency & quality roadmap

A staged program to make the pipeline substantially faster and cleaner
without ever regressing output quality. Each item ships as a small,
independently revertible PR that must pass the validation gate below.

The work mines concrete suggestions from an efficient re-implementation
of MAMMA ([rerun-io/examples-monorepo#48][pr48]) — which reported an
~11.3× speedup at 23.3 mm MPJPE — and ports the *portable* ones here,
fixing bugs and cleaning code along the way. That PR is a reference, not
something we merge.

[pr48]: https://github.com/rerun-io/examples-monorepo/pull/48

## Why

The pipeline runs as a 5-step subprocess DAG
(`ma_cap → ma_masks → ma_2d → ma_3d → ma_vis`) where each step is a fresh
Python process handing off to the next via disk (NPZ/PNG). That boundary
costs, *per sequence*:

- ~3–6 s cold `import torch` + CUDA init, ×5 steps;
- redundant video decode inside `ma_masks`, `ma_2d`, and `ma_vis`;
- 4K mask PNG encode→decode and intermediate NPZ serialization.

In the shipped presets every step already runs in the **same `mamma`
conda env**, so the process isolation buys almost nothing while paying
all of the above.

## Locked decisions

| Topic | Decision |
|---|---|
| Architecture | Run steps **in-process** when they share an env; keep the subprocess path only for different-env / container runs. Models stay **resident** (no load/unload). Intermediate disk writes become an **optional debug toggle**. |
| End goal | Stop at **in-process batch** orchestration. Full causal *streaming* (the PR's realtime sliding-window fitter, SAM2 forgetful bank) is **out of scope**. |
| Acceleration deps | **Portable PyTorch by default.** NVIDIA-only accel (TensorRT, CUDA-graphs, NVDEC/NVENC) is **opt-in behind flags**, added later only where ROI is proven. The public release must run without them. |
| VRAM | Keep models resident; adopt **512-vertex sampled SMPL-X**; bound *transient* buffers. Load/unload is a fallback only if we actually OOM (24 GB is ample for a 4-camera clip). |
| Validation | **Self-consistency vs. current `main`** — no ground truth exists for the in-the-wild example. |

## Validation gate (every PR must pass)

1. **Regression harness** — `scripts/regression_check.py` (run in the `mamma` env):
   compares `ma_2d` landmarks (pixel drift) and `ma_3d` `pred_vertices` /
   `pred_joints` (MPJPE / PVE in mm) against a golden snapshot of `main`,
   plus per-step and total wall-clock.
   - *Numerically-neutral* refactors (in-process handoff, single-decode,
     dead-code removal): drift must be at the noise floor (sub-mm).
   - *Numerically-changing* optimizations (batching with a different
     reduction order, 512-vertex SMPL-X, opt-in accel): bounded drift —
     a few mm MPJPE/PVE at most, far under the PR's 30 mm budget — **and**
     a written justification in the PR description.
   - Total wall-clock must not regress (ideally improves).
2. **Smoke test** — `python scripts/smoke_test.py` (and `--full` for
   architecture PRs).
3. **Unit tests** — `cd segmentation && pytest tests/ -v`; add tests for
   new modules.

### Establishing the golden / noise floor

```bash
# Once, on main:
python scripts/regression_check.py --make-golden \
    --cfg configs/examples/presets/quick.yaml \
    --capture configs/examples/captures/mamma_example.json

# Sanity: main-vs-main measures the determinism noise floor (should be ~0).
python scripts/regression_check.py --check
```

If main-vs-main drift is not ~0 the pipeline has nondeterminism; record
the measured floor and set the neutral-refactor tolerances just above it
(harness flags `--pred_vertices-tol-mm`, `--pred_joints-tol-mm`, `--px-tol`).
On the reference machine (RTX 4090) the floor is **0.000** — the pipeline
is deterministic, so the default sub-mm tolerances hold.

Wall-clock is cache-sensitive (a cold first run is several× slower than a
warm one), so it is only a coarse guard: capture the golden and the
candidate under similar warm/cold conditions, and rely on the drift check
as the real gate.

## PR backlog (smallest risk first)

### Phase 0 — Foundation
- **PR #1 — Regression harness + this roadmap.** No behavior change.
  Adds `scripts/regression_check.py` and `docs/optimization-roadmap.md`.

### Phase 1 — Portable in-step optimizations (no architecture change)
- **PR #2 — Batch `ma_2d` inference.** Remove the hardcoded
  `batch_size=1` DataLoader in `landmarks/run_ma_2d.py`.
- **PR #3 — Sync-free `ma_3d` fitting loop.** Remove host↔GPU syncs /
  `.item()` / CPU NaN-guards in `optimization/utils/{optimization,fitting}.py`.
- **PR #4 — 512-vertex sampled SMPL-X.** Use the existing
  `MAMMA_DOWNSAMPLED_VERTS_PKL` (`verts_512.pkl`) for the optimization
  forward; keep the full mesh for final export.
- **PR #5 — Single-decode reuse / parallel relog.** Extend within-step
  decode reuse and the existing `overlay_num_workers` /
  `rerun_image_num_workers` parallelism in `visualization/`.

### Phase 2 — Architecture: in-process orchestration + I/O toggle
- **PR #6 — Importable step contract + `--debug-io` toggle.** Refactor
  each `run_*.py` into an importable function with an in-memory I/O
  contract; disk writes become optional.
- **PR #7 — In-process executor in `inference/runner.py`.** Used when all
  enabled steps share an env; subprocess engines remain the fallback.

### Phase 3 — Opt-in NVIDIA accel (behind flags, ROI-gated)
- **PR #8+** — TensorRT MammaNet engine, CUDA-graphed fitter,
  NVDEC/NVENC. Each defaults OFF with a portable fallback; pursued only
  where Phase 1–2 measurements show the portable path is the bottleneck.

## Per-PR description template

```
## What & why
<one-paragraph summary>

## PR #48 idea ported
<which suggestion, and the worth-it judgement>

## Validation
- regression_check: PVE <x> mm / MPJPE <y> mm / px <z>  (tol: ...)
- timing: <before>s -> <after>s  (<n>× / <pct>%)
- smoke_test: PASS    pytest: PASS
- numeric change justified: <yes/no — why>
```

## Out of scope

Full causal streaming; vendored SAM2 fork; gated-data plumbing; any
non-portable dependency as a default; GUI changes (beyond thin adapters
forced by a step refactor).
