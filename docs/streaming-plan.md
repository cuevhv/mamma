# Streaming MAMMA — what PR #48 does, and a feasible plan

Companion to `streaming-feasibility.md` (which concluded a *thin wrapper* is
impossible). This goes deeper: exactly what PR #48's streaming includes, what we
can reuse, and a staged plan with two feasibility levels.

## What PR #48's streaming actually is

A **single-process, GPU-resident, windowed** pipeline — no disk I/O between
stages (verified by a `test_no_disk_writes`). Per-tick data flow:

```
NVDEC decode → GPU resize → SAM2 track → MammaNet(TensorRT) → GPU triangulate → sliding-window SMPL-X fit → async H.264/Rerun log
```

Concrete design points:
- **Window:** the SMPL-X fit holds a **W=12-frame** sliding window; each new frame
  triggers **~24 warm Adam iters** over that window (causal, with a small
  `emit_lag` latency), wrapped in a **CUDA graph** (92 → 15.3 ms/tick). A 300-iter
  bootstrap starts it.
- **SAM2 streaming:** a *vendored fork* with a **forgetful memory bank (window=7)**,
  **forward-only** (no reverse pass), **batched across cameras** in lockstep, with
  **sparse re-detection every 30 ticks**.
- **Tensors stay on the GPU** the whole loop; logging/encode happen **async on
  worker threads**; decode runs ahead via a **per-camera NVDEC subprocess pool**.
- **512-vertex sampled SMPL-X** forward in the fit (5e-7 err vs full).

**Reused from the original:** MammaNet *weights*, the SMPL-X model + *loss terms*,
the epipolar scoring *math*. **Rewritten:** SAM2 integration (streaming fork), the
fitter (causal sliding-window + CUDA graphs), decode (NVDEC), and the whole
orchestrator. So PR #48 **reuses the model + math but rewrites the three hardest
pieces** — it is a ground-up streaming engine, not a patch.

Your mental model is exactly right: *a window of frames flows through every stage
on the GPU, with no intermediate disk I/O*. The cost is that "causal window"
changes the **algorithms** (forgetful tracking, online fit), so it cannot be a
wrapper over today's batch stages.

## What we already have that a streaming MAMMA would reuse

The recent work is directly reusable as streaming building blocks:
- **MammaNet TensorRT engine** (item B) — the streaming "Landmarks" stage, already
  5.3× and cached.
- **On-GPU crop preprocessing** (item A, kornia warp/blur/normalize) — the
  resize/crop stage, already GPU-resident.
- **SMPL-X model + loss terms + 512-vertex sampled forward** — already used by the
  fit (PR #4 finding), reusable in a windowed fitter.
- **Epipolar association + triangulation math** — reusable (port the CPU SVD to a
  batched GPU eigh, as PR #48 did).

So roughly half the compute (landmarks + preprocessing + SMPL-X/losses + assoc
math) is already in hand.

## Two feasibility levels

### Level B — chunked in-process (the pragmatic, high-reuse option)
Process the sequence in **overlapping chunks** (e.g. 60–120 frames), running all
stages **in one process, GPU-resident, no intermediate disk** per chunk.

- **Reuses the existing algorithms unchanged:** SAM2 can still run forward+reverse
  **within a chunk** (so *no streaming-SAM2 fork needed*), and `ma_3d` batch-fits a
  chunk (a smaller batch; overlap carries temporal continuity). The temporal
  losses work inside the chunk.
- **Delivers the resource/I/O wins you want:** bounded memory (chunk, not whole
  sequence → no SAM2 OOM at scale), GPU residency, and **no mask-PNG / NPZ
  round-trips**.
- **Does NOT give realtime** (still processes the whole clip, just in memory and in
  chunks) and has **chunk-boundary effects** (mitigated by overlap).
- **The one real cost:** running the three stages in one process hits the
  `utils`/`core` name clash (see `architecture-analysis.md`). That must be resolved
  first — either namespace the step packages (mechanical cross-repo rename) or load
  each stage under an isolated import namespace via `importlib`.

### Level A — full realtime causal streaming (PR #48 parity)
Everything in B **plus**: a **streaming SAM2** (forgetful bank, forward-only,
batched), a **causal sliding-window fitter** (online W-frame warm-started CUDA-graph
optimize, `emit_lag`), and **NVDEC decode + async logging threads**. This is the
big rewrite; it buys **realtime/online** output on top of B's resource wins.

## Staged plan (maximal reuse, each stage independently useful)

1. **Unblock in-process** (prereq for any of this): resolve the `utils`/`core`
   clash — namespace the three step packages (or `importlib` isolation). Validate
   the existing batch DAG still passes the regression + GT harness. *This is the
   gating refactor; nothing streaming-shaped works without it.*
2. **Chunked orchestrator (Level B):** a driver that runs decode→SAM2→ma_2d→ma_3d
   on overlapping chunks in one process, GPU-resident, reusing today's stage code +
   the TensorRT/GPU-preproc work. Validate **GT MPJPE within tolerance** vs the
   batch pipeline (chunk+overlap ≈ batch). Ship as an opt-in `--streaming-chunked`
   runner path; keep the DAG as default.
3. **GPU triangulation** (reused math → batched GPU eigh) — independently useful,
   small, validate drift.
4. **Causal fitter (Level A):** rewrite the fit as an online windowed warm-started
   optimize (reuse SMPL-X + losses), CUDA-graph it. Validate GT.
5. **Streaming SAM2 (Level A):** vendor/adapt a forgetful-bank forward-only
   predictor. Validate masks/identity vs the batch SAM2.
6. **NVDEC decode + async log threads (Level A):** the realtime polish.

Stop after step 2 for the resource/I/O wins; continue to 4–6 only if realtime is a
hard requirement.

## Pros / cons

**Pros (mostly Level A, partly B):** large speedup (PR #48: ~11×, ≥80% realtime),
bounded memory that scales to long / many-camera clips (fixes the SAM2 OOM by
construction), GPU residency (no CPU↔GPU transfers), zero intermediate disk I/O,
and online/realtime emission (A only).

**Cons:** it is **not minimal** — A rewrites SAM2 + fitter + orchestrator; even B
needs the import-isolation refactor. **Results change** (causal/forgetful and
chunk-boundary effects differ from the joint batch fit — validate against GT, do
not expect bit-parity with the DAG). New deps (streaming SAM2 fork, NVDEC/torchcodec,
CUDA graphs). And it is a **second pipeline to maintain** alongside the batch DAG.

## Test the theory *before* building it

Streaming "sounds optimal," but its wins are specific and measurable — so quantify
the overhead it would remove **on the batch DAG** first, cheaply, before
committing to the rewrite:

1. **Intermediate I/O** — total bytes + write/read time of the mask PNGs
   (`ma_masks`→`ma_2d`) and the NPZ handoffs. Streaming removes all of it.
2. **Redundant decode** — the same video is decoded in `ma_masks`, `ma_2d`, and
   `ma_vis`; measure decode time × that redundancy. Streaming decodes once.
3. **Per-step startup** — ~3–6 s `import torch`/CUDA-init × 5 steps, per sequence.
   Streaming pays it once.
4. **CPU↔GPU transfers** — already largely removed within `ma_2d` by item A.

If (1)+(2)+(3) is a large fraction of wall time, streaming's *orchestration* win is
real; if small, the win is dominated by the *compute* optimizations
(TensorRT/CUDA-graphs/GPU-triangulation) — and those we can (and partly did, A+B)
land **inside the DAG** without a streaming rewrite. That decomposition is the
honest test of "is streaming optimal, or just the compute tricks it also ships?".

### Theory test — measured (eval seq, 6 cam / 225 f / 2 ppl)

| overhead streaming removes | measured | note |
|---|---|---|
| mask PNG round-trip | ~5–9 s/cam | 450 PNGs/cam, only 4.6 MB but ~4.7 s just to *read* (file-open bound) |
| redundant decode | ~5 s/cam | ma_masks + ma_2d + ma_vis each decode (~2 extra passes) |
| per-step startup | ~30 s/seq | ~6 s × 5 steps, once |

vs a pipeline dominated by **compute** (`ma_3d` ~230 s, `ma_2d` ~138 s/6 cam). So
the *orchestration* overhead streaming eliminates is **~20–25 %** of wall time —
real but not dominant. The bulk of PR #48's ~11× came from **compute** tricks
(TensorRT, CUDA-graph fitter, GPU triangulation, batched SAM2), which are
**orthogonal to streaming** and landable in the DAG — we already shipped TensorRT
(5.3×) and GPU preprocessing there.

**Conclusion of the test:** streaming is genuinely *optimal* on two axes the DAG
can't match — **bounded memory at scale** (no whole-video OOM) and
**realtime/online** emission. For pure **offline-batch throughput**, the
streaming-specific win is the ~20–25 % orchestration overhead; the larger speedups
are compute optimizations we can keep landing in the DAG more cheaply. So the
theory holds *for scale/realtime*, and is *modest* for offline throughput.

## Is the gating refactor (import isolation) worth it? — investigated: **no**

Renaming the clashing packages turns out to be far more than a mechanical rename:

- **It touches training.** `landmarks/train.py` imports `from utils.util …` and
  `from lib.models …`; ~7 files under `landmarks/`, ~6 under `segmentation/` (and
  similar in `optimization/`) import the to-be-renamed packages. So the refactor
  edits **training code** across three subtrees — exactly what we agreed to keep
  hands off.
- **String references break silently.** `segmentation/tests/...import_module("core.pipeline")`
  and config refs like `utils.paths.string_path_to_windows` aren't caught by a
  static "update the `from X import`" pass, so a rename risks silent breakage.
- **It breaks the modular workflow** — running/developing a single step in its own
  dir (the whole point of the subprocess layout).
- **Payoff is modest.** Per the theory test, Level B's offline-throughput win is
  ~20–25 %, and its genuinely-unique benefit (bounded memory at scale) is reachable
  far more cheaply **per step** (e.g. the SAM2 `init_state` OOM→offload we already
  shipped; an `ma_masks` chunking pass).

**What stays safe:** checkpoint loading (`torch.load(...)['state_dict']` is tensor
names, not pickled classes), and the GUI (it shells out to the step subprocesses,
doesn't import them).

**Verdict:** the import-isolation refactor is **not worth it** — high blast radius
(training + string refs + 3 subtrees), real silent-breakage risk, and a modest /
otherwise-reachable payoff. Get the memory-at-scale win with **targeted per-step
bounds** instead, keep the DAG, and reserve a true rewrite for **Level A** only if
realtime becomes a product requirement.

## Recommendation

- If the goal is **resource + I/O wins at scale** (bounded memory, GPU residency,
  no intermediate I/O): do **Level B** — it reuses the existing algorithms and our
  A/B work, and the only hard prerequisite is the import-isolation refactor. This
  is the "window of data on the GPU without intermediate I/O" you described,
  achieved with batch-within-chunk rather than a causal rewrite.
- If the goal is **realtime/online**: that requires **Level A**, a dedicated
  multi-week program (streaming SAM2 + causal fitter + orchestrator) — worth it
  only if realtime is a product requirement. Reuse stays high (weights, TensorRT,
  GPU preproc, SMPL-X/losses, assoc math), but the three hard pieces are new.
- Either way: **keep the batch DAG as the default**, add streaming as an opt-in
  path, and gate every step on the GT harness (results will differ from the DAG, so
  GT — not self-consistency — is the right yardstick).
