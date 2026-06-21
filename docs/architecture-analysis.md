# Pipeline architecture analysis: the subprocess DAG vs. in-process / streaming

Written while scoping the "Phase-2 decode-once" idea (roadmap item C). The
investigation changed the conclusion, so this records the reasoning.

## What we have today

`inference/runner.py` runs MAMMA as a **5-step subprocess DAG**
(`ma_cap → ma_masks → ma_2d → ma_3d → ma_vis`). Each step is a separate Python
process launched by `engines.py` (conda / docker / apptainer) with `cwd` set to
that step's repo dir. Steps hand off via **disk** (NPZ / PNG / MP4). Order is a
topological sort of declared dependencies; per-`(step, seq)` `.DONE` sentinels
make runs resumable.

## The concrete blocker for "run steps in one process"

The step repos have **conflicting top-level package names**:
`segmentation/utils`, `landmarks/utils`, and `optimization/utils` are three
*different* `utils` packages (same story for `core` / `lib`). Each step works
only because it runs **isolated**, with its own `cwd` first on `sys.path`, so
`import utils` resolves to *its* `utils`. In a single process, the first-imported
`utils` wins in `sys.modules` and every other step gets the wrong module →
breakage. So "all steps in one process" is impossible **without first renaming
those packages** (a mechanical but cross-repo refactor touching every intra-step
import).

## Why the DAG exists — and it is not just legacy

| Benefit | Why it matters here |
|---|---|
| **Memory isolation** | Each step frees its GPU/CPU memory on exit, so we never hold YOLO + SAM2 + MammaNet + SMPL-X (and decoded frames) at once. The SAM2 whole-video OOM we already hit shows memory pressure is real, and the scale targets (≤8 people, ≤3k frames, ≤32 views) make it worse. |
| **Fault tolerance / resume** | A step crash/OOM doesn't kill the run; `.DONE` sentinels resume. |
| **Env / dependency flexibility** | The engine abstraction lets steps run in different conda envs / containers / cluster nodes. |
| **Modularity** | Each step is a self-contained repo you can run, test, and debug alone. |

## What the DAG costs

1. **Per-step startup** — cold `import torch` + CUDA init (~3–6 s) ×5 steps, **per
   sequence**. Significant for short clips; amortized for long ones.
2. **Cross-step redundant decode** — `ma_masks`, `ma_2d`, `ma_vis` each decode the
   source video independently (~2–3× redundant).
3. **Intermediate serialization** — masks PNG, landmarks NPZ, verts NPZ written
   then re-read.
4. **No GPU residency across steps** — frames/tensors can't stay on the GPU
   between steps.

## The decisive insight: decode-once at scale needs *streaming*, not in-process batch

"Decode once and share across steps" only helps if the shared frames live
somewhere all steps can read. In-process batch means **holding the decoded
frames in memory for the whole sequence**. At the scale targets that is
impossible: 32 cams × 3000 frames × 4K × 3 bytes ≈ **terabytes**. So a batch
in-process pipeline cannot hold the frames — it would OOM long before the DAG's
per-step models would.

The only way to truly decode-once across steps at that scale is to **stream**:
push a *window* of frames through all steps and discard it, holding only the
working set. That is exactly the [PR #48][pr48] causal-streaming design — and it
is the large rewrite we explicitly put **out of scope** (end goal = batch).

[pr48]: https://github.com/rerun-io/examples-monorepo/pull/48

So: **in-process batch fusion is blocked (imports) *and* doesn't scale (memory);
the scaling version is streaming, which is out of scope.**

## Options considered

1. **Keep the DAG (recommended).** Robust, memory-isolated, env-flexible, no
   refactor. Keeps the startup/decode/serialize costs.
2. **Full in-process batch fusion.** Needs the package-rename refactor *and* loses
   memory isolation *and* doesn't scale (frames in memory). Net: high cost, breaks
   the very thing (memory headroom) the scale targets need. Not worth it.
3. **Shared-memory frame server.** A decoder process fills `shared_memory`/memmap;
   steps read instead of re-decoding. Keeps subprocess isolation, attacks only
   redundant decode — but the shared buffer is itself memory-bound at scale, and
   it adds real coordination complexity. Marginal.
4. **Persistent warm workers per step.** The runner keeps one long-lived process
   per `(step, env)` that processes sequences sequentially, paying `import torch`
   /CUDA-init **once** instead of per sequence. Preserves all isolation; removes
   the startup cost for **multi-sequence** captures (the common real case).
   Feasible, low-risk, modest — the best DAG-preserving win if startup proves
   material. Does not address decode or GPU-residency.
5. **Streaming rewrite (PR #48).** The only true decode-once-at-scale. Big,
   out of scope, and trades batch simplicity for realtime complexity.

## Recommendation

**Keep the DAG. Do not pursue in-process fusion (drop roadmap item C as framed).**
The subprocess boundary is a deliberate design that buys memory isolation, fault
tolerance, and env flexibility — and at the stated scale, memory isolation is
worth *more* than the fusion speedup, because the fusion would OOM. The
"decode-once" win that motivated C is only reachable via streaming, which is out
of scope.

Direct remaining effort where it pays without touching the architecture:
- **Within-step GPU/portable wins** (already done for `ma_2d`: PR #2, B1, A).
- **Opt-in TensorRT** for the dominant forward (roadmap B), behind a flag.
- **Persistent warm workers** (Option 4) *iff* a full-pipeline measurement shows
  per-step startup is a material fraction for real multi-sequence captures.
- Decode itself: portable options are limited; the big one (NVDEC) is opt-in
  NVIDIA, same bucket as TensorRT.

If we ever want true decode-once / GPU-residency across steps, the honest path is
the **streaming rewrite**, decided as its own program — not a quick in-process
patch.

## What would break if we fused anyway

Import resolution (the `utils`/`core` clashes); peak GPU/CPU memory (OOM at
scale); the run-one-step-in-isolation workflow; env/container flexibility; and
fault-isolation/resume. That is a lot of working machinery to trade for a speedup
that the memory math says won't even hold at the target scale.
