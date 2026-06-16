# Can we add PR #48-style causal streaming with minimal code?

Investigation requested after the `ma_2d` optimization work. Short answer:
**no — not with minimal code / no rewrite**, because the two heaviest stages are
fundamentally *batch and non-causal*. Streaming is a paradigm change, not a patch.
This documents why, and what *is* worth doing to get the streaming *benefits*
(speed, bounded memory, less I/O) at a fraction of the cost.

## Why streaming can't be a thin wrapper over today's code

Causal streaming means: push a *window* of frames through every stage and discard
it, holding only a small working set. That requires every stage to be **online /
causal**. Two of ours are not:

1. **SAM2 segmentation (`ma_masks`) is whole-video and bidirectional.**
   `segmentation/core/pipeline.py` calls `init_state(video)` (loads the whole
   clip — the ~5 GB OOM we hit) then `propagate_in_video` **forward and reverse**
   (lines 289 & 303). The reverse pass uses future frames to fix past masks →
   non-causal by construction. PR #48 had to vendor a *streaming SAM2 fork with a
   forgetful memory bank* to make this causal. That is a new dependency + new
   integration, not a wrapper.

2. **`ma_3d` optimization is a joint batch fit over all frames.**
   `batch_size = frames_len` (the whole sequence), and the objective couples
   frames together: `pts3d_temp_loss` and `angular_acc_loss` (velocity /
   acceleration over consecutive frames), plus
   `mvpose_style_associate_and_triangulate_temporal` (resolves cross-camera label
   swaps over the whole sequence). A causal sliding-window fit (what PR #48 does)
   is a **different algorithm** with different results — not a refactor.

So PR #48 is a ground-up re-implementation: it reuses the *weights* (MammaNet,
SMPL-X) but replaces the SAM2 integration (streaming), the fitter (causal
sliding-window, CUDA-graphed), and the orchestration (single process). On top of
that, our 5 step repos can't even share one process yet (the `utils`/`core`
name-clash from `architecture-analysis.md`). "Minimal code" and "streaming"
genuinely conflict here.

## What streaming would actually buy us (and the cheaper ways to get most of it)

| Streaming benefit | Cheaper, bounded alternative (no full rewrite) |
|---|---|
| Bounded memory / no whole-video OOM | **Chunk SAM2 over the video** (process N-frame windows, carry a small mask-context). Contained change in `ma_masks`; fixes the scale/OOM blocker directly. |
| No CPU↔GPU round-trips, GPU residency | **Within-step GPU residency** — already done for `ma_2d` (item A). |
| Faster forward | **Opt-in TensorRT** for the MammaNet forward (roadmap B). |
| Less intermediate I/O (mask PNGs) | Store masks compactly (RLE / packed bits) instead of 4K PNGs — contained; or skip them where a consumer can read in-stream. |
| Skip per-step process startup | **Persistent warm workers** per step-env (keeps isolation). |

These recover most of the *resource and speed* wins of streaming without the
causal-fitter rewrite, the streaming-SAM2 fork, or the import-isolation refactor —
and each is independently shippable and metrics-gated.

## Recommendation

- **Do not** try to bolt streaming onto the current batch stages — it would be a
  rewrite wearing a wrapper, and the temporal/bidirectional dependencies mean the
  results would change.
- **If** the streaming wins are wanted, treat it as its **own dedicated program**
  (a parallel "streaming" pipeline that reuses the weights), scoped explicitly —
  not a minimal patch. Sequence it after the cheap bounded wins above, and only if
  realtime/throughput becomes a hard requirement.
- **Highest-value next step toward the *same goals*, cheaply:** chunk SAM2 to
  bound `ma_masks` memory (fixes OOM at the 32-cam / 3k-frame scale), then opt-in
  TensorRT for the forward. Both are contained, validatable, and preserve the DAG.

## One-line summary

Streaming's value is real, but it lives in the *algorithms* (causal SAM2 + causal
fitter), so it can't be added cheaply. Chase the same benefits — bounded memory,
GPU residency, less I/O — with the contained, DAG-preserving changes above.
