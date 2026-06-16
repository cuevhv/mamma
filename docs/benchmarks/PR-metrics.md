# PR metrics ledger

Every optimization PR is justified by **measured** before/after numbers — never
estimates. This ledger is the running record of how each change moves the repo
on speed, GPU memory/utilization, CPU RAM, and accuracy. Decisions are
metrics-driven; a change that doesn't measurably help (or that costs more than
it saves) is rejected and recorded as such.

Reference target throughout: **[rerun-io/examples-monorepo#48][pr48]** (the
efficient MAMMA rewrite — ~11.3× end-to-end, GPU-resident). We mine its ideas
and check our numbers against its philosophy.

[pr48]: https://github.com/rerun-io/examples-monorepo/pull/48

## How metrics are produced (so they're reproducible, not hallucinated)

- **Speed / GPU mem / CPU RAM:** `scripts/benchmark.py` wraps a step subprocess,
  samples peak GPU memory (per-process `nvidia-smi`) and peak CPU RSS (`psutil`
  process tree), times wall-clock → appends to `results.jsonl` + `README.md`.
- **GPU utilization:** sampled with `nvidia-smi --query-gpu=utilization.gpu`.
- **Self-consistency accuracy:** `scripts/regression_check.py` — drift of `ma_2d`
  landmarks (px) and `ma_3d` joints/vertices (mm) vs a golden snapshot of `main`.
  (Now wipes outputs each run so it truly recomputes — see observations B3.)
- **Absolute accuracy (planned):** MPJPE/PVE vs ground truth on
  `mamma_eval_dance` (`gt/` dir; `run_ma_3d` computes these when `use_gt`).
- All numbers below are on the RTX 4090 / `mamma` env unless noted, and cite the
  measurement so they can be re-run.

## Ledger

### PR #1 — Regression harness + benchmark infra + roadmap/observations
Enabling infrastructure; **no pipeline behavior change.** Establishes the golden
snapshot, the metrics tooling, and the validation gate. Without it none of the
numbers below would be trustworthy.

### (Rejected by metrics) — Batch the `ma_2d` forward
Measured, then **dropped**: a clean example of a metrics-driven *no*.

| metric (1 cam, 426 frames) | bs=1 (baseline) | bs=16 | verdict |
|---|---:|---:|---|
| wall time | 329 s | 324 s (−1.6 %, noise) | no gain |
| peak GPU memory | ~7.6 GB | ~20 GB (**+12 GB**) | worse |
| forward speedup from batching | — | 1.08× | negligible |

Root cause: `ma_2d` was **CPU-bound** (GPU ~1 % util); the forward is ~2.5 % of
the work, and even that barely batches. Batching cost memory for no speed → cut.

### PR #2 — cv2-ROI anti-alias blur in `ViTDetDataset` (inference-only)
Replace the per-crop full-frame `skimage` gaussian blur with `cv2.GaussianBlur`
on the bbox ROI. This was the real `ma_2d` bottleneck (97 % of per-crop time).

| metric | before (main/skimage) | after (cv2-ROI) | delta |
|---|---:|---:|---|
| blur op / crop (4K) | 341 ms | 3.8 ms | **89× faster** |
| `ma_2d` wall (1 cam, 426 f) | 329 s | 78 s | **4.2× faster** |
| `ma_2d` loop rate | 1.34 frame/s | 6.45 frame/s | **4.8×** |
| peak GPU memory | ~7.6 GB | ~7.6 GB | unchanged |
| peak CPU RSS | 6.26 GB | 6.26 GB | unchanged |
| accuracy: `ma_3d` 3D drift vs main | — | mean ~0.1 mm, max ~1.8 mm | negligible |
| **accuracy: vs GT — MPJPE** | **21.65 mm** | **21.67 mm** | **+0.02 mm** |
| accuracy: vs GT — PA-MPJPE | 18.16 mm | 18.17 mm | +0.01 mm |
| accuracy: vs GT — PVE | 20.30 mm | 20.31 mm | +0.01 mm |

**Absolute accuracy is unchanged to 0.02 mm.** GT eval on
`mamma_eval_dance/250225_WestCoastSwing_Basic_Whip_…` (6 cameras, 225 frames,
2 people), `run_ma_3d` `use_gt` MPJPE/PVE vs the dataset `gt/`. Same `ma_cap`
inputs + provided masks; only `ma_2d` differs (skimage vs cv2-ROI). So PR #2 is
**4.8× faster `ma_2d` at zero accuracy cost** — a metrics-proven win.

After this fix `ma_2d` is **video-decode-bound** (~65 % of the loop is 4K H.264
decode) — the next lever (GPU/NVDEC decode, per PR #48). Patch-level fidelity vs
the old blur: mean |Δ| 0.014 / 255 (visible-joint 2D drift mean 0.05 px).

> Note: `run_ma_3d` hardcodes `use_gt=False` in its CLI entry (line 838); the GT
> eval above flipped it temporarily. A `--use-gt` flag is a small follow-up (the
> GT-accuracy harness task) so this is reproducible without editing code.

### B1 — Skip the unused ViTDet detector build in mask mode
`run_ma_2d.py::main` unconditionally built `DefaultPredictor_Lazy` (cascade
Mask-RCNN ViTDet-H), but in mask mode the detector is never called (boxes come
from the segmentation masks). Guarded the build behind `masks_folder is None`.

| metric (1 cam, masked, startup-dominated) | before | after | delta |
|---|---:|---:|---|
| `ma_2d` process wall time | 13.96 s | 7.66 s | **−6.3 s** |
| peak CPU RSS | 6.26 GB | 1.67 GB | **−4.6 GB** |
| output | — | byte-identical | **0 drift** |

Pure win, zero behavior change in mask mode (detector never ran). The freed RAM
(and the detector's GPU memory) directly helps the memory-scale targets.

### A — GPU crop preprocessing (kornia warp/blur/normalize, on-device)
Replace the per-crop CPU `ViTDetDataset` path in `run_ma_2d.py` with on-GPU
preprocessing: the decoded frame is uploaded to the device **once per frame**,
and every body's crop is produced by a kornia affine `warp_affine` (+ anti-alias
`gaussian_blur2d`) and normalized on-device. Supersedes PR #2's cv2-ROI blur for
`ma_2d` (the GPU path no longer goes through `ViTDetDataset`).

| metric | cv2-ROI (PR #2) | GPU-preproc (A) | result |
|---|---:|---:|---|
| preprocess / crop (isolated) | 3.76 ms | 1.03 ms | **3.6×** |
| `ma_2d` wall (6 cam, 225 f) | 125 s | 105 s | ~16 %¹ |
| peak CPU RSS (6 cam) | 6.26 GB | 3.52 GB | lower¹ |
| **GT MPJPE** (vs `gt/`) | 21.67 mm | **21.66 mm** | **+0.01 mm vs main** |

¹ The 6-cam run also carries B1 (detector skip), which accounts for part of the
wall/RSS gain; the **isolated** preprocessing speedup is the 3.6× micro-benchmark.
Patch fidelity vs the CPU path: mean 0.14 on a 0–255 scale → GT MPJPE moves
+0.01 mm. Strategic value: the frame stays resident on the GPU and there is no
per-crop host→device copy — the foundation for decode-on-GPU (Phase 2 / NVDEC).

> Layering note: with A merged, `ViTDetDataset.__getitem__` (and its PR #2 blur)
> is no longer used by `ma_2d` — a candidate for removal in a follow-up cleanup.

## End-to-end: absolute "seconds saved" (all changes vs. `main`)

All shipped changes (PR #2 + B1 + A + viz-default-off) live in `ma_2d`; the other
four steps are unchanged, so **pipeline seconds-saved = `ma_2d` seconds-saved**.
Measured per camera (`main` vs. the optimized branch), then scaled by camera
count (`ma_2d` is per-camera-sequential → linear in cameras).

| sequence (1 cam) | `main` | optimized | saved/cam | speedup |
|---|---:|---:|---:|---:|
| WestCoastSwing — 2 ppl, 225 f | 69.2 s | 23.0 s | **46 s** | **3.0×** |
| MultiMama — **6 ppl, 743 f** | 298.1 s | 128.5 s | **170 s** | **2.3×** |

Extrapolated to full multi-camera sequences (`ma_2d` step wall time):

| sequence | cams | `main` `ma_2d` | optimized `ma_2d` | **saved** |
|---|---:|---:|---:|---:|
| WestCoastSwing (2 ppl, 225 f) | 6 | 6.9 min | 2.3 min | **~4.6 min** |
| MultiMama (6 ppl, 743 f) | 32 | **~159 min** | **~68 min** | **~90 min** |

Stress-axes covered: **more people** (2→6), **more frames** (225→743), **more
cameras** (6→32). The gain grows with the workload, so the biggest absolute
saving is on the largest capture: a 6-person, 32-camera, 743-frame sequence's
`ma_2d` drops from ~2.6 h to ~1.1 h — **~90 minutes saved**, at GT-accuracy parity
(+0.01 mm) and lower peak RAM. (Numbers are measured per camera on real masks;
the full-sequence rows are the linear ×cameras extrapolation.)

## Pending / next metrics
- Decode optimization / opt-in TensorRT (the new `ma_2d` bottleneck is the forward).
- Minimal-code path to PR #48-style causal streaming (investigation).
- Memory behavior at scale (full 32-cam run, `data/mamma_multi`).
