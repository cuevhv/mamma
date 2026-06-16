# MAMMA optimization — observations log

A running, evidence-based log of findings, bugs, and opportunities spotted
while executing the [optimization roadmap](optimization-roadmap.md). Each
entry records **what** was observed, the **evidence**, and the **date**, so
later PRs can rely on recorded facts rather than recollection. Speculative
items are marked *(unverified)*.

Conventions: measurements are on the reference machine (RTX 4090, 24 GB)
in the `mamma` conda env, on `data/mamma_example`
(`pushing_and_lifting_from_ground`, 4 cameras) unless noted. "Warm" = OS/GPU
caches primed by a prior run; "cold" = first run.

---

## Verified findings

### F1 — The pipeline is deterministic here (noise floor = 0.000)
`scripts/regression_check.py` main-vs-main on the quick preset (frames 60–90)
shows **0.000** drift across all 8 compared arrays (4× `ma_2d` landmarks,
2× `ma_3d` vertices+joints). Implication: numerically-neutral refactors can be
held to ~0 drift; any non-zero drift is a real change, not RNG.
*Evidence:* PR #1 validation, 2026-06-16. *Caveat:* depends on
`cfg.deterministic`/seed in `ma_2d` and the optimizer; re-confirm if those change.

### F2 — Wall-clock is strongly cache-sensitive
Same pipeline: **cold** total 351 s vs **warm** total ~57 s. Per step (cold→warm):
ma_masks 111→9 s, ma_2d 106→12 s, ma_3d 106→8 s, ma_vis 27→27 s.
Implication: only compare timings captured under the same warm/cold state.
The golden's stored timing is cold and must NOT be used as a speed baseline.
*Evidence:* PR #1 golden vs first check, 2026-06-16.

### F3 — Batching the `ma_2d` forward is numerically exact but a small steady-state win
Batched landmark inference (PR #2) gives **0.000** drift vs batch-size-1
golden (bit-identical here). But warm `ma_2d` only moved ~12.5 s → ~12.0 s
(~4%). The GPU forward is **not** the steady-state `ma_2d` bottleneck at this
scale; per-crop CPU preprocessing (mask PNG read, gaussian anti-alias blur,
patch generation in `ViTDetDataset.__getitem__`) dominates.
*Evidence:* PR #2 regression check + A/B, 2026-06-16.
*Action:* to actually speed up `ma_2d`, attack CPU preprocessing (overlap with
GPU via `num_workers`, or move blur/crop to GPU), not just batch the forward.

### F4 — `ma_2d` is CPU-bound, not GPU-bound (GPU ~1% utilized)
During full-sequence `ma_2d` inference, `nvidia-smi` shows **~1% GPU
utilization**. The wall time is spent on CPU: video decode + per-crop
preprocessing. Direct consequence: batching the GPU forward (PR #2) cannot
meaningfully reduce `ma_2d` wall time, even though it is numerically exact.
*Evidence:* live `nvidia-smi` during 426-frame run, 2026-06-16.

**Measured baseline** (1 camera, 426 frames, bs=1, no-draw, warm): **329 s
wall (~1.3 frame/s ≈ 0.38 s/crop), peak CPU RSS 6.3 GB, GPU util ~1%.**
Extrapolates to ~22 min for the 4-camera full sequence. This is the number
every `ma_2d` PR should move. *(`/usr/bin/time -v`, 2026-06-16.)*

### F5 — Batching raises `ma_2d` peak GPU memory (bs=16 ≈ 16 GB)
At batch size 16, the `ma_2d` process peaked near **16 GB** GPU. So bs=32 is
likely to OOM on a 24 GB card, and large batches reduce headroom / increase
contention (a stray bs=16 process contributed to a SAM2 OOM in `ma_masks`).
Implication: the batched path needs a conservative default and is a
*memory/throughput trade-off*, not a free win. *Evidence:* sampler during the
batch-size sweep, 2026-06-16.

### F6 — The dominant `ma_2d` CPU cost is a full-image gaussian blur per crop
`ViTDetDataset.__getitem__` (`landmarks/lib/datasets/vitdet_dataset.py:61,68`)
does `cvimg = self.img_cv2.copy()` then `skimage.filters.gaussian(cvimg, …)`
on the **entire 4K frame** for **every crop**, whenever the person bbox is
large (`downsampling_factor > 1.1`, common in 4K footage). With ~850 crops per
camera this is the `ma_2d` bottleneck. *(Used only by `run_ma_2d` inference,
not training — but numerics-affecting, so validate via harness.)*

**Decomposition (4K frame, /tmp/ma2d_profile.py, 2026-06-16):**

| per-crop op | time |
|---|---|
| CPU preprocessing, blur ON (large bbox) | **386.6 ms** |
| CPU preprocessing, blur OFF (small bbox) | 3.4 ms |
| → full-4K blur+copy overhead | **383.2 ms (~97%)** |
| GPU forward bs=1 | 10.5 ms |
| GPU forward bs=16 | 9.7 ms/crop (1.08× — negligible) |

Whole-run confirmation (1 cam, 426f): bs=1 **329 s** vs bs=16 **324 s**
(1.6%, noise) for +12 GB GPU memory. **The blur is ~97% of `ma_2d`; batching
the forward is irrelevant** (and the forward itself only batches 1.08×, so
batching was dropped).

### F7 — cv2-ROI blur fix: ~4.8× faster `ma_2d`, now decode-bound
Replacing the full-4K `skimage` blur with `cv2.GaussianBlur` on the bbox ROI
(`vitdet_dataset.py`): blur step **341 ms → 3.8 ms (89×)**; whole-camera run
**329 s → 78 s (4.2× total, 4.8× loop: 1.34 → 6.45 frame/s)**. Patch fidelity
vs the old blur: max |Δ| 1.04 / mean 0.014 on a uint8 scale (negligible).
After this fix `ma_2d` is **video-decode-bound** (~65% of the loop is 4K H.264
decode) — the next lever, aligned with GPU/NVDEC decode. *Evidence:*
`/tmp/blur_compare.py` + timed runs, 2026-06-16.

### F8 — GT accuracy unchanged by the cv2-ROI blur (the decisive validation)
Absolute MPJPE/PVE vs ground truth on
`mamma_eval_dance/250225_WestCoastSwing_Basic_Whip_…` (6 cams, 225 frames, 2
people), via `run_ma_3d` `use_gt`: **main 21.65 mm MPJPE vs cv2-ROI 21.67 mm**
(Δ +0.02 mm; PA-MPJPE 18.16→18.17; PVE 20.30→20.31). Same `ma_cap`/masks; only
`ma_2d` differs. So PR #2 is **4.8× faster at zero accuracy cost.** Note:
`run_ma_3d` hardcodes `use_gt=False` at its CLI entry (line 838) — flip it (or
add a `--use-gt` flag) to reproduce. *Evidence:* GT eval, 2026-06-16.

---

## Bugs / smells

### B3 — Harness reused stale step outputs (FIXED) — invalidated early "0.000" checks
`scripts/regression_check.py --check` was reporting **0.000 drift for every
change** because the pipeline steps (`run_ma_2d.py` etc.) **skip when their
output `.npz` already exists**, and the runner's `--force` only clears DONE
sentinels — so each check silently reused the golden run's outputs. Caught via
mtime (golden newer than the "check" output). **Fix:** the harness now wipes the
out-dir before each run so every step recomputes from scratch. Any "0.000"
validation recorded before this fix (2026-06-16) is void. *Lesson: a regression
check must prove it actually recomputed the thing it's checking.*

**Real cv2-ROI drift (post-fix, quick preset, cv2 vs main/skimage):** `ma_2d`
landmarks mean 0.06–0.14 px (max ≤6.7 px); `ma_3d` joints/vertices **mean
~0.1 mm, p99 ~0.5 mm, max ~1.8 mm**. Well within the "numerically-changing"
budget (a few mm). The full-seq 2D drift on visible joints was mean 0.05 px (F7).


### B1 — `ma_2d` builds the heavy ViTDet detector even in mask mode (unused)
`landmarks/run_ma_2d.py::main` unconditionally constructs
`DefaultPredictor_Lazy` (cascade_mask_rcnn ViTDet-H) — but in the pipeline
(mask mode, `--mask_path` set) the detector is **never called**; boxes come
from the segmentation masks. This loads/initializes a large model on every
`ma_2d` process for nothing, inflating per-step startup substantially.
*Evidence:* code path in `main()` (detector built after model load); standalone
`ma_2d` runs dominated by this cost, 2026-06-16. *(startup cost — quantify and
likely guard behind `masks_path is None`.)*

### B2 — `--save_cam_output` defaults to True (debug I/O on by default)
Every `ma_2d` run writes per-body debug JPEGs (frames where `n % 20 == 0`) and
stitches per-body preview MP4s via ffmpeg. This is debug visualization done on
normal pipeline runs — a candidate for the Phase-2 `--debug-io` toggle
(default off). *Evidence:* parser default in `run_ma_2d.py`, 2026-06-16.

---

## Opportunities (to quantify before acting)

- **O1** — Skip the unused detector build in mask mode (see B1). Likely the
  largest single `ma_2d` per-step win; trivial and safe (guarded by mode).
- **O2** — Overlap `ma_2d` crop preprocessing with the GPU (DataLoader
  `num_workers>0`, or GPU-side blur/resize) to cut the steady-state floor (F3).
- **O3** — Make debug I/O (B2) optional under the Phase-2 toggle.
- **O4 (high value)** — Fix the per-crop full-4K blur+copy (F6): blur/copy only
  the crop ROI, or use `cv2.GaussianBlur`, or blur after extracting the patch.
  Likely the single largest `ma_2d` speedup. Numerics-affecting → gate on the
  regression harness. This, not batching, is where `ma_2d` time actually goes.

## Scale targets & memory constraints (keep in mind for every change)

Real captures can be much larger than the example. Soft maxes (not fixed — just
plan for them): **up to ~8 people**, **up to ~3000 frames**, **up to 32 camera
views**. So when a change scales memory with people × frames × views, be
conservative — a default that fits the 4-cam/30-frame example can OOM at scale.

Guidance:
- **Don't grow batch sizes blindly.** Prefer streaming/chunked processing with a
  bounded working set; expose a max (people/frames/views per batch) and split
  into multiple batches when the set won't fit, rather than loading everything.
- Recommend sweet-spot caps for GPU/CPU memory rather than unbounded growth.
- **Known OOM risk: SAM2/SAM3 video inference loads the whole video onto the GPU**
  (`load_video_frames_from_video_file` tried to allocate ~5 GB for 426 frames;
  see the `ma_masks` OOM, 2026-06-16). Long sequences will OOM — needs windowed
  decode / chunked propagation. Handle in a future `ma_masks` PR.
- This mirrors PR #48's philosophy (bounded transient buffers, forgetful memory
  bank, 512-vertex SMPL-X) — keep data on GPU, but keep the *working set* bounded.

## Data & environment notes

- **Larger test data available** (for memory-scaling tests): `data/mamma_multi`
  symlinks the cluster release `…/mamma_markerless_multiple_people`. Example
  6-person sequence `260216_MultiMama_6_social_circle_111111_1`: **32 cameras
  (IOI_01..IOI_32), 743 frames, 2056×1504**. Calib is
  `configs/examples/calib/260216_MultiMama.yaml`. Each sequence offers
  `videos/`, `videos_crf16/`, `videos_crf24/` (use the lightweight `videos_crf24`
  for now) plus precomputed `masks/`, `meta/` (per-cam NPZ), `pred/`.
  6 people × 32 views ≈ 192 crops/frame — a real stress test for the scale
  targets above.
- **Video encoding caveat (note for the future):** the release videos may not be
  encoded in the most decode-friendly way for our libraries (codec/GOP), which
  matters now that `ma_2d` is decode-bound (F7) and SAM2 loads whole videos.
  Re-encoding to a decode-optimized format (e.g. all-intra / smaller GOP, or a
  codec our stack decodes fastest) is a possible future speedup. *(User will
  handle re-encoding; flagged here so we account for it.)*

## Non-issues (verified — don't re-investigate)

- **`ma_3d` scene videos are already skipped.** `run_ma_3d` prints "Scene video
  overlay rendering has moved to run_ma_vis.py … Skipping scene videos" and
  produces 0 `*_smplx_scene.mp4` regardless of `--skip_scene_videos` (now a
  no-op). No redundant per-frame mesh rendering in `ma_3d` to remove; its wall
  time is the optimization itself (~230 s on 6 cam / 225 f / 2 ppl).
  *Verified by measurement, 2026-06-17.*

## Open questions

- Q1 — What is the full `mamma_example` sequence length, and how do the metrics
  (time / GPU mem / CPU RAM) scale vs the 30-frame quick run? *(benchmark in
  progress)*
- Q2 — Where exactly does steady-state `ma_2d` time go (mask read vs blur vs
  patch-gen vs forward)? Needs a per-stage profile to target O2.
