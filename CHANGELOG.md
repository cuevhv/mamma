# Changelog

## v1.1.0

### Segmentation (ma_masks)

- **New `sam3_prompt_light` backend.** SAM3 text-prompt detection driving the lean SAM3 tracker — no YOLO, no 16× multiplex — a lighter option alongside `sam2` / `sam3` / `sam3_prompt`.
- **Unified track-then-match cross-camera identity (epipolar-aware).** Every backend now propagates each camera independently into clean tracklets, then remaps whole tracklets to the init-camera ids using CLIP + epipolar geometry. `--calibration` now injects intrinsics/extrinsics so `--videos_dir` / `--images_root_dir` also get epipolar geometry (previously only `--ma_cap_dir` did); choose the path with `--cross-camera-strategy` (calibration-aware `auto` by default). With calibration present, `sam2` / `sam3` / `sam3_prompt_light` reach `sam3_prompt`-level cross-camera accuracy.
- **Outsider-robust seeding.** Track-then-match seeds every detected person and lets the remap drop bystanders/outsiders, instead of pre-trimming to the expected count by confidence (which could drop a partner-occluded real subject for a cleaner-looking outsider).
- **Long-clip memory efficiency.** Opt-in `--lazy-frames` bounds host RAM to a small LRU window (byte-identical output); `--prune-memory` bounds SAM 2's growing memory-attention state. Both off by default.
- **Faster mask export (byte-identical).** The per-frame overlay/PNG export no longer builds full-frame float64 temporaries per object — the whole ma_masks step runs ~35% faster (and ~20% less host RAM) with byte-identical mask PNGs and overlay videos.
- **Faster and more robust.** `sam3_prompt` reuses SAM3's own detector instead of a redundant YOLO pass. Hardening for the SAM3-text backends: the low-confidence init fallback now actually engages on hard/occluded init frames (and runs on the detector's own device, so CPU/MPS hosts work), plus a fix for a multi-person gallery-lookup crash. Also deterministic `MaskStore` cleanup, an OOM-retry guard, and a forced headless matplotlib backend.

### Landmarks (ma_2d)

- **Opt-in TensorRT FP16 backend (~5×).** `--tensorrt` runs the ma_2d forward pass through a disk-cached TensorRT engine.
- **GPU-side crop preprocessing.** Warp / blur / normalize moved onto the GPU (kornia).
- **Faster by default.** Skips the unused ViTDet detector build in mask mode. Visualizations stay on by default — `--disable-visualizations` turns them off (`--save_cam_output` kept as a deprecated alias).
- **Faster input path (bit-identical).** The per-crop bbox comes from `cv2.boundingRect` instead of a full-frame `np.where`, and frame decode + mask reads are prefetched on a worker thread while the previous frame computes — ma_2d runs ~40–50% faster with bit-identical outputs.

### Optimization (ma_3d)

- **Occlusion-aware gating (new, opt-in).** Fixes implausible fits when a body part is hidden in most views (e.g. legs behind a table coming out with impossible rotations). Before, occluded keypoints were held at a minimum weight, so the many views that *couldn't* see a part out-voted the one that could; gating instead drives occluded detections toward zero so the view that actually sees the part wins. Enable with `--occlusion-aware-weights` or an `occlusion_gating` config block — **off by default, existing runs are unchanged.**
- **VPoser pose prior (new addition).** A gentle regulariser (`vposer_recon_loss`) that keeps joints with little image evidence on realistic poses by penalizing how far they sit from VPoser's learned pose manifold. Requires the VPoser `V02_05` weights.
- **Recommended config for occluded captures.** `config_real_gmf_small_vals_detr_exp_no_vtemplate_occlusion_vposer.yaml` bundles both of the above at validated settings — the go-to setup for sequences with heavily occluded subjects.
- **Rerun contact visualization.** Contact diagnostics now render into the `.rrd` scene (`intermediate_triangulated_points.rrd`), replacing the standalone Plotly HTMLs.
- **Ground-truth evaluation.** `--use-gt` on `run_ma_3d` for reproducible evaluation on the `mamma_eval_*` sequences.
- **Opt-in `--tf32` fast fits.** TF32 tensor-core math for the optimization loop (~2.5× faster on Ampere+). Off by default: the fit lands in a nearby different optimum (~5–6 mm vs FP32; +1.6 mm GT MPJPE measured) — use for iteration/preview, keep FP32 for final fits.
- **Cleaner outputs.** The `verts_joints` npz no longer duplicates `gt_*` arrays during inference; ma_3d debug videos are re-encoded to browser-friendly H.264 / yuv420p so they play inline in the viewer.

### Visualization (ma_vis)

- **Generic lens undistortion.** When overlaying the 3D fit on the original footage, `--undistort` now correctly undistorts `radtan` / `opencv_brown` lens models (previously a silent no-op for these), so the render lines up with the image.
- **H.264 Rerun video backdrop.** `--rerun-video` replaces per-frame JPEG backdrops with a single H.264 stream — now the default, with a safe JPEG fallback and browser-WebCodecs-compatible encoding.
- **All-view overlays and faster rendering.** Render overlays for every view via the `all` sentinel; overlay rendering is parallelized (~2×, `--overlay-num-workers`).

### SMPL-X export (new)

- **Export fitted SMPL-X.** To a Blender-addon npz, and to rigged FBX / ABC / BVH / USD via a portable, auto-downloaded Blender.
- **Orientation handling.** Auto up-axis + floor detection (`--up-axis auto`, the default); "respect the data" orientation by default with an optional Blender-compat fix; meters-only units.

### Calibration & world orientation

- **Multi-format calibration loaders** with an explicit convention and up-axis detection.
- **Signed world up-axis** threaded through ma_3d and ma_vis, fixing scene orientation end-to-end.
- **Generic lens distortion in the ma_cap NPZ.** Each per-camera NPZ carries `distortion_model` + `distortion_coeffs` for every model (radtan / opencv_brown / vicon_radial_2), so `--undistort` works straight from `--ma_cap_dir` (no separate `--calibration`); legacy `vicon_radial_2` still read.

### Presets & performance

- **Preset refactor.** Example presets consolidated to `quick` (~2 s smoke test) + `full` (all frames, low-memory `ma_masks`). Speed knobs are ordinary per-step flags instead of dedicated presets: ma_2d `--tensorrt`, ma_vis `--overlay-num-workers`, ma_masks `--lazy-frames` / `--prune-memory`, ma_3d `--use-gt`.
- **Regression harness.** Output-regression tests and a metrics ledger to track end-to-end timings.

### GUI

- **SMPL-X Exporter.** New Exporter tab (downloads, health check, sequence pick, format select, run), a shared export panel on the Results page, and multi-select batch export with a Blender version check.
- **Redesigned Results page.** Logs/config access, per-step copy-path, an outputs explorer, and #F/#V columns.
- **Task table.** Status ⇄ Timing toggle with per-step times (frames/views recorded per sequence after ma_cap); per-sequence row delete and whole-task Stop/Restart on grouped rows; imported runs share the live Done status tag.
- **New Task form.** Surfaces and edits the preset frame range; video ingestion for "New Capture" configs (`capture_root` vs `ioi_root`); every preset labelled in the quickstart wizard.
- **Calibration UI.** Calibration status, camera-rig 3D preview (`.rrd` in the Rerun viewer), and up-axis controls; clearer Rerun hints (Chromium-only backdrop note, fresh-layout open).
- **Performance & polish.** Cache-first thumbnails fix the slow Captures/Results loads; higher surface/border contrast, friendlier path/error wording (muted, not red), and a soft in-app reload on Capture detail.

### Containers

- **Unified Docker/Apptainer image for the whole pipeline.** `docker/Dockerfile` + `docker/mamma.def` build one image (Python 3.11, CUDA 12.4, torch 2.5.1 — the same pinned env as `requirements/`) that runs all five steps via the per-step `engine: docker` / `engine: apptainer` preset keys; usable from both the CLI and the GUI (select a preset that sets the engine; the command preview shows the exact `docker run` / `apptainer run` line). Setup + preset examples in `docker/README.md`.
- **Runner-contract fixes for the standalone ma_masks image.** `segmentation/Dockerfile` no longer sets `ENTRYPOINT ["python"]` (the runner passes `python` itself, so pipeline invocations became `python python …`); `segmentation/mamma_masks.def`'s runscript now prefers the bind-mounted `/repo` over the baked copy, so pipeline runs execute the checked-out code.
- A relative `sif_path` in a preset now resolves against the repo root (was: against the runner's cwd).

### Datasets & assets

- **MammaSyn from Hugging Face.** The synthetic dataset now downloads from the (gated) HF dataset by default; docs updated for `hf auth login`.
- **Optional VPoser download.** Fetch the VPoser `V02_05` weights via CLI or GUI (needed for the ma_3d pose prior).
