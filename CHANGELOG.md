# Changelog

## v1.1.0

### Segmentation (ma_masks)

- New `sam3_prompt_light` backend: SAM3 text detection driving the lean SAM3 tracker, no YOLO.
- Identity cleanup is now the default: the mask-continuity gate drops transient teleport/duplicate stretches, collapse recovery re-acquires a person lost to a tracker takeover (with a convergence guard so real full-contact overlap is never re-seeded), and seed selection is centered on long clips. Multi-frame init anchors were removed (measured harmful on similar-looking people). All knobs live in the config YAML; the CLI carries no algorithm flags.
- Cross-camera identity scores against every completed camera (not just the init camera), and each completed view enriches the per-person appearance bank.
- Decode-ahead prefetch keeps the GPU from waiting on frame reads; mask PNG writes are verified and resume caches are written atomically.
- Track-then-match cross-camera identity (CLIP + epipolar) for every backend; calibration-aware `auto` strategy.
- Seed prescreen: crowded views shortlist candidates per subject before tracking, so bystanders can't exhaust GPU memory.
- Re-latched tracklets (mid-sequence duplicates) are truncated at the takeover frame.
- Long-clip memory is bounded automatically: decoded frames offload to host RAM, and past 1500 frames the memory bank offloads and frame loading turns lazy (`sam.offload_state_to_cpu` / `sam.lazy_frame_loading`, both `auto`).
- ~35% faster mask export, byte-identical outputs.
- Sturdier SAM3-text init fallback; multi-person gallery-lookup fix; `sam3_prompt` reuses its own detector.

### Landmarks (ma_2d)

- Opt-in TensorRT FP16 backend (~5x faster forward pass).
- GPU-side crop preprocessing and a prefetched input path: ~40-50% faster, bit-identical.
- `--disable-visualizations` to skip viz output (`--save_cam_output` kept as alias).

### Optimization (ma_3d)

- Re-ID phantom fixes: unsupported views gated out, stale duplicates cleared, untriangulatable bodies excluded, association conflicts quarantined. Re-ID debug videos now label each subject.
- Opt-in `--tri-irls`: robust triangulation of the 3D seed.
- Opt-in `--occlusion-aware-weights`: fixes implausible fits for body parts hidden in most views.
- Optional VPoser pose prior; a bundled config recipe for heavily occluded captures.
- `--use-gt` evaluation mode; opt-in `--tf32` fast fits (~2.5x).
- Contact visualization in the Rerun `.rrd`; leaner output npz; browser-friendly debug videos.

### Visualization (ma_vis)

- `--undistort` now handles radtan / opencv_brown lenses (was a silent no-op).
- H.264 Rerun video backdrop by default.
- All-view overlays via the `all` sentinel.

### SMPL-X export (new)

- Export fits to Blender npz, or rigged FBX / ABC / BVH / USD via a portable auto-downloaded Blender.
- Auto up-axis + floor detection; meters-only units.
- ABC exports no longer write face sets that hijack scene materials.

### Calibration & world orientation

- Multi-format calibration loaders with up-axis detection.
- Signed world up-axis threaded through ma_3d and ma_vis.
- Per-camera NPZ carries every lens-distortion model, so `--undistort` works straight from `--ma_cap_dir`.

### Presets & performance

- Presets consolidated to `quick` and `full`; speed knobs are ordinary per-step flags.
- Output-regression tests and a timing ledger.

### GUI

- New SMPL-X Exporter tab, shared export panel, batch export.
- Interactive in-browser 3D viewer for ma_3d results.
- Redesigned Results page: logs, configs, outputs explorer.
- Task table: per-step timings, per-sequence delete, Stop/Restart.
- New Task form: preset editor with command preview, frame-range editing, video ingestion.
- Calibration status, camera-rig 3D preview, up-axis controls.
- Faster page loads (cache-first thumbnails) and UI polish.
- Multi-sequence tasks: a failed sequence no longer cancels the rest; removing a sequence survives Restart.
- Fixes: Tasks-to-Results opens the clicked sequence; one "Heavy occlusion-aware" toggle for ma_3d.

### Containers

- One Docker/Apptainer image for the whole pipeline, selected via `engine: docker|apptainer` preset keys.
- Standalone ma_masks image fixes (entrypoint, bind-mounted repo, relative `sif_path` resolution).

### Datasets & assets

- MammaSyn downloads from Hugging Face (gated) by default.
- Optional VPoser weights download via CLI or GUI.
