# Changelog

## v1.1.0

### Migration from v1.0.x

- **Person detector.** ma_masks detects people with D-FINE-X through Hugging Face `transformers` instead of Ultralytics YOLO12x. Install the updated requirements (`transformers>=4.52,<5.19`, the range that supports the pinned torch 2.5.1; `ultralytics` is no longer listed). The weights (about 240 MB) download from Hugging Face on the first `sam2` / `sam3` run, so compute nodes without internet access must pre-populate the Hugging Face cache or pass `--detector-checkpoint <local dir>` (see `docs/INSTALL.md`).
- **`MAMMA_YOLO_CHECKPOINT` is no longer used**: a value left in `.env.local` is ignored (the runner warns when it is set) and no longer passed as `--yolo-checkpoint`. To keep YOLO, `pip install ultralytics` and add `--detector yolo --detector-checkpoint <file>.pt` to the ma_masks `flags`. `--yolo-checkpoint` is now optional and implies `--detector yolo`.
- **Mask output layout.** ma_masks writes one `<camera>/masks/masks.zip` per camera (the same `mask_FFFF_PP.png` entries inside) instead of loose PNG files. ma_2d and the GUI read either layout. Scripts that glob the PNG files should read the archive (`segmentation/core/mask_archive.py`, or any ZIP reader) or run ma_masks with `--png`.
- **No CPU fallback on a GPU host.** When an NVIDIA device is present but CUDA fails to initialize, ma_masks stops with an error instead of running on the CPU. There is no override: hiding the GPU on such a host (for example `CUDA_VISIBLE_DEVICES=""`) also stops the run. Fix the GPU or run on another machine.
- **Training data root renamed** from `data/mamma` to `data/mammasyn` in the ma_2d training configs (`dataset_path`). Move or symlink existing data, or override `dataset_path`. Hi4D and SignAvatars are no longer offered by the MammaSyn download scripts (`--interactions` / `--hands`).
- **ma_masks results differ from v1.0.x** on the same input: track-then-match identity on calibrated rigs, the mask-continuity gate, collapse recovery (sam2), seed centering and prescreen, re-latch truncation, the rig vote on rigs of 12 or more calibrated cameras, and the new detector are on by default, and multi-frame init anchors were removed. For a closer comparison, pass `--cross-camera-strategy match_then_track --rig_vote false` and set `masks.mask_continuity_gate` / `masks.recover_collapsed_tracklets` to `false` in a copied config YAML.
- **Removed ma_masks config keys** (`matching.init_anchor_enable`, `init_anchor_score_threshold`, `init_anchor_max_per_id`, `sam.fp16_frames`) are ignored without a warning.
- **`segmentation/process_sequence.py` is no longer a script**; use `segmentation/run_ma_masks.py`.
- **GUI API origin check.** The GUI backend refuses API requests sent from other web origins (it allowed any origin before). The bundled UI, the Vite dev server on localhost and SSH tunnels work as before; add any other origin (for example the dev server reached through a LAN hostname, or a reverse proxy that rewrites `Host`) to `MAMMA_ALLOWED_ORIGINS`.
- **Exit code on cancel.** `python -m inference run` exits with 130 when it is cancelled (it returned 0).
- **World up-axis.** The runner detects the up-axis from the camera rig and passes it to ma_3d and ma_vis, so captures whose calibration is not Z-up are oriented differently than in v1.0.x. Set `up_axis` in the capture JSON to pin it.
- **ma_3d outputs.** The `verts_joints` npz carries `gt_*` arrays only with `--use-gt` (read `pred_*`), and contact visualization is a Rerun `.rrd` instead of Plotly HTML files.
- **ma_2d output npz** omits `contacts_gt` / `floor_contacts_gt` when no ground truth is given (check `npz.files`).
- **ma_vis defaults.** `--fps` follows the capture's recorded frame rate (was 30), overlays render in parallel (`--overlay-num-workers` automatic, was 1), and the Rerun backdrop is an H.264 video (needs ffmpeg, else JPEG). Pass `--fps 30`, `--overlay-num-workers 1` or `--no-rerun-video` for the old behaviour.
- **Presets.** A `--cam_names` subset in a step's `flags` now restricts that step (before, the capture's camera list overrode it). Remove stray `--cam_names` flags if all cameras are wanted.
- **Containers.** Rebuild images from `docker/` (one image for all steps) or `segmentation/Dockerfile`: images need `transformers`, the engine contract changed (repo root bound at `/repo`, docker runs as your user, apptainer uses `--nvccli` when available), and the standalone ma_masks image no longer has a `python` entrypoint.

### Segmentation (ma_masks)

- Person detection for sam2/sam3 now uses D-FINE-X (Apache-2.0 code and COCO-only weights, via transformers) instead of Ultralytics YOLO12x (AGPL-3.0). Weights download from Hugging Face on first use. `--detector yolo` keeps the old model (install `ultralytics` yourself), and `--detector package.module:Class` plugs in your own; see `segmentation/core/person_detector.py`. `--detector-checkpoint` selects other weights; `--yolo-checkpoint PATH` still works and selects YOLO.
- New `sam3_prompt_light` backend: SAM3 text detection driving the lean SAM3 tracker, no box detector.
- Per-frame masks are written to one `masks.zip` per camera (same PNG names inside, thousands fewer files); ma_2d and the GUI read either layout. `--png` writes loose PNG files as before.
- Identity cleanup is now the default: the mask-continuity gate drops transient teleport/duplicate stretches, seed selection is centered on long clips, and with sam2 collapse recovery re-acquires a person lost to a tracker takeover (a convergence guard keeps real full-contact overlap from being re-seeded). A whole-clip sweep unifies one person tracked under two ids. Multi-frame init anchors were removed (measured harmful on similar-looking people). The cleanup switches are config keys (`masks.mask_continuity_gate`, `masks.recover_collapsed_tracklets`).
- Track-then-match cross-camera identity (CLIP + epipolar) for every backend. The calibration-aware `auto` strategy uses it for sam3_prompt_light and for calibrated sam2/sam3 rigs; `--cross-camera-strategy` (or `masks.cross_camera_strategy`) overrides it.
- Cross-camera identity scores against every completed camera (not just the init camera), and each completed view enriches the per-person appearance bank.
- New identity repair after tracking (`masks.rig_vote`, CLI `--rig_vote`, on by default with 12 or more calibrated cameras): every camera's masks vote in a shared voxel grid and per-camera identity errors are relabeled against the rest of the rig, for 2 to 6 people. Rewrites only the mask entries (and redraws the overlays of relabeled cameras); reports in `rig_vote_report.json` and `rig_vote_summary.json`. A config without `masks.rig_vote` uses `auto`.
- Seed prescreen: on calibrated rigs, crowded views shortlist candidates per subject before tracking, so bystanders can't exhaust GPU memory.
- Re-latched tracklets (mid-sequence duplicates) are truncated at the takeover frame.
- ma_masks stops with a clear error when an NVIDIA GPU is present but CUDA fails to initialize, instead of silently running on the CPU for hours.
- A camera that fails is reported by name and the run exits non-zero; the other cameras still finish and are reused on the next run.
- On long clips, decode-ahead prefetch keeps the GPU from waiting on frame reads. Failed mask writes stop the run instead of leaving gaps, and resume caches are written atomically (a killed run no longer poisons the next one).
- Long-clip memory is bounded automatically: decoded frames offload to host RAM, and past 1500 frames the memory bank offloads and frame loading turns lazy (`sam.offload_state_to_cpu` / `sam.lazy_frame_loading`, both `auto`).
- Faster mask export: overlays are drawn only inside each mask's bounding box (about 35% shorter ma_masks runs on the measured clips, byte-identical outputs).
- `--calibration` adds epipolar calibration in `--videos_dir` / `--images_root_dir` modes, and `--undistort` falls back to the per-camera distortion in the ma_cap NPZ when no calibration file is given.
- Overlay videos (`masked_outputs.mp4`) and the collage keep every frame at the export frame rate; they were written at 25 fps and dropped about one frame in six.
- Image-folder input: frames named without zero padding (`1.jpg`, `2.jpg`, `10.jpg`) are ordered numerically in ma_cap, ma_masks and ma_vis, the same as the SAM frame loader; before, ma_cap's lexicographic order could attach masks to the wrong frame.
- Multi-person gallery-lookup fix; `sam3_prompt` uses its own SAM3 detector and no longer loads a box detector.
- `segmentation/process_sequence.py` is no longer a script; use `run_ma_masks.py`.

### Landmarks (ma_2d)

- Opt-in `--tensorrt` FP16 backend (~5x faster forward pass; `--tensorrt-fp32` ~2x). Mask (chained) mode only; falls back to PyTorch when unavailable.
- Crop preprocessing runs on the GPU and frame/mask reads are prefetched: ~40-50% less ma_2d time at unchanged accuracy.
- `--disable-visualizations` turns off the debug viz (still on by default); `--save_cam_output` / `--no-save_cam_output` keep working.
- Reads the packed `masks.zip` as well as loose mask PNGs.
- Per-body debug videos get readable banner labels on every tile.
- Inference no longer records an autograd graph for the backbone: lower GPU memory per crop, identical outputs.

### Optimization (ma_3d)

- Re-ID phantom fixes: unsupported views gated out, stale duplicates cleared, untriangulatable bodies excluded, association conflicts quarantined. Re-ID debug videos now label each subject.
- Opt-in `--tri-irls`: robust triangulation of the 3D seed.
- Opt-in `--tri-outlier-gate`: bridges isolated frames where the triangulated body teleports (bad masks or re-ID).
- Opt-in `--occlusion-aware-weights` (alias `--stricter-weights`): fixes implausible fits for body parts hidden in most views.
- Opt-in `--use-vposer` pose prior (install `requirements/requirements-vposer.txt`); a bundled config recipe for heavily occluded captures.
- `--use-gt` evaluation mode; opt-in `--tf32` for ~2.5x faster fits at slightly lower accuracy.
- Each camera's 2D predictions are paired with its calibration by camera name, not by sorted position.
- The number of bodies to fit is the largest over all cameras' 2D files (it was the first camera's), so a person missing from the first view is no longer skipped.
- A corrupt or incomplete 2D prediction file stops the fit with an error instead of silently dropping that camera.
- Contact visualization in the Rerun `.rrd`; leaner output npz; browser-friendly debug videos.

### Visualization (ma_vis)

- `--undistort` (ma_masks, ma_2d, ma_vis) now handles radtan / opencv_brown lenses (was a silent no-op).
- H.264 Rerun video backdrop by default.
- All-view overlays via the `all` sentinel.
- `--fps` defaults to the capture's recorded frame rate (was 30).
- Overlay rendering runs in parallel by default (`--overlay-num-workers` automatic: min(cameras, 4), with a serial retry on failure).
- Overlays render on containers and headless hosts where the default EGL device fails.
- `scene.rrd` opens framed on the people, with a fresh viewer layout per sequence.
- Ground sizing no longer copies every mesh vertex, which saves gigabytes of transient memory on long multi-person sequences.

### SMPL-X export (new)

- Export fits to SMPL-X Blender add-on npz, or rigged FBX / ABC / BVH / USD through a portable Blender 4.5 LTS (`data/download_blender.sh` or the Exporter tab; needs the SMPL-X Blender add-on). See `docs/EXPORT.md`.
- Auto up-axis and floor detection with one shared floor for all bodies (`--no-ground` keeps the raw translation). Meters by default; `--unit cm` on the CLI (the GUI exports meters).
- A failing format no longer discards the others, and exports are checked against the fitted vertices (`--no-validate` skips the check).
- ABC files carry plain geometry (no face sets that pick up a scene material); USD files bake the body shape into the mesh so importers do not clamp it.

### Calibration & world orientation

- New OpenCV FileStorage calibration loader (a YAML file or a directory of per-camera files); MAMMA YAML accepts `extrinsics_convention` and `quaternion_order`; the world up-axis is detected from the camera rig.
- Signed world up-axis threaded through ma_3d and ma_vis; set `up_axis` in the capture JSON to override the auto-detect.
- Per-camera NPZ carries every lens-distortion model, so `--undistort` works straight from `--ma_cap_dir`.
- Image frames on network storage: transient read failures are retried, and a corrupt frame's error names the file.

### Runner & remote cluster

- Remote HTCondor submission (new) from the CLI (`python -m inference submit-remote`, `submit-cluster`; `python -m inference cluster` manages cluster profiles) and from the GUI (cluster profiles, live status on the Tasks board).
- Cluster camera fan-out: `ma_masks.fanout_cameras` and `ma_2d.fanout_cameras` run one job per camera.
- Cluster runs resync their Tasks rows after a GUI backend restart (startup pass and a per-task re-sync button).
- `resources.default.exclude_machines` in a cluster profile keeps jobs off listed hosts.
- Multi-sequence runs: a failed sequence cancels only its own downstream steps; the other sequences keep running (CLI and GUI).
- A `--cam_names` subset in a step's `flags` now overrides the capture's camera list.
- `--force` removes the step's DONE marker before re-running, so a failed forced run no longer leaves the previous run marked done.
- A cancelled run exits with code 130 and no longer logs success; the cancellation does not carry over to the next run in the same process.
- `--footage` with per-camera image folders creates an image capture (it was treated as a video capture and failed).
- The cluster Tasks board reopens a step when DAGMan retries it, instead of keeping the failed state.

### Presets & tools

- `quick` now covers frames 60-120 (was 60-90). Speed options stay opt-in per-step flags (`--tensorrt`, `--tf32`); ma_vis overlay rendering parallelizes automatically.
- Developer tools: `scripts/regression_check.py` (output-drift and timing gate against a golden run) and `scripts/benchmark.py` (wall time and peak GPU / CPU memory per step).
- `scripts/eval_id_continuity.py` scores ma_masks identities against ground-truth silhouettes, per camera.

### GUI

- New SMPL-X Exporter tab, shared export panel, batch export.
- Interactive in-browser 3D viewer for ma_3d results.
- Redesigned Results page: logs, configs, outputs explorer.
- Re-run a task from Results: pick the steps to recompute; unchanged steps reuse the source run's outputs.
- Task table: per-step timings (Status/Timing toggle), frames/views columns, per-sequence delete (honored by Restart), task-level Stop/Restart on grouped rows.
- New Task form: preset editor with common settings and command preview, frame-range editing.
- Calibration status, camera-rig 3D preview, up-axis controls; capture forms show a reference for each supported calibration format and gain a Camera FPS field.
- New Task form: ma_masks gains "Identity repair (rig vote)" (Auto, On, Off) and "Number of people".
- New ma_3d "Heavy occlusion-aware" toggle turns on `--occlusion-aware-weights`, `--tri-irls` and `--tri-outlier-gate` together.
- Faster page loads (cache-first thumbnails) and UI polish.
- Fixes: Tasks-to-Results opens the clicked sequence; captures created from per-camera videos now run (ma_cap failed with "camera dir not found"); deleting a queued task no longer still runs it; failed downloads and exports are no longer reported as success; the step panel's Stop button now stops the task (it only relabelled the step while the work kept running).
- Security: the API refuses requests from other web origins (see Migration), and output HTML/SVG files open in a sandbox, also in a new tab.

### Containers

- One Docker/Apptainer image for all five steps (`docker/`), used through the existing per-step `engine: docker|apptainer` keys; `engine`, `sif_path` and `docker_image` can now be set once in `global`. See `docker/README.md`.
- Containerized steps mirror the conda layout: repo root bound at `/repo`, docker runs as your user (no root-owned outputs), apptainer uses `--nvccli` when `nvidia-container-cli` is available, and a relative `sif_path` resolves against the repo root. Standalone ma_masks image: no `python` entrypoint; runs the bind-mounted checkout.

### Datasets & assets

- MammaSyn downloads from Hugging Face (gated) by default; the GUI download through the MAMMA account now fetches the `.tar` shards, not only the manifests.
- Optional VPoser weights download via CLI or GUI.
- The download scripts verify the server's TLS certificate (they passed `--no-check-certificate`).

### Documentation

- New guides: SMPL-X export (`docs/EXPORT.md`), containers (`docker/README.md`), calibration formats (`capture/loaders/README.md`); `docs/INSTALL.md` covers Docker/Apptainer, TensorRT and VPoser.
