# [ma_masks]: Multi-view Multi-people Segmentation Masks for MAMMA

**Part of [MAMMA: Markerless Accurate Multi-person Motion Acquisition (CVPR 2026)](https://mamma.is.tue.mpg.de/)**.

This module is the segmentation step: it exports per-person binary masks that are used downstream by the MammaNet model.

![collage_masked_outputs.png](assets/collage_masked_outputs.png)

## How It Works

The pipeline processes one camera at a time. The **init camera** (specified via `--cam_init`, or the first available
camera by default) is used to define the people that we want to track. On each **subsequent camera**, it detects people
and matches them to the init camera's IDs.

**Four detection modes are available:**

| Mode                              | Description                                                          | Box detector? |
|-----------------------------------|---------------------------------------------------------------------|:-------------:|
| `--sam_version sam2`              | D-FINE detects people → SAM2 tracks from single-frame box prompts    |  Yes  |
| `--sam_version sam3`              | D-FINE detects people → SAM3 tracks (single bbox prompt)             |  Yes  |
| `--sam_version sam3_prompt`       | SAM3 text prompt `"person"` detects **and** tracks, all in SAM3's full video predictor |  No   |
| `--sam_version sam3_prompt_light` | Same text prompt `"person"`, but detection runs on SAM3's detector and tracking on the lean SAM3 tracker, lighter than `sam3_prompt` (see Performance below) |  No   |

The `sam2` / `sam3` person detector is D-FINE-X (Apache-2.0), downloaded from Hugging Face on first use.
`--detector yolo --detector-checkpoint yolo12x.pt` uses Ultralytics YOLO instead (AGPL-3.0, `pip install ultralytics`),
and `--detector my_pkg.my_module:MyDetector` loads your own class. It is built as `MyDetector(checkpoint, device)`
and needs one method, `detect(image, conf)`, returning `[(xyxy, score), ...]` (see `core/person_detector.py`).
Config thresholds such as `matching.yolo_person_conf` stay on the YOLO12x confidence scale; each detector maps them.

Alternatively, **instead of automatically** defining the people to be tracked, the `--interactive` opens a GUI on the
init camera where you can manually click on the people (keys 0-9 to switch person, left-click to mark, right-click to
remove). This replaces auto-detection on the init camera only - subsequent cameras are still matched automatically.
Works with `sam2`, `sam3` and `sam3_prompt_light` (`sam3_prompt` always initializes from its text prompt). Requires a
display.

In `sam3_prompt` mode, the pipeline first samples a small set of candidate frames, selects a prompt-frame using SAM3
mask-area-based scoring, then propagates from that frame.

**Cross-camera ID matching** uses two signals:

- **CLIP appearance**: encodes person crops with OpenCLIP (ViT-B-32) and matches them against a per-person appearance
  gallery / feature bank
- **Epipolar geometry**: uses the calibration between the current camera and the reference cameras (the init camera,
  plus every camera already processed in the same run) to score geometric consistency

The **Hungarian algorithm** makes the final one-to-one ID assignment across cameras - no duplicate IDs.

**Strategy: track-then-match by default.** Each non-init camera is first
propagated *independently* into clean local tracklets; whole tracklets are then
remapped to the init camera's ids using the CLIP + epipolar signals above
(multi-frame, occlusion-weighted). This replaces the legacy behavior of committing
identity at seed time and lets the remap drop bystanders/outsiders. Control it with
`--cross-camera-strategy {auto,track_then_match,match_then_track}` or
`masks.cross_camera_strategy`. The default `auto` uses track-then-match for
`sam3_prompt_light`, and for `sam2` / `sam3` whenever epipolar calibration is
available; without calibration, `sam2` / `sam3` fall back to match-then-track.
`sam3_prompt` always tracks-then-matches.

**Long-clip memory** is bounded automatically: decoded frames stay on the host
(`sam.offload_video_to_cpu`), the per-frame memory bank offloads past 1500 frames
(`sam.offload_state_to_cpu: auto`), and frame loading turns lazy past the same
length (`sam.lazy_frame_loading: auto`; image input is byte-identical, mp4 input
is streamed through OpenCV and is near-identical, at some extra wall time). All
knobs live in the `sam:` section of the config YAML, including the opt-in
`prune_memory_state` for extreme clips.

**Identity cleanup.** After propagation, each camera's tracklets are cleaned:
near-identical duplicates are merged, a tracklet that re-latches onto another
person is cut at the takeover frame, tracklets that stay tiny are dropped, and the
mask-continuity gate removes short teleports and duplicate stretches. Except for
`sam3_prompt`, one person tracked under two ids is unified and single-frame mask
islands are dropped. With `configs/sam2.yaml`, collapse recovery also re-detects a
person lost to a tracker takeover and re-tracks only that stretch. After all
cameras finish, the [rig vote](#identity-repair-rig-vote) repairs remaining
per-camera identity errors on calibrated rigs of 12 or more cameras.

> **Note**: Identity errors these steps miss, for example a long swap under severe occlusion on a small rig, are
> not corrected in ma_masks. In practice, their impact is often mitigated in the downstream **multi-view SMPL-X
> fitting** stage, where cross-view association is constrained by epipolar geometry and supported by the majority of
> views with correct masks.

## Added after the CVPR code release

The CVPR code release (v1.0.x) shipped the core pipeline described above. The features below were added since; they
make the step more robust and cover more use cases. **Default** = always on, **auto** = on when the condition holds,
**opt-in** = off unless enabled. Config keys that the shipped YAMLs leave out can still be set in the same section.

| Feature | What it does | Status | Control |
|---|---|---|---|
| D-FINE-X person detector | Apache-2.0 box detector for `sam2` / `sam3`, replacing YOLO12x; YOLO or your own class can be plugged in | default | `--detector`, `--detector-checkpoint`, `matching.detector` |
| `sam3_prompt_light` backend | SAM3 text detection with the lean SAM3 tracker; no box detector | opt-in | `--sam_version sam3_prompt_light` |
| Track-then-match identity | Tracks each camera on its own, then remaps whole tracklets by CLIP + epipolar | auto (calibrated `sam2` / `sam3`; always for `sam3_prompt_light`) | `--cross-camera-strategy`, `masks.cross_camera_strategy` |
| Multi-reference remap | Epipolar references include every camera already processed; matched tracklets enrich the appearance bank | default under track-then-match | none |
| Seed everyone | Non-init cameras seed every detected person and let the remap drop the extras | default under track-then-match | `masks.max_seed_tracklets` (ceiling) |
| Seed prescreen | In crowded views, keeps the best candidates per subject before tracking | auto (more detections than 3 per subject, calibrated) | `masks.seed_prescreen` |
| Centered seed frame | Prefers seed and prompt frames away from the clip ends | auto (clips over 1000 frames) | `matching.seed_center_bias` |
| Mask-continuity gate | Removes short teleports and duplicate mask stretches | default | `masks.mask_continuity_gate` |
| Collapse recovery | Re-detects a person lost to a tracker takeover and re-tracks that stretch | default in `sam2.yaml`, off in `sam3.yaml` | `masks.recover_collapsed_tracklets` |
| Re-latch truncation | Cuts a tracklet that jumps onto another person at the takeover frame | default | `masks.truncate_relatched_tracklets` |
| Split-body sweep, island drop | Unifies one person tracked under two ids; drops single-frame mask islands | default (not `sam3_prompt`) | none |
| Re-acquisition | Re-tracks stretches where the cleanup left a person without a mask, from a fresh tracker start where the detector sees someone no mask covers | default (not `sam3_prompt`) | `masks.reacquire_lost_people` |
| Rig vote | Repairs per-camera identity errors by voting across the calibrated rig | auto (12 or more calibrated cameras, 2 to 6 people) | `--rig_vote`, `masks.rig_vote`, `masks.rig_vote_min_cameras` |
| Packed masks | One `masks.zip` per camera instead of thousands of PNG files | default | `--png` for loose PNGs |
| Lazy frame loading | Loads frames on demand, with decode-ahead prefetch, to bound host RAM | auto (1500 frames or more) | `sam.lazy_frame_loading` |
| Memory-state pruning | Bounds SAM2 memory on very long clips | opt-in | `sam.prune_memory_state` |
| Calibration for video / image input | `--calibration` adds epipolar geometry when frames come from videos or image folders; `--undistort` falls back to the NPZ distortion | opt-in | `--calibration`, `--undistort` |
| Camera fan-out | One cluster job per camera; the rig vote and collage run once at the end | opt-in (cluster runs) | `ma_masks.fanout_cameras` in the run config |
| Failure handling | A failing camera is reported and the run exits non-zero; resume caches are written atomically; a broken GPU stops the run instead of falling back to the CPU | default | none |

## Performance & which mode to choose

Measured on one 6-person, 4-camera, 150-frame clip (per camera, 24 GB GPU), scored
against ground-truth masks. Numbers are indicative, not a formal benchmark.

| Mode                | Peak VRAM | Long sequences? | Box detector |
|---------------------|:---------:|:---------------:|:----:|
| `sam2`              | **~4.4 GB** | ✅ best (CPU-offload bounded) | Yes |
| `sam3`              | ~8 GB     | ✅ good (CPU-offload bounded) | Yes |
| `sam3_prompt`       | ~11.5 GB  | ⚠️ limited (~700 frames, then OOM) | No |
| `sam3_prompt_light` | ~8.8 GB   | ✅ good (handles full clips) | No |

On the clip tested, all modes recovered every ground-truth person with comparable
mask quality (mean IoU ~0.96–0.97); the small accuracy differences were within
run-to-run noise, so we don't rank modes by accuracy. The clear, repeatable
differences are **VRAM** and **how long a clip each can process**:

- **Lowest VRAM / longest sequences → `sam2`** (then `sam3`). Both keep decoded
  frames and tracker state on the host via CPU offload, so per-camera VRAM stays
  bounded and clip length is limited by host RAM rather than VRAM (`sam2` ran 1200+
  frames in testing).
- **Avoid `sam3_prompt` for long clips**: its session cache grows with frame count
  and OOMs around ~700 frames on a 24 GB GPU. Use `sam3_prompt_light` instead, which
  handles full-length clips at lower VRAM.
- **No box detector needed → `sam3_prompt` / `sam3_prompt_light`** (they detect via SAM3
  text).

Per-camera memory is driven mainly by sequence length. `sam2`, `sam3` and
`sam3_prompt_light` bound VRAM via CPU offload (`sam.offload_video_to_cpu`,
`offload_state_to_cpu`); `sam3_prompt` (session API) is capped by an
internal cache prune (`sam.sam3_prune_cache_window`).

## Installation

Installation is managed by the parent `mamma_release` repo. See [`docs/INSTALL.md`](../docs/INSTALL.md) for the full env + CUDA + weights setup (including SAM 3's gated Hugging Face access and the optional Apptainer/Docker container recipes shipped in this folder); activate `mamma` before running `run_ma_masks.py`. Default weight locations (used both by the runner and by standalone invocations):

- Person detector (D-FINE-X): downloaded on first use to the local Hugging Face cache
  (`--detector-checkpoint` points to other weights).
- SAM 2: `data/weights/sam2/sam2.1_hiera_large.pt`
- SAM 3: downloaded on first use to the local Hugging Face cache (e.g.
  `~/.cache/huggingface/.../models--facebook--sam3`), not under `data/weights/`. With
  `sam3_prompt`, `--sam_checkpoint <snapshot>/sam3.pt` uses a specific local file (see
  below); `sam3` and `sam3_prompt_light` always load from the Hugging Face cache.

SAM 3 weights come from a **gated** Hugging Face model, so the SAM3 backends
(`sam3`, `sam3_prompt`, `sam3_prompt_light`) need an approved account and a local
`hf auth login`. Inside a container that can't see the host's HF cache,
`sam3_prompt` can bypass cache discovery with an explicit checkpoint file:

```bash
python segmentation/run_ma_masks.py \
  --ma_cap_dir /path/to/dataset --seq_name my_sequence \
  --out output/ --sam_version sam3_prompt --cfg segmentation/configs/sam3.yaml \
  --sam_checkpoint .hf_cache/hf_cache/hub/models--facebook--sam3/snapshots/<SNAPSHOT_ID>/sam3.pt
```

## Quick Start

> **Run from the repository root.** The bare `--cfg` default (`configs/...`) only
> resolves when launched from inside `segmentation/`, so pass the config
> explicitly: add `--cfg segmentation/configs/sam2.yaml` for SAM2 runs, or
> `--cfg segmentation/configs/sam3.yaml` for `sam3` / `sam3_prompt` /
> `sam3_prompt_light`.

### Basic (NPZ input, SAM2)

```bash
python segmentation/run_ma_masks.py \
  --ma_cap_dir /path/to/dataset --seq_name my_sequence \
  --out output/ --cam_init IOI_09
```

### SAM3 text prompt (no box detector needed)

```bash
python segmentation/run_ma_masks.py \
  --ma_cap_dir /path/to/dataset --seq_name my_sequence \
  --out output/ --sam_version sam3_prompt --cam_init IOI_09
```

### From image directories

```bash
# Expected: images_root_dir/<cam_name>/<frame>.jpg
python segmentation/run_ma_masks.py \
  --ma_cap_dir /path/to/dataset --seq_name my_sequence \
  --out output/ --images_root_dir /path/to/frames/ --cam_init cam_01
```

### From MP4 videos

```bash
python segmentation/run_ma_masks.py \
  --ma_cap_dir /path/to/dataset --seq_name my_sequence \
  --out output/ --videos_dir /path/to/videos/ --cam_init camera_01
```

When both `--videos_dir` and `--ma_cap_dir` are provided, frames are read from videos (faster) and camera calibration (
K, R, T) is loaded from NPZ files (for epipolar geometry).

### Frame range

```bash
python segmentation/run_ma_masks.py \
  --ma_cap_dir /path/to/data --seq_name my_sequence \
  --out output/ --start 10 --end 60
```

### Subset of cameras

```bash
python segmentation/run_ma_masks.py \
  --ma_cap_dir /path/to/dataset --seq_name my_sequence \
  --out output/ --cam_names IOI_01 IOI_02 IOI_09
```

By default, all cameras in the sequence are processed. Use `--cam_names` to
restrict processing to a specific list (names must match the NPZ / MP4 / image
directory names).

### Interactive (click on people via GUI)

```bash
python segmentation/run_ma_masks.py \
  --ma_cap_dir /path/to/dataset --seq_name my_sequence \
  --out output/ --cam_init IOI_09 --interactive
```

Keys **0-9** switch person ID, **left-click** marks a person, close window when done. Only the init camera uses the GUI;
other cameras are matched automatically. Requires a display.

### Skip overlay visualization

```bash
python segmentation/run_ma_masks.py \
  --ma_cap_dir /path/to/dataset --seq_name my_sequence \
  --out output/ --sam_version sam3_prompt --skip_masked_outputs
```

Saves time and disk by skipping overlay images and MP4. Masks and `masks.npy` are still produced.

## Configuration

Config files in `configs/` control all parameters. Each argument is documented inline.

| File (pass via `--cfg`, from repo root)  | --sam_version                                  |
|------------------------------------------|------------------------------------------------|
| `segmentation/configs/sam2.yaml`         | For `sam2` (default for it)                    |
| `segmentation/configs/sam3.yaml`         | For `sam3` + `sam3_prompt` + `sam3_prompt_light` (default for them) |
| `segmentation/configs/sam3_default.yaml` | Same backends, SAM3's unmodified tracking defaults (experimental) |

Key options:

```yaml
scene:
  expected_subjects: null        # null = auto-detect, or set fixed N

matching:
  start_frame: 10                # sam3_prompt prompt search / legacy matching skip early frames
  sample_interval: 50            # legacy match_then_track: sample every Nth frame
  detector: null                 # null or dfine (D-FINE-X), yolo, or package.module:Class
  yolo_person_conf: 0.5          # person confidence, YOLO12x scale (lower = catch more)
  min_mask_area_ratio: 0.01      # small-mask threshold (sam3_prompt prompt frame, remap crop weights)
  sam3_redetect_samples: 30      # max candidate prompt frames (sam3_prompt)
  clip_weight: 0.4               # appearance matching weight
  epipolar_weight: 0.6           # geometric matching weight

masks:
  merge_duplicate_tracklets: true  # merge near-identical tracklets (same person tracked twice)
  merge_iou_threshold: 0.95        # mask IoU threshold for merging (very conservative)
  mask_continuity_gate: true       # drop short teleport / duplicate mask stretches
  recover_collapsed_tracklets: true  # re-seed a person lost to a takeover (false in sam3.yaml)
  reacquire_lost_people: true      # re-track people the cleanup left maskless (not sam3_prompt)
  discard_tiny_tracklets: true     # remove tracklets that are too small on most frames
  tiny_tracklet_min_area_ratio: 0.005  # min mask area as fraction of image
  tiny_tracklet_min_frame_ratio: 0.2   # discard if small on >80% of frames
  cross_camera_strategy: auto      # auto | track_then_match | match_then_track
  rig_vote: auto                   # identity repair: auto | true | false
  rig_vote_min_cameras: 12         # calibrated cameras needed for auto
  max_seed_tracklets: 20           # ceiling on people seeded per non-init camera

sam:
  propagate_reverse: true        # propagate backward from init frame

# sam3_prompt only: override SAM3 internal tracking parameters
sam3_tracking:
  new_det_thresh: 0.85           # higher = fewer spurious new tracklets
  trk_assoc_iou_thresh: 0.25     # lower = stickier existing tracks
  init_trk_keep_alive: 60        # frames a track survives without re-matching
  max_trk_keep_alive: 60
```

### Identity repair (rig vote)

After all cameras are segmented, a rig vote repairs identity errors across the
calibrated rig: every camera's masks label a voxel grid of the capture volume,
each camera is checked against the vote of the other cameras, and per-camera
errors are relabeled, from a camera mislabeled for the whole clip to a swap
after two people touch. It relabels the mask entries only (inside `masks.zip`,
or the loose PNGs) and writes `rig_vote_report.json` per relabeled camera plus
`rig_vote_summary.json`. The overlays of relabeled cameras, and so the collage,
are redrawn from the repaired masks; `masks.npy` keeps the tracked ids. ma_2d
reads the repaired masks.

| Key under `masks:` | CLI override | Default |
|---|---|---|
| `rig_vote` | `--rig_vote auto/true/false` | `auto`: on with at least `rig_vote_min_cameras` calibrated cameras |
| `rig_vote_min_cameras` | none | `12` |

It handles 2 to 6 people and needs at least 5 cameras with masks; with fewer
cameras the voxel vote is too ambiguous to trust, and the tracking stands as
is. `true` runs it on any calibrated rig that meets those limits. On a camera
fan-out it runs once, in the finalize cell.

### Overriding from the CLI

A few common settings can be passed directly on the command line without
editing the YAML:

- `--cfg segmentation/configs/sam3_default.yaml`: use a different config file
  (e.g. SAM3 with its unmodified tracking defaults instead of the tuned values
  in `segmentation/configs/sam3.yaml`).
- `--expected_subjects N`: tell the pipeline to expect exactly N people in
  the scene. Otherwise N is auto-inferred from init-camera detections.
- `--detector`, `--detector-checkpoint`: person detector and its weights
  (`matching.detector`, `matching.detector_checkpoint`).
- `--cross-camera-strategy auto|track_then_match|match_then_track`: overrides
  `masks.cross_camera_strategy`.
- `--rig_vote auto|true|false`: overrides `masks.rig_vote`.
- `--png`: write loose PNG masks instead of `masks.zip` (`masks.export_png`).

## Output

```
<out>/<dataset_name>/<seq_name>/   # dataset_name defaults to basename(--ma_cap_dir)
    run.log                      # full log file for this run
    config.yaml                  # the config YAML plus CLI overrides used
    subject_feature_bank.npy     # cached CLIP feature bank
    initialize/                  # init-camera subject initialization visuals
    videos/collage.mp4           # all cameras' overlays side by side
    rig_vote_summary.json        # identity repair summary (when the rig vote runs)
    cross_camera_summary/        # per-person cross-camera images (--debug_crop_summary)
    <camera>/
        masks/masks.zip          # per-frame per-person masks (loose PNGs with --png)
        masks.npy                # matching / resume cache (completion marker)
        masked_outputs/          # colored overlay video (MP4)
        anchor_visualizations/   # seed-frame prompt visuals
        anchor_report.json       # anchor prompts (legacy match_then_track only)
        pre_remap/               # raw SAM3 IDs before remapping (sam3_prompt)
        rig_vote_report.json     # identity relabels applied to this camera
        person_XX_crop_summary.png  # per-person crop samples (--debug_crop_summary)
```

### Mask layout

Each camera's masks are one `masks.zip` in `<camera>/masks/`: a ZIP archive
stored without compression, whose entries are grayscale PNGs named
`mask_{frame:04d}_{person:02d}.png` (person ids start at 1; 255 = person,
0 = background), the same names the loose files have. `--png` (or
`masks.export_png: true`) writes those PNGs as loose files in `masks/` instead,
and removes a stale `masks.zip` there. ma_2d reads either layout and prefers
`masks.zip` when both exist; the GUI file browser opens `masks.zip` like a
folder. In your own code, use
`MaskArchiveReader` from `core/mask_archive.py` (ma_2d carries a copy in
`landmarks/utils/mask_archive.py`), or any ZIP reader.

## Project Structure

```
run_ma_masks.py              # CLI entry point
process_sequence.py          # sequence driver (process_seq), called by run_ma_masks.py
core/
    pipeline.py              # main segmentation engine
    predictor_factory.py     # SAM2/SAM3 predictor builders + adapters
    person_detector.py       # person box detectors (D-FINE-X, YOLO, custom)
    mask_archive.py          # masks.zip writer / reader
    mask_store.py            # disk-backed mask store used during tracking
    geometry.py              # epipolar geometry helpers
    exceptions.py            # per-camera failure type
    frame_source.py          # unified frame access (images / MP4)
    video_reader.py          # in-memory MP4 reader
    logging.py               # loguru config (INFO→stdout, WARN+→stderr)
    rig_vote.py              # rig-vote identity repair (post-process)
configs/
    sam2.yaml                # SAM2 config (documented)
    sam3.yaml                # SAM3 config: tuned tracking defaults
    sam3_default.yaml        # SAM3 config: unmodified upstream tracking defaults
utils/                       # drawing, paths, GUI, math, visualization helpers
scripts/                     # standalone helper tools (e.g. check_ready.py)
tests/                       # unit tests
```

## Tests

```bash
pytest tests/ -v
```

Citation, license, and acknowledgments live in the [top-level README](../README.md).
