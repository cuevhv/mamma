# Configs

## What's a preset?

A **preset** is a YAML file that tells MAMMA *which* pipeline steps to run and *with what* hyperparameters. It's **capture-independent** — one preset can drive runs against many different captures.

Two are shipped under [`configs/examples/presets/`](../configs/examples/presets/):

- [`quick.yaml`](../configs/examples/presets/quick.yaml) — restricted to 30 frames, ~5 min on one GPU. The smoke fixture.
- [`full.yaml`](../configs/examples/presets/full.yaml) — full-frame, no slicing.

Shape (excerpted from `quick.yaml`):

```yaml
global:
  out_dir: output
  conda_env: mamma
  start_frame: 60
  end_frame: 90
ma_cap:
  enabled: true
  engine: conda
  script: run_ma_cap.py
  repo_path: capture
  flags: []
ma_masks:
  enabled: true
  engine: conda
  dependencies: [ma_cap]
  script: run_ma_masks.py
  repo_path: segmentation
  flags: [--sam_version sam2]
# ... ma_2d, ma_3d, ma_vis
```

Each top-level key controls one part of the run:

| Section    | What it controls                                                                 | Common flags · source                                                                                          |
|------------|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `global`   | Run-wide settings: output dir, conda env, frame range, log dir, default engine.  | [Schema reference → `global`](#schema-reference)                                                                |
| `ma_cap`   | Loads the multi-view capture (videos + calibration → per-camera NPZs).           | [common ↓](#ma_cap) · [`capture/run_ma_cap.py`](../capture/run_ma_cap.py)                   |
| `ma_masks` | Per-person segmentation (SAM + YOLO).                                            | [common ↓](#ma_masks) · [`segmentation/run_ma_masks.py`](../segmentation/run_ma_masks.py) |
| `ma_2d`    | 2D landmark detection (MammaNet).                                                | [common ↓](#ma_2d) · [`landmarks/run_ma_2d.py`](../landmarks/run_ma_2d.py)               |
| `ma_3d`    | Multi-view SMPL-X optimization (the actual body fit).                            | [common ↓](#ma_3d) · [`optimization/run_ma_3d.py`](../optimization/run_ma_3d.py)      |
| `ma_vis`   | Per-camera overlays + interactive Rerun scene.                                   | [common ↓](#ma_vis) · [`visualization/cli.py`](../visualization/cli.py)                    |

The per-step `flags:` list in a preset takes any of the argv switches documented in the linked source files (argparse `help=` strings). Each step block also declares its `engine`, `dependencies`, and where the script lives — see [Per-step](#per-step-ma_cap-ma_masks-ma_2d-ma_3d-ma_vis) in the Schema reference for those structural fields.

## Modify a shipped preset

Copy one and edit:

```bash
cp configs/examples/presets/quick.yaml my_preset.yaml
# edit `flags:` to tweak per-step hyperparameters, or set `enabled: false` to skip a step
python -m inference run \
  --cfg my_preset.yaml \
  --footage <dir> --seq_name <name> --calib <file>.yaml \
  --out-tag run01 -v
```

Common edits:

- **Skip a step:** set `<step>.enabled: false`. Downstream steps either skip too (if they depended on the disabled one) or carry on (if they have another path to their inputs).
- **Tune a step:** add `<step>.flags: ["--<flag> <value>"]`. The string is `shlex`-split, so `"--sam_version sam3"` becomes two argv tokens.
- **Limit frames:** set `global.start_frame` / `global.end_frame`. Only `ma_cap` reads these; downstream steps pick up the range via the per-camera NPZ.
- **Change conda env:** `global.conda_env: my_other_env`.

### Common per-step flags

A hand-curated subset — the most-tuned flags per step. The complete list lives in each entry script's argparse `help=` strings; run `python <script> --help` (or open the linked source) for the full surface.

#### `ma_cap`

Source: [`capture/run_ma_cap.py`](../capture/run_ma_cap.py).

- `--start N` / `--end N` — frame-range slice (per-camera). Usually set via `global.start_frame`/`end_frame` instead.
- `--fps N` — override the FPS recorded in `global.npz`. Defaults to the capture's `cam_fps`.
- `-v` / `-vv` — INFO / DEBUG logging.

#### `ma_masks`

Source: [`segmentation/run_ma_masks.py`](../segmentation/run_ma_masks.py).

- `--sam_version sam2|sam3|sam3_prompt` — SAM backend; quick.yaml ships `sam2`.
- `--expected_subjects N` — force the person count (auto-detected when unset).
- `--init_frame N` — frame index used for person-detection initialisation.
- `--interactive` — click-to-init through a GUI instead of YOLO auto-detect.
- `--undistort` — apply Vicon-radial-2 undistortion before segmentation.

#### `ma_2d`

Source: [`landmarks/run_ma_2d.py`](../landmarks/run_ma_2d.py).

- `--tensorrt` — compile the landmark network to a TensorRT engine for a faster
  forward (~5× FP16). **Opt-in, NVIDIA-only**; falls back to plain PyTorch when
  `torch-tensorrt` is unavailable, so configs stay portable. The compiled engine is
  cached to disk (keyed by weights/shape/precision/GPU), so after the first run it
  loads in ~2 s — it pays off most on long / many-camera sequences. Needs the
  optional dep in [`requirements/requirements-tensorrt.txt`](../requirements/requirements-tensorrt.txt).
- `--tensorrt-fp32` — with `--tensorrt`, use the near-exact FP32 engine (~2×) instead
  of FP16 (~5×).
- `--save_cam_output` — write per-body debug viz frames + a preview video. **Off by
  default** (these artifacts aren't consumed downstream); pass it to enable inspection.
  `--no-save_cam_output` is the explicit off.
- `--video_fps F` — FPS for generated viz videos (default 5).
- `--undistort` — undistort frames before landmark inference.

#### `ma_3d`

Source: [`optimization/run_ma_3d.py`](../optimization/run_ma_3d.py).

- `--config_file <path>` — alternate optimizer YAML (default
  `config_files/config.yaml`).
- `--ignore_start_frames N` — exclude the first N frames from the optimization
  loss (still rendered in the output).
- `--skip_detection_analysis` — skip post-run 2D-detection plots + CSVs.
- `--detection_analysis_top_k K` — top-K least-confident frames to report
  (default 30).
- `--use-gt` — evaluation only: also compute MPJPE/PVE/PA-MPJPE against the
  ground-truth SMPL-X poses in `<ma_cap_dir>/<seq>/gt/global.npz` (e.g. the
  `mamma_eval_dance` sequences). **Off by default** — normal in-the-wild captures
  have no GT.

#### `ma_vis`

Source: [`visualization/cli.py`](../visualization/cli.py).

- `--overlay-num-workers N` — parallelize per-camera overlay rendering (measured
  ~2× at `N=4` on 6 cameras). Default `1` (serial) to stay safe where concurrent
  offscreen pyrender/EGL is finicky; the example presets ship `4`.
- `--up-axis x|y|z` — world up axis (default `z`).
- `--fps N` — Rerun timeline + overlay video FPS (default 30).
- `--cam-names-overlay <list>` — restrict overlay rendering to a camera subset.
- `--rerun-light` — skip the heavy Rerun scene; only emit overlays.

For the full set of editable fields, see [Schema reference](#schema-reference) below.

## Preset vs run config

| Role              | What                                                          | Capture-bound? | Format          |
|-------------------|---------------------------------------------------------------|----------------|-----------------|
| **Preset**        | Reusable template: which steps run + per-step hyperparameters.| **No**         | YAML (shipped) or JSON (GUI-saved) |
| **Run config**    | Concrete capture-bound execution snapshot (preset + capture). | **Yes**        | JSON (always — mechanical write)   |

A **run** is the executed work. A **run config** is the frozen file that captures everything needed to reproduce it.

The pipeline runner takes one path either way:

```
python -m inference run --preset <preset>.yaml --capture <capture>.json --out-tag run01 -v
# or, for an already-bound run config (e.g. one saved by the GUI):
python -m inference run --task gui/var/interface/run_configs/run_42.json
```

The CLI rejects mixing `--task` with `--preset/--capture`. `--preset` requires `--capture` and vice versa. Internally, `--preset+--capture` is merged in memory by `inference.config.materialize_run_config()` and written to a temp file before dispatch — the same code path the GUI's submit flow uses.

Shipped presets ([`configs/examples/presets/`](../configs/examples/presets/)) are YAML; everything the GUI writes (saved presets, frozen run configs) is JSON. Both loaders (`inference/config.py:load_run_config`, `gui/backend/config_io.py:load_config_file`) accept either format.

## Canonical smoke fixture (Breakdance quick)

[`examples/presets/quick.yaml`](../configs/examples/presets/quick.yaml)
+ [`examples/captures/140725_Breakdance.json`](../configs/examples/captures/140725_Breakdance.json)
— drives all five steps (`ma_cap → ma_masks → ma_2d → ma_3d → ma_vis`)
end-to-end against the Breakdance capture, restricted to 30 frames
(60–90) so the full pipeline completes in ~5 minutes:

```
python -m inference run \
  --preset configs/examples/presets/quick.yaml \
  --capture configs/examples/captures/140725_Breakdance.json \
  --out-tag smoke -v
```

`scripts/smoke_test.py` uses this same fixture for its end-to-end
pipeline-walk check and for every builder-shape test. Swap the
`--capture` argument for any other JSON under
`configs/examples/captures/` (13 shipped capture manifests — the
preset is capture-independent). Use `configs/examples/presets/full.yaml`
for full-frame runs.

The example capture manifests and calibration files ship ready to use
under `configs/examples/`. The GUI auto-discovers them — see
[`gui/README.md`](../gui/README.md) for the multi-root scan rules.
For the on-disk data layout, see [`docs/INSTALL.md`](INSTALL.md).

## Schema reference

> Skip this section unless you're authoring a preset from scratch or debugging an unfamiliar field. The [Modify a shipped preset](#modify-a-shipped-preset) recipe covers the common edits.

### `global` (required for runs; presets may omit `capture_json` and `seq_ids`)

| Field           | Type         | Required | Description |
|-----------------|--------------|----------|-------------|
| `dataset_name`  | string       | yes (run; presets omit) | Used to compose output paths — see [Output path layout](#output-path-layout). Presets omit this field; the materializer derives it from the capture filename at submit time. |
| `capture_json`  | path         | yes (run)| Capture-config JSON listing all sequences. **Presets omit this** — the materializer fills it from `--capture`. |
| `seq_ids`       | int list     | no       | Subset of sequence ids to run. Empty / omitted = all sequences in `capture_json`. Override on the CLI with `--seqs`. |
| `out_dir`       | path         | yes      | Root output directory. Override on the CLI with `--out-dir`. |
| `cam_names`     | string list  | yes (run; presets omit) | Camera names. Forwarded as `--cam_names` to most steps. Presets omit this; the materializer derives it from `capture.cams` at submit time. |
| `conda_env`     | string       | no       | Default conda env for the `conda` engine (default `mamma`). |
| `jobs_log_dir`  | path         | no       | Where per-(step, seq) `.log/.out/.err` files go. Falls back to `$MAMMA_DATA_DIR/logs` or `~/.mamma/logs`. |
| `username`      | string       | no       | Inserted into log paths so multi-user setups don't collide. Falls back to `$USER`. |
| `bind`          | string list  | no       | Extra `apptainer --bind` / `docker -v` entries. |

### Per-step (`ma_cap`, `ma_masks`, `ma_2d`, `ma_3d`, `ma_vis`)

| Field           | Type         | Required | Description |
|-----------------|--------------|----------|-------------|
| `enabled`       | bool         | yes      | Steps with `enabled: false` are skipped (and dropped from the pipeline). |
| `dependencies`  | step-id list | yes      | Other step ids this one depends on. Cycles raise `RuntimeError`. |
| `engine`        | string       | yes      | One of `conda`, `apptainer`, `docker`. |
| `script`        | string       | yes      | Filename relative to `repo_path` to invoke under `python`. |
| `repo_path`     | path         | yes      | Filesystem path to the upstream repo. Absolute paths are recommended; `${VAR}` and `~` are also expanded if used. |
| `flags`         | string list  | no       | Extra argv added after the step's standard flags. Each string is `shlex`-split, so `"--sam_version sam3"` becomes two argv tokens. |
| `sif_path`      | path         | apptainer only | `.sif` file for `apptainer run`. |
| `docker_image`  | string       | docker only | Image reference for `docker run`. |
| `submit_cfg.gpus` | int        | apptainer only | `> 0` adds `--nv` to `apptainer run`. |
| Step-specific   | mixed        | varies   | E.g. `config_path`, `weights` for `ma_2d`; `config_file` for `ma_3d`. See the shipped presets under `examples/presets/`. |

#### Capture-derived step fields (presets omit; materializer fills)

Two `ma_cap` fields are special: they're capture-coupled, so the
shipped presets leave them out and the materializer fills them in
from the bound capture JSON at submit/run time.

| Step field            | Required | Derived from (when preset is silent)                                                                  |
|-----------------------|----------|--------------------------------------------------------------------------------------------------------|
| `ma_cap.videos_dir`   | yes      | `<capture.capture_root>/{seq_name}/<capture.videos_subdir>` — defaults to `videos_crf24`. iphones captures set `videos_subdir: videos` since they ship raw videos. |
| `ma_cap.calibration`  | yes      | `capture.calib`, resolved relative to the capture JSON's parent dir.                                   |

Setting either field on the preset overrides the derivation — useful
when you want a one-off video tier (e.g. `videos_light/`) or an
alternate calibration.

### Output path layout

Every step writes under the same 4-segment shape, emitted by
`inference/steps/base.py:step_out_dir()` and consumed by both the
runner (for DONE-sentinel resolution + input lookup) and the GUI
(for `gui/backend/sync.py` enumeration):

```
<out_dir>/<step>/<output_id>/<dataset_name>/<seq>/
```

| Segment           | Source                                                                 | Why it's load-bearing                                                                 |
|-------------------|------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `<out_dir>`       | `global.out_dir` in the run config; overridable via CLI `--out-dir`.   | Root of the on-disk output tree. Many runs across many datasets share this root.      |
| `<step>`          | The step name (`ma_cap`, `ma_masks`, `ma_2d`, `ma_3d`, `ma_vis`).      | Per-step isolation; lets each step's outputs live next to (not inside) the others.    |
| `<output_id>`     | CLI: `--out-tag` (default `local`). GUI: form's "Output ID" field, default equal to `task_id` but reusable to extend a previous run group. | Per-submission namespace. Two runs of the same preset against the same capture stay isolated. **Not the same as `task_id`** — `task_id` is the DB row id, never appears in paths; the GUI exposes a separate `output_id` so users can fold related runs into one tree. |
| `<dataset_name>`  | `global.dataset_name` in the run config; the materializer derives it from the capture filename when the preset is silent. | Per-capture namespace. The same preset against two different captures lands in two different subtrees, so downstream-step input resolution (e.g. `ma_2d` reading `ma_cap` outputs from the "standard location") doesn't need explicit per-capture overrides. |
| `<seq>`           | Sequence name from the capture JSON's `sequences[...].name`/`ioi`.     | Per-sequence isolation. Sequence names embed the capture prefix today (e.g. `140725_Breakdance_Improv_1_…`), so collisions are unlikely, but the segment keeps the layout self-describing. |

DONE sentinels live at `…/<seq>/DONE` (same path; one extra file).
Per-(step, seq) log files live in a different root under
`global.jobs_log_dir`: `<jobs_log_dir>/<user>/<output_id>/<step>/<seq>.{log,out,err}`
— no `<dataset_name>` segment there, since logs are short-lived.

### Step input overrides

Each step takes its inputs from the standard layout
`<out_dir>/<dep>/<output_id>/<dataset_name>/`. To point a step at a *different*
prior-step output, set `<step>.<dep>_dir` (e.g. `ma_2d.ma_cap_dir`) — empty
string means "use the standard location."

## DONE-sentinel re-run semantics

After a `(step, seq)` pair finishes successfully, the runner writes
`<out_dir>/<step>/<output_id>/<dataset_name>/<seq>/DONE`. Subsequent runs **skip**
that pair. This makes re-running the pipeline after a failure cheap:
unfinished work resumes, finished work is left alone.

Pass `--force` to ignore DONE sentinels and re-run everything.

## Engines

| Engine    | When to use | Notes |
|-----------|-------------|-------|
| `conda`   | Default for local runs. | Each step's `repo_path` becomes the working directory; `python <script>` runs in `<conda_env>`. |
| `apptainer` | Mirrors the cluster setup. | Requires `sif_path` per step. `repo_path` is bind-mounted at `/repo`. |
| `docker`  | Same shape as apptainer for non-HPC machines. | Requires `docker_image` per step. Always passes `--gpus all`. |
