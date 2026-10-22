# Registration (ma_3d)

Fits SMPL-X bodies to multi-view sequences using per-camera 2D keypoints, camera metadata, and optimization settings defined in YAML configs. This is the `ma_3d` step of the MAMMA pipeline.

**Entry point:** `run_ma_3d.py`, fits SMPL-X to multi-person, multi-camera sequences.

For the end-to-end pipeline see the [top-level README](../README.md); for the
environment + CUDA + weights setup see [`docs/INSTALL.md`](../docs/INSTALL.md)
(activate `mamma` before running `run_ma_3d.py` directly). This README documents
running `ma_3d` standalone, its config structure, data contracts, and outputs.

> **Run all commands from the repository root** (the same launch dir as
> `python -m inference`). Script and config paths below are written relative to
> the root.

## Fitting Configs

`config_files/contact_configs/config_real_gmf_small_vals_detr_exp_no_vtemplate.yaml` should be enough to run most models. See [`docs/CONFIGS.md`](../docs/CONFIGS.md) for how `ma_3d` slots into a pipeline preset.

Every YAML under `config_files/` controls:

- **Model setup**: `n_betas`, `use_v_template`, `use_bun_model` determine which SMPL-X meshes are loaded, whether a custom template is injected, and how many body-shape coefficients can change.
- **Optimization stages**: each block under `optim` (`first_run`, `second_run`, `third_run`, ...) sets which parameters are free (`pose/betas/trans`), and the losses used for that stage (reprojection, prior, temporal, intersection, etc.).

To customize a config:

1. Copy the closest template, e.g. `config_files/contact_configs/config_real_gmf_small_vals_detr_exp_no_vtemplate.yaml`.
2. Adjust `optim` stages, change iteration counts, learning rates, or loss weights.
3. Toggle `use_v_template` or `n_betas` if you have subject-specific templates or want more/less shape freedom.

Pass your edited file through `--config_file` when running the script.

## Usage

```bash
python optimization/run_ma_3d.py \
    --seq_name   <sequence_name> \
    --ma_cap_dir <path/to/ma_cap> \
    --ma_2d_dir  <path/to/2d_predictions> \
    --config_file optimization/config_files/contact_configs/config_real_gmf_small_vals_detr_exp_no_vtemplate.yaml \
    --out_path   <output_root> \
    [--cam_names IOI_01 IOI_02 ...] \
    [--cam_name_prefix IOI_]
```

The script internally constructs:
- `metadata_data_pth = <ma_cap_dir>/<seq_name>/gt`
- `ldmks_pred_path  = <ma_2d_dir>/<seq_name>`

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--seq_name` | yes | - | Sequence subdirectory name inside both `--ma_cap_dir` and `--ma_2d_dir`. Also used as the output folder name. |
| `--ma_cap_dir` | yes | - | Root of the MA-CAP data tree. Sequence GT data lives at `<ma_cap_dir>/<seq_name>/gt/`. |
| `--ma_2d_dir` | yes | - | Root of the 2D landmark prediction tree. Per-sequence predictions live at `<ma_2d_dir>/<seq_name>/`. |
| `--config_file` | yes (in practice) | - | YAML optimization config (see *Fitting Configs* above). The argparse default points at a `config_files/config.yaml` that isn't shipped, so always pass a preset, e.g. `optimization/config_files/contact_configs/config_real_gmf_small_vals_detr_exp_no_vtemplate.yaml`. |
| `--out_path` | yes | - | Output root. Results are written to `<out_path>/<seq_name>/`. |
| `--cam_names` | no | `[]` | Explicit list of camera names to use (e.g. `IOI_01 IOI_02`). Overrides `--cam_name_prefix`. |
| `--cam_name_prefix` | no | `IOI_` | Glob prefix for camera selection when `--cam_names` is empty. |
| `--start_frame` | no | `0` | First frame index to use. All earlier frames are excluded from the data entirely. |
| `--end_frame` | no | `None` | End frame index (exclusive). All later frames are excluded from the data entirely. |
| `--ignore_start_frames` | no | `0` | Number of initial frames to exclude from optimization losses (but keep in the output). Unlike `--start_frame` which removes frames entirely, this keeps them but does not optimize for them; after fitting, they are filled by copying from the first optimized frame. Applied after `--start_frame`/`--end_frame` slicing. |

### Example

```bash
python optimization/run_ma_3d.py \
    --seq_name 260216_MultiMama_3_pass_clap_001101_1 \
    --ma_cap_dir /path/to/ma_cap \
    --ma_2d_dir  /path/to/2d_predictions \
    --config_file optimization/config_files/contact_configs/config_real_gmf_small_vals_detr_exp_no_vtemplate.yaml \
    --out_path /scratch/my_user/results \
    --cam_name_prefix IOI_
```

## Data Directory Layout

### MA-CAP GT directory (`<ma_cap_dir>/<seq_name>/gt/`)

```
gt/
├── global.npz          # Sequence-level metadata (frames_len, people_len, gender, ...)
├── IOI_01.npz          # Per-camera metadata
├── IOI_02.npz
└── ...
```

Each per-camera NPZ contains:

| Key | Description |
|---|---|
| `cam_int` | 3x3 intrinsics matrix |
| `cam_ext` | 4x4 extrinsics matrix (world-to-camera). Values >200 are assumed to be in mm and are automatically converted to metres. |
| `cam_img_w` / `cam_img_h` | Image resolution in pixels |
| `img_abs_path` | Per-frame absolute image paths (array of strings) |
| `img_rel_path` | Per-frame relative image paths (used if `img_abs_path` is absent) |

`global.npz` is optional. When absent (e.g. for MPI_Dance sequences), the batch size and number of people are inferred from the first prediction file.

### 2D Landmark Predictions (`<ma_2d_dir>/<seq_name>/`)

One NPZ file per camera, named with the same prefix as the camera metadata files:

```
<seq_name>/
├── IOI_01_<anything>.npz
├── IOI_02_<anything>.npz
└── ...
```

Each prediction NPZ should contain:

| Key | Shape | Description |
|---|---|---|
| `landmarks` | `[T, N, L, 3]` | 2D keypoints. Channel 2 is log-variance; converted to std in `fitting.py` via `exp` then `sqrt`. If shape is `[T, N, L, 2]` the uncertainty channel is dropped. |
| `visibilities` | `[T, N, L]` | Per-landmark confidence scores (optional). |
| `contacts` | `[T, N, L]` | Per-landmark contact labels (optional). |
| `floor_contacts` | `[T, N, L]` | Per-landmark floor-contact labels (optional). |

`T` = frames, `N` = people, `L` = landmarks.

## Features

### Frame Subsampling

Long sequences (>600 frames) are automatically subsampled: one frame every `round(T/600)` frames is used for optimization, keeping GPU memory bounded.

### Resume / Skip

If `<out_path>/<seq_name>/smplx_params_body_id-XX.npz` already exists for **all** bodies in the sequence, the script exits immediately without re-running. This makes it safe to re-queue interrupted jobs.

### Camera Selection

Two modes:
1. **By name list** (`--cam_names IOI_01 IOI_02 ...`): the specified cameras are used in that order.
2. **By prefix** (`--cam_name_prefix IOI_`): all `<prefix>*.npz` files in the GT directory are globbed and sorted.

Prediction files are matched to camera files by the same criterion and aligned by sort order. If the number of prediction files is smaller than the number of camera files, the camera list is trimmed to match.

### Cross-View Re-Identification (Multi-Person)

For multi-person sequences, the 2D detector assigns body IDs independently per camera, so the same physical person may receive different integer IDs in different camera views. Before optimization the pipeline runs a **temporal multi-view re-ID** step (`utils/epipolar_association.py: mvpose_style_associate_and_triangulate_temporal`) that:

1. Triangulates each body's 3D trajectory from all cameras.
2. Builds a geometric affinity matrix across views using reprojection error.
3. Solves Hungarian-matching problems and merges them with Union-Find to produce consistent cross-camera group assignments.
4. Runs `propagate_ids_via_reprojection` to fill in cameras where a person was not detected.
5. Applies a majority-vote across cameras to choose a canonical body ID for each group, then remaps `pts2d`, `pts2d_vis_weight`, `contacts`, and `floor_contacts` accordingly.
6. Re-triangulates 3D points with the corrected assignments.

The re-ID runs **once** before the optimization loop begins (not per stage). It operates at whole-tracklet granularity (one ID per camera, not per-frame).

## Outputs

All outputs land under `<out_path>/<seq_name>/`:

| File | Description |
|---|---|
| `run_args.json` | Full CLI arguments used for this run (for reproducibility). |
| `run_config.yaml` | Copy of the YAML config file used. |
| `smplx_params_body_id-XX.npz` | Optimized SMPL-X parameters: `smplx_pose`, `smplx_betas`, `smplx_translation`, `triangulated_3d_pts`, `smplx_contact`, `smplx_floor_contact`, `v_template_pred`. |
| `verts_joints_body_id-XX.npz` | GT and predicted joint & vertex trajectories. |
| `body_id-XX.csv` | Frame-wise MPJPE / PVE statistics. |
| `reid_debug/reid_cam*.mp4` | Before/after re-ID overlay videos (top=before, bottom=after, yellow separator). |
| `reid_debug/reid_stats.png` | Per-camera re-ID correction statistics. |

## Weight & asset paths

Paths to model weights and data assets are passed as CLI flags (gathered into a
`PathsConfig` dataclass, `utils/paths_config.py`, at the entry point and threaded
through the call graph). For where the weights themselves live, see
[`docs/INSTALL.md`](../docs/INSTALL.md).

| CLI flag | Required | Description |
|---|---|---|
| `--smplx-models` | always | Directory containing the SMPL-X locked-head body models. |
| `--downsampled-verts` | always | Path to the downsampled-vertex matrix pickle (e.g. `verts_512.pkl`). |
| `--bun-models` | only if `use_bun_model: True` in the YAML config | Directory with the BUN-variant SMPL-X body models. |
| `--part-mesh` | only if using `MultiSDF` defaults (`losses/sdf.py`) | Folder containing the per-part `.ply` segmentation meshes. |

When spawned by the MAMMA inference runner, the runner-side wrapper
(`inference/steps/ma_3d.py`) translates the corresponding `MAMMA_*` environment
variables into these flags before launching the subprocess, and strips
`MAMMA_*` from the child env, so subprocesses must consume paths via argv only.
For direct invocation outside the runner, pass the flags yourself.
