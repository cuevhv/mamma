# MammaNet — 2D landmark detection (`ma_2d`)

Training, evaluation, and standalone inference for the MammaNet 2D landmark
detector — the `ma_2d` step of the MAMMA pipeline. For the end-to-end pipeline
and dataset downloads, see the [top-level README](../README.md); for the
environment + CUDA + weights setup, see [`docs/INSTALL.md`](../docs/INSTALL.md).

Built on PyTorch Lightning (training loop), Hydra (config composition), and
WebDataset (data loading).

> **Run all commands from the repository root** (the same launch dir as
> `python -m inference`). Script paths and config paths below are written
> relative to the root.

## Environment Variables

Most paths are baked into the repo under `data/` (see
[Body models and weights](#body-models-and-weights)). The few remaining knobs —
SAM2 mask root, optional W&B key — live in a `.env` file at the repo root; see
the table in [`configs/constants.py`](configs/constants.py) for the canonical
list.

`.env` is auto-loaded via
[python-dotenv](https://github.com/theskumar/python-dotenv) when
`configs/constants.py` (or `run_ma_2d.py`) is imported. Exporting in your shell
works too.

The webdataset root itself is **not** an env var — pass it as a Hydra CLI
override:

```bash
python landmarks/train.py dataset_path=/your/path/to/BEDLAM_LAB_WD
```

(default: `data/mammasyn`; the BEDLAM masks webdataset is expected at
`${dataset_path}/BEDLAM_MASKS_WD/`).


## Configuration Files
`hydra` is used to read the configuration files. You can find them in
`configs/train/models_2d`, structured as follows:

```
configs/train/models_2d
|-- config.yaml
|-- config_mammanet_mask_512.yaml
|-- config_mammanet_mask_512_no_contact.yaml
|-- config_mammanet_no-mask_512.yaml
|-- config_hrnet_no-mask_512.yaml
|-- config_camerahmr_mask_512.yaml
|-- config_camerahmr_no-mask_512.yaml
|-- data/
|   |-- BEDLAM_WD.yaml
|   |-- BEDLAM_WD_all.yaml
|   |-- BEDLAM_WD_all_moyo.yaml
|   |-- BEDLAM_WD_all_moyo_no-hi4d.yaml
|   |-- BEDLAM_WD_all_plus_bedlam.yaml
|   |-- BEDLAM_WD_all_plus_bedlam_moyo_no-hi4d.yaml
|   |-- BEDLAM_WD_hands.yaml
|   |-- BEDLAM_WD_interaction.yaml
|   |-- BEDLAM_WD_interaction_no-hi4d.yaml
|   \-- BEDLAM_WD_only_bedlam.yaml
|-- model/
|   |-- pose_mammanet.yaml
|   |-- pose_mammanet_normal_no_contact.yaml
|   |-- pose_hrnet.yaml
|   \-- pose_camerahmr.yaml
|-- optimizer/
|   \-- adam.yaml
```
### Config.yaml
`config.yaml` holds the top-level configuration. Its important fields are:
- model: select any yaml file inside the `model` folder (default `pose_mammanet`).
- data: the configuration file where the datasets are called. they are inside the `data` folder (default `BEDLAM_WD_all_plus_bedlam`).

You can also change variables like the input image size, or if you want to train with the mask encoder.
Also there are general configuration variables like number of gpus `gpus_n` number of workers `workers_per_gpu` and number of samples per gpu `samples_per_gpu`.

### models
As mentioned above, the models can be found in the folder `model`.
The models are divided in 2.

- `backbone` which we recommend not to change anything.
- `decoder` from which you can change the transformer decoder parameters and also if the decoder will output `uncertainty`, `visibility`, `contact`, `floor_contact`.

All the variables that have the format of `${}`, means that they inherit the value from the main `config.yaml` file, e.g. `use_mask`.

### data
The dataset configuration files can be found in `data`. In the dataset you can find a general weighting for our dataset subsets and `bedlam`.

The configuration file is split in two main parts, `train` and `val`. Both have as arguments the name of the datasets, e.g. `BEDLAM_WD`.

Inside each dataset we find the arguments: `weights`, `hand_weight` and `label_path`. `weights` is the general weighting of a given dataset. We use `hand_weight` to weight the loss for the hands, this is because unfortunately some datasets do not have hand motions, just bodies. As a consequence, this causes penetration between bodies. Weighting down the hands is a way to prevent the network from learning this. `label_path` is the path to the dataset.


## Datasets
The synthetic training data and processed evaluation datasets live on the MAMMA
project page; see the [top-level README](../README.md#mamma-datasets) and
[`docs/DATASETS.md`](../docs/DATASETS.md) for the download scripts.

Our dataset is stored in the `webdataset` format. For each dataset (e.g. `hi4d_1_NC_200_00_contact`) the folder structure is as follows.

```bash
hi4d_1_NC_200_00_contact
|-- be_bcWkpdSJTmS4_seq_000000
|-- be_bcWkpdSJTmS4_seq_000001
|   |-- 000000.tar
|   |-- 000001.tar
|   \-- metadata.json
|-- be_bcWkpdSJTmS4_seq_000002
|-- be_bcWkpdSJTmS4_seq_000003
|-- be_bcWkpdSJTmS4_seq_000004
|-- ...
|-- be_bcWkpdSJTmS4_seq_000199
|-- get_dataset_list.sh
|-- tar_train_list.txt
\-- train_data.txt
```

Each subfolder of the dataset corresponds to one sequence. For each sequence `webdataset` saves its per-frame metadata in `.tar` files.

`tar_train_list.txt` has a list of all the `.tar` files of the dataset.

## Body models and weights

Training reuses the same `data/` layout as inference (see
[`docs/INSTALL.md`](../docs/INSTALL.md) for the full setup). The assets training
needs:

- `data/body_models/smplx_locked_head/` — SMPL-X locked-head model.
- `data/body_models/downsampled_verts/verts_512.pkl` — the 512-vertex SMPL-X
  subsampling loaded by `BEDLAM_WD` (also used by `DrawUV` for visualization).
- `data/weights/vitpose/vitpose-b-multi-coco.pth` — pretrained ViTPose backbone
  ([download](https://drive.google.com/file/d/1sCkVDSSqyzltPyGDaBKsTwY-Adag2Vgr/view?usp=sharing)).
- `data/weights/hrnet/pose_hrnet_w48_256x192.pth` — COCO-pretrained HRNet-W48,
  only needed if you train the HRNet variant.

## Training

Training is driven by `train.py` (Hydra + PyTorch Lightning) with configs under
`configs/train/models_2d/`. A separate `train_hrnet.py` entry point exists for
the HRNet backbone — it's a stripped-down variant of `train.py` (no `bf16-mixed`
precision, no per-step checkpointing) tuned for HRNet's memory characteristics.

### Prerequisites

1. **`data/` assets** — body models, SMPL-X subsampling files, and pretrained
   ViTPose backbone. See [Body models and weights](#body-models-and-weights) for
   the full layout.
2. **Training data**: point `dataset_path` (Hydra CLI override; default
   `data/mammasyn`) at your MAMMA webdataset root. The BEDLAM masks webdataset
   must sit at `${dataset_path}/BEDLAM_MASKS_WD/`.
3. **HRNet backbone (only if training HRNet)**: place
   `pose_hrnet_w48_256x192.pth` at `data/weights/hrnet/pose_hrnet_w48_256x192.pth`.

### Quickstart

```
python landmarks/train.py
```

Uses `config.yaml` defaults: model `pose_mammanet`, data `BEDLAM_WD_all_plus_bedlam`,
optimizer `adam`. All run artifacts (experiment logs/viz, checkpoints, wandb,
Hydra's run dir) land under `landmark_outputs/` at the launch dir. Default scale
is 4-GPU DDP (see [Multi-GPU and debug mode](#multi-gpu-and-debug-mode) below).

### Hydra overrides

Swap presets from the CLI:

```
# different data preset
python landmarks/train.py data=BEDLAM_WD_all

# different model
python landmarks/train.py model=pose_camerahmr

# combine multiple overrides
python landmarks/train.py model=pose_camerahmr data=BEDLAM_WD_hands use_mask_encoder=false
```

Available presets in `configs/train/models_2d/`:

- **model/**: `pose_mammanet` (default), `pose_mammanet_normal_no_contact`,
  `pose_hrnet`, `pose_camerahmr`
- **data/**: `BEDLAM_WD`, `BEDLAM_WD_all`, `BEDLAM_WD_all_plus_bedlam` (default),
  `BEDLAM_WD_all_moyo`, `BEDLAM_WD_all_moyo_no-hi4d`,
  `BEDLAM_WD_all_plus_bedlam_moyo_no-hi4d`, `BEDLAM_WD_hands`,
  `BEDLAM_WD_interaction`, `BEDLAM_WD_interaction_no-hi4d`, `BEDLAM_WD_only_bedlam`
- **optimizer/**: `adam`

Scalar fields can be overridden directly: `python landmarks/train.py scale_image=1
total_epochs=10 samples_per_gpu=12`.

### Multi-GPU and debug mode

`config.yaml` defaults to 4-GPU DDP (`gpus_n: ${if:${debug_mode}, 1, 4}`).
Override `gpus_n` to change the count and use `CUDA_VISIBLE_DEVICES` to pick
which GPUs are exposed.

For a single-GPU sanity check that exercises the full pipeline without burning
GPU-hours:

```
python landmarks/train.py debug_mode=true
```

`debug_mode=true` switches to 1 GPU, fewer workers, smaller per-step batch,
`log_steps=1`, and reduced loss weights. Use it after touching configs or env
vars to confirm the data / body-model plumbing is correct.

### Training HRNet

For the HRNet backbone, use `train_hrnet.py` with the matching model preset:

```
python landmarks/train_hrnet.py model=pose_hrnet model_name=DenseLdmks2DHRNet
```

(`model_name` is a top-level cfg field that selects the Lightning module via
`build_model`; the default is `DenseLdmks2DViT`.)

### Logging

If `WANDB_API_KEY` is set in your `.env` / shell, training logs to
Weights & Biases. Leave it unset to disable wandb logging.

## Standalone 2D inference

`run_ma_2d.py` runs the trained MammaNet over a multi-camera sequence and writes
dense 2D landmarks (plus per-landmark visibility, contact, and floor-contact) —
one NPZ per camera. This is the standalone form of the `ma_2d` pipeline step; its
output is exactly the input contract the `ma_3d` step consumes (see
[`optimization/README.md`](../optimization/README.md#2d-landmark-predictions-ma_2d_dirseq_name)).

It takes **exactly one** of three input modes:

- `--img_folder <ma_cap_root>` — chained mode: reads the per-camera NPZ manifests
  at `<img_folder>/<seq_name>/gt/IOI_*.npz` (each carrying per-frame
  `img_abs_path`). Use with `--seq_name`.
- `--videos_dir <dir>` — one `<cam_name>.mp4` per camera.
- `--images_root_dir <dir>` — one `<cam_name>/` frame directory per camera.

Key arguments:

- `--config_path` — model config (e.g. `landmarks/configs/train/models_2d/config_mammanet_mask_512.yaml`).
- `--weights` (required) — trained MammaNet checkpoint.
- `--downsampled-verts` — path to `verts_512.pkl` (used by the UV visualization);
  pass `data/body_models/downsampled_verts/verts_512.pkl` (the bundled location).
- `--mask_path` — folder of segmentation masks
  (`<mask_path>/<cam>/masks/mask_<frame>_<bodyid>.png`); one body is exported per
  mask ID. If omitted, Detectron2 detects people online and keeps the single box
  closest to the image center (single-person assumption).
- `--out_folder` (default `out`) — outputs; with `--seq_name` set, results are
  scoped under `<out_folder>/<seq_name>/`.
- `--cam_names IOI_01 IOI_02 …` — restrict to specific cameras.
- `--calibration` + `--undistort` — apply Vicon-radial-2 undistortion before
  inference (videos / images modes).
- `--start` / `--end` — frame range (videos / images modes only; NPZ mode uses
  the manifest's range).
- `--save_cam_output` / `--no-save_cam_output` — per-camera viz frames + preview MP4.

Example (from the repo root) — per-camera videos, masks from the `ma_masks` step:

```bash
python landmarks/run_ma_2d.py \
  --videos_dir /path/to/videos \
  --config_path landmarks/configs/train/models_2d/config_mammanet_mask_512.yaml \
  --weights /path/to/mammanet.ckpt \
  --downsampled-verts data/body_models/downsampled_verts/verts_512.pkl \
  --mask_path /path/to/ma_masks_output \
  --out_folder out
```

Chained NPZ mode (consuming `ma_cap` output), with online Detectron2 detection:

```bash
python landmarks/run_ma_2d.py \
  --img_folder /path/to/ma_cap_out --seq_name my_sequence \
  --config_path landmarks/configs/train/models_2d/config_mammanet_mask_512.yaml \
  --weights /path/to/mammanet.ckpt \
  --downsampled-verts data/body_models/downsampled_verts/verts_512.pkl \
  --out_folder out
```

## Evaluation

The benchmark evaluation scripts will be released separately.
