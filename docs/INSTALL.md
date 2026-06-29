# Installation

Targets **Python 3.11** and **CUDA 12.4**. Other versions may work but are untested.

After completing this guide, return to [README.md → Quick demo](../README.md#quick-demo) for a first run.

---

## 1. Environment installation

`micromamba` or `conda` both work — the env file is a plain conda spec. Examples use `micromamba`; for `conda` just substitute the binary name (`conda create …`, `conda activate …`).

### 1a. Create and activate the env

```bash
micromamba create -f requirements/mamma_conda.yaml -y
micromamba activate mamma
```

> **Heads-up if you used `micromamba` and also have `conda` installed.** The pipeline runner shells out via `conda run -n mamma …` (`inference/engines.py`), and conda only searches its own `envs_dirs` (typically `~/miniconda3/envs/`). Envs created by `micromamba` land under `$MAMBA_ROOT_PREFIX/envs/` and are invisible to `conda` by default, so `ma_cap` will fail with `EnvironmentLocationNotFound: Not a conda environment: …/envs/mamma`. Tell conda where to look once:
>
> ```bash
> conda config --append envs_dirs "$MAMBA_ROOT_PREFIX/envs"
> conda env list   # should now list `mamma`
> ```
>
> Written to `~/.condarc`; reversible with `conda config --remove envs_dirs "$MAMBA_ROOT_PREFIX/envs"`.

### 1b. Make CUDA 12.4 reachable

The pip layer in §1c compiles CUDA kernels (`detectron2`, `pytorch_sdf`), so the **CUDA 12.4 toolkit** must be on `PATH` *before* that step runs.

**Option A — system toolkit:**

```bash
export CUDA_HOME=/path/to/cuda-12.4    # e.g. `module load cuda/12.4` on HPC
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version                         # must report release 12.4
```

**Option B — install the toolkit into the env** (`mamma` already activated from §1a):

```bash
micromamba install -n mamma -c nvidia/label/cuda-12.4.1 cuda-toolkit -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version                         # must report release 12.4
```

The labeled channel pin (`nvidia/label/cuda-12.4.1`) matters — without it, conda resolves to the latest 13.x package and the kernel compile fails with `CUDA version (13.x) mismatches PyTorch (12.4)`.

### 1c. Install pip layers

With CUDA reachable, install the two pip layers:

```bash
pip install -r requirements/requirements.txt
pip install --no-build-isolation   -r requirements/requirements_no_build_isolation.txt
```

The env also bundles `nodejs=20`, Flask, Flask-CORS, and python-dotenv, so the GUI runs in the same env — no second env to manage.

---

## 2. Model weights

Two recommended paths to fetch everything (covered by [README → Get the data](../README.md#get-the-data)):

- **GUI:** start the GUI (`bash gui/scripts/dev.sh`), open the *Pipeline assets* panel on the Home page, sign in once where prompted, and use the one-click download buttons. Easiest path.
- **CLI:** shell scripts under [`data/`](../data/) prompt for credentials and download into `<repo>/data/`:
  ```bash
  bash data/download_mamma_weights.sh --all       # MAMMA landmark ckpt + downsampled SMPL-X verts (MAMMA account)
  bash data/download_smplx_locked_head.sh         # SMPL-X locked-head body model (SMPL-X account)
  ```
  The two scripts use different gates: `download_mamma_weights.sh` authenticates against the **MAMMA** account (register at <https://mamma.is.tue.mpg.de/>); `download_smplx_locked_head.sh` against the **SMPL-X** account (register at <https://smpl-x.is.tue.mpg.de/>).

Target layout:

```
data/
├── body_models/
│   ├── smplx_locked_head/
│   └── downsampled_verts/verts_512.pkl
└── weights/
    ├── ma_2d/mamma_mask_full_cvpr.ckpt
    ├── sam2/sam2.1_hiera_large.pt
    ├── yolo/yolo12x.pt
    ├── vitpose/...                # training-only
    └── hrnet/...                  # training-only (HRNet variant)
```

<!-- TODO: link to the SMPL-X locked-head model + downsampled-verts download page once it's public. -->

### Manual fallback (no GUI, no scripts)

YOLO and SAM 2 are public and can be wget'd directly:

```bash
mkdir -p data/weights/yolo data/weights/sam2 configs/sam2.1

wget -O data/weights/yolo/yolo12x.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12x.pt

wget -O data/weights/sam2/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

wget -O configs/sam2.1/sam2.1_hiera_l.yaml \
  https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
```

The MAMMA landmark checkpoint and the SMPL-X locked-head model are gated. Use the GUI's *Pipeline assets* panel, or the CLI scripts (`data/download_mamma_weights.sh --ckpt` for the landmark ckpt, `data/download_smplx_locked_head.sh` for the body model).

### SAM 3 (optional, Hugging Face)

The shipped presets default to **SAM 2**. To switch to SAM 3 (gated, weights downloaded lazily from the HuggingFace cache):

```bash
pip install "setuptools<81"
pip install git+https://github.com/facebookresearch/sam3.git
hf auth login                      # account must have approved access
```

Then edit the preset to use SAM 3: change `ma_masks.flags` to `- --sam_version sam3_prompt`. Weights download on first run; no env var is needed.

### TensorRT for `ma_2d` (optional, NVIDIA-only)

The `ma_2d --tensorrt` flag (used by the `full_tensorrt.yaml` preset) compiles the landmark
network to a TensorRT FP16 engine for a ~5× faster forward. It's optional and
NVIDIA-only — without it the flag falls back to plain PyTorch, so configs stay portable.

```bash
pip install -r requirements/requirements-tensorrt.txt
```

The compiled engine is cached to disk on first run (keyed by weights/shape/precision/GPU),
so subsequent runs load it in ~2 s. See [`docs/CONFIGS.md`](CONFIGS.md#common-per-step-flags) for `--tensorrt` / `--tensorrt-fp32`.

### Backbones for training only (skip for inference)

Skip this section unless you intend to retrain.

- **ViTPose** (`vitpose-b-multi-coco.pth`, ~330 MB) — initializes the `ma_2d` landmark detector during training. Inference ignores it (the trained ckpt overwrites the backbone).
  ```bash
  mkdir -p data/weights/vitpose
  gdown "https://drive.google.com/file/d/1sCkVDSSqyzltPyGDaBKsTwY-Adag2Vgr/view?usp=sharing" \
    -O data/weights/vitpose/
  ```
  Path is hardcoded in `landmarks/configs/constants.py:PATHS.PRETRAINED_VITPOSE_CKPT_PTH`; edit there to relocate (no env override).

- **HRNet** (`pose_hrnet_w48_256x192.pth`) — COCO-pretrained HRNet-W48 backbone, needed only when training the HRNet variant (`landmarks/train_hrnet.py`). Fetch from the official [`deep-high-resolution-net.pytorch` model zoo](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) (Google Drive; documented in the [upstream README](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch#download-pretrained-models)) — open `models/pytorch/pose_coco/` inside the folder and download `pose_hrnet_w48_256x192.pth`. Place it at `data/weights/hrnet/pose_hrnet_w48_256x192.pth`; the path is hardcoded in `landmarks/lib/models/models_2d/w48_256x192_adam_lr1e-3.yaml`.

---

## 3. Verify

```bash
micromamba activate mamma
python -m inference doctor
```

`doctor` prints each resolved `MAMMA_*` path and where it came from (`DEFAULTS`, `.env.local`, or shell). Anything flagged red needs fixing before a pipeline run.

Once `doctor` prints `PASS — environment looks healthy.`, **you're ready to run the [Quick demo](../README.md#quick-demo)!** (~5 min on one GPU — a 4-cam, 30-frame end-to-end smoke.)

---

## Customising paths

If you keep weights outside the repo (shared lab path, scratch mount, etc.), override defaults via a `.env.local` file at the repo root:

```bash
cp .env.example .env.local            # commented snapshot of the defaults
# edit .env.local: uncomment the keys you want to override
bash gui/scripts/dev.sh               # restart the GUI so it re-reads .env.local
```

`.env.local` is gitignored — your local edits never end up in commits. Run `python -m inference doctor` to confirm; the `SOURCE` column reads `[.env.local]` for keys you've changed.

Override-able env keys (declared in [`inference/assets.py`](../inference/assets.py)):

| Env key                       | Default                                                       | Consumed by                  |
|-------------------------------|---------------------------------------------------------------|------------------------------|
| `MAMMA_YOLO_CHECKPOINT`       | `data/weights/yolo/yolo12x.pt`                                | ma_masks (`--yolo-checkpoint`) |
| `MAMMA_SAM2_CHECKPOINT`       | `data/weights/sam2/sam2.1_hiera_large.pt`                     | ma_masks (`--sam_checkpoint`, only when `--sam_version sam2`) |
| `MAMMA_SMPLX_LOCKHEAD_MODELS` | `data/body_models/smplx_locked_head`                          | ma_3d (`--smplx-models`)     |
| `MAMMA_DOWNSAMPLED_VERTS_PKL` | `data/body_models/downsampled_verts/verts_512.pkl`            | ma_2d + ma_3d (`--downsampled-verts`) |
| `MAMMA_MA2D_CHECKPOINT`       | `data/weights/ma_2d/mamma_mask_full_cvpr.ckpt`                | ma_2d (`--weights`, after task.json) |
| `MAMMA_BUN_MODELS`            | (none — set only when `use_bun_model: True`)                  | ma_3d (`--bun-models`)       |
| `MAMMA_PART_MESH_PATH`        | (none — set only when the SDF loss is enabled)                | ma_3d (`--part-mesh`)        |

> Training-side paths in [`landmarks/configs/constants.py`](../landmarks/configs/constants.py) are **not** overridable via `.env.local`. To relocate training assets, edit that file directly.

**SAM checkpoint resolution.** When `--sam_version sam2`, the runner injects `--sam_checkpoint <MAMMA_SAM2_CHECKPOINT>` automatically. When `--sam_version sam3` or `sam3_prompt`, no env var is needed — the subprocess loads SAM 3 from the HuggingFace cache. To pin a specific checkpoint, add `--sam_checkpoint <path-or-hf-id>` to the preset's `flags`; the runner detects an explicit preset entry and skips its own injection.

The GUI's *Pipeline assets* panel probes the **default** locations only. With overrides in `.env.local`, a row may still show "missing" while your runs succeed — the runner trusts the env vars, the panel shows defaults.

---

## Step repositories

Each step's code lives under its own top-level directory. All five trees are tracked directly by `mamma_release` — there are no submodules; `git clone` gets the full source in one shot.

| Step      | Directory          |
|-----------|--------------------|
| ma_cap    | `capture/`         |
| ma_masks  | `segmentation/`    |
| ma_2d     | `landmarks/`       |
| ma_3d     | `optimization/`    |
| ma_vis    | `visualization/`   |

See [`steps.md`](steps.md) for the per-step builder mapping and the pipeline diagram.
