# Running the pipeline in Docker / Apptainer

The runner supports three engines per step, `conda` (default), `docker`,
`apptainer` (`inference/engines.py`). The files here provide **one unified
image that runs every step** (ma_cap, ma_masks, ma_2d, ma_3d, ma_vis),
mirroring the `mamma` conda env from [docs/INSTALL.md](../docs/INSTALL.md)
(Python 3.11, CUDA 12.4, torch 2.5.1).

The image contains the **environment only**. At run time the pipeline
bind-mounts the repo root at `/repo` and runs each step from
`/repo/<step subdir>`, the same shape as the conda engine, and passes
absolute host paths (weights, footage, output) on the command line, so the
container also needs those host paths visible at the *same* locations (see
*Binds* below). Weights and data stay on the host; nothing is baked in.

## 1. Build

**Docker** (from the repo root; `.dockerignore` keeps the context tiny):

```bash
docker build -f docker/Dockerfile -t mamma:v1.1.0 .
```

The docker image is about 25 GB and the `.sif` about 8 GB; the conversion needs about 30 GB free in
Apptainer's temp dir (move it with `APPTAINER_TMPDIR`).

**Apptainer**, two routes:

```bash
# a) From the docker image (preferred when docker exists: identical env).
#    NOTE: use the wrapper .def, not docker-daemon:// directly: the direct
#    conversion adopts the NVIDIA base image's ENTRYPOINT as the runscript,
#    which breaks the runner's `apptainer run <sif> <script> <args…>` contract:
apptainer build mamma.sif docker/mamma_from_docker.def

# b) Directly from the definition file (no docker needed), as root:
sudo apptainer build mamma.sif docker/mamma.def
#    --fakeroot instead of sudo works only where fakeroot builds work (an
#    /etc/subuid entry for your user); without one this build fails.
```

> If the build dies in the final squashfs step (`mksquashfs command failed:
> exit status 134/139`, a squashfs-tools crash seen on large images), retry
> single-threaded: `apptainer build --mksquashfs-args "-processors 1 -mem 8G" …`
> (about 20 minutes for this image).

## 2. Configure a preset

Copy a shipped preset (e.g. `configs/examples/presets/quick.yaml`) and set,
**per step**, the engine keys ([docs/CONFIGS.md](../docs/CONFIGS.md) §step
keys). All five steps can use the same image / sif.

**Docker**, every step gets `engine` + `docker_image`, and `global.bind`
carries an identity mount of the repo root so the absolute host paths in the
command line resolve inside the container:

```yaml
global:
  # REQUIRED for docker: identity-mount the repo root (covers data/weights,
  # output/, configs). Footage outside the repo needs its own entry.
  # The second entry shares your Hugging Face cache (the container's HOME is
  # /tmp): SAM 3 needs it for your `hf auth login` token, and it keeps the
  # D-FINE detector download across runs.
  bind: ["/abs/path/to/mamma:/abs/path/to/mamma",
         "/home/<you>/.cache/huggingface:/tmp/.cache/huggingface"]
ma_masks:            # same three lines for ma_cap, ma_2d, ma_3d, ma_vis
  engine: docker
  docker_image: mamma:v1.1.0
```

The runner then executes, per step (`--user` keeps output files owned by you,
not root):

```
docker run --rm --gpus all --user <uid>:<gid> -w /repo/<step> -v <repo root>:/repo \
    -v /abs/path/to/mamma:/abs/path/to/mamma mamma:v1.1.0 python <script> <args…>
```

**Apptainer**, `sif_path` instead of `docker_image`; GPU steps additionally
need `submit_cfg.gpus: 1`, which adds `--nv` (every step except ma_cap wants
it, ma_vis renders through EGL too):

```yaml
ma_masks:
  engine: apptainer
  sif_path: /abs/path/to/mamma.sif   # relative = resolved against the repo root
  submit_cfg: { gpus: 1 }
```

GPU binding mode: when `nvidia-container-cli` exists on the host the runner
adds `--nvccli` next to `--nv` (docker-style driver-lib injection). The
legacy `--nv` mode injects the host's whole GL stack, which crashes
containers older than the host glibc (`cv2` import fails with
`GLIBC_2.3x not found`). Override per step with `submit_cfg: { nvccli: true|false }`.

Apptainer auto-binds `$HOME` and the working directory, so when the repo and
footage live under your home directory no `bind` entries are needed; anything
elsewhere (scratch filesystems, cluster storage) goes into `global.bind`
exactly like docker.

Engines mix freely, e.g. `ma_cap` on `conda`, GPU steps in a container.

## 3. Run

**Terminal**, exactly like a conda run:

```bash
python -m inference run --cfg my_docker_preset.yaml \
  --footage data/mamma_example --seq_name pushing_and_lifting_from_ground \
  --calib configs/examples/calib/iphones_outdoors.yaml --out-tag docker01 -v
```

**GUI**: the engine is part of the preset, not a per-run widget: save your
preset (drop the YAML into `configs/examples/presets/` or save it as a user
preset from the New Task form), then select it when creating the task. The
command preview in the New Task form shows the full `docker run …` /
`apptainer run …` invocation, what you preview is byte-identical to what
runs. The GUI itself (Flask backend + browser) stays on the host env.

## 4. Requirements on the host

- **Docker:** an NVIDIA driver + the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  (the runner always passes `--gpus all`). The image requests the
  `graphics` capability so EGL offscreen rendering (ma_vis overlays,
  ma_3d contact viz) works.
- **Apptainer:** ≥ 1.1 with `--nv` support and a working NVIDIA driver.
- The host needs **no** conda env, CUDA toolkit, or Python for the
  containerized steps, only for the GUI/runner itself (any Python ≥ 3.10
  with PyYAML can drive `python -m inference run`).

## Notes

- TensorRT (`ma_2d --tensorrt`) is not installed in the image, add
  `requirements/requirements-tensorrt.txt` to the Dockerfile if you need it
  containerized.
- The first ma_masks run downloads the D-FINE person detector (about 240 MB)
  into the Hugging Face cache, like SAM 3 (with docker, only with the cache bind
  above; otherwise on every run); ma_2d may download weights into the
  bound repo `data/` directory, same as the conda engine.
