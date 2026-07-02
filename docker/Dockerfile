# MAMMA unified pipeline image — one image runs ALL five steps
# (ma_cap, ma_masks, ma_2d, ma_3d, ma_vis), mirroring the `mamma` conda env
# (docs/INSTALL.md: Python 3.11, CUDA 12.4, torch 2.5.1).
#
# Build (from the repo root; .dockerignore keeps the context tiny):
#
#   docker build -f docker/Dockerfile -t mamma:v1.1.0 .
#
# Runner contract (inference/engines.py): the pipeline invokes
#
#   docker run --rm --gpus all -w /repo -v <step repo_path>:/repo \
#       [-v <extra binds>] <image> python <script> <args...>
#
# so this image must provide `python` on PATH with every step's deps — the
# step CODE is bind-mounted at /repo at run time, never baked in. Model
# weights / footage / output paths arrive as ABSOLUTE HOST PATHS in argv:
# add an identity bind of the repo root to the preset so they resolve, e.g.
#
#   global:
#     bind: ["/abs/path/to/mamma:/abs/path/to/mamma"]
#
# See docker/README.md for a full preset example (CLI + GUI usage).
#
# devel base (not runtime): the no-build-isolation pip layer compiles CUDA
# kernels (detectron2, pytorch_sdf) and needs nvcc 12.4 — same toolkit the
# torch cu124 wheels target.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL org.opencontainers.image.title="MAMMA pipeline" \
      org.opencontainers.image.description="Markerless multi-person motion capture — unified env for all pipeline steps" \
      org.opencontainers.image.source="https://github.com/cuevhv/mamma"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    # pyrender/EGL offscreen rendering (ma_3d contact viz, ma_vis overlays):
    # the NVIDIA container runtime only injects libEGL_nvidia when the
    # 'graphics' capability is requested — without it every render falls
    # back to (or fails on) software GL.
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYOPENGL_PLATFORM=egl \
    # Headless matplotlib everywhere (same rationale as engines._child_env).
    MPLBACKEND=Agg

# System deps: build tools for the CUDA-kernel pip layer; ffmpeg for
# ma_cap/ma_vis video IO; EGL/GL runtime libs for pyrender offscreen.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash git curl wget ca-certificates bzip2 \
        build-essential ninja-build \
        ffmpeg \
        libgl1 libglib2.0-0 libegl1 libgles2 libglvnd0 \
        libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# micromamba — smallest way to get a pinned Python 3.11 with conda-forge.
RUN mkdir -p /opt/micromamba && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# Env spec mirrors requirements/mamma_conda.yaml minus host-only extras
# (nodejs is for the GUI frontend, which runs on the host, not in here).
RUN micromamba create -y -n mamma -c conda-forge \
        python=3.11 pip "numpy<2" wget && \
    micromamba clean --all -y

ENV PATH=/opt/micromamba/envs/mamma/bin:/opt/micromamba/bin:$PATH

# Torch layer first (pinned, cu124) — requirements.txt packages (sam2/sam3,
# and the no-build-isolation layer) need torch importable at install time.
RUN python -m pip install --upgrade pip "setuptools<81" wheel && \
    python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 torchvision==0.20.1 decord

# Main dependency layer (docs/INSTALL.md §1c, first half).
COPY requirements/requirements.txt /opt/mamma/requirements/requirements.txt
RUN python -m pip install -r /opt/mamma/requirements/requirements.txt

# CUDA-kernel layer (detectron2, pytorch_sdf). No GPU is visible during
# `docker build`, so tell the extensions to compile anyway and for which
# architectures (Ampere/Ada/Hopper + PTX forward-compat).
COPY requirements/requirements_no_build_isolation.txt /opt/mamma/requirements/requirements_no_build_isolation.txt
RUN CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    python -m pip install --no-build-isolation \
        -r /opt/mamma/requirements/requirements_no_build_isolation.txt

# Optional VPoser prior for ma_3d (small, pure-python — bake it in so the
# occlusion+vposer recipe works out of the box; weights still come from the
# host's data/ bind).
COPY requirements/requirements-vposer.txt /opt/mamma/requirements/requirements-vposer.txt
RUN python -m pip install -r /opt/mamma/requirements/requirements-vposer.txt

# No ENTRYPOINT on purpose: the runner passes the full `python <script>
# <args>` command itself (an ENTRYPOINT ["python"] would make it
# `python python <script>` and break every step).
WORKDIR /repo
CMD ["python", "--version"]
