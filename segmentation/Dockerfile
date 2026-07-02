FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

LABEL description="ma_masks: SAM2+SAM3 segmentation pipeline for MAMMA"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    SAM_ENV_NAME=sam_env

ENV PATH=/opt/micromamba/envs/${SAM_ENV_NAME}/bin:/opt/micromamba/bin:$PATH

# System deps
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bash git curl wget ca-certificates \
        gcc g++ ninja-build \
        ffmpeg libgl1 libglib2.0-0 bzip2 && \
    rm -rf /var/lib/apt/lists/*

# 1. Install micromamba
RUN mkdir -p /opt/micromamba && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# 2. Create Python 3.12 environment
RUN micromamba create -y -n "${SAM_ENV_NAME}" -c conda-forge python=3.12 pip && \
    micromamba run -n "${SAM_ENV_NAME}" \
        python -m pip install --upgrade pip "setuptools<81" wheel

# 3. Install PyTorch (CUDA 12.6)
RUN micromamba run -n "${SAM_ENV_NAME}" \
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. Install project dependencies (SAM2, YOLO, CLIP, etc.)
COPY requirements.txt /opt/masks/requirements.txt
RUN micromamba run -n "${SAM_ENV_NAME}" \
    python -m pip install -r /opt/masks/requirements.txt

# 5. Install SAM3 and its dependencies (optional — comment out if not needed)
RUN micromamba run -n "${SAM_ENV_NAME}" \
        python -m pip install decord && \
    micromamba run -n "${SAM_ENV_NAME}" \
        python -m pip install git+https://github.com/facebookresearch/sam3.git || \
        echo "WARNING: SAM3 installation failed (may need HF_TOKEN at runtime)"

# 6. Download SAM2 weights
RUN mkdir -p /opt/masks/weights && \
    wget -q -O /opt/masks/weights/sam2.1_hiera_large.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt || \
        echo "WARNING: SAM2 weights download failed. Bind-mount or set SAM2_CHECKPOINT."

# Clean caches
RUN micromamba clean --all -y || true

# Copy source code
COPY run_ma_masks.py process_sequence.py /opt/masks/
COPY configs /opt/masks/configs
COPY core /opt/masks/core
COPY utils /opt/masks/utils
COPY scripts /opt/masks/scripts

WORKDIR /opt/masks

# No ENTRYPOINT on purpose. The pipeline runner (inference/engines.py) invokes
#   docker run … -w /repo -v <segmentation>:/repo <image> python <script> <args>
# i.e. it passes `python` itself — an ENTRYPOINT ["python"] would turn that
# into `python python <script>` and break every pipeline run. Standalone use:
#   docker run --rm --gpus all <image> python run_ma_masks.py --help
CMD ["python", "--version"]
