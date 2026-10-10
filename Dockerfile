# Lightweight Python 3.14 base, matching the tested environment.
# The heavy nvidia/cuda base image is not needed: the torch wheel
# brings its own CUDA runtime, and the GPU is accessed through Docker's
# GPU support (Docker Desktop with WSL2 GPU passthrough) at run time.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# System libs Pillow/Gradio commonly need for image handling, plus
# build-essential (gcc) — torch 2.13's Triton backend compiles CUDA
# kernels on first use and needs a C compiler present to do it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Step 1: install torch/torchvision and their CUDA runtime libs from
# local wheels (cu126, cp314). This is the combination tested on an
# RTX 3050 (4GB). The wheels are not included in this repo; see the
# README for the exact filenames.
COPY *.whl /tmp/wheels/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        /tmp/wheels/nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
        /tmp/wheels/nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl \
        /tmp/wheels/nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl \
        /tmp/wheels/torch-2.13.0+cu126-cp314-cp314-manylinux_2_28_x86_64.whl \
        /tmp/wheels/torchvision-0.28.0+cu126-cp314-cp314-manylinux_2_28_x86_64.whl && \
    rm -rf /tmp/wheels

# --- Step 2: install the rest from PyPI.
# NOTE: transformers requires huggingface-hub<1.0, while gradio requires
# >=1.2.0. This conflict is known and does not break the app in testing.
# Gradio is installed first with its full dependency tree, then
# huggingface-hub is forced down to the version that works with this code.
RUN pip install --no-cache-dir gradio==6.20.0 gradio_client==2.5.0 && \
    pip install --no-cache-dir --force-reinstall --no-deps huggingface-hub==0.36.2 && \
    pip install --no-cache-dir \
        transformers==4.57.6 \
        accelerate==1.14.0 \
        qwen-vl-utils==0.0.14 \
        pillow==12.3.0 \
        numpy==2.5.1

# --- Step 3: copy the app code (a single self-contained file)
COPY queen_ui.py .

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "queen_ui.py"]