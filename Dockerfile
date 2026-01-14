FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates bzip2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# micromamba
RUN curl -L https://micro.mamba.pm/install.sh | bash
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=/root/.local/bin:${PATH}

# Create env (without torch)
COPY env.yaml /tmp/env.yaml
RUN micromamba create -y -n nwm -f /tmp/env.yaml && micromamba clean -a -y
ENV PATH=${MAMBA_ROOT_PREFIX}/envs/nwm/bin:${PATH}

# Install PyTorch nightly cu126 (README 기준)
RUN python -m pip install --upgrade pip && \
    pip install --pre torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/nightly/cu126

# Copy code
COPY . /workspace/nwm
WORKDIR /workspace/nwm

CMD ["bash"]
