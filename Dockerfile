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
ENV CONDA_PREFIX=${MAMBA_ROOT_PREFIX}/envs/nwm
ENV PATH=${CONDA_PREFIX}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# Install a matched PyTorch stack.
RUN python -m pip install --upgrade pip && \
    pip install \
      torch==2.10.0 \
      torchvision==0.25.0 \
      torchaudio==2.10.0

# Copy code
COPY . /workspace/nwm
WORKDIR /workspace/nwm

CMD ["bash"]
