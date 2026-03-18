#!/usr/bin/env bash

set -euo pipefail

CONTAINER_NAME="${NWM_CONTAINER_NAME:-nwm_dev}"
CONTAINER_WORKDIR="${NWM_CONTAINER_WORKDIR:-/workspace/nwm}"

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 '<command>'"
  echo "Example: $0 'python train.py --config config/nwm_cdit_xl.yaml'"
  exit 1
fi

if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container '$CONTAINER_NAME' does not exist."
  echo "Create it first with: ./scripts/nwm-start.sh"
  echo "Example for 2 GPUs: NWM_GPU_REQUEST='device=0,1' ./scripts/nwm-start.sh"
  exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != "true" ]; then
  echo "Starting container '$CONTAINER_NAME'..."
  docker start "$CONTAINER_NAME" >/dev/null
fi

DOCKER_TTY_ARGS=(-i)
if [ -t 0 ] && [ -t 1 ]; then
  DOCKER_TTY_ARGS=(-it)
fi

docker exec "${DOCKER_TTY_ARGS[@]}" -w "$CONTAINER_WORKDIR" "$CONTAINER_NAME" bash -lc \
  "export CONDA_PREFIX=/opt/micromamba/envs/nwm; \
   export PATH=\$CONDA_PREFIX/bin:\$PATH; \
   export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; \
   $*"
