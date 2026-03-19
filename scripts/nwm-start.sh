#!/usr/bin/env bash

set -euo pipefail

CONTAINER_NAME="${NWM_CONTAINER_NAME:-nwm_dev}"
CONTAINER_WORKDIR="${NWM_CONTAINER_WORKDIR:-/workspace/nwm}"
IMAGE_NAME="${NWM_IMAGE_NAME:-nwm:cu126}"
GPU_REQUEST="${NWM_GPU_REQUEST:-all}"
PORT_MAPPING="${NWM_PORT_MAPPING:-8888:8888}"
HOST_MODELS_DIR="${NWM_HOST_MODELS_DIR:-}"
CONTAINER_MODELS_DIR="${NWM_CONTAINER_MODELS_DIR:-${CONTAINER_WORKDIR}/external_models}"
DOCKER_GPU_REQUEST="$GPU_REQUEST"

if [[ "$GPU_REQUEST" == device=* ]]; then
  DOCKER_GPU_REQUEST="\"$GPU_REQUEST\""
fi

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" = "true" ]; then
    echo "Container '$CONTAINER_NAME' is already running."
  else
    echo "Starting existing container '$CONTAINER_NAME'..."
    docker start "$CONTAINER_NAME" >/dev/null
  fi

  echo "If you want to change visible GPUs, remove and recreate the container first:"
  echo "  docker rm -f $CONTAINER_NAME"
  echo "  NWM_GPU_REQUEST='device=0,1' ./scripts/nwm-start.sh"
  exit 0
fi

echo "Creating container '$CONTAINER_NAME' from image '$IMAGE_NAME'..."
echo "GPU request: $GPU_REQUEST"

DOCKER_RUN_ARGS=(
  -d
  --name "$CONTAINER_NAME"
  --gpus "$DOCKER_GPU_REQUEST"
  -p "$PORT_MAPPING"
  -v "$PWD":"$CONTAINER_WORKDIR"
)

if [ -n "$HOST_MODELS_DIR" ]; then
  if [ ! -d "$HOST_MODELS_DIR" ]; then
    echo "Host models directory '$HOST_MODELS_DIR' does not exist."
    exit 1
  fi
  echo "Mounting host models: $HOST_MODELS_DIR -> $CONTAINER_MODELS_DIR"
  DOCKER_RUN_ARGS+=(-v "$HOST_MODELS_DIR":"$CONTAINER_MODELS_DIR":ro)
fi

docker run "${DOCKER_RUN_ARGS[@]}" \
  "$IMAGE_NAME" \
  tail -f /dev/null >/dev/null

echo "Container '$CONTAINER_NAME' is up."
