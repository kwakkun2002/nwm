#!/usr/bin/env bash
# Start Jupyter Notebook on port 8888.
# Usage: ./scripts/user_friendly/start_jupyter.sh
# Example: NWM_JUPYTER_PORT=8890 ./scripts/user_friendly/start_jupyter.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONTAINER_NAME="${NWM_CONTAINER_NAME:-nwm_dev}"
CONTAINER_WORKDIR="${NWM_CONTAINER_WORKDIR:-/workspace/nwm}"
WEIGHTS_ROOT="${NWM_WEIGHTS_DIR:-${CONTAINER_WORKDIR}/weights}"
CACHE_ROOT="${NWM_CACHE_DIR:-${WEIGHTS_ROOT}/cache}"
PORT="${NWM_JUPYTER_PORT:-8888}"
LOG_FILE="${NWM_JUPYTER_LOG:-/tmp/nwm_jupyter_${PORT}.log}"

"${REPO_ROOT}/scripts/docker/nwm-start.sh"

echo "Stopping any existing Jupyter server on port ${PORT}..."
docker exec "${CONTAINER_NAME}" bash -lc "python - <<'PY'
import os
import signal

port = '${PORT}'
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    try:
        cmdline = open(f'/proc/{pid}/cmdline', 'rb').read().decode(errors='ignore').replace('\x00', ' ')
    except OSError:
        continue
    if 'jupyter-notebook' in cmdline and f'--port {port}' in cmdline:
        os.kill(int(pid), signal.SIGTERM)
PY"

echo "Starting Jupyter Notebook on port ${PORT}..."
docker exec -d -w "${CONTAINER_WORKDIR}" "${CONTAINER_NAME}" bash -lc "\
  mkdir -p '${CACHE_ROOT}/torch' '${CACHE_ROOT}/xdg' '${CACHE_ROOT}/huggingface'; \
  export CONDA_PREFIX=/opt/micromamba/envs/nwm; \
  export PATH=\$CONDA_PREFIX/bin:\$PATH; \
  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; \
  export TORCH_HOME='${CACHE_ROOT}/torch'; \
  export XDG_CACHE_HOME='${CACHE_ROOT}/xdg'; \
  export HF_HOME='${CACHE_ROOT}/huggingface'; \
  export HUGGINGFACE_HUB_CACHE=\"\$HF_HOME/hub\"; \
  export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"; \
  nohup jupyter notebook --ip 0.0.0.0 --port ${PORT} --no-browser --allow-root > '${LOG_FILE}' 2>&1 &"

echo "Waiting for Jupyter to come up..."
for _ in $(seq 1 30); do
  if docker exec "${CONTAINER_NAME}" bash -lc "test -f '${LOG_FILE}' && grep -q 'http://127.0.0.1:${PORT}/tree?token=' '${LOG_FILE}'"; then
    break
  fi
  sleep 1
done

URL="$(docker exec "${CONTAINER_NAME}" bash -lc "grep -o 'http://127.0.0.1:${PORT}/tree?token=[^[:space:]]*' '${LOG_FILE}' | tail -n 1" || true)"

if [[ -z "${URL}" ]]; then
  echo "Failed to extract Jupyter URL. Recent log:"
  docker exec "${CONTAINER_NAME}" bash -lc "tail -n 40 '${LOG_FILE}'"
  exit 1
fi

echo
echo "Open this URL in your browser:"
echo "${URL}"
echo
echo "Notebook root inside the container: ${CONTAINER_WORKDIR}"
