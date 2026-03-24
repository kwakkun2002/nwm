#!/usr/bin/env bash
# Stop Jupyter Notebook on port 8888.
# Usage: ./scripts/user_friendly/stop_jupyter.sh
# Example: NWM_JUPYTER_PORT=8890 ./scripts/user_friendly/stop_jupyter.sh

set -euo pipefail

CONTAINER_NAME="${NWM_CONTAINER_NAME:-nwm_dev}"
PORT="${NWM_JUPYTER_PORT:-8888}"
LOG_FILE="${NWM_JUPYTER_LOG:-/tmp/nwm_jupyter_${PORT}.log}"

if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container '${CONTAINER_NAME}' does not exist."
  exit 0
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
  echo "Container '${CONTAINER_NAME}' is not running."
  exit 0
fi

echo "Stopping Jupyter Notebook on port ${PORT}..."
STOP_RESULT="$(docker exec "${CONTAINER_NAME}" bash -lc "python - <<'PY'
import os
import signal

port = '${PORT}'
stopped = []
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    try:
        cmdline = open(f'/proc/{pid}/cmdline', 'rb').read().decode(errors='ignore').replace('\x00', ' ')
    except OSError:
        continue
    if 'jupyter-notebook' in cmdline and f'--port {port}' in cmdline:
        os.kill(int(pid), signal.SIGTERM)
        stopped.append(pid)

print(' '.join(stopped))
PY")"

if [[ -z "${STOP_RESULT// }" ]]; then
  echo "No Jupyter process found on port ${PORT}."
else
  echo "Stopped Jupyter process(es): ${STOP_RESULT}"
fi

docker exec "${CONTAINER_NAME}" bash -lc "rm -f '${LOG_FILE}'" >/dev/null 2>&1 || true
