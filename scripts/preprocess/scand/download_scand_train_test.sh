#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

LOCAL_DIR="${1:-datasets/raw/scand}"
WORKERS="${2:-1}"

export HF_HUB_DISABLE_XET=1

date
PYTHONUNBUFFERED=1 python scripts/preprocess/scand/download_scand_hf_subset.py \
  --split-dir data/splits/scand/train \
  --local-dir "$LOCAL_DIR" \
  --workers "$WORKERS"

PYTHONUNBUFFERED=1 python scripts/preprocess/scand/download_scand_hf_subset.py \
  --split-dir data/splits/scand/test \
  --local-dir "$LOCAL_DIR" \
  --workers "$WORKERS"
date
