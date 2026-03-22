#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CUDA_DEVICE="${CUDA_DEVICE:-1}"
WEIGHTS_ROOT="${NWM_WEIGHTS_DIR:-/workspace/nwm/weights}"
MODEL_PATH="${MODEL_PATH:-${WEIGHTS_ROOT}/pretrained/Qwen2-VL-7B-Instruct}"
PROMPT_MODE="${PROMPT_MODE:-scene_only}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
TEST_SHARDS="${TEST_SHARDS:-16}"
TRAIN_SHARDS="${TRAIN_SHARDS:-64}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/workspace/nwm/datasets/recon_1fps_test}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/workspace/nwm/datasets/recon_1fps_train}"
TEST_MANIFEST="${TEST_MANIFEST:-/workspace/nwm/artifacts/phase1/recon_test_1fps_manifest.jsonl}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/workspace/nwm/artifacts/phase1/recon_train_1fps_manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/nwm/artifacts/phase1/qwen}"

run_split() {
  local split="$1"
  local manifest="$2"
  local data_root="$3"
  local num_shards="$4"
  local split_root="$OUTPUT_ROOT/$split"
  local shard_root="$split_root/shards"
  local merged_output="$split_root/all.jsonl"

  mkdir -p "$shard_root"

  for ((shard_index=0; shard_index<num_shards; shard_index++)); do
    local shard_output
    shard_output="$(printf "%s/%03d.jsonl" "$shard_root" "$shard_index")"
    if [[ -s "$shard_output" ]]; then
      echo "[$split] skip shard $shard_index -> $shard_output"
      continue
    fi

    echo "[$split] start shard $shard_index/$((num_shards-1))"
    ./scripts/nwm-run.sh "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python scripts/generate_qwen_captions.py \
      --manifest $manifest \
      --data-root $data_root \
      --output $shard_output \
      --model-name-or-path $MODEL_PATH \
      --prompt-mode $PROMPT_MODE \
      --batch-size $BATCH_SIZE \
      --dtype $DTYPE \
      --num-shards $num_shards \
      --shard-index $shard_index"
  done

  python - "$shard_root" "$merged_output" <<'PY'
import glob
import os
import sys

shard_root = sys.argv[1]
merged_output = sys.argv[2]
paths = sorted(glob.glob(os.path.join(shard_root, "*.jsonl")))
with open(merged_output, "w", encoding="utf-8") as dst:
    for path in paths:
        with open(path, "r", encoding="utf-8") as src:
            for line in src:
                dst.write(line)
print(f"merged {len(paths)} shards -> {merged_output}")
PY
}

run_split "recon_test_1fps" "$TEST_MANIFEST" "$TEST_DATA_ROOT" "$TEST_SHARDS"
run_split "recon_train_1fps" "$TRAIN_MANIFEST" "$TRAIN_DATA_ROOT" "$TRAIN_SHARDS"
