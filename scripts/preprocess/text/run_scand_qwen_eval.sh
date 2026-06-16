#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_PATH="${MODEL_PATH:-weights/pretrained/Qwen2-VL-7B-Instruct}"
MANIFEST="${MANIFEST:-artifacts/summaries/preprocess/scand_text/scand_eval_dense_manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/bulk/preprocess/scand_text/qwen_eval_dense}"
PROMPT_MODE="${PROMPT_MODE:-scene_only}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
NUM_SHARDS="${NUM_SHARDS:-32}"
SHARD_START="${SHARD_START:-0}"
SHARD_STEP="${SHARD_STEP:-1}"

mkdir -p "$OUTPUT_ROOT/shards"

for ((shard_index=SHARD_START; shard_index<NUM_SHARDS; shard_index+=SHARD_STEP)); do
  shard_output="$(printf "%s/shards/%03d.jsonl" "$OUTPUT_ROOT" "$shard_index")"
  if [[ -s "$shard_output" ]]; then
    echo "[scand_eval_dense] skip shard $shard_index -> $shard_output"
    continue
  fi

  echo "[scand_eval_dense] start shard $shard_index/$((NUM_SHARDS-1)) on CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/preprocess/text/generate_qwen_captions.py \
    --manifest "$MANIFEST" \
    --output "$shard_output" \
    --model-name-or-path "$MODEL_PATH" \
    --prompt-mode "$PROMPT_MODE" \
    --batch-size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$shard_index"
done
