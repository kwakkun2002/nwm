#!/usr/bin/env bash

set -euo pipefail

MODEL_SIZE="${MODEL_SIZE:-s}"
if [ "$MODEL_SIZE" != "s" ] && [ "$MODEL_SIZE" != "b" ]; then
  echo "MODEL_SIZE must be 's' or 'b'." >&2
  exit 2
fi

LOG_ROOT="${LOG_ROOT:-logs/async}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOG_EVERY="${LOG_EVERY:-10}"
CKPT_EVERY="${CKPT_EVERY:-500}"
EVAL_EVERY="${EVAL_EVERY:-1000000}"
TORCH_COMPILE="${TORCH_COMPILE:-1}"
TRAIN_EPOCHS_TO_30K="${TRAIN_EPOCHS_TO_30K:-4}"
TARGET_TRAIN_STEPS="${TARGET_TRAIN_STEPS:-30000}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-8308}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
PLANNING_ACTION_SAMPLER="${PLANNING_ACTION_SAMPLER:-repeat}"
TIME_NUM_WORKERS="${TIME_NUM_WORKERS:-0}"
PLANNING_NUM_WORKERS="${PLANNING_NUM_WORKERS:-0}"

mkdir -p "$LOG_ROOT"

run_logged() {
  local name="$1"
  shift

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local log_path="${LOG_ROOT}/${name}_${timestamp}.out"

  echo "[$(date --iso-8601=seconds)] starting ${name}"
  echo "+ $*"
  "$@" 2>&1 | tee "$log_path"
  echo "[$(date --iso-8601=seconds)] finished ${name}; log=${log_path}"
}

stage_is_running() {
  local pattern="$1"
  pgrep -af "$pattern" \
    | grep -v "run_cdit_recon_64_pipeline.sh" \
    | grep -v "pgrep -af" \
    >/dev/null
}

wait_for_stage() {
  local pattern="$1"
  local label="$2"

  while stage_is_running "$pattern"; do
    echo "[$(date --iso-8601=seconds)] waiting for ${label} to finish..."
    sleep "$WAIT_INTERVAL_SECONDS"
  done
}

train_stage() {
  local name="$1"
  local config="$2"
  local epochs="$3"

  run_logged "$name" \
    torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" scripts/train.py \
      --config "$config" \
      --epochs "$epochs" \
      --log-every "$LOG_EVERY" \
      --ckpt-every "$CKPT_EVERY" \
      --eval-every "$EVAL_EVERY" \
      --torch-compile "$TORCH_COMPILE"
}

next_epochs_for_target() {
  local checkpoint="$1"

  python - "$checkpoint" "$TARGET_TRAIN_STEPS" "$TRAIN_STEPS_PER_EPOCH" <<'PY'
import math
import sys

import torch

checkpoint_path = sys.argv[1]
target_steps = int(sys.argv[2])
steps_per_epoch = int(sys.argv[3])

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
epoch = int(checkpoint.get("epoch", -1))
train_steps = int(checkpoint.get("train_steps", 0))
remaining_steps = max(0, target_steps - train_steps)
additional_epochs = max(1, math.ceil(remaining_steps / steps_per_epoch))
print(epoch + 1 + additional_epochs)
PY
}

train_until_checkpoint() {
  local label="$1"
  local checkpoint_dir="$2"
  local target_checkpoint="$3"
  local config="$4"

  local latest_checkpoint="${checkpoint_dir}/latest.pth.tar"
  while [ ! -f "$target_checkpoint" ]; do
    wait_for_stage "scripts/train.py --config ${config}" "${label} training"

    if [ -f "$target_checkpoint" ]; then
      break
    fi

    if [ ! -f "$latest_checkpoint" ]; then
      train_stage \
        "${label}_initial_to_30k" \
        "$config" \
        "$TRAIN_EPOCHS_TO_30K"
    else
      local next_epochs
      next_epochs="$(next_epochs_for_target "$latest_checkpoint")"
      train_stage \
        "${label}_resume_to_30k" \
        "$config" \
        "$next_epochs"
    fi
  done
}

planning_eval_name() {
  if [ "$PLANNING_ACTION_SAMPLER" = "sequence" ]; then
    echo "CEM_seq_N32_K5_RS1_rep1_OPT1"
  elif [ "$PLANNING_ACTION_SAMPLER" = "repeat" ]; then
    echo "CEM_repeat_N32_K5_RS1_rep1_OPT1"
  else
    echo "CEM_N32_K5_RS1_rep1_OPT1"
  fi
}

plan_stage() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  local output_dir="$4"
  local exp_name="$5"

  local eval_tag
  eval_tag="$(planning_eval_name)"
  local metric_json="${output_dir}/${exp_name}/recon_${eval_tag}.json"
  if [ -f "$metric_json" ]; then
    echo "[$(date --iso-8601=seconds)] existing planning result: ${metric_json}"
    return
  fi

  local action_args=()
  if [ "$PLANNING_ACTION_SAMPLER" != "legacy" ]; then
    action_args=(--action_sampler "$PLANNING_ACTION_SAMPLER")
  fi

  run_logged "$name" \
    torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" scripts/plan_eval.py \
      --exp "$config" \
      --ckp "$checkpoint" \
      --datasets recon \
      --rollout_stride 1 \
      --batch_size 1 \
      --num_samples 32 \
      --topk 5 \
      "${action_args[@]}" \
      --num_workers "$PLANNING_NUM_WORKERS" \
      --output_dir "$output_dir" \
      --opt_steps 1 \
      --num_repeat_eval 1
}

NO_TEXT_EXP="nwm_cdit_${MODEL_SIZE}_recon_64"
TEXT_EXP="nwm_cdit_${MODEL_SIZE}_recon_64_text_dense"
NO_TEXT_CONFIG="configs/experiment/${NO_TEXT_EXP}.yaml"
TEXT_CONFIG="configs/experiment/${TEXT_EXP}.yaml"
NO_TEXT_DIR="weights/checkpoints/${NO_TEXT_EXP}"
TEXT_DIR="weights/checkpoints/${TEXT_EXP}"
NO_TEXT_10K="${NO_TEXT_DIR}/0010000.pth.tar"
NO_TEXT_30K="${NO_TEXT_DIR}/0030000.pth.tar"
TEXT_30K="${TEXT_DIR}/0030000.pth.tar"

train_until_checkpoint \
  "$NO_TEXT_EXP" \
  "$NO_TEXT_DIR" \
  "$NO_TEXT_30K" \
  "$NO_TEXT_CONFIG"

if [ ! -f "$NO_TEXT_10K" ]; then
  echo "Missing required 10k no-text checkpoint for text warm-start: ${NO_TEXT_10K}" >&2
  exit 1
fi

train_until_checkpoint \
  "$TEXT_EXP" \
  "$TEXT_DIR" \
  "$TEXT_30K" \
  "$TEXT_CONFIG"

run_logged "nwm_cdit_${MODEL_SIZE}_recon_64_time_checkpoint_sweep" \
  python scripts/eval/run_time_checkpoint_sweep.py \
    --experiments "${NO_TEXT_EXP},${TEXT_EXP}" \
    --include_latest 1 \
    --num_workers "$TIME_NUM_WORKERS"

run_logged "nwm_cdit_${MODEL_SIZE}_recon_64_time_plot" \
  python scripts/analysis/plot_recon_time_over_checkpoints.py --model-size "$MODEL_SIZE"

plan_stage \
  "planning_recon64_${MODEL_SIZE}_notext_0030000_full_n32" \
  "$NO_TEXT_CONFIG" \
  "0030000" \
  "artifacts/bulk/planning/recon64_${MODEL_SIZE}_notext_0030000_full_n32" \
  "$NO_TEXT_EXP"

plan_stage \
  "planning_recon64_${MODEL_SIZE}_text_dense_0030000_full_n32" \
  "$TEXT_CONFIG" \
  "0030000" \
  "artifacts/bulk/planning/recon64_${MODEL_SIZE}_text_dense_0030000_full_n32" \
  "$TEXT_EXP"

run_logged "nwm_cdit_${MODEL_SIZE}_planning_64_summary" \
  python scripts/analysis/summarize_planning_text_comparison.py \
    --model-size "$MODEL_SIZE" \
    --action-sampler "$PLANNING_ACTION_SAMPLER" \
    --images 64
