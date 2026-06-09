#!/usr/bin/env bash

set -euo pipefail

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
EXTERNAL_LOCK_STALE_SECONDS="${EXTERNAL_LOCK_STALE_SECONDS:-900}"
PLANNING_ACTION_SAMPLER="${PLANNING_ACTION_SAMPLER:-repeat}"
PLANNING_NUM_WORKERS="${PLANNING_NUM_WORKERS:-12}"

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
    | grep -v "run_cdit_b_recon_pipeline.sh" \
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

lock_value() {
  local lock_path="$1"
  local key="$2"

  awk -F= -v key="$key" '$1 == key { print substr($0, length(key) + 2) }' "$lock_path"
}

wait_for_external_lock() {
  local lock_path="$1"
  local target_checkpoint="$2"
  local label="$3"

  while [ -f "$lock_path" ] && [ ! -f "$target_checkpoint" ]; do
    local log_path
    log_path="$(lock_value "$lock_path" "log" || true)"

    if [ -n "$log_path" ] && [ -f "$log_path" ]; then
      if grep -Eq "Done!|ChildFailedError|Traceback|RuntimeError|Exception" "$log_path"; then
        echo "[$(date --iso-8601=seconds)] external ${label} log is no longer active: ${log_path}"
        break
      fi

      local now
      local mtime
      now="$(date +%s)"
      mtime="$(stat -c %Y "$log_path")"
      if [ $((now - mtime)) -gt "$EXTERNAL_LOCK_STALE_SECONDS" ]; then
        echo "[$(date --iso-8601=seconds)] external ${label} log is stale: ${log_path}"
        break
      fi
    fi

    echo "[$(date --iso-8601=seconds)] waiting for external ${label}: ${lock_path}"
    sleep "$WAIT_INTERVAL_SECONDS"
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

NO_TEXT_10K="weights/checkpoints/nwm_cdit_b_recon_128/0010000.pth.tar"
NO_TEXT_30K="weights/checkpoints/nwm_cdit_b_recon_128/0030000.pth.tar"
TEXT_30K="weights/checkpoints/nwm_cdit_b_recon_128_text_dense/0030000.pth.tar"
RAW_TEXT_30K="weights/checkpoints/nwm_cdit_b_recon_raw_text_dense/0030000.pth.tar"
RAW_TEXT_EXTERNAL_LOCK="${RAW_TEXT_EXTERNAL_LOCK:-${LOG_ROOT}/nwm_cdit_b_recon_raw_text_dense.external.lock}"

if [ ! -f "$NO_TEXT_30K" ]; then
  train_until_checkpoint \
    "nwm_cdit_b_recon_128" \
    "weights/checkpoints/nwm_cdit_b_recon_128" \
    "$NO_TEXT_30K" \
    "configs/experiment/nwm_cdit_b_recon_128.yaml"
fi

if [ ! -f "$NO_TEXT_10K" ]; then
  echo "Missing required 10k no-text checkpoint: ${NO_TEXT_10K}" >&2
  exit 1
fi

if [ ! -f "$TEXT_30K" ]; then
  train_until_checkpoint \
    "nwm_cdit_b_recon_128_text_dense" \
    "weights/checkpoints/nwm_cdit_b_recon_128_text_dense" \
    "$TEXT_30K" \
    "configs/experiment/nwm_cdit_b_recon_128_text_dense.yaml"
fi

if [ ! -f "$RAW_TEXT_30K" ]; then
  wait_for_stage "scripts/train.py --config configs/experiment/nwm_cdit_b_recon_raw_text_dense.yaml" "nwm_cdit_b_recon_raw_text_dense training"
  wait_for_external_lock "$RAW_TEXT_EXTERNAL_LOCK" "$RAW_TEXT_30K" "nwm_cdit_b_recon_raw_text_dense training"

  if [ ! -f "$RAW_TEXT_30K" ]; then
    train_until_checkpoint \
      "nwm_cdit_b_recon_raw_text_dense" \
      "weights/checkpoints/nwm_cdit_b_recon_raw_text_dense" \
      "$RAW_TEXT_30K" \
      "configs/experiment/nwm_cdit_b_recon_raw_text_dense.yaml"
  fi
fi

run_logged "nwm_cdit_b_time_checkpoint_sweep" \
  python scripts/eval/run_time_checkpoint_sweep.py \
    --experiments nwm_cdit_b,nwm_cdit_b_recon_128,nwm_cdit_b_recon_128_text_dense,nwm_cdit_b_recon_raw_text_dense \
    --include_latest 1

run_logged "nwm_cdit_b_recon_time_plot" \
  python scripts/analysis/plot_recon_time_over_checkpoints.py --model-size b

plan_stage \
  "planning_recon128_b_notext_0030000_full_n32" \
  "configs/experiment/nwm_cdit_b_recon_128.yaml" \
  "0030000" \
  "artifacts/bulk/planning/recon128_b_notext_0030000_full_n32" \
  "nwm_cdit_b_recon_128"

plan_stage \
  "planning_recon128_b_text_dense_0030000_full_n32" \
  "configs/experiment/nwm_cdit_b_recon_128_text_dense.yaml" \
  "0030000" \
  "artifacts/bulk/planning/recon128_b_text_dense_0030000_full_n32" \
  "nwm_cdit_b_recon_128_text_dense"

plan_stage \
  "planning_recon224_b_notext_0100000_full_n32" \
  "configs/experiment/nwm_cdit_b.yaml" \
  "0100000" \
  "artifacts/bulk/planning/recon224_b_notext_0100000_full_n32" \
  "nwm_cdit_b"

plan_stage \
  "planning_recon224_b_text_dense_0030000_full_n32" \
  "configs/experiment/nwm_cdit_b_recon_raw_text_dense.yaml" \
  "0030000" \
  "artifacts/bulk/planning/recon224_b_text_dense_0030000_full_n32" \
  "nwm_cdit_b_recon_raw_text_dense"

run_logged "nwm_cdit_b_planning_summary" \
  python scripts/analysis/summarize_planning_text_comparison.py \
    --model-size b \
    --action-sampler "$PLANNING_ACTION_SAMPLER"
