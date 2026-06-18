#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CONTAINER_NAME="${NWM_CONTAINER_NAME:-nwm_dev}"
CONTAINER_WORKDIR="${NWM_CONTAINER_WORKDIR:-/workspace/nwm}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
WEIGHTS_ROOT="${NWM_WEIGHTS_DIR:-/workspace/nwm/weights}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${WEIGHTS_ROOT}/pretrained/Qwen2-VL-7B-Instruct}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-openai/clip-vit-base-patch32}"
PROMPT_MODE="${PROMPT_MODE:-nav_prediction}"
CAPTION_BATCH_SIZE="${CAPTION_BATCH_SIZE:-1}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
DTYPE="${DTYPE:-bfloat16}"
EMBED_DTYPE="${EMBED_DTYPE:-float16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
MAX_LENGTH="${MAX_LENGTH:-77}"
TEST_SHARDS="${TEST_SHARDS:-16}"
TRAIN_SHARDS="${TRAIN_SHARDS:-64}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/workspace/nwm/datasets/recon_1fps_test}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/workspace/nwm/datasets/recon_1fps_train}"
TEST_MANIFEST="${TEST_MANIFEST:-/workspace/nwm/artifacts/summaries/preprocess/phase1/recon_test_1fps_manifest.jsonl}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/workspace/nwm/artifacts/summaries/preprocess/phase1/recon_train_1fps_manifest.jsonl}"
DENSE_TEST_ROOT="${DENSE_TEST_ROOT:-/workspace/nwm/datasets/recon_raw/recon_release}"
DENSE_TRAIN_ROOT="${DENSE_TRAIN_ROOT:-/workspace/nwm/datasets/recon_raw/recon_release}"
TEST_TRAJ_NAMES="${TEST_TRAJ_NAMES:-/workspace/nwm/data/splits/recon/test/traj_names.txt}"
TRAIN_TRAJ_NAMES="${TRAIN_TRAJ_NAMES:-/workspace/nwm/data/splits/recon/train/traj_names.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/nwm/datasets/derived/phase2_nav_pred_qwen_clip}"

CAPTION_ROOT="$OUTPUT_ROOT/captions"
CLEAN_ROOT="$OUTPUT_ROOT/captions_clean"
SPARSE_EMBED_ROOT="$OUTPUT_ROOT/text_embeds_sparse"
DENSE_EMBED_ROOT="$OUTPUT_ROOT/text_embeds_dense"

run_in_container() {
  local command="$1"
  local cache_root="${NWM_CACHE_DIR:-${WEIGHTS_ROOT}/cache}"
  if [[ "${NWM_IN_CONTAINER:-0}" == "1" ]]; then
    mkdir -p "$cache_root/torch" "$cache_root/xdg" "$cache_root/huggingface"
    export CONDA_PREFIX=/opt/micromamba/envs/nwm
    export PATH="$CONDA_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export TORCH_HOME="$cache_root/torch"
    export XDG_CACHE_HOME="$cache_root/xdg"
    export HF_HOME="$cache_root/huggingface"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    bash -lc "$command"
    return
  fi

  docker exec -w "$CONTAINER_WORKDIR" "$CONTAINER_NAME" bash -lc \
    "mkdir -p '$cache_root/torch' '$cache_root/xdg' '$cache_root/huggingface'; \
     export CONDA_PREFIX=/opt/micromamba/envs/nwm; \
     export PATH=\$CONDA_PREFIX/bin:\$PATH; \
     export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; \
     export TORCH_HOME='$cache_root/torch'; \
     export XDG_CACHE_HOME='$cache_root/xdg'; \
     export HF_HOME='$cache_root/huggingface'; \
     export HUGGINGFACE_HUB_CACHE=\"\$HF_HOME/hub\"; \
     export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"; \
     $command"
}

run_qwen_split() {
  local split="$1"
  local manifest="$2"
  local data_root="$3"
  local num_shards="$4"
  local split_root="$CAPTION_ROOT/$split"
  local shard_root="$split_root/shards"
  local merged_output="$split_root/all.jsonl"

  mkdir -p "$shard_root"

  for ((shard_index=0; shard_index<num_shards; shard_index++)); do
    local shard_output
    shard_output="$(printf "%s/%03d.jsonl" "$shard_root" "$shard_index")"
    if [[ -s "$shard_output" ]]; then
      echo "[$split] skip qwen shard $shard_index -> $shard_output"
      continue
    fi

    echo "[$split] start qwen shard $shard_index/$((num_shards-1))"
    run_in_container "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python scripts/preprocess/text/generate_qwen_captions.py \
      --manifest $manifest \
      --data-root $data_root \
      --output $shard_output \
      --model-name-or-path $QWEN_MODEL_PATH \
      --prompt-mode $PROMPT_MODE \
      --batch-size $CAPTION_BATCH_SIZE \
      --max-new-tokens $MAX_NEW_TOKENS \
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
os.makedirs(os.path.dirname(merged_output), exist_ok=True)
with open(merged_output, "w", encoding="utf-8") as dst:
    for path in paths:
        with open(path, "r", encoding="utf-8") as src:
            for line in src:
                dst.write(line)
print(f"merged {len(paths)} shards -> {merged_output}")
PY
}

clean_split() {
  local split="$1"
  local input="$CAPTION_ROOT/$split/all.jsonl"
  local output="$CLEAN_ROOT/${split}_clean.jsonl"
  mkdir -p "$CLEAN_ROOT"
  run_in_container "python scripts/preprocess/text/clean_captions.py \
    --input $input \
    --output $output \
    --format structured"
}

embed_split() {
  local split="$1"
  local clean_jsonl="$CLEAN_ROOT/${split}_clean.jsonl"
  local output_root="$SPARSE_EMBED_ROOT/$split"
  run_in_container "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python scripts/preprocess/text/precompute_text_embeddings.py \
    --input $clean_jsonl \
    --output-root $output_root \
    --model-name-or-path $CLIP_MODEL_PATH \
    --batch-size $EMBED_BATCH_SIZE \
    --max-length $MAX_LENGTH \
    --dtype $EMBED_DTYPE \
    --local-files-only $LOCAL_FILES_ONLY"
}

align_split() {
  local split="$1"
  local dense_data_root="$2"
  local traj_names="$3"
  local output_root="$4"
  run_in_container "python scripts/preprocess/text/align_text_embeddings_to_dense.py \
    --dense-data-root $dense_data_root \
    --traj-names $traj_names \
    --sparse-embedding-root $SPARSE_EMBED_ROOT/$split \
    --output-root $output_root \
    --source-frame-stride 4 \
    --mode ffill"
}

run_qwen_split "recon_test_1fps" "$TEST_MANIFEST" "$TEST_DATA_ROOT" "$TEST_SHARDS"
run_qwen_split "recon_train_1fps" "$TRAIN_MANIFEST" "$TRAIN_DATA_ROOT" "$TRAIN_SHARDS"

clean_split "recon_test_1fps"
clean_split "recon_train_1fps"

embed_split "recon_test_1fps"
embed_split "recon_train_1fps"

align_split "recon_test_1fps" "$DENSE_TEST_ROOT" "$TEST_TRAJ_NAMES" "$DENSE_EMBED_ROOT/recon_all_raw_rel"
align_split "recon_train_1fps" "$DENSE_TRAIN_ROOT" "$TRAIN_TRAJ_NAMES" "$DENSE_EMBED_ROOT/recon_all_raw_rel"
