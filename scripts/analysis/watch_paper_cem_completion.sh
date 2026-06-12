#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INTERVAL_SECONDS="${PAPER_CEM_WATCH_INTERVAL_SECONDS:-600}"

EXPECTED_FILES=(
  "artifacts/bulk/planning_paper_cem/recon224_b_notext_0100000_full_n120_rep3/nwm_cdit_b/recon_CEM_repeat_N120_K5_RS1_rep3_OPT1.json"
  "artifacts/bulk/planning_paper_cem/recon224_b_text_dense_0030000_full_n120_rep3/nwm_cdit_b_recon_raw_text_dense/recon_CEM_repeat_N120_K5_RS1_rep3_OPT1.json"
  "artifacts/bulk/planning_paper_cem/recon128_b_text_dense_0030000_full_n120_rep3/nwm_cdit_b_recon_128_text_dense/recon_CEM_repeat_N120_K5_RS1_rep3_OPT1.json"
)

echo "[$(date --iso-8601=seconds)] watching for paper-style CEM results"
printf '  %s\n' "${EXPECTED_FILES[@]}"

last_done_count=-1

while true; do
  missing=()
  for path in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "$path" ]; then
      missing+=("$path")
    fi
  done

  done_count=$((${#EXPECTED_FILES[@]} - ${#missing[@]}))
  if [ "$done_count" -ne "$last_done_count" ]; then
    echo "[$(date --iso-8601=seconds)] detected ${done_count}/${#EXPECTED_FILES[@]} completed result(s); refreshing summary and deck"
    python scripts/analysis/summarize_paper_cem_planning.py
    python scripts/analysis/build_nwm_text_conditioned_deck.py
    python scripts/analysis/verify_paper_cem_planning_outputs.py
    last_done_count="$done_count"
  fi

  if [ "${#missing[@]}" -eq 0 ]; then
    break
  fi

  echo "[$(date --iso-8601=seconds)] waiting; missing ${#missing[@]} result(s)"
  sleep "$INTERVAL_SECONDS"
done

echo "[$(date --iso-8601=seconds)] paper-style CEM post-processing complete"
