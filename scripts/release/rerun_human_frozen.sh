#!/usr/bin/env bash
# Retrain and evaluate the controlled suite on the human-adjudicated freeze.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

DATA=${DATA:-data/grounded_human_frozen}
OUT=${OUT:-artifacts/predictions_human_frozen}
CKPT=${CKPT:-artifacts/river_l_human_frozen}
GENERATOR=${GENERATOR:-ollama/qwen3.5:4b}
LOCAL_BASE_URL=${LOCAL_BASE_URL:-http://127.0.0.1:11434/v1}
MODEL_TIMEOUT=${MODEL_TIMEOUT:-300}
DEVICE=${DEVICE:-cpu}
NO_GENERATION=${NO_GENERATION:-0}
SUMMARY=${SUMMARY:-artifacts/rerun/structured_results.json}
mkdir -p "$OUT" "$CKPT"

log() { echo "[$(date -Is)] $*"; }

for fixture in "$DATA"/river_bench_{train,dev,test}.jsonl; do
  test -s "$fixture" || { echo "Missing frozen fixture: $fixture" >&2; exit 2; }
done

log "training RIVER-L on frozen train/dev splits"
log "fitting RIVER-D thresholds on the frozen dev split"
uv run python scripts/release/tune_river_d.py \
  --dev "$DATA/river_bench_dev.jsonl" \
  --base-config configs/baselines/river_d.json \
  --output "$OUT/river_d_frozen.json"

for seed in 11 42 101 202 314; do
  uv run --extra learned river-l-train \
    --train "$DATA/river_bench_train.jsonl" \
    --validation "$DATA/river_bench_dev.jsonl" \
    --config configs/learned/river_l_full.json \
    --output "$CKPT/river_l_seed${seed}.pt" \
    --manifest "$CKPT/river_l_seed${seed}.manifest.json" \
    --model-card "$CKPT/river_l_seed${seed}.model-card.md" \
    --seed "$seed" --device "$DEVICE"
  uv run --extra learned river-l-validate \
    --checkpoint "$CKPT/river_l_seed${seed}.pt" \
    --fixture "$DATA/river_bench_dev.jsonl" \
    --output "$CKPT/river_l_seed${seed}.validation.json"
  printf '{"checkpoint":"%s/river_l_seed%s.pt","device":"%s"}\n' \
    "$CKPT" "$seed" "$DEVICE" > "$CKPT/river_l_seed${seed}.json"
done

run() {
  local name=$1 config=${2:-} seed=${3:-42}
  local output="$OUT/${name}_test_seed${seed}.jsonl"
  local args=(--baseline "$name" --fixture "$DATA/river_bench_test.jsonl"
    --generator "$GENERATOR" --local-base-url "$LOCAL_BASE_URL"
    --model-timeout "$MODEL_TIMEOUT" --seed "$seed" --output "$output")
  test -z "$config" || args+=(--config "$config")
  test "$NO_GENERATION" = 0 || args+=(--no-generation)
  log "running $name seed=$seed"
  uv run python -m experiments.run "${args[@]}"
}

run naive_rag
run stateless_rag
run last_write_wins
run recency_weighted configs/baselines/recency_weighted.json
run river_d "$OUT/river_d_frozen.json"
for seed in 11 42 101 202 314; do
  run river_l "$CKPT/river_l_seed${seed}.json" "$seed"
done

if test "$NO_GENERATION" = 0; then
  log "scoring generated answers and generating result tables"
  uv run --extra test python scripts/release/score_answer_text.py "$OUT"/*_test_seed*.jsonl \
    --fixture "$DATA/river_bench_test.jsonl"
  uv run --extra test python -m analysis.generate_tables "$OUT" \
    --output experiments/sourced_tables_human_frozen --no-figures
else
  log "structured-only mode: skipping answer-text scoring and final-looking result tables"
fi
uv run python scripts/release/summarize_structured_rerun.py --predictions "$OUT" \
  --output "$SUMMARY"
log "human-frozen rerun complete"
