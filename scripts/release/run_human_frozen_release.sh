#!/usr/bin/env bash
set -Eeuo pipefail

# Release pipeline. It reuses the frozen structured decisions and reruns only
# rendering, strict scoring, pinned MiniCheck, the audit and the tables.
# Run from a clean committed repository root.

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

: "${ANNOTATION_WORKBOOK:?Set ANNOTATION_WORKBOOK to the final 1,780-row workbook path}"
: "${REVIEW_APPROVER:?Set REVIEW_APPROVER to the person who verified the stale-note resolution}"

fixture="${RIVER_FIXTURE:-data/grounded_human_frozen/river_bench_test.jsonl}"
train_fixture="${RIVER_TRAIN_FIXTURE:-data/grounded_human_frozen/river_bench_train.jsonl}"
dev_fixture="${RIVER_DEV_FIXTURE:-data/grounded_human_frozen/river_bench_dev.jsonl}"
combined_fixture="${RIVER_COMBINED_FIXTURE:-data/grounded_human_frozen/river_bench_1502.jsonl}"
adjudication_ledger="${RIVER_ADJUDICATION_LEDGER:-data/grounded_human_frozen/adjudication_ledger.jsonl}"
corpus_manifest="${RIVER_CORPUS_MANIFEST:-data/grounded_human_frozen/manifest.json}"
source_predictions="${SOURCE_PREDICTIONS:-artifacts/predictions_human_frozen}"
model="${RIVER_GENERATOR_MODEL:-qwen3.5:4b}"
generator_name="${RIVER_GENERATOR_NAME:-ollama/qwen3.5:4b}"
base_url="${RIVER_GENERATOR_BASE_URL:-http://127.0.0.1:11434/v1}"
expected_model_digest="${RIVER_GENERATOR_DIGEST:-2a654d98e6fb}"
release_root="${1:-artifacts/release_human_frozen}"

if [[ -e "$release_root" ]]; then
  echo "Refusing to overwrite existing release directory: $release_root" >&2
  exit 2
fi
mkdir -p "$release_root"/{logs,predictions_rendered,predictions_scored,predictions_minicheck,tables,provenance}

finish() {
  code=$?
  printf '%s\n' "$code" > "$release_root/provenance/pipeline.exit_code"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$release_root/provenance/finish_utc.txt"
}
trap finish EXIT
date -u +%Y-%m-%dT%H:%M:%SZ > "$release_root/provenance/start_utc.txt"
exec > >(tee "$release_root/logs/release_pipeline.log") 2>&1

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Release requires a clean committed repository." >&2
  git status --short >&2
  exit 2
fi
git rev-parse HEAD > "$release_root/provenance/code_commit.txt"

for required in \
  "$fixture" \
  "$train_fixture" \
  "$dev_fixture" \
  "$combined_fixture" \
  "$adjudication_ledger" \
  "$corpus_manifest" \
  "$ANNOTATION_WORKBOOK"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 2; }
done

check_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

# These are the hashes recorded by the immutable human-frozen manifest used by
# the 2026-09-21 run. Do not silently rebuild or substitute any of these inputs.
check_sha256 "$ANNOTATION_WORKBOOK" "4fcbf5843411a9301b85010bdcd3088d055083a63404e899c8b4b4701aa15b23"
check_sha256 "$adjudication_ledger" "ae1c14e77264012802a477276c608790073fb2d8794c5d7601d16cc34fa0290d"
check_sha256 "$combined_fixture" "52ff3ffc0a3f0beac41a1f1fb2100c2f27fc1a3a380463deaac44376b84993ea"
check_sha256 "$dev_fixture" "a8627d785a0878855d59cc117d8f2cecdb0de11c6f3b614bc9c849b569c177f5"
check_sha256 "$fixture" "ac42915c30444ec90bbad1c84555c4086a99dbc3f2688219183d91e27eb1666d"
check_sha256 "$train_fixture" "90c80e3d3f83966462802bd67bfa0e27c574be32f03b851fb704163b40f8d18a"
check_sha256 "$corpus_manifest" "61e23614d2bec417bfd25ae0c8322ae14dc19e670fb810431096efe93f3d1eca"

declare -A source_prediction_hashes=(
  [last_write_wins_test_seed42.jsonl]="3629748eace82bbe1752a774fa2af451814166da3c82ef56b87e1b233ffae7a3"
  [naive_rag_test_seed42.jsonl]="932702596435b19c4a4108794d47f83bd1d513ec3a14ec02131c48db0330e05f"
  [recency_weighted_test_seed42.jsonl]="58aeaee77029f3f817e30db38807a12d2a6d4455e4882a0cf39f97975c17b328"
  [river_d_test_seed42.jsonl]="55096a4abd6c592b50e01c66d8f30c3c29737e9d06ca880ac0d5dcee93d1c36c"
  [river_l_test_seed101.jsonl]="6bd09de1d4aa0ce10f203263879c03a8cc18e8776c91bf12d899a2ca08e6ef1b"
  [river_l_test_seed11.jsonl]="cee255c53c46b75deb18e4c5f4f24d1388ebe29b2172e6ff06aefbd53e8a118e"
  [river_l_test_seed202.jsonl]="11badaca00a21d793c1b7ac4d1903e4c1acc8adaa070fb2c71692c77794a2923"
  [river_l_test_seed314.jsonl]="672e0617790def68a624af8a2c1769b3d01ace6aa14e7075455210e1fdc1375b"
  [river_l_test_seed42.jsonl]="952c1a73c6eb2001b938da18e16defc48c7ea5f390ec7fa790fa69946f18d4a3"
  [stateless_rag_test_seed42.jsonl]="5003f7a6f91b8755bcec75f8e19ca040155591c355780c162ea50b2666dfbc52"
)
for filename in "${!source_prediction_hashes[@]}"; do
  source_file="$source_predictions/$filename"
  [[ -f "$source_file" ]] || { echo "Missing source prediction: $source_file" >&2; exit 2; }
  check_sha256 "$source_file" "${source_prediction_hashes[$filename]}"
done

actual_digest="$(ollama list | awk -v model="$model" '$1 == model {print $2}')"
if [[ -z "$actual_digest" || "$actual_digest" != "$expected_model_digest"* ]]; then
  echo "Generator digest mismatch: expected prefix $expected_model_digest, got ${actual_digest:-missing}" >&2
  exit 2
fi

uv sync --frozen --extra test --extra learned --extra minicheck
uv_run=(uv run --no-sync)
"${uv_run[@]}" pytest -q | tee "$release_root/logs/pytest_pre_release.log"

"${uv_run[@]}" python scripts/release/build_review_note_resolution.py \
  --fixture "$train_fixture" "$dev_fixture" "$fixture" \
  --annotation-workbook "$ANNOTATION_WORKBOOK" \
  --approved-by "$REVIEW_APPROVER" \
  --output "$release_root/provenance/review_note_resolution.json"

shopt -s nullglob
primary_files=("$source_predictions"/*.jsonl)
if [[ ${#primary_files[@]} -ne 10 ]]; then
  echo "Expected exactly 10 primary prediction files, found ${#primary_files[@]}" >&2
  exit 2
fi

for source in "${primary_files[@]}"; do
  destination="$release_root/predictions_rendered/$(basename "$source")"
  "${uv_run[@]}" python scripts/release/rerender_from_predictions.py \
    --input "$source" \
    --fixture "$fixture" \
    --output "$destination" \
    --model "$model" \
    --generator-name "$generator_name" \
    --base-url "$base_url" \
    --temperature 0 \
    --max-output-tokens 256 \
    --timeout 300 \
    --expected-examples 248
done

rendered_files=("$release_root/predictions_rendered"/*.jsonl)
"${uv_run[@]}" python scripts/release/score_answer_text.py \
  --fixture "$fixture" \
  --output-dir "$release_root/predictions_scored" \
  "${rendered_files[@]}" | tee "$release_root/logs/strict_answer_scoring.log"

# This wrapper loads an explicit local snapshot path and records the resolved
# revision plus every weight-file SHA-256.
"${uv_run[@]}" python -m experiments.run \
  --baseline minicheck_no_revision \
  --fixture "$fixture" \
  --output "$release_root/predictions_minicheck/minicheck_no_revision_test_seed42.jsonl" \
  --seed 42 \
  --generator "$generator_name" \
  --config configs/baselines/minicheck_no_revision.json \
  --no-generation | tee "$release_root/logs/minicheck.log"

table_args=(
  --table overall_controlled_baseline_comparison
  --table revision_state_comparison
  --table results_by_intervention
  --table paired_recency_decomposition
  --table river_d_vs_river_l
  --table conflict_detection_negative_controls
  --table conflict_versus_temporal_evolution
  --table nei_selective_abstention
  --table external_dataset_generalization
  --table statistical_comparison_strongest_baseline
  --table computational_cost
)
"${uv_run[@]}" python -m analysis.generate_tables \
  "$release_root/predictions_scored" \
  "$release_root/predictions_minicheck" \
  --output "$release_root/tables" \
  --no-figures \
  --bootstrap-resamples 10000 \
  --seed 13 \
  "${table_args[@]}" | tee "$release_root/logs/generate_tables.log"

"${uv_run[@]}" python scripts/release/audit_release_results.py \
  --fixture "$fixture" \
  --source-dir "$source_predictions" \
  --primary-dir "$release_root/predictions_scored" \
  --minicheck-dir "$release_root/predictions_minicheck" \
  --review-note-resolution "$release_root/provenance/review_note_resolution.json" \
  --expected-examples 248 \
  --output "$release_root/provenance/release_audit.json" | tee "$release_root/logs/release_audit.log"

"${uv_run[@]}" pytest -q | tee "$release_root/logs/pytest_post_release.log"
# Hash immutable scientific outputs. Live logs and EXIT-trap timestamps are
# intentionally excluded because they can still receive buffered bytes.
find \
  "$release_root/predictions_rendered" \
  "$release_root/predictions_scored" \
  "$release_root/predictions_minicheck" \
  "$release_root/tables" \
  "$release_root/provenance/review_note_resolution.json" \
  "$release_root/provenance/release_audit.json" \
  "$release_root/provenance/code_commit.txt" \
  -type f -print0 | sort -z | xargs -0 sha256sum > "$release_root/SHA256SUMS"

echo "Release pipeline completed successfully: $release_root"
