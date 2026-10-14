# Reproducing the results

There are four levels, from checking the released numbers to rerunning the
full pipeline. Levels 1–3 need no GPU and no language model.

| Level | What it shows | Needs | Time |
|---|---|---|---|
| 1. Verify the release | The released files are the ones the pipeline produced | `sha256sum`, `tar` | Seconds |
| 2. Regenerate the tables and endpoint statistics | Tables and reported numbers follow from the archived predictions | CPU | Seconds (statistics); ~25 min (tables) |
| 3. Rerun the decisions | Structured results follow from the corpus and code | CPU | ~4 min |
| 4. Full release pipeline | Rendering, strict scoring, MiniCheck and audit | GPU, Ollama | ~40 min on one A100 |

Setup for every level:

```bash
uv sync --extra test --extra learned
uv run python scripts/benchmark/rehydrate_pat.py
```

The second command restores the PAT-Questions text that the public release
leaves out ([`data/README.md`](../data/README.md#pat-questions-text)). Until it
has run, the corpus and the archives under `results/human_frozen/` hold
placeholders and their checksums do not match.

## 1. Verify the release

```bash
cd results/human_frozen
sha256sum -c RIVER_human_frozen_release_20260922.tar.xz.sha256
mkdir -p /tmp/river_release
tar -xJf RIVER_human_frozen_release_20260922.tar.xz -C /tmp/river_release
(cd /tmp/river_release && sha256sum -c "$OLDPWD/SHA256SUMS")   # 70 files: OK
```

The archive holds every rendered, scored and MiniCheck prediction file, the
tables, and the release audit. `SHA256SUMS` lists them under their original
paths inside the archive.

## 2. Regenerate the tables and endpoint statistics

The primary endpoints, intervals, confusion matrix and pair outcomes, in a few
seconds:

```bash
R=$(echo /tmp/river_release/artifacts/release_human_frozen_*)
uv run python scripts/release/summarize_release_endpoints.py --release-dir "$R" \
  --output artifacts/release_endpoints.json
```

[`results_provenance.md`](results_provenance.md) says which output field gives
which reported number.

The release tables, about 25 minutes on one core:

```bash
uv run --extra test python -m analysis.generate_tables \
  "$R/predictions_scored" "$R/predictions_minicheck" \
  --output artifacts/tables_check --no-figures \
  --bootstrap-resamples 10000 --seed 13 \
  --table overall_controlled_baseline_comparison --table revision_state_comparison \
  --table results_by_intervention --table paired_recency_decomposition \
  --table river_d_vs_river_l --table conflict_detection_negative_controls \
  --table conflict_versus_temporal_evolution --table nei_selective_abstention \
  --table external_dataset_generalization \
  --table statistical_comparison_strongest_baseline --table computational_cost
diff -r artifacts/tables_check/tables results/human_frozen/tables && echo identical
```

All 44 table files (CSV, JSON, Markdown and LaTeX) regenerate byte-identical.

## 3. Rerun the structured decisions on CPU

The quickest check fits RIVER-D on the dev split and runs it on the test split
(about 15 seconds):

```bash
uv run python scripts/release/tune_river_d.py \
  --dev data/grounded_human_frozen/river_bench_dev.jsonl \
  --output artifacts/river_d_frozen.json
uv run python -m experiments.run --baseline river_d \
  --fixture data/grounded_human_frozen/river_bench_test.jsonl \
  --config artifacts/river_d_frozen.json --no-generation \
  --output artifacts/quickstart/river_d_test_seed42.jsonl
uv run python scripts/release/summarize_structured_rerun.py \
  --predictions artifacts/quickstart --output artifacts/quickstart/summary.json
```

This gives state accuracy 0.536, active selection 0.631 and joint accuracy
0.536, the RIVER-D row of the results table in the README.

The full structured suite retrains RIVER-L for five seeds and runs every
controlled system:

```bash
NO_GENERATION=1 DEVICE=cpu \
  OUT=artifacts/rerun/predictions CKPT=artifacts/rerun/checkpoints \
  SUMMARY=artifacts/rerun/structured_results.json \
  bash scripts/release/rerun_human_frozen.sh
```

The deterministic systems reproduce the released results exactly. RIVER-L is
retrained, so its numbers vary with hardware: a CPU rerun gave joint accuracy
0.786 ± 0.019 against the released 0.779 ± 0.007, which was trained on GPU.
The exact released decisions are the archived predictions from level 1.

## 4. Full release pipeline

This reproduces the released directory end to end. It requires:

- a GPU (the release used one NVIDIA A100 80GB);
- Ollama serving `qwen3.5:4b` with digest prefix `2a654d98e6fb` (the script
  checks it; the release used Ollama 0.33.2);
- the source structured predictions:
  ```bash
  mkdir -p artifacts
  tar -xJf results/human_frozen/inputs/predictions_human_frozen.tar.xz -C artifacts
  sha256sum -c results/human_frozen/inputs/EXPECTED_SOURCE_PREDICTION_HASHES.sha256
  ```
- the human annotation workbook (SHA-256 recorded in
  `data/grounded_human_frozen/manifest.json`), which is **not distributed**,
  and the name of the person approving the review-note resolution.

```bash
ANNOTATION_WORKBOOK=/path/to/workbook.xlsx REVIEW_APPROVER="Name, role" \
  bash scripts/release/run_human_frozen_release.sh artifacts/release_rerun
```

The script refuses a dirty working tree and an existing output directory. It
re-renders answers from the frozen structured decisions, scores them, runs the
pinned MiniCheck model, audits the result and writes the tables. When the
release was made, these scripts lived directly under `scripts/`; see the
[release notes](release/RELEASE_NOTES.md).

## Generating the structured decisions with a generator

`scripts/release/rerun_human_frozen.sh` without `NO_GENERATION=1` also renders
answers with a local generator (default `ollama/qwen3.5:4b` at
`http://127.0.0.1:11434/v1`). Any OpenAI-compatible local server works through
`--local-model` and `--local-base-url`; `experiments/run.py` rejects
non-local URLs.

## Rules the pipeline enforces

1. Thresholds, checkpoints and early stopping use the dev split only;
   calibration fails on train or test rows.
2. Both variants of a pair stay in the same split.
3. `controlled` and `native` rows are never combined in one metric.
4. Missing labels stay missing (`null`/`NA`), never zero.
5. Tables are generated only from prediction records, and the generator refuses
   authored or synthetic data unless `--allow-engineering-data` is passed (for
   smoke tests only).
