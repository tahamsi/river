# Human-frozen release (2026-09-22)

The final RIVER results. Every reported number comes from this directory and
from the structured decisions it re-rendered
(`experiments/primary_run_2026-09-21/structured_results.json`); see
[`docs/results_provenance.md`](../../docs/results_provenance.md).

- Corpus: `data/grounded_human_frozen/`, 1,502 examples / 751 pairs after
  human adjudication, split 1,000 / 254 / 248 (train / dev / test).
- Evaluated split: test (248 examples).
- Generator: `qwen3.5:4b` via local Ollama 0.33.2, digest prefix `2a654d98e6fb`,
  temperature 0.
- Hardware: NVIDIA A100 80GB PCIe.
- Run window (UTC): 2026-09-22T20:22:26Z – 2026-09-22T21:01:13Z.
- Pipeline exit code: `0`; `provenance/release_audit.json` status `pass`.
- Tests before and after the run: 145 passed, 6 skipped, 3 xfailed.

## Contents

| Path | What it is |
| --- | --- |
| `tables/` | Result tables in CSV, JSON, Markdown and LaTeX |
| `manuscript/numbers.json` | Every table cell, machine-readable |
| `manuscript/provenance.json` | Each cell mapped to prediction files, run ids, config and dataset hashes, and example ids |
| `provenance/release_audit.json` | Release audit: prediction contract, decision faithfulness, MiniCheck pins, RIVER-L seed summary |
| `provenance/review_note_resolution.json` | Signed addendum binding the reviewer and workbook hash to every flagged corpus row |
| `provenance/code_commit.txt`, `start_utc.txt`, `finish_utc.txt`, `pipeline.exit_code` | Run provenance |
| `logs/` | Full pipeline, test, scoring, MiniCheck, table and audit logs |
| `SHA256SUMS` | Checksums of every file in the original release directory (70 files) |
| `RIVER_human_frozen_release_20260922.tar.xz` | The complete release directory, including all rendered, scored and MiniCheck prediction files |

The tables, manuscript, provenance and logs here are extracted from the archive
unchanged. `SHA256SUMS` records them under their original paths inside the archive, where
`tables/` and `manuscript/` were under `tables/tables/` and
`tables/manuscript/`.

## Verify

Run `uv run python scripts/benchmark/rehydrate_pat.py` from the repository root
first; until then the archives hold placeholders for PAT-Questions text and the
checksums below fail.

```bash
cd results/human_frozen
sha256sum -c RIVER_human_frozen_release_20260922.tar.xz.sha256
mkdir -p /tmp/river_release
tar -xJf RIVER_human_frozen_release_20260922.tar.xz -C /tmp/river_release
cd /tmp/river_release && sha256sum -c "$OLDPWD/SHA256SUMS"
```

## Headline numbers

RIVER-L over five seeds (11, 42, 101, 202, 314), from `release_audit.json`:

- Joint revision accuracy: 0.7790 (sample SD 0.0066)
- Strict answer accuracy: 0.8013 (sample SD 0.0165)

Table 1 (`tables/table_01_overall_controlled_baseline_comparison.md`) gives the
controlled baseline comparison with seed-clustered bootstrap intervals
(10,000 resamples, seed 13).

## Code

`provenance/code_commit.txt` records the development commit the release was
produced from. That commit is in the private development history, not in this
repository. The release history, including the known decision-faithfulness
caveat (16 of 1,428 active answers do not literally match the selected
hypothesis), is in
[`docs/release/RELEASE_NOTES.md`](../../docs/release/RELEASE_NOTES.md),
and the full pipeline is described in
[`docs/reproducing.md`](../../docs/reproducing.md#4-full-release-pipeline).
