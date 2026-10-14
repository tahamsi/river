# Release notes

## 2026-09-22

Results: [`results/human_frozen/`](../../results/human_frozen/).
Built from development commit `5af87ec`
(`results/human_frozen/provenance/code_commit.txt`). That commit belongs to the
private development history and is not part of this repository. Since then the
scripts have moved into subdirectories of `scripts/` (see
[Reproducing](../reproducing.md#4-full-release-pipeline)).

This release reuses the structured decisions of the primary run (2026-09-21,
[`experiments/primary_run_2026-09-21/`](../../experiments/primary_run_2026-09-21/))
unchanged and reruns the downstream steps. The release audit checks that each
release prediction file has the same structured digest as its source.

### Changes from the primary run

- Rendering: for an `active` decision, the generator receives only the selected
  hypothesis and its evidence. Non-active decisions return a fixed reply
  without a generator call.
- Answer scoring: an answer must begin with `FINAL_ANSWER: <answer>` and match
  the gold answer or an alias after normalisation. The primary run scored structured fields.
- MiniCheck: package commit `b58b9fa69acbd1015ec970fa65dd752413a053d2`, model
  `lytang/MiniCheck-Flan-T5-Large` at revision
  `96eafd01cee2d16cf81aaa2fb226b14f422a37b3`, loaded from a local snapshot with
  recorded weight hashes.
- Confidence intervals: bootstrap resampling clustered by test pair, so the
  five RIVER-L seeds are not counted as independent rows.
- Release audit added before table generation.

### Fixes during the release

These affect reporting only. No corpus row, gold label, structured prediction
or model weight changed.

- `usr` and `estimated_cost` are marked not applicable for MiniCheck's
  `--no-generation` run.
- Decision faithfulness is reported as a rate rather than used as a release
  gate.

### Known issues

- 16 of 1,428 active answers (1.1%) do not literally match the selected
  hypothesis. Rerunning these rows with the same settings (temperature 0, same
  seed) gave identical output. They are listed under `decision_faithfulness` in
  `provenance/release_audit.json` and are scored as errors.

### Build history

| Attempt | Date | Exit code | Result |
| --- | --- | --- | --- |
| 1 | 2026-09-21 | 2 | Table step failed on `usr` / `estimated_cost` for MiniCheck |
| 2 | 2026-09-22 | 2 | Audit failed on decision faithfulness |
| 3 | 2026-09-22 | 0 | Released |

### Environment

NVIDIA A100 80GB PCIe; Ollama 0.33.2 serving `qwen3.5:4b` (digest prefix
`2a654d98e6fb`, checked by the release script). To rerun, see
[`reproducing.md`](../reproducing.md#4-full-release-pipeline).
