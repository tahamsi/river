# Results provenance

Where each reported number comes from. Release files are under
`results/human_frozen/`; every one is covered by `SHA256SUMS`.

`scripts/release/summarize_release_endpoints.py` recomputes the endpoints,
intervals, confusion counts and pair outcomes below from the release
predictions in a few seconds on CPU (see
[`reproducing.md`](reproducing.md#2-regenerate-the-tables-and-endpoint-statistics)).
Point estimates, standard deviations, confusion counts and pair outcomes are
deterministic. Bootstrap bounds are Monte Carlo estimates and move in the third
decimal place with the random seed.

## Two denominators

The primary endpoints (the results table in the README) use **all 248 test
examples**, or the **157 active examples** for active selection and strict
answer accuracy.

The release tables (`tables/table_*`) aggregate over **246 rows**: they exclude
the two `retroactive_correction` test rows, which are pre-registered as
qualitative-only (`QUALITATIVE_ONLY_INTERVENTIONS` in
`river/evaluation/metrics.py`). Their `n` for RIVER-L is 5 × 246 = 1,230
row–seed decisions. Where a quantity appears in both, small differences come
from this denominator. For example, naive-RAG strict answer accuracy is 0.682
over the 157 active rows, and 0.686 in release Table 1, which scores
answer-or-revision accuracy over 246 rows.

The structured decisions behind every number were produced once, by the
2026-09-21 run, and kept byte-for-byte by the final release: the release audit
checks that the structured digest of each release prediction file equals that
of its source.

## Primary endpoints

| Quantity | Source |
|---|---|
| State accuracy, active selection, joint accuracy (per system; RIVER-L mean ± SD over five seeds) | `summarize_release_endpoints.py` (`systems`); also `experiments/primary_run_2026-09-21/structured_results.json` |
| Strict answer accuracy | `summarize_release_endpoints.py` (`strict_answer_accuracy`); also `provenance/release_audit.json` (`primary_runs`, `river_l_seed_summary`) |
| Recency flip rate | `summarize_release_endpoints.py`: `rfr_pooled` pools `order_reversal` and `expired_evidence` pairs, as in release Table 1; `rfr_order_reversal` uses order-reversal pairs only, as in release Table 3. They differ only for naive and stateless RAG (0.432 and 0.095) |
| 95% intervals, including RIVER-L minus last-write-wins (joint) and RIVER-L minus naive RAG (strict answer) | `summarize_release_endpoints.py` (`intervals`): 10,000 draws resampling test pairs with replacement and RIVER-L seeds within each draw |
| Joint accuracy by intervention | `experiments/primary_run_2026-09-21/structured_results.json` (`by_intervention_joint_accuracy`; RIVER-L pooled over seeds) |
| Order-reversal pair outcomes (29 pairs) | `summarize_release_endpoints.py` (`order_reversal_pairs`); also `tables/table_03_paired_recency_decomposition.*` |
| RIVER-L state confusion matrix and per-state recall | `summarize_release_endpoints.py` (`river_l_confusion_pooled_over_seeds`, `river_l_state_recall`), pooled over five seeds |

## Diagnostics

| Quantity | Source |
|---|---|
| Conflict macro-F1, abstention F1 | `tables/table_01_revision_state_comparison.*` |
| Expected calibration error | `tables/table_06_river_d_vs_river_l.*` (`ece`) |
| Insufficient-evidence abstention F1, accuracy at 80% coverage | `tables/table_11_nei_selective_abstention.*` |
| Results by intervention with coverage | `tables/table_02_results_by_intervention.*` |
| Conflict negative controls; conflict versus evolution | `tables/table_09_*`, `tables/table_10_*` |
| Results by source dataset | `tables/table_12_external_dataset_generalization.*` |
| Paired tests against the strongest baseline | `tables/table_14_statistical_comparison_strongest_baseline.*` |
| Latency, model calls and cost | `tables/table_16_computational_cost.*` |
| Rendering audit: 1,412 of 1,428 active answers express the selected hypothesis | `provenance/release_audit.json` (`decision_faithfulness`) |
| Sensitivity of RIVER-L joint accuracy to the one test pair whose human gold contradicts the source 2WikiMultihopQA answer (`grounded-372a2f2e5a86cf6740`; 0.7790 → 0.7805) | `summarize_release_endpoints.py` (`source_answer_sensitivity`) |

## Corpus

| Quantity | Source |
|---|---|
| 1,502 examples, 751 pairs, ten interventions; review pool, exclusions, splits, state and family counts | `data/grounded_human_frozen/manifest.json` (`counts`) |
