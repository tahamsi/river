# Integrity checks and corrections

Several problems were found and corrected during development, before the final
corpus was frozen. They are recorded here because each one could have produced
a misleading result, and each now has a guard in the test suite.

| Finding | Correction | Guard |
|---|---|---|
| In an early corpus, the stored `entailment`/`contradiction` scores were written from the gold label at construction time, so a rule reading only those scores recovered the gold hypothesis in every contested row | Scores are now measured against the evidence text with `microsoft/deberta-large-mnli` (`scripts/benchmark/recompute_nli_scores.py`); every evidence item records `verifier_scores_measured_by` | `tests/test_verifier_scores_are_not_labels.py`, `tests/test_evidence_features_carry_signal.py` |
| Benchmark identifiers could reveal the gold answer | Identifiers are hashed (`scripts/benchmark/anonymise_ids.py`) | `tests/test_identifiers_do_not_reveal_gold.py` |
| An evaluation narrative contained numbers without prediction records behind them | Withdrawn. Every reported number is now produced by `analysis.generate_tables` from prediction records, with a provenance entry for each cell | `tests/test_manuscript_tables.py` |
| Answer accuracy was derived from hypothesis selection, and RFR compared empty signatures, so systems that never answered scored well | Strict answer scoring of the rendered text under a `FINAL_ANSWER:` contract (`scripts/release/score_answer_text.py`); for systems that select no hypothesis, RFR compares the answer read back from the rendered text, and pairs where neither member answered are not counted | `tests/test_answer_text_scoring.py` |
| Some baselines are identical on this benchmark: stateless RAG subclasses naive RAG, and with two evidence items per row recency weighting reduces to last-write-wins | Reported as identical rather than as independent results | `tests/test_baseline_distinctness.py` |
| Some families could not be anchored in their cited source record | `retroactive_correction` is pre-registered as qualitative-only and excluded from aggregates | `tests/test_qualitative_only_families.py` |
| The first run on the frozen corpus let the renderer see every hypothesis, so a rendered answer could disagree with the structured decision | The final release renders `ACTIVE` answers from the selected hypothesis only and audits decision faithfulness (98.9%; failures are counted as errors) | `scripts/release/audit_release_results.py`, `tests/test_release_audit.py` |

## Residual signal in the frozen corpus

On the frozen corpus, taking the hypothesis with the highest measured
entailment selects the gold hypothesis in 438 of 756 contested `active` rows
(58%). It is higher in two small families by construction:
`unsupported_late_assertion` (10/10) and `repeated_weak_assertion` (19/20)
present the competing claim without independent support, so low entailment is
the evidential signal those families are designed to test.

`provenance_reliability` and `extraction_confidence` are constant (1.0) in the
frozen corpus, so the mechanisms that read them have no effect there.
