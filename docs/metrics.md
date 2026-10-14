# Evaluation metrics

Metrics are computed only from normalised `PredictionRecord` JSONL files
(`river/schemas.py`). Implementations and missing-label behaviour are in
`river/evaluation/metrics.py`; paired statistics are in
`river/evaluation/statistics.py`. Missing labels produce `NA`/`null`, never
zero, and `controlled` and `native` comparison modes are never mixed.

## Primary endpoints

| Endpoint | Definition | Rows |
|---|---|---|
| State accuracy | Predicted state equals `gold_state` | All test rows |
| Active selection accuracy | Selected hypothesis equals `gold_hypothesis_id`; an abstention counts as wrong | Rows whose gold state is `active` |
| Joint revision accuracy | Correct state and, for `active` rows, the correct hypothesis | All test rows |
| Strict answer accuracy | The answer satisfies the `FINAL_ANSWER: <answer>` contract and matches the gold answer or an alias after normalisation (`scripts/release/score_answer_text.py`) | Rows whose gold state is `active` |
| Recency flip rate (RFR) | Fraction of pairs whose `(state, hypothesis)` signature changes between the two arrival orders; pairs where neither member expressed an answer are skipped | `order_reversal` and `expired_evidence` pairs (release Table 1); `order_reversal` only in release Table 3 |

RFR must be read with accuracy: a constant predictor is invariant. Pair
outcomes are therefore also reported as stable-correct, stable-wrong and
flipped.

RFR, USR, repetition failure, the temporal task accuracies and the
stable/flipped pair categories are defined by this project. Established
components are cited where they appear below.

## Correctness and recency robustness

Let a prediction signature be `(predicted_state,
predicted_hypothesis_id)`. `revision_correct` requires the predicted state to
equal `gold_state` and, when a gold hypothesis exists, the predicted hypothesis
to equal `gold_hypothesis_id`.

| Metric | Definition | Direction |
|---|---|:---:|
| Answer Accuracy | Mean `answer_correct` where a normalised answer label exists; revision accuracy is used only by tables that explicitly request `answer_or_revision_accuracy` | ↑ |
| Recency Flip Rate (RFR) | Fraction of countable pairs whose two prediction signatures differ (see above for which pairs) | ↓ |
| Order-Reversal Consistency | `1 - RFR` | ↑ |
| Mean Probability JSD / Probability Recency Sensitivity | Mean base-2 Jensen–Shannon divergence between paired state-probability distributions; unavailable when a system does not emit comparable probabilities | ↓ |
| Stable-Correct | Fraction of order pairs with identical signatures and both variants correct | ↑ |
| Stable-Wrong | Fraction with identical signatures and both variants wrong | ↓ |
| Flipped-to-Correct | Fraction with different signatures where the first variant is wrong and the second correct | Diagnostic |
| Flipped-Wrong | Fraction with different signatures where the first variant is correct and the second wrong | ↓ |

Jensen–Shannon divergence follows [Lin (1991)](https://doi.org/10.1109/18.61115).
Every pair falls into exactly one category; pairs that fit none of the four
named patterns are counted as `other`.

## Unsupported evidence, updates, and time

| Metric | Definition | Direction |
|---|---|:---:|
| Unsupported Supersession Rate (USR) | Fraction of late-unsupported variants selecting `intervention_hypothesis_id` | ↓ |
| Repetition Failure Rate | Fraction of repeated-weak variants selecting the repeated intervention hypothesis; it is the complement of implemented `repetition_resistance` | ↓ |
| Independent-Support Precision | Precision of claims counted as independently corroborated, where corroborating evidence must come from distinct independence groups | ↑ |
| Genuine Update Accuracy (GUA) | Revision accuracy on supported-update variants whose intervention hypothesis is the gold hypothesis | ↑ |
| Retroactive-Correction Accuracy | Revision accuracy on correction variants whose corrected hypothesis is gold | ↑ |
| Evolution Accuracy | Revision accuracy on `temporal_evolution` rows | ↑ |
| Dispute / Concurrent-Dispute Accuracy | Revision accuracy on `concurrent_disagreement` rows | ↑ |
| NEI Abstention Accuracy | Fraction of `insufficient_evidence_nei` rows predicted `abstain` | ↑ |
| Historical-/Current-State Accuracy | Revision accuracy restricted to rows whose gold temporal slice is respectively `historical` or `current` | ↑ |
| Stale-Belief Rate | Fraction of rows with a declared stale hypothesis where that stale ID is selected | ↓ |
| False-Dispute Rate | `FP / (FP + TN)` when `disputed` is the positive state; equivalently `1 - specificity` | ↓ |
| Incorrect Supersession | Fraction of evolution cases collapsed into an incorrect single replacement | ↓ |
| Native Temporal Accuracy | Micro-average of joint temporal correctness on canonical update, correction, evolution, historical, and current examples: both the normalised answer/selected hypothesis and normalised state must be correct; counterfactual pair duplicates do not add weight | ↑ |

The task-specific accuracies above apply `revision_correct` to the rows of the
named intervention. RFR should always be reported with GUA: a system that never
changes its answer has an RFR of zero while failing every genuine update.

## Classification, abstention, and selective prediction

Conflict and abstention precision, recall, F1, specificity, and macro-F1 use
ordinary confusion-matrix definitions. Precision is `TP/(TP+FP)`, recall is
`TP/(TP+FN)`, specificity is `TN/(TN+FP)`, and F1 is the harmonic mean of
precision and recall. These are standard measures; see [van Rijsbergen (1979)](https://www.dcs.gla.ac.uk/Keith/Preface.html).

| Metric | Definition | Direction |
|---|---|:---:|
| Coverage | Fraction of emitted states that are not `abstain` | ↑, subject to risk |
| Selective Risk | `1 - accuracy` among non-abstained predictions | ↓ |
| Risk-Coverage Curve | Sort labelled predictions by confidence descending and trace empirical risk as progressively lower-confidence items are included | — |
| AURC | Trapezoidal area under that risk-coverage curve | ↓ |
| Accuracy at 80% Coverage | Accuracy of the highest-confidence `ceil(0.8 n)` predictions | ↑ |
| Risk at 80% Coverage | `1 - accuracy_at_80_percent_coverage`; this complement is what the results table labels as risk | ↓ |

Selective risk and coverage follow the selective-classification framework; see
[Geifman and El-Yaniv (2017)](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).
The evaluator reports accuracy at fixed coverage, so "Risk at 80% Coverage"
is its complement.

## Citation and claim verification

Citation metrics operate on generated atomic claims and their exact cited
`(source_id, chunk_id)` pairs. Gold claim labels come from the dataset, never
from the verifier being evaluated.

| Metric | Definition | Direction |
|---|---|:---:|
| Citation Compliance | Fraction of evaluated answers for which every generated claim includes at least one citation | ↑ |
| Citation Support Precision / Recall / F1 | One-vs-rest precision, recall, and F1 for predicted `supported` versus gold claim labels | ↑ |
| Contradiction, Neutral/Insufficient, Temporal-Mismatch F1 | One-vs-rest F1 for the corresponding claim label | ↑ |
| Citation Support Rate | Fraction of rows whose normalised `citation_supported` flag is true | ↑ |
| Supported and Correct | Fraction with both supported citations and a correct answer | ↑ |
| Compliant but Unsupported | Fraction citing every claim while failing citation support | ↓ |
| Supported but Wrong | Fraction with supported citations but an incorrect answer | ↓ |
| Unsupported Claim Rate | `1 - citation_support_rate`, as shown in the results tables | ↓ |

"Citation F1" in the results means `citation_support_f1`, not citation-string
overlap or document-level retrieval F1.

## Calibration, efficiency, and statistical reporting

The Brier score is the mean squared difference between confidence and binary
correctness, following [Brier (1950)](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2).
Expected Calibration Error (ECE) uses ten equal-width confidence bins and the
sample-count-weighted absolute confidence/accuracy gap; reliability output lists
each bin's bounds, count, mean confidence, accuracy, and gap. For calibration of neural
networks see [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html).

Efficiency fields are directly instrumented means: end-to-end `latency_ms`,
revision latency in milliseconds, total tokens, model calls, NLI calls, peak
memory in MB, and estimated USD cost. The results document displays end-to-end
latency in seconds, peak memory in GB, and relative cost; those are unit
conversions or normalisation against the declared reference system, not new
evaluator metrics.

Confidence intervals use percentile bootstrap resampling, following
[Efron (1979)](https://doi.org/10.1214/aos/1176344552). The release tables
draw 10,000 resamples (seed 13) with each test pair as one cluster, so the five
RIVER-L predictions of an example are never treated as five independent test
examples (`analysis/generate_tables.py`).
Binary paired correctness uses the exact McNemar procedure
([McNemar, 1947](https://doi.org/10.1007/BF02295996)); continuous paired
contrasts use sign-flip permutation tests. Multiple comparisons use
[Holm (1979)](https://www.jstor.org/stable/4615733) or
[Benjamini and Hochberg (1995)](https://www.jstor.org/stable/2346101).
The code also reports paired mean difference, Cohen's `dz`, seed mean/SD/SEM,
and a one-sided paired GUA non-inferiority test with the predeclared margin.

