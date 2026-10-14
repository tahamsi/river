# RIVER-Bench annotation contract

Annotators must use only statements present in the source-native record. Do not
infer dates, updates, corrections, provenance quality, or alternative answers.
If a required fact is absent, set the decision to `needs_external_evidence` or
`unsuitable`; never complete the field from general knowledge.

## Required base labels

Each accepted base record requires:

- source record ID, source dataset, question, and source-native record hash;
- canonical gold answer and explicitly supported aliases;
- atomic subject, predicate, and object hypotheses;
- exact evidence chunks with source/chunk IDs;
- claim-level gold labels: `supported`, `contradicted`, `neutral`, or
  `temporal_mismatch`;
- gold claim-to-chunk citation links;
- transaction time with its documented basis;
- valid-time start/end only when the source states or entails them;
- temporal slice (`historical`, `current`, `timeline`, or `not_applicable`);
- gold revision state and selected hypothesis, when one is selectable;
- stale hypothesis ID only when an old value is explicitly superseded;
- intervention suitability and a short evidence-grounded rationale;
- annotation status, annotator, review status, and review notes.

`provenance_reliability`, `extraction_confidence`, and NLI probabilities are
model inputs, not factual gold labels. They must be derived by a documented
scoring procedure and must not be presented as human gold annotations.

## Task eligibility

| Task | Minimum evidence required |
|---|---|
| `order_reversal` | A complete evidence set and fixed gold label; the paired row changes order only. |
| `unsupported_late_assertion` | A supported claim plus a source-recorded later assertion that lacks supporting evidence. |
| `repeated_weak_assertion` | A source-recorded weak assertion repeated through the same traceable source or dependence group. |
| `supported_genuine_update` | Explicit old and new values, independent support for the new value, and grounded valid-time boundaries. |
| `retroactive_correction` | An original value plus an explicit later-issued correction applying to the earlier period. |
| `temporal_evolution` | At least two supported values with explicit, non-overlapping valid-time intervals. |
| `concurrent_disagreement` | At least two supported incompatible claims about the same scope with overlapping or identical applicability. |
| `evidence_removal` | A supported control whose decisive evidence can be removed; the removal variant must then lack sufficient support. |
| `insufficient_evidence_nei` | A source-native NEI label, or a record whose supplied evidence is explicitly insufficient or irrelevant. |

Synthetic unsupported assertions may be used only in a separately identified
adversarial fixture. They cannot be represented as naturally observed source
annotations.

## Pair and split rules

- Both variants must share a `pair_id`, base record, task, and split.
- A pair must never cross training and validation.
- Entity/event duplicates must remain in one split.
- Human review accepts, corrects, or rejects the whole pair.
- Final metrics may use only accepted pairs and the labels required by that
  metric.
