# RIVER-Bench

RIVER-Bench is a paired, human-adjudicated benchmark for evidence revision in
retrieval-augmented generation. Each example gives a question, an optional
target time, candidate hypotheses, and assessed evidence; the task is to return
one of five revision states and, for an `ACTIVE` decision, the selected
hypothesis. Examples come in pairs that differ in one revision-relevant factor.

The released corpus is `data/grounded_human_frozen/` (schema
`river-bench-grounded-v4-human-frozen`).

## Files

| File | Contents |
|---|---|
| `river_bench_train.jsonl` | 1,000 examples (500 pairs) |
| `river_bench_dev.jsonl` | 254 examples (127 pairs); used for threshold fitting and early stopping |
| `river_bench_test.jsonl` | 248 examples (124 pairs); read once, after freezing |
| `river_bench_1502.jsonl` | All 1,502 examples |
| `adjudication_ledger.jsonl` | Every reviewed input row, whether it was kept, and why |
| `manifest.json` | Counts, freeze policy, workbook hash, input and output SHA-256 |

Splits are made at the pair level, and leakage keys (source entity and
relation) never cross a split boundary.

## Composition

**Revision states**

| State | All | Test | Meaning |
|---|---:|---:|---|
| `active` | 950 | 157 | One hypothesis is supported and valid at the target time |
| `abstain` | 350 | 61 | The evidence does not license a decision |
| `disputed` | 147 | 20 | Independently supported, incompatible hypotheses remain in conflict |
| `evolution` | 35 | 6 | Supported values hold over non-overlapping valid times |
| `unknown` | 20 | 4 | The available evidence cannot determine the requested fact |

**Intervention families**

| Intervention | Examples | What it tests |
|---|---:|---|
| `evidence_removal` | 388 | Withdrawing a decision when decisive support is removed |
| `order_reversal` | 368 | Invariance when only transaction order is reversed |
| `expired_evidence` | 314 | Rejecting a late-arriving claim that is no longer valid |
| `concurrent_disagreement` | 148 | Preserving unresolved, contemporaneous conflict |
| `insufficient_evidence_nei` | 142 | Abstaining when support is inadequate |
| `temporal_evolution` | 40 | Distinguishing genuine change from contradiction |
| `supported_genuine_update` | 38 | Revising when new evidence is admissible |
| `repeated_weak_assertion` | 36 | Not treating repetition as independent corroboration |
| `unsupported_late_assertion` | 22 | Resisting an unsupported claim that arrives last |
| `retroactive_correction` | 6 | Correcting the account of an earlier state |

**Sources**

| Source | Examples | Upstream licence | Reference |
|---|---:|---|---|
| PAT-Questions | 702 | None stated upstream | Meem et al. (2024), [repository](https://github.com/jannatmeem95/PAT-Questions) |
| 2WikiMultihopQA | 368 | Apache-2.0 | Ho et al. (2020), [repository](https://github.com/Alab-NII/2wikimultihop) |
| WikiContradict | 206 | MIT | Hou et al. (2024), [dataset](https://huggingface.co/datasets/ibm-research/Wikipedia_contradict_benchmark) |
| FEVER | 142 | CC BY-SA 3.0 | Thorne et al. (2018), [dataset](https://fever.ai/dataset/fever.html) |
| LongMemEval | 82 | MIT | Wu et al. (2025), [dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) |
| Wikipedia revisions | 2 | CC BY-SA 4.0 | Pinned revisions via the [MediaWiki compare API](https://www.mediawiki.org/wiki/API:Compare) |

Licences are as declared by each upstream release. Because PAT-Questions
states no licence, its questions and templates are replaced by placeholders in
this release and restored from upstream by `scripts/benchmark/rehydrate_pat.py`
([`data/README.md`](../data/README.md#pat-questions-text)). Every row keeps its
`source_dataset`, `source_id` and source-native record, so downstream users can
honour the original terms.

The RIVER-Bench annotations are released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), as the
share-alike terms of the FEVER and Wikipedia sources require. Each row remains
subject to its source's licence as well.

## Record format

One test example, abbreviated:

```json
{
  "example_id": "expired-026ba9a38748d6e2:a",
  "pair_id": "expired-026ba9a38748d6e2:expired_evidence",
  "split": "test",
  "intervention": "expired_evidence",
  "source_dataset": "pat_questions",
  "query": "@@PAT:936002b2119b4409:r@@",
  "target_time": "2024-03-01T00:00:00Z",
  "hypotheses": [
    {"hypothesis_id": "…:h_4b224fd4d470", "subject": "Volodymyr Gerun", "predicate": "play for",
     "object": "Real Betis Baloncesto", "evidence_ids": ["…:e_fa1152cedc91"]},
    {"hypothesis_id": "…:h_b663b313c702", "subject": "Volodymyr Gerun", "predicate": "play for",
     "object": "Sendai 89ers", "evidence_ids": ["…:e_b22ee2dc4e81"]}
  ],
  "evidence": [
    {"evidence_id": "…:e_fa1152cedc91", "hypothesis_id": "…:h_4b224fd4d470",
     "text": "Volodymyr Gerun plays for Real Betis Baloncesto.",
     "transaction_time": "2026-09-02T00:00:00Z",
     "valid_time": {"start": "2015-01-01T00:00:00Z", "end": "2022-01-01T00:00:00Z", "confidence": 1.0},
     "entailment": 0.825, "contradiction": 0.001, "neutral": 0.173,
     "provenance_reliability": 1.0, "extraction_confidence": 1.0,
     "independence_group": "…:e_fa1152cedc91",
     "verifier_scores_measured_by": "microsoft/deberta-large-mnli"}
  ],
  "gold_state": "active",
  "gold_hypothesis_id": "…:h_b663b313c702",
  "intervention_hypothesis_id": "…:h_4b224fd4d470",
  "transformation_metadata": {"annotation_status": "human_adjudicated_frozen", "…": "…"}
}
```

Here the expired claim arrives last, but only the other claim is valid at the
target time, so the gold decision is `active` on the Sendai 89ers hypothesis.

| Field | Meaning |
|---|---|
| `transaction_time` | When the evidence entered memory. Kept for auditing; RIVER never uses it to select truth |
| `valid_time` | Interval over which the proposition holds, with an interval-confidence value |
| `entailment`, `contradiction`, `neutral` | Three-way NLI probabilities measured with `microsoft/deberta-large-mnli` against the evidence text |
| `independence_group` | Evidence sharing a group is dependent; repetition within a group is discounted |
| `provenance_reliability`, `extraction_confidence` | Present for the interface; constant (1.0) in this release |
| `gold_hypothesis_id` | Selected hypothesis for `active` rows; null otherwise |
| `intervention_hypothesis_id` | The hypothesis the intervention introduces or targets |
| `transformation_metadata` | Construction method, source-native record, review checksums, leakage keys |

The full schema is `river_bench.schemas.BenchmarkExample`:

```python
import json
from river_bench.schemas import BenchmarkExample

with open("data/grounded_human_frozen/river_bench_test.jsonl") as f:
    examples = [BenchmarkExample.model_validate(json.loads(line)) for line in f]
```

## Construction

1. **Source corpus.** A deterministic composite of source-native records from
   the datasets above (`scripts/benchmark/build_river_bench_1800.py`,
   `download_all_datasets.py`, `download_retroactive_corrections.py`).
2. **Grounded annotation.** Only source-provided answers, aliases, labels and
   citation coordinates are extracted; chronology-sensitive records and
   corrections are reviewed individually
   (`extract_grounded_annotations.py`, `merge_grounded_annotations.py`).
   The governing rules are in `data/considerations/ANNOTATION_GUIDE.md`.
3. **Paired interventions.** Each base record yields two variants that differ in
   one revision-relevant factor (`build_grounded_river_bench.py`,
   `build_expired_evidence.py`), producing `data/grounded/` and
   `data/grounded_expired/`.
4. **Measured verifier scores.** NLI probabilities are measured against the
   evidence text (`recompute_nli_scores.py`), producing `data/grounded_nli/` and
   `data/grounded_expired_nli/`. Identifiers are hashed so none reveals the gold
   answer (`anonymise_ids.py`).
5. **Human adjudication and freeze.** Three annotators inspected all 1,780
   candidate examples (890 pairs), each recording a five-way label decision and
   a separate construction decision. Annotators were assigned disjoint subsets,
   so this is a complete review rather than an agreement study. A pair is kept
   only if both members are clearly labelled and neither construction is
   rejected (`freeze_adjudicated_corpus.py`). This removed 139 pairs: 121 for
   construction rejection, 14 for inconsistent pair labels and 4 for unresolved
   labels.

The intermediate corpora from steps 3 and 4 and the source archive needed to
rebuild steps 1–3 are not distributed, because they contain PAT-Questions text
throughout. The annotation workbook from step 5 is not distributed either; its
SHA-256 is recorded in `manifest.json`.

## Intended use and limitations

- RIVER-Bench is a behavioural test of revision decisions under controlled
  interventions. Intervention frequencies are constructed, not sampled from
  deployment traffic.
- Validity intervals are supplied as metadata. Some boundaries are construction
  dates that instantiate a known before/after relation
  (`valid_time_boundary: templated`), so the benchmark tests revision given
  explicit validity, not temporal fact extraction from free text.
- The rarest families and states are small: `retroactive_correction` has 6
  examples, `evolution` 35 and `unknown` 20 (6 and 4 on test).
- English only.
- Verifier scores were measured separately for each variant of a pair, so in
  11 `order_reversal` pairs (2 in test) a score differs between the two
  members by 1e-6. The evidence text, validity and labels are identical, and
  `river_bench.validation` compares scores to within 1e-5.
- `retroactive_correction` is pre-registered as a qualitative case study
  (`QUALITATIVE_ONLY_INTERVENTIONS` in `river/evaluation/metrics.py`): its
  evidence prose could not be anchored in the cited record. The release tables
  therefore aggregate over 246 test rows, while the primary endpoints use all
  248. See [`results_provenance.md`](results_provenance.md).
