# RIVER: Recency-Invariant, Validity-Sensitive Evidence Revision

Persistent RAG systems store evidence in an order that need not match the
chronology of the world it describes, so the most recently ingested record is
not necessarily the one that holds at query time. RIVER is a post-retrieval
revision layer that separates **transaction time** (when evidence entered
memory) from **valid time** (when its proposition holds). It assigns one of five
revision states and selects a hypothesis only when the state licenses one. Its
learned variant is permutation-invariant and cannot observe arrival order.

For example, suppose memory ingests a 2024 report that a company's CEO is B,
and later ingests a 2019 article saying the CEO is A. A last-write-wins system
answers A because that record arrived last. RIVER answers B because B's
evidence is valid at the query time. If a later record only asserts a change
without credible support, RIVER keeps its current belief.

| State | Meaning |
|---|---|
| `ACTIVE` | One hypothesis is supported and valid at the target time; it is selected |
| `ABSTAIN` | The evidence is insufficient to license a decision |
| `DISPUTED` | Independently supported, incompatible hypotheses remain in conflict |
| `EVOLUTION` | Supported values hold over non-overlapping valid times |
| `UNKNOWN` | The available evidence cannot determine the requested fact |

Only `ACTIVE` calls the answer generator. Every other state returns a fixed,
deterministic reply.

This repository contains:

- **RIVER-D**, a deterministic revision operator, and **RIVER-L**, a learned
  Deep Sets operator;
- **RIVER-Bench**, 1,502 human-adjudicated examples in 751 paired trials across
  ten interventions;
- the controlled baselines, the evaluation pipeline, and a checksummed release
  of every prediction behind the reported results.

## Results

Frozen test set (248 examples; active selection and strict answer use the 157
`active` examples). RIVER-L is the mean ± sample SD over five seeds.

| System | State acc. ↑ | Active selection ↑ | Joint acc. ↑ | Strict answer ↑ | RFR ↓ |
|---|---:|---:|---:|---:|---:|
| Naive RAG | – | – | – | .682 | .432 |
| Stateless RAG | – | – | – | .682 | .432 |
| Last-write-wins | .633 | .618 | .391 | .624 | 1.000 |
| Recency-weighted | .633 | .618 | .391 | .624 | 1.000 |
| MiniCheck | .750 | – | – | – | .000 |
| RIVER-D | .536 | .631 | .536 | .631 | .000 |
| **RIVER-L** | **.856 ± .007** | **.796 ± .022** | **.779 ± .007** | **.801 ± .016** | **.000** |

RFR is the recency flip rate: how often a decision changes when only the
arrival order of the evidence changes. The values above pool order-reversal and
expired-evidence pairs; on order-reversal pairs alone, naive and stateless RAG
score .095 and every other value is unchanged.

The released predictions, tables, audit and checksums are in
[`results/human_frozen/`](results/human_frozen/);
[`docs/results_provenance.md`](docs/results_provenance.md) traces every
reported number to its source file.

## Quickstart

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). No GPU is needed.

```bash
uv sync --extra test --extra learned
uv run python scripts/benchmark/rehydrate_pat.py   # restores PAT-Questions text (see below)
uv run --extra test pytest -m "not slow and not native"
```

Reproduce the RIVER-D row of the table above on CPU, in about 15 seconds:

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

`artifacts/quickstart/summary.json` should report `state_accuracy` 0.536 and
`active_selection_accuracy` 0.631, which match the RIVER-D row.

[`docs/reproducing.md`](docs/reproducing.md) covers verifying the release
checksums, regenerating the tables, retraining RIVER-L, and the full GPU
pipeline with answer generation.

## RIVER-Bench

The benchmark is in [`data/grounded_human_frozen/`](data/grounded_human_frozen/)
as JSONL, split 1,000 / 254 / 248 at the pair level. Each example has a
question, a target time, candidate hypotheses, and evidence carrying a
transaction time, a validity interval, NLI scores and an independence group.
The two members of a pair differ in one revision-relevant factor: evidence
order, expiry, removal, repetition, an unsupported late claim, a supported
update, concurrent disagreement, temporal evolution, insufficient evidence or a
retroactive correction.

Examples come from FEVER, PAT-Questions, WikiContradict, LongMemEval,
2WikiMultihopQA and Wikipedia revisions. Three annotators adjudicated all 1,780
candidate examples; a pair-level gate removed 139 pairs. The dataset card,
including record format, sources and licences, construction and limitations,
is [`docs/benchmark.md`](docs/benchmark.md).

PAT-Questions publishes no licence, so its question text is not redistributed
here. `scripts/benchmark/rehydrate_pat.py` downloads it from the pinned upstream
release and restores the corpus and result archives byte-for-byte; see
[`data/README.md`](data/README.md#pat-questions-text).

## Repository layout

```text
river/              RIVER: schemas, ledger, RIVER-D (revision.py), RIVER-L (learned/),
                    baselines, verifiers, evaluation metrics and statistics
river_bench/        Benchmark schema, validation, PAT-Questions rehydration and unit fixtures
experiments/        The run.py runner and the record of the run behind the results
analysis/           Table and figure generation with per-cell provenance
configs/            Baseline, ablation, RIVER-L and native-system configurations
data/               RIVER-Bench and the annotation guide
results/            The checksummed release of the reported results
checkpoints/        The five RIVER-L checkpoints behind the reported results
scripts/
  benchmark/        Download, build, score and freeze the benchmark
  release/          Structured rerun, rendering, strict scoring and release audit
  diagnostics/      Leakage and ablation-effect audits
  serving/          Local model servers and native-baseline setup
external/           JSON-stdin wrappers for native external systems
infrastructure/     Service definitions for native baselines
tests/              Test suite, including guards against label leakage
docs/               Documentation (below)
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/method.md`](docs/method.md) | RIVER-D, RIVER-L and selected-evidence rendering |
| [`docs/benchmark.md`](docs/benchmark.md) | Dataset card for RIVER-Bench |
| [`docs/metrics.md`](docs/metrics.md) | Metric definitions and statistical procedures |
| [`docs/baselines.md`](docs/baselines.md) | Controlled baselines, MiniCheck and native external systems |
| [`docs/reproducing.md`](docs/reproducing.md) | Reproduction at four levels, from checksums to the full pipeline |
| [`docs/results_provenance.md`](docs/results_provenance.md) | Source file for every reported number |
| [`docs/integrity.md`](docs/integrity.md) | Problems found during development, their corrections and test guards |
| [`docs/release/`](docs/release/) | Release notes |

## Licence

The code is released under the [MIT licence](LICENSE).

The RIVER-Bench annotations in `data/grounded_human_frozen/` are released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), which the
share-alike terms of the FEVER and Wikipedia sources require. Every example
also remains subject to the terms of the dataset it was derived from, listed
with their licences in [`docs/benchmark.md`](docs/benchmark.md#composition).
PAT-Questions text is not redistributed (see [RIVER-Bench](#river-bench)).
