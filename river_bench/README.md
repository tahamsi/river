# river_bench

Schema, validation and build tooling for RIVER-Bench. The benchmark itself is
in `data/grounded_human_frozen/` and is described in
[`docs/benchmark.md`](../docs/benchmark.md). The fixture in `processed/` is a
small authored corpus for unit tests and is not evaluation data.

```text
river_bench/
  schemas.py          BenchmarkExample, the record schema
  validation.py       Schema, pairing, order-equivalence and split-leakage checks
  river_bench.py      Command-line interface
  pat_redaction.py    PAT-Questions redaction and rehydration
  loaders/            Download hooks for the source datasets
  transformations/    Deterministic construction of the authored fixture
  manifests/          Seed records for the authored fixture
  processed/          The authored unit-test fixture
```

## Validation

```bash
uv run python -m river_bench.river_bench validate-all \
  --fixture data/grounded_human_frozen/river_bench_1502.jsonl
```

`validate-all` runs the individual checks, which are also available on their
own: `validate-schema`, `validate-pairs`, `validate-order-equivalence` and
`validate-split-leakage`. `count-interventions` prints the family counts.
Without `--fixture`, each command runs on the authored unit fixture.

## Authored unit fixture

```bash
uv run python -m river_bench.river_bench build-authored \
  --seed-manifest river_bench/manifests/river_bench_seed_fixtures.jsonl \
  --output river_bench/processed/river_bench_unit.jsonl
```

## Source datasets

Downloads are explicit and are written under `data/`, never into `processed/`.

```bash
uv run python -m river_bench.river_bench list-external
uv run python -m river_bench.river_bench download-external --dataset fever --limit 100
```

FEVER, 2WikiMultihopQA, WikiContradict, LongMemEval and TempRAGEval are fetched
from Hugging Face. PAT-Questions has no packaged release and needs a local
export passed with `--manual-file`. The 2WikiMultihopQA hook loads
2WikiMultihopQA itself, not a relabelled HotpotQA.
