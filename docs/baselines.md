# Baselines

All systems implement `river.baselines.base.Baseline` and are registered by
name in `river/registry.py`. Any of them can be run with `experiments/run.py`
(see [`reproducing.md`](reproducing.md)).

The reported results cover the controlled systems: naive RAG, stateless RAG,
last-write-wins, recency-weighted, MiniCheck without revision, RIVER-D and
RIVER-L. Adapters exist for several native external systems, but none of them
contributes rows to the reported results.

| ID | Role | Status |
|---|---|---|
| `naive_rag` | Retrieval and generation without memory or verification | Built in |
| `stateless_rag` | Per-turn control with no persistent belief state | Built in |
| `last_write_wins` | The most recently arriving value supersedes earlier ones | Built in |
| `recency_weighted` | Relevance combined with recency-weighted memory | Built in |
| `minicheck_no_revision` | Retrieval, generation, atomic claims and exact-chunk MiniCheck, with retry or abstention; no RIVER revision | Built in; requires the `minicheck` extra |
| `river_d` | Deterministic RIVER | Built in |
| `river_l` | Learned, permutation-invariant RIVER | Requires a trained checkpoint |
| `self_rag` | Self-RAG | Adapter; requires the upstream environment |
| `crag` | Corrective RAG | Adapter; requires the upstream environment |
| `graphrag_local` / `graphrag` | Microsoft GraphRAG, local search | Adapter; requires the upstream environment |
| `graphrag_global` | Microsoft GraphRAG, global search | Adapter; native mode only |
| `memgpt` | MemGPT/Letta long-term memory | Adapter; requires the upstream environment |
| `zep` | Zep/Graphiti temporal graph memory | Adapter; requires the upstream environment |

## Comparison modes

Every prediction record carries a `comparison_mode`:

- `controlled`: shared evidence, generator, budget, seed and decoding settings,
  wherever that is technically possible;
- `native`: the system's released configuration, with its own retrieval,
  memory and indexing.

The evaluator refuses to combine the two in one metric. If a native system does
not emit an abstention, dispute or evolution state, the record marks that state
as unsupported or evaluator-derived rather than attributing it to the system.

## Native external systems

Upstream repositories and model weights are not vendored. Each adapter calls a
RIVER-authored wrapper in `external/river_*.py` over a JSON-stdin contract
(`external/river_runner_contract.py`). A manifest with no safe default runner
leaves `default_command` empty, and the adapter then fails with a setup
message; it never falls back to a simplified local reimplementation.

The manifests in `configs/native_baselines/` record the upstream repository and
commit, paper, licence, installation command, model and retriever requirements,
credentials, supported modes, output mapping and state support.

| Manifest | Pin | Licence | Modes | Requirements |
|---|---|---|---|---|
| `self_rag` | `AkariAsai/self-rag@1fcdc420e48f` | MIT | controlled, native | Pinned checkout, model and retriever resources, `HF_TOKEN`, JSON wrapper |
| `crag` | `HuskyInSalt/CRAG@de7c2961ae62` | See upstream | controlled, native | Pinned checkout, credentials, JSON wrapper |
| `graphrag_local` | `microsoft/graphrag@14a00ad88fc3` | MIT | controlled, native | Isolated environment, index build, local model endpoint, JSON wrapper |
| `graphrag_global` | `microsoft/graphrag@14a00ad88fc3` | MIT | native | As for `graphrag_local` |
| `memgpt` | `letta-ai/letta@b76da9092518` | Apache-2.0 | controlled, native | Isolated environment, configured memory state, provider credentials, JSON wrapper |
| `zep` | `getzep/graphiti@0b4bcf1284ee` | Apache-2.0 | controlled, native | Isolated environment, graph store and provider credentials, JSON wrapper |

To inspect the adapters:

```bash
uv run python -m river.baselines.native_cli list-availability
uv run python -m river.baselines.native_cli diagnose-missing-dependency --baseline self_rag
uv run python -m river.baselines.native_cli validate-credentials --baseline graphrag_local
uv run python -m river.baselines.native_cli print-manifest --baseline self_rag --json
```

### Setup

Native adapters use only a local OpenAI-compatible endpoint, set by
`LOCAL_LLM_BASE_URL` (Ollama, vLLM or llama.cpp); hosted model APIs are out of
scope. To install the pinned checkouts in isolated environments and start the
supporting services:

```bash
RIVER_PYTHON_BIN=python3.11 bash scripts/serving/setup_native_baselines.sh
cp infrastructure/native.env.example infrastructure/native.env
# Set the Neo4j password in native.env; the file is not tracked.
set -a; source infrastructure/native.env; set +a
docker compose --env-file infrastructure/native.env -f infrastructure/compose.yaml up -d
```

The installer checks out the revisions pinned in `configs/native_baselines/`
for Self-RAG, CRAG, GraphRAG, Letta/MemGPT and Graphiti, and installs the
MiniCheck extra. It does not download large weights, build indexes or write the
JSON wrappers. Self-RAG and CRAG need their released weights, and an `HF_TOKEN`
where the upstream host gates access. GraphRAG runs through its LiteLLM path,
and the local model must emit GraphRAG's structured JSON reliably. Graphiti
uses the local Neo4j service and a local LLM and embedding endpoint, and Letta
is configured against the same provider.

An external runner configuration looks like this:

```json
{
  "comparison_mode": "controlled",
  "runner_command": ["external/envs/self-rag/bin/python", "external/self-rag/run_river_json.py"],
  "official_checkout_path": "external/self-rag",
  "timeout_seconds": 120,
  "max_retries": 2,
  "rate_limit_per_minute": 20,
  "cost_per_input_token": 0.0,
  "cost_per_output_token": 0.0
}
```

## MiniCheck

`minicheck_no_revision` is a citation-verification baseline, not a revision
system, and does not use the RIVER ledger or revision operator:

```text
retrieve -> generate -> atomic claims -> exact-chunk MiniCheck -> accept / retry / abstain
```

Each claim is checked only against the `(source_id, chunk_id)` pairs it cites,
never against the whole document, another retrieved chunk or uncited context.
The prediction record stores per-claim and per-citation labels and scores in
`claim_verification_results`.

The configuration in `configs/baselines/minicheck_no_revision.json` pins:

| Item | Value |
|---|---|
| Package | `minicheck @ git+https://github.com/Liyan06/MiniCheck.git@b58b9fa69acbd1015ec970fa65dd752413a053d2` (version 0.1.0, Apache-2.0) |
| Model | `lytang/MiniCheck-Flan-T5-Large` at revision `96eafd01cee2d16cf81aaa2fb226b14f422a37b3` (MIT), passed to MiniCheck as `flan-t5-large` |
| Device | `auto` |
| Batch size | 16 |
| Maximum input length | 2,048 |

```bash
uv sync --extra minicheck
uv run python -m experiments.run \
  --baseline minicheck_no_revision \
  --fixture data/grounded_human_frozen/river_bench_test.jsonl \
  --config configs/baselines/minicheck_no_revision.json \
  --seed 42 --no-generation \
  --output artifacts/predictions/minicheck_no_revision_test_seed42.jsonl
```

The first run downloads the model into the cache. To run offline, populate the
cache on a connected machine with the pinned package and revision, then set
`HF_HUB_OFFLINE=1`.

Verifier thresholds may be calibrated only on dev rows: if
`verifier.calibration_file` or `verifier.calibration_rows` contains train or
test rows, calibration fails.
