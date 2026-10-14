#!/usr/bin/env bash
# Answer a fixture's questions against one shared GraphRAG index.
#
# Indexes the corpus once, then runs each question through local search. Defaults
# to a small slice so the shared-index path is proved before hours are spent on
# the full split.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="$PATH:$HOME/.local/bin"
export GRAPHRAG_API_KEY=${GRAPHRAG_API_KEY:-ollama}

FIXTURE=${FIXTURE:-data/grounded_nli/river_bench_test.jsonl}
LIMIT=${LIMIT:-10}
OUT=${OUT:-artifacts/predictions_nli/graphrag_local_test.jsonl}
SLICE=/tmp/graphrag_slice.jsonl

head -"$LIMIT" "$FIXTURE" > "$SLICE"
export GRAPHRAG_SHARED_CORPUS="$SLICE"
mkdir -p "$(dirname "$OUT")"
: > "$OUT"

echo "[$(date -Is)] indexing $LIMIT examples once, then querying each"
python3 - "$SLICE" "$OUT" <<'PYEOF'
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

fixture, output = Path(sys.argv[1]), Path(sys.argv[2])

CHAT_BASE = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
EMBED_BASE = os.environ.get("LOCAL_EMBEDDING_BASE_URL", CHAT_BASE)


def generator_label() -> str:
    """Name the server and checkpoint that will answer, by asking the endpoint.

    A hardcoded label silently lies the moment the endpoint moves, and the
    quantisation differs between servers, so the checkpoint alone is not enough.
    """

    try:
        with urllib.request.urlopen(CHAT_BASE.rstrip("/") + "/models", timeout=30) as response:
            entry = json.loads(response.read())["data"][0]
    except Exception:
        return os.environ.get("LOCAL_LLM_MODEL", "qwen3.5:4b")
    server = entry.get("owned_by") or "unknown"
    return f"{server}/{entry.get('root') or entry.get('id')}"


def _metrics(base: str) -> dict[str, float]:
    """Cumulative request and token counters from a vLLM server, or {} elsewhere.

    GraphRAG is a third-party system run as a subprocess, so it reports no call
    counts of its own. The endpoint it calls does. Examples run strictly one at
    a time here, so the delta across one example is that example's usage.
    """

    root = base.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=10) as response:
            body = response.read().decode()
    except Exception:
        return {}
    totals = {"requests": 0.0, "prompt_tokens": 0.0, "generation_tokens": 0.0}
    for line in body.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        try:
            number = float(value)
        except ValueError:
            continue
        if name.startswith("vllm:request_success_total"):
            totals["requests"] += number
        elif name.startswith("vllm:prompt_tokens_total"):
            totals["prompt_tokens"] += number
        elif name.startswith("vllm:generation_tokens_total"):
            totals["generation_tokens"] += number
    return totals


def _index_exists() -> bool:
    """Whether a shared-corpus index is already on disk for this configuration."""

    root = Path(os.environ.get("GRAPHRAG_WORKSPACE", "artifacts/graphrag"))
    return any(root.glob("shared-*/output/entities.parquet"))


def _delta(before: dict[str, float], after: dict[str, float], key: str) -> float | None:
    if not before or not after or key not in before or key not in after:
        return None
    return after[key] - before[key]


def run_one(row: dict) -> tuple[subprocess.CompletedProcess, float, dict]:
    payload = {
        "baseline_id": "graphrag_local",
        "comparison_mode": "native",
        "session_id": row["example_id"],
        "query": row["query"],
        "target_time": row.get("target_time"),
        "evidence": row["evidence"],
        "budgets": {},
        "native_config": {},
    }
    chat_before, embed_before = _metrics(CHAT_BASE), _metrics(EMBED_BASE)
    start = time.time()
    done = subprocess.run(
        ["external/envs/graphrag/bin/python", "external/river_graphrag_runner.py"],
        input=json.dumps(payload), capture_output=True, text=True, timeout=7200,
    )
    elapsed = time.time() - start
    chat_after, embed_after = _metrics(CHAT_BASE), _metrics(EMBED_BASE)
    usage = {
        "model_calls": _delta(chat_before, chat_after, "requests"),
        "embedding_calls": _delta(embed_before, embed_after, "requests"),
        "input_tokens": _delta(chat_before, chat_after, "prompt_tokens"),
        "output_tokens": _delta(chat_before, chat_after, "generation_tokens"),
    }
    return done, elapsed, usage


GENERATOR = generator_label()
print(f"generator: {GENERATOR}", flush=True)
rows = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]

# Building the shared index happens inside the first query. Charging it to that
# one example would make its telemetry incomparable to the other 249, so it is
# paid for here, once, before any row is written. Its cost is a property of the
# corpus rather than of any example, so it is logged rather than copied onto
# every row; a warm workspace makes this pass a no-op, and the log says which
# it was.
index_prebuilt = _index_exists()
print(f"preparing the shared index (workspace already built: {index_prebuilt})", flush=True)
_, index_seconds, index_usage = run_one(rows[0])
print(f"  index pass took {index_seconds:.0f}s, model calls {index_usage['model_calls']}", flush=True)

with output.open("w") as handle:
    for index, row in enumerate(rows, 1):
        done, elapsed, usage = run_one(row)
        if done.returncode != 0:
            print(f"  {index}/{len(rows)} FAILED: {done.stderr[-300:]}", flush=True)
            continue
        result = json.loads(done.stdout)
        transformation = row.get("transformation_metadata") or {}
        metadata = {
            "native_execution": {
                "system": "graphrag_local",
                "model_calls": usage["model_calls"],
                "embedding_calls": usage["embedding_calls"],
                "index_shared_across_examples": len(rows),
                "index_prebuilt_before_run": index_prebuilt,
            },
            # GraphRAG runs no entailment verifier at all, so this is a measured
            # zero, matching how experiments/run.py records a baseline with no verifier.
            "nli_calls": 0,
            # Local open-weight inference under the plan's $0.00 budget.
            "estimated_cost_usd": 0.0,
        }
        for key in (
            "gold_answer",
            "gold_answer_aliases",
            "gold_claims",
            "claim_gold_labels",
            "citation_gold_labels",
            "temporal_slice",
            "stale_hypothesis_id",
            "annotation_status",
            "human_review_status",
        ):
            if transformation.get(key) is not None:
                metadata[key] = transformation[key]
        result.update(example_id=row["example_id"], base_example_id=row["base_example_id"],
                      pair_id=row.get("pair_id"), intervention=row.get("intervention"),
                      intervention_hypothesis_id=row.get("intervention_hypothesis_id"),
                      split=row.get("split"), gold_state=row.get("gold_state"),
                      gold_hypothesis_id=row.get("gold_hypothesis_id"),
                      dataset=row.get("dataset"), source_dataset=row.get("source_dataset"),
                      baseline="graphrag_local", comparison_mode="native",
                      generator=GENERATOR, seed=42,
                      latency_ms=elapsed * 1000.0,
                      input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
                      metadata=metadata)
        handle.write(json.dumps(result) + "\n")
        handle.flush()
        print(f"  {index}/{len(rows)} ok in {elapsed:.0f}s", flush=True)
PYEOF
echo "[$(date -Is)] GRAPHRAG BATCH DONE rows=$(wc -l < "$OUT" 2>/dev/null || echo 0)"
