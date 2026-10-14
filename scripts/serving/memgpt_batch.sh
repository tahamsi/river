#!/usr/bin/env bash
# Answer a fixture's questions with the MemGPT / Letta native baseline.
#
# One agent per example, so each question sees only its own evidence in archival
# memory. Requires a Letta server (scripts/serving/start_letta.sh) and the PostgreSQL it
# insists on.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="$PATH:$HOME/.local/bin"

FIXTURE=${FIXTURE:-data/grounded_nli/river_bench_test.jsonl}
LIMIT=${LIMIT:-10}
OUT=${OUT:-artifacts/predictions_nli/memgpt_test.jsonl}
SLICE=/tmp/memgpt_slice.jsonl

head -"$LIMIT" "$FIXTURE" > "$SLICE"
mkdir -p "$(dirname "$OUT")"

echo "[$(date -Is)] running $LIMIT examples through Letta"
python3 - "$SLICE" "$OUT" <<'PYEOF'
import json, os, subprocess, sys, time
from pathlib import Path

fixture, output = Path(sys.argv[1]), Path(sys.argv[2])
CHAT_BASE = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1")


def generator_label() -> str:
    """Name the server and checkpoint the agent answers with, by asking it."""

    import urllib.request

    try:
        with urllib.request.urlopen(CHAT_BASE.rstrip("/") + "/models", timeout=30) as response:
            data = json.loads(response.read())["data"]
    except Exception:
        return os.environ.get("LOCAL_LLM_MODEL", "qwen3.5:4b")
    wanted = os.environ.get("LOCAL_LLM_MODEL", "qwen3.5:4b")
    entry = next((item for item in data if item.get("id") == wanted), data[0])
    # Ollama reports its registry namespace here rather than the server, which
    # would put these rows in a generator group of their own and stop them being
    # compared against the controlled baselines that ran on the same server.
    server = entry.get("owned_by") or "unknown"
    if server == "library":
        server = "ollama"
    return f"{server}/{entry.get('root') or entry.get('id')}"


GENERATOR = generator_label()
print(f"generator: {GENERATOR}", flush=True)
rows = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]

with output.open("w") as handle:
    for index, row in enumerate(rows, 1):
        payload = {
            "baseline_id": "memgpt",
            "comparison_mode": "native",
            "session_id": row["example_id"],
            "query": row["query"],
            "target_time": row.get("target_time"),
            "evidence": row["evidence"],
            "budgets": {},
            "native_config": {},
        }
        start = time.time()
        done = subprocess.run(
            ["external/envs/letta/bin/python", "external/river_memgpt_runner.py"],
            input=json.dumps(payload), capture_output=True, text=True, timeout=1800,
        )
        elapsed = time.time() - start
        if done.returncode != 0:
            print(f"  {index}/{len(rows)} FAILED: {done.stderr[-300:]}", flush=True)
            continue
        result = json.loads(done.stdout)
        transformation = row.get("transformation_metadata") or {}
        metadata = {
            "native_execution": {
                "system": "memgpt",
                "server": "letta-0.16.8",
                "model_calls": result.get("model_calls"),
            },
            # Letta runs no entailment verifier, so this is a measured zero.
            "nli_calls": 0,
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
                      baseline="memgpt", comparison_mode="native",
                      generator=GENERATOR, seed=42,
                      latency_ms=elapsed * 1000.0, metadata=metadata)
        handle.write(json.dumps(result) + "\n")
        handle.flush()
        print(f"  {index}/{len(rows)} ok in {elapsed:.0f}s", flush=True)
PYEOF
echo "[$(date -Is)] MEMGPT BATCH DONE rows=$(wc -l < "$OUT" 2>/dev/null || echo 0)"
