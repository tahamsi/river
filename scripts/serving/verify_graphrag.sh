#!/usr/bin/env bash
# End-to-end check that GraphRAG can answer a query.
#
# Builds one small index and asks one question, which confirms that indexing
# and search agree on the embedding dimension.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH=$PATH:$HOME/.local/bin
export GRAPHRAG_API_KEY=${GRAPHRAG_API_KEY:-ollama}

log() { echo "[$(date -Is)] $*"; }

payload=$(python3 - <<'PYEOF'
import json
row = json.loads(open("data/grounded/river_bench_test.jsonl").readline())
print(json.dumps({
    "baseline_id": "graphrag_local",
    "comparison_mode": "controlled",
    "session_id": "verify-dim-fix",
    "query": row["query"],
    "target_time": row.get("target_time"),
    "evidence": row["evidence"],
    "budgets": {},
    "native_config": {},
}))
PYEOF
)

log "running graphrag local search on one example (indexing included)"
printf '%s' "$payload" | timeout 3600 external/envs/graphrag/bin/python external/river_graphrag_runner.py
log "graphrag exit=$?"
