#!/usr/bin/env bash
# Serve Letta (MemGPT) for the native baseline runner.
#
# Letta 0.16.8 reaches for PostgreSQL during application startup even when the
# config names SQLite, so a database is not optional. The documented setup uses
# a pgvector container, which on some hosts needs AppArmor confinement
# relaxed for the docker daemon; the distribution's own postgresql-16-pgvector
# package gives the same server with no such change. Create it once with:
#
#   apt-get install -y postgresql postgresql-16-pgvector
#   su - postgres -c "psql -c \"CREATE USER letta WITH PASSWORD 'letta' SUPERUSER;\""
#   su - postgres -c "createdb -O letta letta"
#   su - postgres -c "psql -d letta -c 'CREATE EXTENSION IF NOT EXISTS vector;'"
#   (cd external/letta && LETTA_PG_URI=... external/envs/letta/bin/alembic upgrade head)
set -euo pipefail

PORT="${PORT:-8283}"
LOG_DIR="${LOG_DIR:-$(dirname "${BASH_SOURCE[0]}")/../../logs}"
VENV="${VENV:-$(dirname "${BASH_SOURCE[0]}")/../../external/envs/letta}"
export LETTA_PG_URI="${LETTA_PG_URI:-postgresql://letta:letta@127.0.0.1:5432/letta}"
# The agent's model handles resolve through this provider. Ollama serves the same
# Q4_K_M checkpoint the controlled baselines ran against, so the comparison is
# not confounded by quantisation.
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"

mkdir -p "$LOG_DIR"

# Letta persists the provider row on first start and serves its model list from
# that row afterwards, so changing the environment alone does not move a
# provider that already exists. Embeddings are issued through the OpenAI client,
# which posts to <base_url>/embeddings; a base_url without /v1 therefore 404s on
# every archival insert. The provider's own raw_base_url strips the suffix again
# for the native /api endpoints, so storing it with /v1 is correct for both.
if command -v psql > /dev/null 2>&1; then
    PGPASSWORD="${LETTA_PG_PASSWORD:-letta}" psql -q -h 127.0.0.1 -U letta -d letta \
        -c "update providers set base_url = '$OLLAMA_BASE_URL' where name = 'ollama' and base_url <> '$OLLAMA_BASE_URL';" \
        > /dev/null 2>&1 || true
fi

if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/health/" -o /dev/null; then
    echo "letta already serving on $PORT"
    exit 0
fi

setsid nohup "$VENV/bin/letta" server --port "$PORT" \
    >"$LOG_DIR/letta.log" 2>&1 < /dev/null &

echo "launched, waiting for readiness"
for i in $(seq 1 30); do
    sleep 10
    if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/health/" -o /dev/null; then
        echo "READY after ~$((i * 10))s"
        curl -s "http://127.0.0.1:$PORT/v1/health/"
        exit 0
    fi
done

echo "did not become ready; last log lines:"
tail -20 "$LOG_DIR/letta.log"
exit 1
