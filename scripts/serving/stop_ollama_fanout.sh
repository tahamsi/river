#!/usr/bin/env bash
# Stop every fan-out Ollama daemon and proxy. Leaves the systemd ollama on
# :11434 alone -- that one is managed by systemctl.
set -uo pipefail

BASE_PORT="${BASE_PORT:-11435}"
LAST_PORT="${LAST_PORT:-11450}"
PROXY_PORTS="${PROXY_PORTS:-11500 11501}"

pkill -f ollama_fanout_proxy.py && echo "stopped proxies" || echo "no proxy running"

for port in $(seq "$BASE_PORT" "$LAST_PORT") $PROXY_PORTS; do
    pids=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u)
    for pid in $pids; do
        kill "$pid" 2>/dev/null && echo "killed pid $pid on port $port"
    done
done

sleep 6
for port in $(seq "$BASE_PORT" "$LAST_PORT") $PROXY_PORTS; do
    pids=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u)
    for pid in $pids; do
        kill -9 "$pid" 2>/dev/null && echo "force killed pid $pid on port $port"
    done
done

sleep 3
echo "remaining ollama serve processes: $(pgrep -cf 'ollama serve')"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
