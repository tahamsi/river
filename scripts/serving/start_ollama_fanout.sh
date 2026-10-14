#!/usr/bin/env bash
# Start pools of independent Ollama daemons plus a round-robin proxy per pool.
#
# Ollama 0.33.2 pins every qwen3-family model to a single llama-server slot
# ("model architecture does not currently support parallel requests"), so one
# daemon serialises requests regardless of OLLAMA_NUM_PARALLEL. Running several
# daemons and fanning out across them restores concurrency without overriding
# that per-architecture restriction.
#
# Chat and embeddings get separate pools because context length is a per-daemon
# setting: at 32k the 639MB embedding model reserves 5.8GB of KV cache, which is
# pure waste. Embedding daemons run with a short context instead.
set -euo pipefail

CHAT_INSTANCES="${CHAT_INSTANCES:-6}"
EMBED_INSTANCES="${EMBED_INSTANCES:-3}"
CHAT_BASE_PORT="${CHAT_BASE_PORT:-11435}"
EMBED_BASE_PORT="${EMBED_BASE_PORT:-11445}"
CHAT_PROXY_PORT="${CHAT_PROXY_PORT:-11500}"
EMBED_PROXY_PORT="${EMBED_PROXY_PORT:-11501}"
CHAT_MODEL="${CHAT_MODEL:-qwen3.5:4b}"
EMBED_MODEL="${EMBED_MODEL:-qwen3-embedding:0.6b}"
CHAT_CONTEXT="${CHAT_CONTEXT:-32768}"
EMBED_CONTEXT="${EMBED_CONTEXT:-2048}"
LOG_DIR="${LOG_DIR:-$(dirname "${BASH_SOURCE[0]}")/../../logs/fanout}"
PROXY="${PROXY:-$(dirname "${BASH_SOURCE[0]}")/ollama_fanout_proxy.py}"

mkdir -p "$LOG_DIR"

# Route every daemon through CUDA MPS so their kernels share one GPU context
# instead of being time-sliced. Harmless if the MPS daemon is not running.
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"

start_pool() {
    local count="$1" base="$2" context="$3" label="$4"
    local backends="" port
    for i in $(seq 0 $((count - 1))); do
        port=$((base + i))
        if ss -ltnH "sport = :$port" | grep -q LISTEN; then
            echo "  $label port $port already listening, reusing" >&2
        else
            OLLAMA_HOST="127.0.0.1:$port" \
            OLLAMA_MODELS=/usr/share/ollama/.ollama/models \
            OLLAMA_NUM_PARALLEL=1 \
            OLLAMA_MAX_LOADED_MODELS=1 \
            OLLAMA_KEEP_ALIVE=24h \
            OLLAMA_CONTEXT_LENGTH="$context" \
            nohup ollama serve >"$LOG_DIR/$label-$port.log" 2>&1 &
            echo "  started $label instance on $port (pid $!)" >&2
        fi
        backends="${backends:+$backends,}http://127.0.0.1:$port"
    done
    for i in $(seq 0 $((count - 1))); do
        port=$((base + i))
        for _ in $(seq 1 60); do
            curl -sf "http://127.0.0.1:$port/api/tags" -o /dev/null && break
            sleep 1
        done
    done
    echo "$backends"
}

start_proxy() {
    local backends="$1" port="$2" label="$3"
    if ss -ltnH "sport = :$port" | grep -q LISTEN; then
        echo "  $label proxy already on $port"
    else
        FANOUT_BACKENDS="$backends" FANOUT_PORT="$port" \
            nohup python3 "$PROXY" >"$LOG_DIR/proxy-$port.log" 2>&1 &
        echo "  started $label proxy on $port (pid $!)"
    fi
}

echo "chat pool ($CHAT_INSTANCES x $CHAT_MODEL, ctx $CHAT_CONTEXT):"
CHAT_BACKENDS=$(start_pool "$CHAT_INSTANCES" "$CHAT_BASE_PORT" "$CHAT_CONTEXT" chat)

echo "embed pool ($EMBED_INSTANCES x $EMBED_MODEL, ctx $EMBED_CONTEXT):"
EMBED_BACKENDS=$(start_pool "$EMBED_INSTANCES" "$EMBED_BASE_PORT" "$EMBED_CONTEXT" embed)

echo "preloading models"
for i in $(seq 0 $((CHAT_INSTANCES - 1))); do
    port=$((CHAT_BASE_PORT + i))
    curl -sf "http://127.0.0.1:$port/api/generate" \
        -d "{\"model\":\"$CHAT_MODEL\",\"prompt\":\"ok\",\"stream\":false,\"keep_alive\":\"24h\",\"options\":{\"num_predict\":1}}" \
        -o /dev/null &
done
for i in $(seq 0 $((EMBED_INSTANCES - 1))); do
    port=$((EMBED_BASE_PORT + i))
    curl -sf "http://127.0.0.1:$port/api/embed" \
        -d "{\"model\":\"$EMBED_MODEL\",\"input\":\"ok\",\"keep_alive\":\"24h\"}" \
        -o /dev/null &
done
wait

start_proxy "$CHAT_BACKENDS" "$CHAT_PROXY_PORT" chat
start_proxy "$EMBED_BACKENDS" "$EMBED_PROXY_PORT" embed
sleep 2

echo "chat  proxy  http://127.0.0.1:$CHAT_PROXY_PORT  -> $CHAT_BACKENDS"
echo "embed proxy  http://127.0.0.1:$EMBED_PROXY_PORT  -> $EMBED_BACKENDS"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
