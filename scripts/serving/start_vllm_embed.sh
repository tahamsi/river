#!/usr/bin/env bash
# Serve Qwen3-Embedding-0.6B through vLLM alongside the chat server.
# The chat server holds 0.70 of the GPU, so this one takes a small slice.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Embedding-0.6B}"
SERVED_NAME="${SERVED_NAME:-qwen3-embedding:0.6b}"
PORT="${PORT:-8001}"
MAX_LEN="${MAX_LEN:-2048}"
GPU_FRACTION="${GPU_FRACTION:-0.10}"
LOG_DIR="${LOG_DIR:-$(dirname "${BASH_SOURCE[0]}")/../../logs/vllm}"
VENV="${VENV:-$HOME/vllm-env}"

mkdir -p "$LOG_DIR"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" -o /dev/null; then
    echo "embedding server already on $PORT"
    exit 0
fi

setsid nohup "$VENV/bin/vllm" serve "$MODEL" \
    --port "$PORT" --host 127.0.0.1 \
    --runner pooling \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_FRACTION" \
    --served-model-name "$SERVED_NAME" \
    >"$LOG_DIR/embed-server.log" 2>&1 < /dev/null &

echo "launched, waiting for readiness"
for i in $(seq 1 60); do
    sleep 10
    if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" -o /dev/null; then
        echo "READY after ~$((i * 10))s"
        exit 0
    fi
done

echo "did not become ready; last log lines:"
tail -20 "$LOG_DIR/embed-server.log"
exit 1
