#!/usr/bin/env bash
# Serve Qwen3.5-4B through vLLM's OpenAI-compatible API.
#
# Ollama pins qwen3-family models to a single inference slot, capping aggregate
# throughput at ~48 tok/s regardless of concurrency. vLLM batches continuously,
# so the same model can use the full GPU.
#
# Note this serves bf16 weights, whereas Ollama served a Q4_K_M quantisation of
# the same checkpoint. Outputs are not bit-identical to earlier Ollama runs.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
SERVED_NAME="${SERVED_NAME:-qwen3.5:4b}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_FRACTION="${GPU_FRACTION:-0.70}"
# Qwen3.5 is a reasoning model: left in thinking mode it prefixes every answer
# with its own "Thinking Process: ...", which would be scored as the answer.
# Turned off in the chat template rather than stripped afterwards.
ENABLE_THINKING="${ENABLE_THINKING:-false}"
# The checkpoint ships no generation_config.json, so vLLM would fall back to
# unconstrained sampling and the model degenerates into mixed-language text.
# These are the parameters Ollama applied, kept identical so this endpoint
# generates like the one the other systems ran against.
SAMPLING='{"temperature": 1.0, "top_k": 20, "top_p": 0.95, "presence_penalty": 1.5}'
LOG_DIR="${LOG_DIR:-$(dirname "${BASH_SOURCE[0]}")/../../logs/vllm}"
VENV="${VENV:-$HOME/vllm-env}"

mkdir -p "$LOG_DIR"

# FlashInfer JIT-compiles its sampling kernels on first use, which needs ninja
# and a CUDA toolchain. The PyTorch-native sampler is used instead and gives
# the same results here.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" -o /dev/null; then
    echo "vLLM already serving on $PORT"
    exit 0
fi

setsid nohup "$VENV/bin/vllm" serve "$MODEL" \
    --port "$PORT" --host 127.0.0.1 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_FRACTION" \
    --served-model-name "$SERVED_NAME" \
    --default-chat-template-kwargs "{\"enable_thinking\": $ENABLE_THINKING}" \
    --override-generation-config "$SAMPLING" \
    >"$LOG_DIR/server.log" 2>&1 < /dev/null &

echo "launched, waiting for readiness"
for i in $(seq 1 90); do
    sleep 10
    if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" -o /dev/null; then
        echo "READY after ~$((i * 10))s"
        curl -s "http://127.0.0.1:$PORT/v1/models"
        exit 0
    fi
done

echo "did not become ready; last log lines:"
tail -20 "$LOG_DIR/server.log"
exit 1
