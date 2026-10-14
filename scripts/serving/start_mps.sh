#!/usr/bin/env bash
# Start the CUDA Multi-Process Service.
#
# Without MPS the driver time-slices CUDA contexts, so N separate llama-server
# processes on one GPU run strictly one at a time -- concurrency gains nothing.
# MPS funnels their kernels through a single context so they execute together.
set -euo pipefail

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

if pgrep -f nvidia-cuda-mps-control >/dev/null; then
    echo "MPS control daemon already running"
else
    nvidia-cuda-mps-control -d
    sleep 2
    echo "MPS control daemon started"
fi

echo get_server_list | nvidia-cuda-mps-control || true
pgrep -af nvidia-cuda-mps-control || echo "warning: control daemon not visible"
