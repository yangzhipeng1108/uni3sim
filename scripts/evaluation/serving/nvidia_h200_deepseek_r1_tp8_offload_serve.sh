#!/usr/bin/env bash
# vLLM 0.30 DeepSeek-R1-0528 TP8 server for the KV host-offload group.
#
# Same model, dummy FP8 weights, TP8, routing simulation, BF16 KV, chunk 32768,
# --max-num-seqs 128 and cache directories as nvidia_h200_deepseek_r1_tp8_serve.sh (section 3b); what differs:
#   --enable-prefix-caching (3b: off, InferenceX 8K/1K --disable-radix-cache) -- the AgentX
#       workload is multi-turn and every arm of this group needs GPU prefix caching;
#   --max-model-len $DSR1_MAX_MODEL_LEN (163840 = DeepSeek-R1 max_position_embeddings; AgentX
#       sessions reach 100K+ tokens);
#   --enable-prompt-tokens-details --enable-per-request-metrics (per-request cached tokens and
#       server-side queue/TTFT/ITL on the final usage chunk);
#   the offload arm's flags come as extra arguments (--kv-offloading-size <GiB>
#       --kv-offloading-backend native), VLLM_USE_SIMPLE_KV_OFFLOAD selects
#       OffloadingConnector (0) or SimpleCPUOffloadConnector (1) for backend native.
#
# Required environment (set by nvidia_h200_deepseek_r1_tp8_capture_runner.py run --phases offload):
#   VLLM_MOE_ROUTING_SIMULATION_STRATEGY  uniform_random | normal_routing
#   DSR1_KV_CACHE_DTYPE                   auto | fp8
#   DSR1_GPU_MEMORY_UTILIZATION           e.g. 0.95
#   DSR1_PORT                             server port
#   DSR1_MAX_MODEL_LEN                    163840
#   VLLM_USE_SIMPLE_KV_OFFLOAD            0 | 1
# Evaluation only; must run under with_gpu_locks.sh 0,1,2,3,4,5,6,7.
set -euo pipefail
: "${VLLM_MOE_ROUTING_SIMULATION_STRATEGY:?must be set (uniform_random)}"
: "${DSR1_KV_CACHE_DTYPE:?must be set (auto or fp8)}"
: "${DSR1_GPU_MEMORY_UTILIZATION:?must be set}"
: "${DSR1_PORT:?must be set}"
: "${DSR1_MAX_MODEL_LEN:?must be set}"
: "${VLLM_USE_SIMPLE_KV_OFFLOAD:?must be set (0 or 1)}"
B=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}
RUNTIME_PYTHON=${MIDLAYER_H200_RUNTIME_PYTHON:?set MIDLAYER_H200_RUNTIME_PYTHON to the selected runtime executable}
MODEL=$B/models/DeepSeek-R1-0528-config
export HOME=$B/h200_microbench/home XDG_CACHE_HOME=$B/cache/xdg HF_HOME=$B/cache/hf \
       VLLM_CACHE_ROOT=$B/cache/vllm_dsr1 TRITON_CACHE_DIR=$B/cache/triton_dsr1 \
       TORCHINDUCTOR_CACHE_DIR=$B/cache/inductor_dsr1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export CUDA_HOME=$B/tools/cuda-12.9
export PATH="$CUDA_HOME/bin:$(dirname -- "$RUNTIME_PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec "$RUNTIME_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name dsr1-fp8 --load-format dummy --tensor-parallel-size 8 \
  --max-model-len "$DSR1_MAX_MODEL_LEN" --gpu-memory-utilization "$DSR1_GPU_MEMORY_UTILIZATION" \
  --max-num-batched-tokens 32768 --max-num-seqs 128 --enable-prefix-caching \
  --enable-prompt-tokens-details --enable-per-request-metrics \
  --kv-cache-dtype "$DSR1_KV_CACHE_DTYPE" --generation-config vllm --port "$DSR1_PORT" "$@"
