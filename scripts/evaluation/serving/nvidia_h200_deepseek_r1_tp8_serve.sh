#!/usr/bin/env bash
# Real vLLM 0.30 DeepSeek-R1-0528 server on 8x H200, TP8, dummy weights.
#
# Model dir: config/tokenizer-only copy of ModelScope deepseek-ai/DeepSeek-R1-0528
# (the model of the InferenceX dsr1-fp8-h200-sglang[-mtp] runs; config.json is
# the DeepSeek-V3 config except transformers_version: DeepseekV3ForCausalLM,
# 61 layers, 256 experts, one NextN/MTP layer, FP8 e4m3 block [128,128]
# dynamic activations; no *.safetensors).  vLLM 0.30 serves it through
# model_executor/models/deepseek_v2 with --load-format dummy (native
# transformers 5 DeepseekV3Config, no remote code).
#
# Same flags as the GLM-5.2 TP8 server (nvidia_h200_glm_5_2_tp8_serve.sh) except the
# model, the caller-chosen --gpu-memory-utilization (BF16 MLA KV for
# C=64 x 9,216 tokens needs ~38.6 GiB per GPU next to ~79.5 GiB of FP8
# weights), the port and the cache directories.  --max-num-batched-tokens
# 32768 is the InferenceX SGLang recipe's --chunked-prefill-size; prefix
# caching off like its --disable-radix-cache.  --generation-config vllm
# ignores the R1 generation_config.json (temperature 0.6 / top_p 0.95); the
# load generator sends temperature 0 like the InferenceX client.
#
# Required environment (set by nvidia_h200_deepseek_r1_tp8_capture_runner.py):
#   VLLM_MOE_ROUTING_SIMULATION_STRATEGY  uniform_random | normal_routing
#   DSR1_KV_CACHE_DTYPE                   auto | fp8
#   DSR1_GPU_MEMORY_UTILIZATION           e.g. 0.95
#   DSR1_PORT                             server port
# Extra arguments are passed to `vllm serve` (--speculative-config for MTP
# on, --profiler-config under nsys).
# Evaluation only; must run under with_gpu_locks.sh 0,1,2,3,4,5,6,7.
set -euo pipefail
: "${VLLM_MOE_ROUTING_SIMULATION_STRATEGY:?must be set (uniform_random)}"
: "${DSR1_KV_CACHE_DTYPE:?must be set (auto or fp8)}"
: "${DSR1_GPU_MEMORY_UTILIZATION:?must be set}"
: "${DSR1_PORT:?must be set}"
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
  --max-model-len 10240 --gpu-memory-utilization "$DSR1_GPU_MEMORY_UTILIZATION" \
  --max-num-batched-tokens 32768 --max-num-seqs 128 --no-enable-prefix-caching \
  --kv-cache-dtype "$DSR1_KV_CACHE_DTYPE" --generation-config vllm --port "$DSR1_PORT" "$@"
