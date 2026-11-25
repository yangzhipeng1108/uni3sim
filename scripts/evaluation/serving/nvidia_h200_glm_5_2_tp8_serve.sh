#!/usr/bin/env bash
# Real vLLM 0.30 GLM-5.2 server on 8x H200, TP8, dummy weights.
#
# Model dir: config/tokenizer-only copy of ModelScope ZhipuAI/GLM-5.2-FP8
# (glm_moe_dsa, 78 layers, 256 experts, FP8 e4m3 block [128,128] dynamic
# activations; no *.safetensors).  vLLM 0.30 serves it through
# vllm/models/deepseek_v32 (GlmMoeDsaForCausalLM) with --load-format dummy.
# FP8 is required: BF16 weights need ~176 GiB per rank at TP8 (> 140 GiB
# HBM), FP8 weights ~88 GiB per rank.
#
# Same flags as the DSV3-L32 H200 server (serve_dsv3_l32.sh) except: TP8,
# vLLM 0.30 env, --max-model-len 10240 (8192 + 1024 plus margin for the
# random dataset's re-encode mismatch), --kv-cache-dtype from the caller,
# and --generation-config vllm (the GLM repo ships top_p=0.95; the DSV3-L32
# config dir had no generation_config.json, so this keeps vLLM's default
# sampling params and the same sampler path as the DSV3 captures).
#
# Required environment (set by nvidia_h200_glm_5_2_tp8_capture_runner.py):
#   VLLM_MOE_ROUTING_SIMULATION_STRATEGY  uniform_random | normal_routing
#   GLM52_KV_CACHE_DTYPE                  fp8_ds_mla | auto
#   GLM52_PORT                            server port
# Extra arguments are passed to `vllm serve` (e.g. --profiler-config).
# Evaluation only; must run under with_gpu_locks.sh 0,1,2,3,4,5,6,7.
set -euo pipefail
: "${VLLM_MOE_ROUTING_SIMULATION_STRATEGY:?must be set (uniform_random)}"
: "${GLM52_KV_CACHE_DTYPE:?must be set (fp8_ds_mla or auto)}"
: "${GLM52_PORT:?must be set}"
B=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}
RUNTIME_PYTHON=${MIDLAYER_H200_RUNTIME_PYTHON:?set MIDLAYER_H200_RUNTIME_PYTHON to the selected runtime executable}
MODEL=$B/models/GLM-5.2-FP8-config
export HOME=$B/h200_microbench/home XDG_CACHE_HOME=$B/cache/xdg HF_HOME=$B/cache/hf \
       VLLM_CACHE_ROOT=$B/cache/vllm_glm52 TRITON_CACHE_DIR=$B/cache/triton_glm52 \
       TORCHINDUCTOR_CACHE_DIR=$B/cache/inductor_glm52 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export CUDA_HOME=$B/tools/cuda-12.9
export PATH="$CUDA_HOME/bin:$(dirname -- "$RUNTIME_PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec "$RUNTIME_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name glm52-fp8 --load-format dummy --tensor-parallel-size 8 \
  --max-model-len 10240 --gpu-memory-utilization 0.90 --max-num-batched-tokens 32768 \
  --max-num-seqs 128 --no-enable-prefix-caching --kv-cache-dtype "$GLM52_KV_CACHE_DTYPE" \
  --generation-config vllm --port "$GLM52_PORT" "$@"
