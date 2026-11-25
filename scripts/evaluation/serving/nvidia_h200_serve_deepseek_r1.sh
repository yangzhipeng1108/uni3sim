#!/usr/bin/env bash
# Real vLLM 0.24 DeepSeek-R1 architecture (dummy FP8 block-128 weights) on H200.
# MODEL: model dir (full DeepSeek-V3/R1 config, or a per-rank proxy written by
# nvidia_h200_deepseek_r1_rank_proxy_config.py); TP: tensor-parallel size; MTP: 0 or the
# number of MTP speculative tokens (2 = SGLang EAGLE steps 2 / 3 draft tokens,
# verify q_len 3).  Run from the operator-selected capture directory.
# Usage: MODEL=<dir> TP=<n> MTP=<0|k> PORT=<port> nvidia_h200_serve_deepseek_r1.sh [extra vllm serve args]
set -euo pipefail
B=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}
RUNTIME_PYTHON=${MIDLAYER_H200_RUNTIME_PYTHON:?set MIDLAYER_H200_RUNTIME_PYTHON to the selected runtime executable}
export HOME=$B/h200_microbench/home XDG_CACHE_HOME=$B/cache/xdg HF_HOME=$B/cache/hf \
       VLLM_CACHE_ROOT=$B/cache/vllm_dsr1 TRITON_CACHE_DIR=$B/cache/triton_dsr1 \
       TORCHINDUCTOR_CACHE_DIR=$B/cache/inductor_dsr1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
       FLASHINFER_WORKSPACE_BASE=$B/cache/flashinfer
mkdir -p $HOME $XDG_CACHE_HOME $HF_HOME $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $FLASHINFER_WORKSPACE_BASE
export CUDA_HOME=$B/tools/cuda-12.9
export PATH="$CUDA_HOME/bin:$(dirname -- "$RUNTIME_PYTHON"):$PATH"
SPEC=()
if [ "$MTP" != 0 ]; then
  SPEC=(--speculative-config "{\"method\": \"deepseek_mtp\", \"num_speculative_tokens\": $MTP}")
fi
exec "$RUNTIME_PYTHON" -m vllm.entrypoints.cli.main serve $MODEL \
  --served-model-name dsr1 --load-format dummy --tensor-parallel-size $TP \
  --max-model-len 9216 --gpu-memory-utilization ${GPU_MEM_UTIL:-0.90} --max-num-batched-tokens 32768 \
  --max-num-seqs 128 --no-enable-prefix-caching --trust-remote-code --port $PORT "${SPEC[@]}" "$@"
