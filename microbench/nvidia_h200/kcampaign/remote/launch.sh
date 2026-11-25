#!/bin/bash
# usage: launch.sh <gpu> <plan_name> [extra nsys_campaign_runner args...]
# Canonical A800 protocol (nsys_nvtx_kernel_median, 3 windows, 1980 MHz gate);
# one flock per physical GPU so campaigns on different GPUs never share a lock.
# PY, NSYS, CUDA_HOME and REPO are caller inputs; PYTHONPATH may add dependencies.
set -e
PY=${PY:?set PY to the capture runtime Python executable}
NSYS=${NSYS:?set NSYS to the capture profiler executable}
export CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to the capture toolkit}
R=${REPO:?set REPO to the capture source snapshot}
K=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/kcampaign
export XDG_CACHE_HOME=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache
export TRITON_CACHE_DIR=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/triton_kc_gpu$1
export VLLM_CACHE_ROOT=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/vllm_kc_gpu$1
export TORCHINDUCTOR_CACHE_DIR=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/inductor_kc_gpu$1
export PATH="$CUDA_HOME/bin:$PATH"
GPU=$1; PLAN=$2; shift 2
"$PY" "$K/write_provenance.py" "$R" "$K/plans/$PLAN.json" "$K/runs/$PLAN" "$GPU"
cd "$R"
exec "$PY" scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py \
  --plans "$K/plans/$PLAN.json" --output-dir "$K/runs/$PLAN" \
  --physical-gpu "$GPU" --nsys "$NSYS" \
  --python "$PY" --gpu-lock "$K/runs/gpu$GPU.lock" "$@"
