#!/usr/bin/env bash
# DeepSeek-R1 on 8x H200, 2026-09-27 8-GPU capture window.
# Runs, back to back on physical GPUs 0-7 (all must be free and owned):
#   1. real vLLM 0.24 DSR1 TP8/EP1 colocated (full 61-layer DeepSeek-V3/R1
#      config, dummy FP8 block-128 weights, uniform-random routing, CUDA
#      graphs), MTP off: pure-decode nsys windows at C = 1..64 (8K/1K
#      InferenceX lengths) via nvidia_h200_deepseek_r1_nsys_decode_capture.sh;
#   2. the same server with MTP on (deepseek_mtp, 2 speculative tokens =
#      SGLang EAGLE 3 draft tokens, verify q_len 3);
#   3. the 8-rank collective mechanisms: FlashInfer trtllm all-reduce +
#      residual RMSNorm (one-shot Lamport, the vLLM 0.24 TP8 decode fusion)
#      and vLLM custom all-reduce, world 8, token rows up to 192 (3 x 64).
# The DeepGEMM JIT cache from the 2026-09-26 single-GPU per-rank proxy runs
# (same per-rank N, K) is reused through the shared cache directories.
# Run from the operator-selected capture directory.
# Usage: nvidia_h200_deepseek_r1_tp8_day2.sh <source commit> [off|on|ar ...]   (default: off on ar)
set -u
B=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}; D=$B/h200_microbench/dsr1_mtpoff
RUNTIME_PYTHON=${MIDLAYER_H200_RUNTIME_PYTHON:?set MIDLAYER_H200_RUNTIME_PYTHON to the selected runtime executable}
COMMIT=$1; shift
STEPS=${*:-off on ar}
OUT=$D/day2; mkdir -p $OUT
# One process per GPU: hold every card's whole-card process lock (the lock
# kcampaign/launch.sh takes) for the whole run, so no other capture can create
# a CUDA context on these GPUs while this script measures.
K=$B/h200_microbench/kcampaign
for GPU_INDEX in 0 1 2 3 4 5 6 7; do
  exec {LOCK_FD}>>$K/runs/gpu$GPU_INDEX.proc.lock
  flock -x $LOCK_FD
done
MODEL=$B/models/DeepSeek-V3-config
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
log() { echo "$(date '+%F %T') $*" | tee -a $OUT/day2.log; }
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a $OUT/day2.log
for STEP in $STEPS; do
  case $STEP in
    off|on)
      MTP=0; [ $STEP = on ] && MTP=2
      log "start TP8 MTP $STEP capture"
      MODEL=$MODEL TP=8 MTP=$MTP PORT=8031 GPU_MEM_UTIL=0.93 \
        $D/bin/nvidia_h200_deepseek_r1_nsys_decode_capture.sh $OUT/tp8_mtp$STEP 1 2 4 8 16 32 64 \
        > $OUT/capture_mtp$STEP.log 2>&1
      log "end TP8 MTP $STEP capture rc=$?"
      $D/bin/nvidia_h200_export_nsys_windows.sh $OUT/tp8_mtp$STEP >> $OUT/day2.log 2>&1
      ;;
    ar)
      export HOME=$B/h200_microbench/home XDG_CACHE_HOME=$B/cache/xdg \
             FLASHINFER_WORKSPACE_BASE=$B/cache/flashinfer CUDA_HOME=$B/tools/cuda-12.9
      log "start 8-rank all-reduce probes"
      "$RUNTIME_PYTHON" $D/microbench/flashinfer_allreduce_fusion_runner.py \
        --world-sizes 8 --output-dir $OUT/ar_flashinfer_fusion_ws8 --source-commit $COMMIT \
        --token-counts 1 2 3 4 6 8 12 16 24 32 48 64 96 128 192 > $OUT/ar_flashinfer.log 2>&1
      log "flashinfer fusion rc=$?"
      "$RUNTIME_PYTHON" $D/microbench/custom_allreduce_latency_runner.py \
        --world-sizes 8 --output-dir $OUT/ar_custom_ws8 --source-commit $COMMIT \
        > $OUT/ar_custom.log 2>&1
      log "custom all-reduce rc=$?"
      ;;
    *) log "unknown step $STEP"; exit 2 ;;
  esac
done
log "done"
