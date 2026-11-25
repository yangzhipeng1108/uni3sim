#!/bin/bash
PY=${PY:?set PY to the capture runtime Python executable}
# Clock-gated capture driver for sm90_fp8_block_tile_wave_refill_sglang_runner.py.
# usage: sm90_fp8_block_tile_wave_refill_sglang_capture.sh GPU RUN_DIR SRC_DIR [L2_PROGRAMS [DRAM_PROGRAMS]]
# L2-resident multi-wave refill of every SGLang tile program, and the
# DRAM-fed variant at the DSR1 TP8 per-rank W2 depth (K 256/512/1024) for the
# TMA W2 programs; two fresh processes each, at most four attempts.
GPU=$1; RUN=$2; SRC=$3
L2_PROGRAMS=${4:-"sgl_bm16_bn64_bk128_w4_s5 sgl_bm16_bn64_bk128_w4_s3 sgl_bm16_bn64_bk128_w4_s4 sgl_bm16_bn128_bk128_w4_s3 sgl_bm16_bn128_bk128_w4_s4 sgl_tma_bm16_bn128_bk128_w4_s2 sgl_tma_bm16_bn128_bk64_w4_s3 sgl_tma_bm16_bn256_bk64_w4_s3"}
DRAM_PROGRAMS=${5:-"sgl_tma_bm16_bn128_bk128_w4_s2 sgl_tma_bm16_bn128_bk64_w4_s3 sgl_tma_bm16_bn256_bk64_w4_s3"}
mkdir -p $RUN; cd $RUN
export PYTHONNOUSERSITE=1 PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/sgl_tile/triton_gpu$GPU
run() {  # tag args...
  local base=$1; shift
  for attempt in 1 2 3 4; do
    tag=${base}_a${attempt}
    CUDA_VISIBLE_DEVICES=$GPU "$PY" $SRC/sm90_fp8_block_tile_wave_refill_sglang_runner.py "$@" --trials 9 --clock-log $tag.clock.csv > $tag.csv 2> $tag.err
    if grep -q "^CLOCK,$GPU,.*,1$" $tag.csv; then echo "ACCEPT $tag"; return; else echo "REJECT $tag"; fi
  done
}
for proc in 1 2; do
  for prog in $L2_PROGRAMS; do
    run gpu${GPU}_${prog}_l2_p${proc} --program $prog
  done
  for prog in $DRAM_PROGRAMS; do
    run gpu${GPU}_${prog}_dram_p${proc} --program $prog --k-values 256 512 1024 --dram
  done
done
echo DONE
