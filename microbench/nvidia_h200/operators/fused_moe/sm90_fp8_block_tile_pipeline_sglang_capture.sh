#!/bin/bash
PY=${PY:?set PY to the capture runtime Python executable}
# Clock-gated capture driver for sm90_fp8_block_tile_pipeline_sglang_runner.py.
# usage: sm90_fp8_block_tile_pipeline_sglang_capture.sh GPU RUN_DIR SRC_DIR [PROGRAMS]
# Two fresh processes per SGLang tile program on one owned GPU, in the
# SGLang 0.5.12 environment; a process whose CLOCK row is not accepted (any
# sample < 1980 MHz) is kept on disk and retried, at most four attempts.
# SRC_DIR holds this runner and sm90_fp8_block_tile_pipeline_runner.py.
# PROGRAMS (space separated) defaults to the eight decode-range programs.
GPU=$1; RUN=$2; SRC=$3
PROGRAMS=${4:-"sgl_bm16_bn64_bk128_w4_s5 sgl_bm16_bn64_bk128_w4_s3 sgl_bm16_bn64_bk128_w4_s4 sgl_bm16_bn128_bk128_w4_s3 sgl_bm16_bn128_bk128_w4_s4 sgl_tma_bm16_bn128_bk128_w4_s2 sgl_tma_bm16_bn128_bk64_w4_s3 sgl_tma_bm16_bn256_bk64_w4_s3"}
mkdir -p $RUN; cd $RUN
export PYTHONNOUSERSITE=1 PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/sgl_tile/triton_gpu$GPU
for proc in 1 2; do
  for prog in $PROGRAMS; do
    for attempt in 1 2 3 4; do
      tag=gpu${GPU}_${prog}_p${proc}_a${attempt}
      CUDA_VISIBLE_DEVICES=$GPU "$PY" $SRC/sm90_fp8_block_tile_pipeline_sglang_runner.py --program $prog --trials 9 --clock-log $tag.clock.csv > $tag.csv 2> $tag.err
      if grep -q "^CLOCK,$GPU,.*,1$" $tag.csv; then echo "ACCEPT $tag"; break; else echo "REJECT $tag"; fi
    done
  done
done
echo DONE
