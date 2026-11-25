#!/bin/bash
PY=${PY:?set PY to the capture runtime Python executable}
# Clock-gated capture driver for sm90_fp8_block_tile_wave_refill_runner.py.
# usage: sm90_fp8_block_tile_wave_refill_capture.sh GPU RUN_DIR
# Two fresh processes of each FP8 tile program on one owned GPU; a process
# whose CLOCK row is not accepted (any sample < 1980 MHz) is kept on disk and
# retried, at most four attempts.  The remote execution copy keeps both
# runners at $ROOT/src/probe/.
GPU=$1; RUN=$2
ROOT=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/wp11_moe_fp8
mkdir -p $RUN; cd $RUN
export TRITON_CACHE_DIR=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/wp11/triton_gpu$GPU
export PYTHONPATH="$ROOT/src/probe${PYTHONPATH:+:$PYTHONPATH}"
for proc in 1 2; do
  for prog in bm16_bn128_bk128_w4_s3 bm16_bn128_bk128_w8_s4 bm16_bn128_bk128_w4_s4 bm16_bn128_bk128_w4_s5 bm32_bn128_bk128_w4_s3 bm64_bn128_bk128_w4_s4 bm64_bn128_bk128_w4_s3; do
    for attempt in 1 2 3 4; do
      tag=gpu${GPU}_${prog}_p${proc}_a${attempt}
      CUDA_VISIBLE_DEVICES=$GPU "$PY" $ROOT/src/probe/sm90_fp8_block_tile_wave_refill_runner.py --program $prog --trials 9 --clock-log $tag.clock.csv > $tag.csv 2> $tag.err
      if grep -q "^CLOCK,$GPU,.*,1$" $tag.csv; then echo "ACCEPT $tag"; break; else echo "REJECT $tag"; fi
    done
  done
done
echo DONE
