#!/bin/bash
PY=${PY:?set PY to the capture runtime Python executable}
# Clock-gated serial capture of the FP8 multi-wave refill probe at the TP8
# per-rank W2 depth (K = 256, two K blocks) next to K = 512 and 1024, with L2-resident
# (wrapped) and DRAM-streamed operands, for the tile program the DSR1 TP8 W2
# decode launches resolve to (bm16 bn128 bk128, 4 warps, 3 stages).  One
# process at a time on one owned GPU; a process whose CLOCK row is not
# accepted is kept and retried (at most four attempts).
# usage: sm90_fp8_block_tile_wave_refill_k256_capture.sh GPU RUN_DIR PROBE_DIR [PROGRAM]
# PROGRAM defaults to bm16_bn128_bk128_w4_s3 (TP8 W2 decode); the TP8
# prefill W2 launches resolve to bm64_bn128_bk128_w4_s3.
GPU=$1; RUN=$2; PROBE=$3; PROGRAM=${4:-bm16_bn128_bk128_w4_s3}
mkdir -p $RUN; cd $RUN
export TRITON_CACHE_DIR=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/dsr1_mtpoff/triton_gpu$GPU
export PYTHONPATH="$PROBE${PYTHONPATH:+:$PYTHONPATH}"
for mode in l2 dram; do
  flag=""; [ $mode = dram ] && flag="--dram"
  for proc in 1 2; do
    for attempt in 1 2 3 4; do
      tag=gpu${GPU}_${PROGRAM}_${mode}_p${proc}_a${attempt}
      CUDA_VISIBLE_DEVICES=$GPU "$PY" $PROBE/sm90_fp8_block_tile_wave_refill_runner.py --program $PROGRAM --trials 9 --k-values 256 512 1024 $flag --clock-log $tag.clock.csv > $tag.csv 2> $tag.err
      if grep -q "^CLOCK,$GPU,.*,1$" $tag.csv; then echo "ACCEPT $tag"; break; else echo "REJECT $tag"; fi
    done
  done
done
echo DONE
