#!/bin/bash
# AIV instruction-fetch rate sweep.
#
# Six binaries hold the same AIV work (one 64-element GM->UB copy, one
# UB-resident vector chain, one UB->GM copy) behind a different number of
# straight-line vector instructions, so they differ only in kernel-text bytes.
# Each process issues LAUNCHES standalone launches of one binary: the first
# runs with a cold AI-vector instruction cache, the rest re-execute resident
# text.  msprof records per-task duration and aiv_icache_miss_rate, so the PMU
# decides which tasks were cold rather than the driver asserting it.
set -u
BASE=${BASE:?BASE must point at the deployed build directory}
OUT=${OUT:?OUT must point at the capture directory}
DEVICE=${DEVICE:?DEVICE must name one idle NPU}
BD=${BD:-48}
LAUNCHES=${LAUNCHES:-8}
REPEATS=${REPEATS:-4}
mkdir -p "$OUT"
for n in 1 32 96 192 384 768; do
  for rep in $(seq 1 "$REPEATS"); do
    d="$OUT/n${n}_r${rep}"
    rm -rf "$d"
    ASCEND_RT_VISIBLE_DEVICES=$DEVICE ASCEND_DEVICE_ID=0 \
      msprof --output="$d" --task-time=on --ai-core=on \
      --application="$BASE/aiv_text_footprint_${n}_bench $BD $LAUNCHES" \
      > "$OUT/n${n}_r${rep}.log" 2>&1
    echo "done n=$n rep=$rep"
  done
done
