#!/usr/bin/env bash
set -euo pipefail

task_root=${MIDLAYER_ASCEND_CONTAINER_TASK_ROOT:?set MIDLAYER_ASCEND_CONTAINER_TASK_ROOT}
binary=$task_root/build_cold/split_fuse_mte2_interleaved
output=$task_root/raw_cold/msprof_counter_g8c4
if [[ -e "$output" ]]; then
    echo "counter output already exists: $output" >&2
    exit 1
fi
if [[ "${ASCEND_RT_VISIBLE_DEVICES:?}" != 0 || "${MIDLAYER_ASCEND_DEVICE:?}" != 0 ]]; then
    echo "counter check requires isolated physical NPU 0" >&2
    exit 1
fi
msprof op --aic-metrics=L2Cache,Memory --replay-mode=application \
    --warm-up=0 --kernel-name=split_fuse_mte2_interleaved \
    --launch-count=31 --output="$output" \
    --application="$binary 8 4 0"
