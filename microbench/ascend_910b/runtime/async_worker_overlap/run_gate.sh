#!/usr/bin/env bash
set -euo pipefail
container=${MIDLAYER_ASCEND_CONTAINER:?set MIDLAYER_ASCEND_CONTAINER}
physical_device=${MIDLAYER_ASCEND_DEVICE_ID:?set MIDLAYER_ASCEND_DEVICE_ID to a nonnegative integer}
case "$physical_device" in
  (''|*[!0-9]*) echo "MIDLAYER_ASCEND_DEVICE_ID must be a nonnegative integer" >&2; exit 2;;
esac

task_dir=${MIDLAYER_ASCEND_TASK_ROOT:?set MIDLAYER_ASCEND_TASK_ROOT}
container_dir=${MIDLAYER_ASCEND_CONTAINER_TASK_ROOT:?set MIDLAYER_ASCEND_CONTAINER_TASK_ROOT}
wake_runner=${MIDLAYER_ASCEND_ASYNC_WAKE_RUNNER:?set MIDLAYER_ASCEND_ASYNC_WAKE_RUNNER}
tokenizer=${MIDLAYER_ASCEND_TOKENIZER:?set MIDLAYER_ASCEND_TOKENIZER}
timeout 240 docker exec \
  -e ASCEND_RT_VISIBLE_DEVICES="$physical_device" \
  -e MIDLAYER_ASCEND_CONTAINER_TASK_ROOT="$container_dir" \
  -e MIDLAYER_ASCEND_TOKENIZER="$tokenizer" \
  -e MID_WORKER_OVERLAP_GATE=1 \
  -e MID_WORKER_OVERLAP_GATE_NODES=512 \
  -e MID_WORKER_OVERLAP_GATE_MODE=graph \
  -e MID_WORKER_OVERLAP_OUTPUT_DIR="$container_dir/output_gate3" \
  -w "$container_dir" \
  "$container" \
  bash -lc "PYTHONPATH=\"$container_dir/source:\$PYTHONPATH\" python \"$wake_runner\"" \
  > "$task_dir/run_gate3.log" 2>&1
