#!/usr/bin/env bash
set -euo pipefail
container=${MIDLAYER_ASCEND_CONTAINER:?set MIDLAYER_ASCEND_CONTAINER}
physical_device=${MIDLAYER_ASCEND_DEVICE_ID:?set MIDLAYER_ASCEND_DEVICE_ID to a nonnegative integer}
case "$physical_device" in
  (''|*[!0-9]*) echo "MIDLAYER_ASCEND_DEVICE_ID must be a nonnegative integer" >&2; exit 2;;
esac
task_dir=${MIDLAYER_ASCEND_TASK_ROOT:?set MIDLAYER_ASCEND_TASK_ROOT}
container_dir=${MIDLAYER_ASCEND_CONTAINER_TASK_ROOT:?set MIDLAYER_ASCEND_CONTAINER_TASK_ROOT}
tokenizer=${MIDLAYER_ASCEND_TOKENIZER:?set MIDLAYER_ASCEND_TOKENIZER}
timeout 180 docker exec \
  -e ASCEND_RT_VISIBLE_DEVICES="$physical_device" \
  -e MIDLAYER_ASCEND_CONTAINER_TASK_ROOT="$container_dir" \
  -e MIDLAYER_ASCEND_TOKENIZER="$tokenizer" \
  -w "$container_dir" \
  "$container" \
  sh -c 'PYTHONPATH="$PWD:$PYTHONPATH" python run_tiny_async.py' \
  > "$task_dir/run2.log" 2>&1
