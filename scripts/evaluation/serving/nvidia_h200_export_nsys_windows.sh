#!/usr/bin/env bash
# sqlite-export every window_N.nsys-rep of one capture directory and write the
# device-0 kernel aggregation (aggregate_nvidia_h200_nsys_window_kernels.py) next to
# it as window_N_dev0_share.json.  Remote copy:
# the operator-selected capture directory.  Evaluation only.
# Usage: nvidia_h200_export_nsys_windows.sh <capture dir>
set -u
B=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}; D=$B/h200_microbench/dsr1_mtpoff
RUNTIME_PYTHON=${MIDLAYER_H200_RUNTIME_PYTHON:?set MIDLAYER_H200_RUNTIME_PYTHON to the selected runtime executable}
for f in $1/window_*.nsys-rep; do
  s=${f%.nsys-rep}.sqlite
  [ -f $s ] || $B/tools/nsys_2025.6.1/bin/nsys export --type sqlite --output $s $f > /dev/null 2>&1
  "$RUNTIME_PYTHON" $D/bin/aggregate_nvidia_h200_nsys_window_kernels.py $s > ${f%.nsys-rep}_dev0_share.json
done
echo done
