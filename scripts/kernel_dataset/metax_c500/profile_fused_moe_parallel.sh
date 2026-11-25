#!/usr/bin/env bash
# Compatibility entry point for segmented fused-MoE profiling.
#
# Despite the historical filename, layer-1 mcProfiler is a host-level
# singleton and MUST NOT run concurrently across devices or ports. This script
# therefore executes the four ranges serially on one selected device.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
PY=${PYTHON_BIN:?set PYTHON_BIN to the selected capture runtime executable}
BUILD=scripts/kernel_dataset/metax_c500/build_dataset.py
CFG=configs/dataset/metax_c500.yaml
LOG_DIR="$REPO_ROOT/profiler_output/kernel_dataset/metax_c500/logs"
mkdir -p "$LOG_DIR"

: "${MACA_PATH:?MACA_PATH is required}"
: "${MACA_ETC_PATH:?MACA_ETC_PATH is required}"
: "${MACA_VISIBLE_DEVICES:?select one verified-idle C500 device}"
export MACA_PATH MACA_ETC_PATH MACA_VISIBLE_DEVICES

TOTAL=176
CHUNK=$(( (TOTAL + 3) / 4 ))

for segment in 0 1 2 3; do
    START=$(( segment * CHUNK + 1 ))
    END=$(( (segment + 1) * CHUNK ))
    if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi
    PORT=50123
    LOG="$LOG_DIR/profile_fused_moe_segment${segment}.log"

    echo "Device $MACA_VISIBLE_DEVICES, serial segment $segment: " \
         "shapes $START-$END (port $PORT) -> $LOG"

    "$PY" -u "$BUILD" \
        --config "$CFG" \
        --kernel fused_moe \
        --stage profile \
        --start-idx "$START" \
        --end-idx "$END" \
        --port "$PORT" \
        2>&1 | tee "$LOG"
done

echo "=== ALL SERIAL SEGMENTS DONE ==="
