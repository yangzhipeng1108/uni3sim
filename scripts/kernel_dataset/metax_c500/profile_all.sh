#!/usr/bin/env bash
# Run full mcProfiler + mcTracer profiling for all kernel types.
# Run inside the C500 profiler container from any checkout location.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
PY=${PYTHON_BIN:?set PYTHON_BIN to the selected capture runtime executable}

# Device selection is deliberately explicit: the operator must first verify
# that this physical C500 is idle. Layer-1 mcProfiler is host-global, so a
# different visible device or port does not make concurrent collection safe.
: "${MACA_VISIBLE_DEVICES:?select one verified-idle C500 device}"
: "${MACA_PATH:?MACA_PATH is required}"
: "${MACA_ETC_PATH:?MACA_ETC_PATH is required}"
export MACA_VISIBLE_DEVICES MACA_PATH MACA_ETC_PATH
BUILD=scripts/kernel_dataset/metax_c500/build_dataset.py
CFG=configs/dataset/metax_c500.yaml
LOG_DIR="$REPO_ROOT/profiler_output/kernel_dataset/metax_c500/logs"
mkdir -p "$LOG_DIR"

echo "=== Starting full profile run at $(date) ==="

for kernel in gemm flash_attn fused_moe vector; do
    echo ">>> Profiling ${kernel}..."
    "$PY" "$BUILD" --config "$CFG" --kernel "${kernel}" --stage profile \
        2>&1 | tee "$LOG_DIR/profile_${kernel}.log"
done

echo "=== Profiling completed at $(date), running aggregate ==="

for kernel in gemm flash_attn fused_moe vector; do
    echo ">>> Aggregating ${kernel}..."
    "$PY" "$BUILD" --config "$CFG" --kernel "${kernel}" --stage aggregate \
        2>&1 | tee "$LOG_DIR/aggregate_${kernel}.log"
done

echo "=== Full pipeline completed at $(date) ==="
