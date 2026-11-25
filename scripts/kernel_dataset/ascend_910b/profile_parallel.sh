#!/usr/bin/env bash
# Parallel msprof-op profiling across the 8 Ascend 910B NPUs.
# Splits each kernel's shape list into 8 chunks, one per NPU device.
#
# Usage (inside the Ascend runtime container):
#   bash scripts/kernel_dataset/ascend_910b/profile_parallel.sh gemm
#   bash scripts/kernel_dataset/ascend_910b/profile_parallel.sh flash_attn
#   bash scripts/kernel_dataset/ascend_910b/profile_parallel.sh fused_moe
#   bash scripts/kernel_dataset/ascend_910b/profile_parallel.sh all
#   PROFILE_RUN=align-fa-rope-YYYYMMDD \
#     bash scripts/kernel_dataset/ascend_910b/profile_parallel.sh align-fa-rope
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
PY=python3
BUILD=scripts/kernel_dataset/ascend_910b/build_dataset.py
CFG=configs/dataset/ascend_910b.yaml
NDEV=8
LOG_DIR="$REPO_ROOT/profiler_output/kernel_dataset/ascend_910b/logs"
mkdir -p "$LOG_DIR"

# Shape counts (workload_configs × layers), used to compute per-device chunks.
declare -A TOTAL=(
    [gemm]=221
    [flash_attn]=306
    [fused_moe]=176
    [vector]=296
)

run_kernel () {
    local kernel=$1
    local total=${TOTAL[$kernel]}
    local chunk=$(( (total + NDEV - 1) / NDEV ))
    local pids=()
    echo "=== $kernel: $total shapes, $chunk/device across $NDEV NPUs ==="
    for dev in $(seq 0 $((NDEV-1))); do
        local start=$(( dev * chunk + 1 ))
        local end=$(( (dev + 1) * chunk ))
        [ "$end" -gt "$total" ] && end=$total
        [ "$start" -gt "$total" ] && continue
        local log="$LOG_DIR/profile_${kernel}_device${dev}.log"
        echo "  dev $dev: shapes $start-$end -> $log"
        ASCEND_RT_VISIBLE_DEVICES=$dev "$PY" "$BUILD" --config "$CFG" \
            --kernel "$kernel" --stage profile \
            --start-idx "$start" --end-idx "$end" --device-id "$dev" \
            > "$log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "At least one Ascend profiling worker failed for $kernel" >&2
        return 1
    fi
    echo "=== $kernel profiling done ==="
}

run_alignment_reprofile () {
    local profile_run=${PROFILE_RUN:?PROFILE_RUN is required for align-fa-rope}
    local alignment_log_dir="$REPO_ROOT/profiler_output/kernel_dataset/ascend_910b/profile_runs/$profile_run/logs"
    local pids=()
    mkdir -p "$alignment_log_dir"
    echo "=== alignment reprofile: 56 FA across $NDEV NPUs ==="
    for dev in $(seq 0 $((NDEV-1))); do
        local fa_start=$(( dev * 7 + 1 ))
        local fa_end=$(( fa_start + 6 ))
        local log="$alignment_log_dir/device${dev}.log"
        echo "  dev $dev: FA $fa_start-$fa_end -> $log"
        "$PY" "$BUILD" --config "$CFG" \
            --kernel flash_attn --stage profile \
            --target-flash-attn-reprofile --profile-run "$profile_run" \
            --start-idx "$fa_start" --end-idx "$fa_end" \
            --device-id "$dev" > "$log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "At least one alignment FA profiling worker failed" >&2
        return 1
    fi
    echo "=== FA reprofile done; collecting 32 RoPE cases serially ==="
    "$PY" "$BUILD" --config "$CFG" \
        --kernel vector --stage profile --target-vector-op rope \
        --profile-run "$profile_run" --device-id 0 \
        > "$alignment_log_dir/rope_device0.log" 2>&1
    echo "=== alignment reprofile done ==="
}

if [ "${1:-}" = "align-fa-rope" ]; then
    run_alignment_reprofile
elif [ "${1:-}" = "all" ]; then
    for k in gemm flash_attn fused_moe vector; do run_kernel "$k"; done
else
    kernel=${1:-}
    if [ -z "$kernel" ] || [ -z "${TOTAL[$kernel]:-}" ]; then
        echo "usage: $0 {gemm|flash_attn|fused_moe|vector|all|align-fa-rope}" >&2
        exit 2
    fi
    run_kernel "$kernel"
fi

echo "ALL DONE. Publish complete datasets with:"
echo "  $PY $BUILD --config $CFG --kernel gemm,flash_attn,fused_moe,vector --stage aggregate"
