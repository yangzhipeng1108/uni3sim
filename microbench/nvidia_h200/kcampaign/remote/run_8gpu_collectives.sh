#!/bin/bash
# usage: run_8gpu_collectives.sh full|quick
#
# 8-rank H200 collective measurements on vLLM 0.30 (whole machine, GPUs 0-7).
# Re-executes itself under with_gpu_locks.sh so all eight gpu<N>.proc.lock
# are held (ascending) for the whole job; one process tree per GPU, nothing
# else may run on any GPU meanwhile.  Clocks cannot be locked: nvidia-smi
# samples every GPU every 500 ms for the whole job and summarize_clock_log.py
# flags every row that overlaps a busy non-1980 MHz sample.
#
#   quick : one 8-rank tp_collectives pass, tokens {1,512}, 3 trials (~2 min)
#   full  : custom AR probe ws 2/4/8 (same probe as the 4-rank reference),
#           vLLM 0.30 dispatch runner hidden 7168 ws 2/4/8 (auto FlashInfer
#           backend), hidden 7168 ws 8 with the trtllm FlashInfer backend
#           forced (cross-check against the 0.24 trtllm reference), hidden
#           6144 ws 8 (GLM-5.2), then a repeat of custom AR ws 8 and dispatch
#           hidden 7168 ws 8 for repeatability (~10-15 min total)
#
# Output: ${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/collectives_8gpu/<mode>_<stamp>/
set -euo pipefail
PY=${PY:?set PY to the capture runtime Python executable}
export CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to the capture toolkit}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(cd "$HERE/../../../.." && pwd)
if [ "${1:-}" != "--locked" ]; then
  case "${1:-}" in (full|quick) ;; (*) echo "usage: $0 full|quick" >&2; exit 2;; esac
  exec bash "$HERE/with_gpu_locks.sh" 0,1,2,3,4,5,6,7 "${KC_LOCK_WAIT_S:-900}" -- bash "$HERE/run_8gpu_collectives.sh" --locked "$1"
fi
MODE=$2
COMMIT=$(cat "$SRC/DEPLOYED_COMMIT")
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/collectives_8gpu/${MODE}_${STAMP}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1

export XDG_CACHE_HOME=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache
export TRITON_CACHE_DIR=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/triton_coll8
export VLLM_CACHE_ROOT=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/vllm_coll8
export TORCHINDUCTOR_CACHE_DIR=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/inductor_coll8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset VLLM_FLASHINFER_ALLREDUCE_BACKEND

CAR=$SRC/microbench/nvidia_h200/communication/custom_allreduce/custom_allreduce_latency_runner.py
TPC=$SRC/microbench/nvidia_h200/communication/tp_collectives/vllm_tp_collectives_runner.py
echo "[coll8] $(date '+%F %T') mode=$MODE commit=$COMMIT out=$OUT"

# Machine facts (NVML only, no CUDA context).
{
  nvidia-smi
  nvidia-smi topo -m
  nvidia-smi --query-gpu=index,name,pci.bus_id,serial,driver_version,clocks.max.sm,clocks.applications.graphics,fabric.state,fabric.status --format=csv
  nvidia-smi -q | grep -A3 -E '^\s+Fabric$' || echo 'no Fabric section in nvidia-smi -q'
} > "$OUT/machine.txt"
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$NGPU" != "8" ]; then echo "[coll8] FAIL: host shows $NGPU GPUs, need 8" >&2; exit 4; fi
"$PY" -c "import vllm, flashinfer, torch; print('vllm', vllm.__version__, 'flashinfer', flashinfer.__version__, 'torch', torch.__version__)" > "$OUT/versions.txt"

nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu,clocks_event_reasons.active \
  --format=csv,noheader,nounits -lms 500 > "$OUT/clocks.csv" &
CLOCK_PID=$!
trap 'kill $CLOCK_PID 2>/dev/null || true' EXIT

run() {  # run <label> <cmd...>
  echo "[coll8] $(date '+%F %T') start $1"
  local t0=$SECONDS
  "${@:2}"
  echo "[coll8] $(date '+%F %T') done $1 ($((SECONDS - t0)) s)"
}

REPORTS=()
if [ "$MODE" = quick ]; then
  run tpc_quick_h7168_ws8 "$PY" "$TPC" --world-sizes 8 --hidden 7168 --quick \
    --output-dir "$OUT/tpc_quick_h7168" --source-commit "$COMMIT"
  REPORTS+=("$OUT/tpc_quick_h7168/report.json")
else
  run car_ws2_4_8 "$PY" "$CAR" --world-sizes 2 4 8 \
    --output-dir "$OUT/car_ws248" --source-commit "$COMMIT"
  run tpc_h7168_auto_ws2_4_8 "$PY" "$TPC" --world-sizes 2 4 8 --hidden 7168 \
    --output-dir "$OUT/tpc_h7168_auto" --source-commit "$COMMIT"
  run tpc_h7168_trtllm_ws8 env VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
    "$PY" "$TPC" --world-sizes 8 --hidden 7168 --port 29821 \
    --output-dir "$OUT/tpc_h7168_trtllm" --source-commit "$COMMIT"
  run tpc_h6144_auto_ws8 "$PY" "$TPC" --world-sizes 8 --hidden 6144 --port 29841 \
    --output-dir "$OUT/tpc_h6144_auto" --source-commit "$COMMIT"
  run car_ws8_repeat "$PY" "$CAR" --world-sizes 8 --port 29631 \
    --output-dir "$OUT/car_ws8_repeat" --source-commit "$COMMIT"
  run tpc_h7168_auto_ws8_repeat "$PY" "$TPC" --world-sizes 8 --hidden 7168 \
    --port 29831 --output-dir "$OUT/tpc_h7168_auto_repeat" --source-commit "$COMMIT"
  REPORTS+=("$OUT/car_ws248/report.json" "$OUT/tpc_h7168_auto/report.json"
            "$OUT/tpc_h7168_trtllm/report.json" "$OUT/tpc_h6144_auto/report.json"
            "$OUT/car_ws8_repeat/report.json" "$OUT/tpc_h7168_auto_repeat/report.json")
fi

kill $CLOCK_PID; wait $CLOCK_PID 2>/dev/null || true
trap - EXIT
python3 "$HERE/summarize_clock_log.py" --clock-log "$OUT/clocks.csv" \
  --gpus 0,1,2,3,4,5,6,7 --report "${REPORTS[@]}" --output "$OUT/clock_summary.json"
echo "[coll8] $(date '+%F %T') ALL DONE out=$OUT"
