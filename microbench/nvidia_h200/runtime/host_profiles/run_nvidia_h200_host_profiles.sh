#!/bin/bash
# usage: run_nvidia_h200_host_profiles.sh <gpuA> <gpuB>        (run ON target host)
#   NO_FLOCK=1   skip `flock -x runs/gpu<N>.proc.lock` (caller already holds both GPUs)
#   DRY=1        print every command, run nothing, create nothing
#   LOCK_WAIT_S  seconds to wait per GPU lock when flocking (default 600)
#   PY, TORCHRUN, MODEL_TOKENIZER, CUDA_HOME must be provided by the caller.
#   VLLM_VERSION selects 0.11.0 or 0.30.0; PYTHONPATH may add caller dependencies.
#
# Per-hardware host profiles the A800-style ShareGPT serving evaluation needs,
# measured sequentially (one GPU process tree at a time, never two processes
# on one GPU), in priority order:
#   1 host_launch            GPU A   torch eager / CUDA Graph replay enqueue (2 repeats)
#   2 scheduler_prefill      GPU A   vLLM 0.11 Scheduler prefill sweep, native platform (rep1 train qps4, rep2 held-out qps16)
#   3 runtime_prepare        GPU A   host_runtime_prepare_runner (needs eager enqueue from step 1)
#   4 collective_prepare     GPU A,B collective_host_prepare_runner, torchrun 2 ranks
#   5 request_path           GPU A visible, host-only: primitives x2, admission/IPC regions x3, delivery region probe
#   6 nccl_ring_protocol     GPU A,B NCCL Ring LL/LL128/Simple + auto, 2 ranks, CUDA-graph nodes
# Every step runs even when an earlier one failed (except those needing the
# eager enqueue value); the exit code is non-zero if any step failed.
# Output: ${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/host_profiles/<stamp>/
set -uo pipefail
if [ "$#" -ne 2 ]; then echo "usage: $0 <gpuA> <gpuB>" >&2; exit 2; fi
A=$1; B=$2
for g in "$A" "$B"; do
  case "$g" in (''|*[!0-9]*) echo "bad gpu id '$g'" >&2; exit 2;; esac
done
if [ "$A" = "$B" ]; then echo "gpuA and gpuB must differ" >&2; exit 2; fi
DRY=${DRY:-0}; NO_FLOCK=${NO_FLOCK:-0}; LOCK_WAIT_S=${LOCK_WAIT_S:-600}

PUBLIC=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}
# The deployed snapshot this script lives in, with SOURCE_COMMIT.
SNAP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
VLLM_VERSION=${VLLM_VERSION:-0.11.0}
case "$VLLM_VERSION" in (0.11.0|0.30.0) ;; (*) echo "VLLM_VERSION must be 0.11.0 or 0.30.0, got '$VLLM_VERSION'" >&2; exit 2;; esac
PY=${PY:?set PY to the runtime Python executable}
TORCHRUN=${TORCHRUN:?set TORCHRUN to the distributed runtime executable}
LOCKS=$PUBLIC/h200_microbench/kcampaign/runs
MODEL_TOKENIZER=${MODEL_TOKENIZER:?set MODEL_TOKENIZER to the tokenizer path}
SCHED_MODEL_CONFIG=$SNAP/configs/model/llama-3.1-8b-instruct.json
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$PUBLIC/h200_microbench/host_profiles/${STAMP}_vllm$VLLM_VERSION
RAW=$OUT/profiler_output/microbench/nvidia_h200
if [ ! -f "$SNAP/SOURCE_COMMIT" ]; then echo "$SNAP has no SOURCE_COMMIT; run the deployed snapshot copy" >&2; exit 2; fi
COMMIT=$(cat "$SNAP/SOURCE_COMMIT")
if ! [[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then echo "bad SOURCE_COMMIT in $SNAP" >&2; exit 2; fi

LAUNCH_RUNNER=microbench/nvidia_h200/runtime/launch/torch_eager_graph_runner.py
SCHED_RUNNER=scripts/measure_external_vllm_scheduler_service.py
PREPARE_RUNNER=microbench/nvidia_a800/runtime/launch/host_runtime_prepare_runner.py
COLL_PREPARE_RUNNER=microbench/nvidia_a800/communication/collectives/collective_host_prepare_runner.py
REQUEST_RUNNER=microbench/nvidia_a800/runtime/host_request_path/vllm_host_request_path_runner.py
DELIVERY_RUNNER=microbench/nvidia_a800/runtime/host_request_path/vllm_delivery_region_runner.py
NCCL_RUNNER=microbench/nvidia_h200/communication/nccl_protocol_sweep/nccl_ring_protocol_runner.py
RUNNERS="$LAUNCH_RUNNER $SCHED_RUNNER $PREPARE_RUNNER $COLL_PREPARE_RUNNER $REQUEST_RUNNER $DELIVERY_RUNNER $NCCL_RUNNER microbench/nvidia_h200/runtime/host_profiles/run_nvidia_h200_host_profiles.sh"

export XDG_CACHE_HOME=$PUBLIC/cache
export TRITON_CACHE_DIR=$PUBLIC/cache/triton_hostprof
export VLLM_CACHE_ROOT=$PUBLIC/cache/vllm_hostprof
export TORCHINDUCTOR_CACHE_DIR=$PUBLIC/cache/inductor_hostprof
export CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to the capture toolkit}
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$SNAP${PYTHONPATH:+:$PYTHONPATH}"
# Each step sets CUDA_VISIBLE_DEVICES itself; host_launch uses physical ids.
unset CUDA_VISIBLE_DEVICES NCCL_ALGO NCCL_PROTO NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE

FAILED=()
say() { echo "[hostprof] $(date '+%F %T') $*"; }

# step <label> <gpu_csv|-> <env assignments...> -- <command...>
# gpu_csv "-" means no lock.  Runs from $SNAP so runner paths are repo-relative.
step() {
  local label=$1 gpus=$2; shift 2
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  local cmd=("$@")
  if [ "$gpus" != "-" ] && [ "$NO_FLOCK" != 1 ]; then
    local g locked=()
    IFS=',' read -r -a glist <<< "$gpus"
    for g in "${glist[@]}"; do locked+=(flock -x -w "$LOCK_WAIT_S" "$LOCKS/gpu$g.proc.lock"); done
    cmd=("${locked[@]}" "${cmd[@]}")
  fi
  if [ "$DRY" = 1 ]; then
    echo "## $label (gpus=$gpus)"
    printf 'cd %q && env' "$SNAP"; printf ' %q' "${envs[@]}" "${cmd[@]}"; printf '\n'
    return 0
  fi
  say "start $label gpus=$gpus"
  local t0=$SECONDS rc
  (cd "$SNAP" && env "${envs[@]}" "${cmd[@]}") > "$OUT/logs/$label.log" 2>&1
  rc=$?
  printf '%s\t%s\t%s\t%s\n' "$label" "$gpus" "$rc" "$((SECONDS - t0))" >> "$OUT/steps.tsv"
  say "done $label rc=$rc ($((SECONDS - t0)) s)"
  if [ "$rc" != 0 ]; then FAILED+=("$label"); tail -20 "$OUT/logs/$label.log"; fi
  return $rc
}

if [ "$DRY" != 1 ]; then
  "$PY" -c 'import importlib.metadata as m, sys; actual=m.version("vllm").split("+", 1)[0]; expected=sys.argv[1]; sys.exit(0 if actual == expected else f"vLLM version {actual!r} does not match selected {expected!r}")' "$VLLM_VERSION" || exit 2
  mkdir -p "$OUT/logs" "$OUT/provenance" "$RAW"
  exec > >(tee -a "$OUT/job.log") 2>&1
  say "gpus=$A,$B no_flock=$NO_FLOCK snapshot=$SNAP commit=$COMMIT out=$OUT"
  printf 'label\tgpus\trc\tseconds\n' > "$OUT/steps.tsv"
  # Provenance: NVML/CPU facts only, no CUDA context.
  P=$OUT/provenance
  echo "$COMMIT" > "$P/source_commit"
  echo "$SNAP" > "$P/snapshot_dir"
  (cd "$SNAP" && sha256sum $RUNNERS configs/model/llama-3.1-8b-instruct.json) > "$P/files.sha256"
  echo "$0 $*" > "$P/command"
  env | sort > "$P/environment.txt"
  hostname > "$P/hostname"; uname -a > "$P/uname"; cat /etc/os-release > "$P/os-release"
  lscpu > "$P/lscpu.txt"; nproc > "$P/nproc"; free -g > "$P/free.txt"; cat /proc/loadavg > "$P/loadavg_start"
  nvidia-smi > "$P/nvidia-smi.txt"; nvidia-smi -q > "$P/nvidia-smi-q.txt"; nvidia-smi topo -m > "$P/nvidia-smi-topo.txt"
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv > "$P/compute_apps_start.csv"
  "$PY" -V > "$P/python_version" 2>&1
  CUDA_VISIBLE_DEVICES="" "$PY" -c 'import importlib.metadata as m, json, torch; print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "nccl": ".".join(map(str, torch.cuda.nccl.version())), **{p: m.version(p) for p in ("vllm", "transformers", "nvidia-nccl-cu12", "pyzmq", "uvicorn", "zstandard")}}, indent=1, sort_keys=True))' > "$P/versions.json" 2> "$P/versions.err"
  "$PY" -m pip freeze > "$P/pip_freeze.txt" 2>/dev/null
  nvidia-smi -i "$A,$B" --query-gpu=timestamp,index,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader,nounits -lms 500 > "$OUT/clocks.csv" &
  CLOCK_PID=$!
  trap 'kill $CLOCK_PID 2>/dev/null || true' EXIT
fi

# 1 host launch (physical device id; the runner checks name "NVIDIA H200" and idleness via nvidia-smi -i).
for r in 1 2; do
  step host_launch_r$r "$A" -- "$PY" "$LAUNCH_RUNNER" --output-dir "$RAW/host_launch_r$r" --device-id "$A" \
    --baseline-commit "$COMMIT" --expected-source-sha256 "$( [ "$DRY" = 1 ] && echo '<sha256 of runner>' || sha256sum "$SNAP/$LAUNCH_RUNNER" | cut -d' ' -f1)"
done
EAGER_US=""
if [ "$DRY" = 1 ]; then
  EAGER_US='<eager:512 median ns/1000 from host_launch_r1>'
elif [ -f "$RAW/host_launch_r1/manifest.json" ]; then
  EAGER_US=$("$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); assert d["status"] == "complete", d["status"]; print(repr(d["summary"]["eager:512"]["median"] / 1000.0))' "$RAW/host_launch_r1/manifest.json") || EAGER_US=""
  say "eager enqueue (eager:512 median) = ${EAGER_US:-MISSING} us"
  echo "$EAGER_US" > "$OUT/eager_enqueue_us.txt"
fi

# 2 scheduler prefill sweep: the A800 20260918 control workload, native platform.
for r in 1 2; do
  qps=$([ "$r" = 1 ] && echo 4 || echo 16)
  step scheduler_prefill_rep$r "$A" CUDA_VISIBLE_DEVICES="$A" -- "$PY" "$SCHED_RUNNER" --hardware-id nvidia_h200 \
    --model-config "$SCHED_MODEL_CONFIG" --output "$OUT/scheduler_host_prefill/sched_prefill_nvidia_h200_rep$r.json" \
    --sample-count 6000 --max-batched-tokens 8192 --max-requests 8 --num-kv-blocks 16384 --kv-block-size-tokens 16 \
    --max-model-len 8192 --prompt-tokens 729 --output-tokens 6 --platform native --enable-prefix-caching --log-stats \
    --admission-batch-sizes 1 2 4 8 --trace-root "$OUT/scheduler_host_prefill/dataset_root" \
    --trace-model-id scheduler-microbench --trace-case-id prefill_sweep_rep$r --trace-target-qps $qps --trace-phases prefill
done

# 3/4 runtime prepare (1 GPU) and collective host prepare (2 GPUs) need the eager enqueue value.
if [ -n "$EAGER_US" ]; then
  step runtime_prepare_r1 "$A" CUDA_VISIBLE_DEVICES="$A" -- "$PY" "$PREPARE_RUNNER" --hardware-id nvidia_h200 \
    --device-id 0 --eager-enqueue-us "$EAGER_US" --trials 15 --output-dir "$RAW/host_runtime_prepare_r1"
  step collective_prepare_r1 "$A,$B" CUDA_VISIBLE_DEVICES="$A,$B" -- "$TORCHRUN" --standalone --nproc-per-node 2 \
    "$COLL_PREPARE_RUNNER" --hardware-id nvidia_h200 --eager-enqueue-us "$EAGER_US" \
    --output-dir "$RAW/collective_host_prepare_r1"
else
  say "SKIP runtime_prepare_r1 and collective_prepare_r1: no eager enqueue value"; FAILED+=(runtime_prepare_r1 collective_prepare_r1)
fi

# 5 request path: host-only (no model, no device work); GPU A stays visible for vLLM platform detection.
for r in 1 2; do
  step request_path_primitives_r$r "$A" CUDA_VISIBLE_DEVICES="$A" -- "$PY" "$REQUEST_RUNNER" --hardware-id nvidia_h200 \
    --tokenizer "$MODEL_TOKENIZER" --trials 120 --warmup 15 --output "$RAW/host_request_path_r$r/manifest.json"
done
for r in 1 2 3; do
  step request_path_admission_ipc_r$r "$A" CUDA_VISIBLE_DEVICES="$A" -- "$PY" "$REQUEST_RUNNER" --hardware-id nvidia_h200 \
    --tokenizer "$MODEL_TOKENIZER" --mode admission_ipc_regions --region-prompt-tokens 86 --trials 200 --warmup 20 \
    --output "$RAW/admission_ipc_region_r$r/manifest.json"
done
step request_path_delivery_region "$A" CUDA_VISIBLE_DEVICES="" -- "$PY" "$DELIVERY_RUNNER" "$MODEL_TOKENIZER" \
  1,2,4,8,16,32,64 40 3 "$RAW/output_delivery_region_r1.json"

# 6 NCCL Ring per-protocol two-rank sweep (ProtoBridge TP2 inputs).
step nccl_ring_protocol_ws2 "$A,$B" CUDA_VISIBLE_DEVICES="$A,$B" -- "$PY" "$NCCL_RUNNER" \
  --output-dir "$RAW/nccl_ring_protocol_ws2" --source-commit "$COMMIT"

if [ "$DRY" = 1 ]; then exit 0; fi
kill $CLOCK_PID 2>/dev/null; wait $CLOCK_PID 2>/dev/null; trap - EXIT
cat /proc/loadavg > "$OUT/provenance/loadavg_end"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv > "$OUT/provenance/compute_apps_end.csv"
(cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name job.log -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
if [ "${#FAILED[@]}" -ne 0 ]; then say "FAILED steps: ${FAILED[*]} out=$OUT"; exit 1; fi
say "ALL DONE out=$OUT"
