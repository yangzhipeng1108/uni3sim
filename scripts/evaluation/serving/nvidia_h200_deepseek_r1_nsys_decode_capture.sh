#!/usr/bin/env bash
# Capture pure-decode nsys windows of the real vLLM DeepSeek-R1 server on H200
# (nvidia_h200_serve_deepseek_r1.sh; MODEL/TP/MTP/PORT and CUDA_VISIBLE_DEVICES are passed
# through).  Same protocol as nvidia_h200_nsys_pure_decode_capture.sh: each
# concurrency sends exactly C requests (one wave, 8K/1K InferenceX lengths),
# the ~1.5 s window starts after all C prefills drained and ends before the
# shortest request (>= 819 output tokens) finishes, so every captured step is
# decode-only (MTP on: draft + verify).  SM clock / power are logged at 100 ms
# for the 1980 MHz gate.  Run from the operator-selected capture directory.
# Evaluation only.
# Usage: VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random MODEL=.. TP=.. MTP=.. PORT=.. \
#            nvidia_h200_deepseek_r1_nsys_decode_capture.sh <out dir> <C>...
set -euo pipefail
B=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}; D=$B/h200_microbench/dsr1_mtpoff
RUNTIME_PYTHON=${MIDLAYER_H200_RUNTIME_PYTHON:?set MIDLAYER_H200_RUNTIME_PYTHON to the selected runtime executable}
OUT=$1; shift; mkdir -p $OUT
export PATH=$B/tools/nsys_2025.6.1/bin:$PATH
echo "MODEL=$MODEL TP=$TP MTP=$MTP PORT=$PORT CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} routing=${VLLM_MOE_ROUTING_SIMULATION_STRATEGY:-model}" > $OUT/config.txt
nsys profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=repeat-shutdown:$# \
  --trace-fork-before-exec=true -o $OUT/window_%n --force-overwrite true \
  $D/bin/nvidia_h200_serve_deepseek_r1.sh --profiler-config '{"profiler": "cuda"}' > $OUT/server.log 2>&1 &
NSYS=$!
until grep -q "Application startup complete" $OUT/server.log; do
  kill -0 $NSYS 2>/dev/null || { echo "server exited before startup" >&2; exit 1; }
  sleep 10
done
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw --format=csv -lms 100 > $OUT/telemetry.csv &
TEL=$!
for C in "$@"; do
  "$RUNTIME_PYTHON" -m vllm.entrypoints.cli.main bench serve --backend vllm \
    --model dsr1 --tokenizer $MODEL --trust-remote-code --port $PORT \
    --dataset-name random --random-input-len 7373 --random-output-len 922 --random-range-ratio 0.1111111111 \
    --num-prompts $C --max-concurrency $C --ignore-eos > $OUT/load_c$C.log 2>&1 &
  LOAD=$!
  until curl -s --max-time 5 http://localhost:$PORT/metrics | awk -v c=$C '
      /^vllm:num_requests_running/ {r += $NF} /^vllm:num_requests_waiting/ {w += $NF}
      END {exit !(r == c && w == 0)}'; do sleep 0.5; done
  sleep 2
  echo "C=$C start $(date +%T.%N)" >> $OUT/windows.txt
  curl -s --max-time 120 -X POST http://localhost:$PORT/start_profile
  sleep 1.5
  curl -s --max-time 120 -X POST http://localhost:$PORT/stop_profile || true
  echo "C=$C stop $(date +%T.%N)" >> $OUT/windows.txt
  if [ "$C" = "${@: -1}" ]; then
    # repeat-shutdown stops the server after the last range: never wait on
    # the load generator of the final window (it would block forever).
    kill $LOAD 2>/dev/null || true
  else
    wait $LOAD || true
    sleep 5
  fi
done
kill $TEL
# nsys writes every window, then terminates the server; give it the time to
# finish and make sure no server process outlives the capture.
for _ in $(seq 120); do kill -0 $NSYS 2>/dev/null || break; sleep 5; done
kill -INT $NSYS 2>/dev/null || true
wait $NSYS 2>/dev/null || true
pkill -P $$ 2>/dev/null || true
# The API server runs under the nsys launcher, not this shell: stop the vLLM
# server of this capture (served model dsr1 on this port) if it outlived nsys.
pkill -f -- "--served-model-name dsr1 .*--port $PORT" 2>/dev/null || true
